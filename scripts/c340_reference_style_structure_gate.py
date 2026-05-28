from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import c269_adaptive_length_gate_audit as c269


EXPERIMENT_ID = "C340"
EXPERIMENT_SLUG = "C340_reference_style_structure_gate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C340_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C340 S1 reference-style structure gate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "zip": out_dir.with_suffix(".zip"),
    }


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return float(ordered[lo])
        return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))

    mean = statistics.fmean(ordered)
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "count": len(ordered),
        "mean": mean,
        "std": std,
        "std_over_mean": std / mean if mean else None,
        "min": min(ordered),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": max(ordered),
    }


def style_flags(question: str, reference: str) -> dict[str, bool]:
    ref = str(reference).strip()
    ref_l = ref.lower().replace("ё", "е")
    feats = c269.question_features(str(question))
    return {
        "answer_marker_start": bool(re.match(r"^\s*(ответ|answer)\s*[:：-]", ref_l)),
        "answer_marker_any": bool(re.search(r"(^|\n)\s*(ответ|answer)\s*[:：-]", ref_l)),
        "multiline": "\n" in ref,
        "list_like": bool(re.search(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", ref)),
        "formula_or_equation": bool(re.search(r"[=<>≤≥±√]|\\frac|\\sqrt|\b(?:sin|cos|tg|log)\b", ref_l)),
        "numeric_heavy": bool(re.search(r"\d", ref)) and len(re.findall(r"\d", ref)) >= 2,
        "short_answer": word_count(ref) <= 12,
        "medium_answer": 13 <= word_count(ref) <= 160,
        "long_answer": word_count(ref) > 160,
        "formal_russian_marker": bool(
            re.search(
                r"\b(следовательно|таким образом|данн(?:ое|ый|ая)|получаем|имеем|решение|поскольку|отсюда)\b",
                ref_l,
            )
        ),
        "hedging_marker": bool(re.search(r"\b(возможно|вероятно|примерно|около|probably|maybe|approximately)\b", ref_l)),
        "first_person_marker": bool(re.search(r"\b(я считаю|я думаю|i think|i believe)\b", ref_l)),
        "open_question": bool(feats["open_cue"]),
        "numeric_question": bool(feats["digit"]),
        "expr_question": bool(feats["expr"]),
        "route_long_question": bool(feats["route_long"]),
    }


def bucket_name(question: str) -> str:
    feats = c269.question_features(question)
    return "|".join(
        [
            "open" if feats["open_cue"] else "closed",
            "subject" if feats["subject_long_cue"] else "general",
            "num" if feats["digit"] else "nonnum",
            "expr" if feats["expr"] else "noexpr",
            "qlong" if feats["chars"] >= 180 else "qshort",
            "routelong" if feats["route_long"] else "routebase",
        ]
    )


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def summarize_counter(counter: Counter[str], rows: int) -> dict[str, Any]:
    return {name: {"count": int(value), "rate": rate(int(value), rows)} for name, value in sorted(counter.items())}


def top_bucket_summary(rows_by_bucket: dict[str, int], flags_by_bucket: dict[str, Counter[str]], lengths_by_bucket: dict[str, list[int]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket, rows in sorted(rows_by_bucket.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        flags = flags_by_bucket[bucket]
        style_rates = {name: rate(int(value), rows) for name, value in sorted(flags.items())}
        dominant = sorted(style_rates.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        out[bucket] = {
            "rows": int(rows),
            "length_stats": stats(lengths_by_bucket[bucket]),
            "dominant_style_rates": {name: value for name, value in dominant},
        }
    return out


def run_audit() -> dict[str, Any]:
    import pandas as pd

    data = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    data = data.rename(columns={"query": "question", "answer": "reference_answer"})
    data = data.dropna(subset=["question", "reference_answer"]).copy()

    rows = int(len(data))
    flag_counts: Counter[str] = Counter()
    lengths: list[int] = []
    rows_by_bucket: Counter[str] = Counter()
    flags_by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    lengths_by_bucket: defaultdict[str, list[int]] = defaultdict(list)
    broad_stable_strata: list[dict[str, Any]] = []

    for _, row in data.iterrows():
        question = str(row["question"])
        reference = str(row["reference_answer"])
        length = word_count(reference)
        lengths.append(length)
        bucket = bucket_name(question)
        rows_by_bucket[bucket] += 1
        lengths_by_bucket[bucket].append(length)
        flags = style_flags(question, reference)
        for name, value in flags.items():
            if value:
                flag_counts[name] += 1
                flags_by_bucket[bucket][name] += 1

    for bucket, bucket_rows in rows_by_bucket.items():
        if bucket_rows < max(100, int(rows * 0.02)):
            continue
        for name, count in flags_by_bucket[bucket].items():
            r = rate(int(count), int(bucket_rows))
            if r >= 0.55:
                broad_stable_strata.append({"bucket": bucket, "rows": int(bucket_rows), "feature": name, "rate": r})

    length_summary = stats(lengths)
    heterogeneity_gate = (length_summary.get("std_over_mean") or 0.0) <= 2.5
    global_marker_gate = any(rate(int(flag_counts[name]), rows) >= 0.30 for name in ["answer_marker_start", "multiline", "list_like", "formal_russian_marker"])
    bucket_strata_gate = len(broad_stable_strata) > 0
    gate = {
        "rows_at_least_1000": rows >= 1000,
        "length_heterogeneity_not_extreme": heterogeneity_gate,
        "global_style_feature_over_30pct": global_marker_gate,
        "bucket_stable_style_strata_exist": bucket_strata_gate,
        "no_raw_artifact": True,
    }
    gate["decision_recommendation"] = "MUTATE" if gate["rows_at_least_1000"] and bucket_strata_gate else "KILL"

    return {
        "status": "completed",
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": True,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "prompts_returned": False,
        "references_returned": False,
        "outputs_returned": False,
        "model_loaded": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "rows": rows,
        "reference_length_stats": length_summary,
        "global_style_features": summarize_counter(flag_counts, rows),
        "top_buckets": top_bucket_summary(dict(rows_by_bucket), flags_by_bucket, lengths_by_bucket),
        "broad_stable_strata": sorted(broad_stable_strata, key=lambda item: (-item["rows"], -item["rate"], item["feature"]))[:30],
        "gate": gate,
        "recommended_next": {
            "if_mutate": "Prototype only the broadest stable style feature, preserving C111 content and avoiding shortening/replacement.",
            "if_kill": "Park deterministic S1 postprocessing and return to S3 only with a materially different training harness.",
        },
    }


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    report = f"""# C340 S1 Reference-Style Structure Gate

## Objective
- Complete formal S1 Gate 1: reference answer structure/style analysis.
- Remote-only aggregate audit; no raw prompts, references, outputs, row ids, datasets, adapters, or weights returned.

## Result
- status: `{summary.get("status")}`
- decision recommendation: `{summary.get("gate", {}).get("decision_recommendation")}`
- rows: `{summary.get("rows")}`

## Gate
`{json.dumps(summary.get("gate", {}), ensure_ascii=False)}`

## Reference Length Stats
`{json.dumps(summary.get("reference_length_stats", {}), ensure_ascii=False)}`

## Global Style Features
`{json.dumps(summary.get("global_style_features", {}), ensure_ascii=False)}`

## Broad Stable Strata
`{json.dumps(summary.get("broad_stable_strata", []), ensure_ascii=False)}`

## Top Buckets
`{json.dumps(summary.get("top_buckets", {}), ensure_ascii=False)}`

## Recommended Next
`{json.dumps(summary.get("recommended_next", {}), ensure_ascii=False)}`

## Hygiene
- raw task data read remote only: `{summary.get("raw_task_data_read_remote_only")}`
- raw examples returned: `{summary.get("raw_examples_returned")}`
- row ids returned: `{summary.get("row_ids_returned")}`
- prompts returned: `{summary.get("prompts_returned")}`
- references returned: `{summary.get("references_returned")}`
- outputs returned: `{summary.get("outputs_returned")}`
- model loaded: `{summary.get("model_loaded")}`
- training started: `{summary.get("training_started")}`
- adapter weights returned: `{summary.get("adapter_weights_returned")}`
"""
    report_path.write_text(report, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        summary = {
            "status": "dry_run",
            "leaderboard_submission": False,
            "raw_task_data_read_remote_only": True,
            "raw_examples_returned": False,
            "row_ids_returned": False,
            "prompts_returned": False,
            "references_returned": False,
            "outputs_returned": False,
            "model_loaded": False,
            "training_started": False,
            "adapter_weights_returned": False,
            "rows": 0,
            "gate": {"decision_recommendation": "DRY_RUN"},
        }
    else:
        try:
            summary = run_audit()
        except Exception as exc:
            summary = {
                "status": "failed",
                "leaderboard_submission": False,
                "raw_task_data_read_remote_only": False,
                "raw_examples_returned": False,
                "row_ids_returned": False,
                "prompts_returned": False,
                "references_returned": False,
                "outputs_returned": False,
                "model_loaded": False,
                "training_started": False,
                "adapter_weights_returned": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
                "gate": {"decision_recommendation": "INVESTIGATE"},
            }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, paths["report"])
    if paths["zip"].exists():
        paths["zip"].unlink()
    shutil.make_archive(str(paths["out_dir"]), "zip", paths["out_dir"])
    return 0 if summary.get("status") in {"completed", "dry_run"} else 1


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
