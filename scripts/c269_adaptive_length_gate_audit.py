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


EXPERIMENT_ID = "C269"
EXPERIMENT_SLUG = "C269_adaptive_length_gate_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C269_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C269 S2 adaptive-length zero-GPU gate audit.")
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


def question_features(question: str) -> dict[str, Any]:
    text = str(question)
    q = text.lower().replace("ё", "е")
    chars = len(q)
    tokens = word_count(q)
    digit = bool(re.search(r"\d", q))
    expr = bool(re.search(r"[=<>]|(?:\d\s*[+*/:×-]\s*\d)", q))
    open_cue = bool(
        re.search(
            r"(объясн|почему|зачем|опиши|определ[ие]|охарактериз|расскаж|"
            r"сочин|эссе|напиши|перечисл|сравн|докаж|обосну|аргумент|"
            r"проанализ|describe|explain|why|write|essay|list|compare|prove)",
            q,
        )
    )
    subject_long_cue = bool(
        re.search(
            r"(литератур|истори|биолог|географ|обществозн|текст|стихотвор|"
            r"геро[йя]|автор|произведен|эколог|культура)",
            q,
        )
    )
    closed_numeric_cue = bool(
        digit
        and re.search(r"(найд|вычисл|сколько|реши|уравнен|периметр|площад|процент|calculate|find|solve)", q)
    )
    route_long = bool(
        open_cue
        or subject_long_cue and chars >= 90
        or chars >= 260 and not closed_numeric_cue
        or tokens >= 45 and not expr
    )
    conservative_long = bool(open_cue or (chars >= 350 and not closed_numeric_cue))
    return {
        "chars": chars,
        "tokens": tokens,
        "digit": digit,
        "expr": expr,
        "open_cue": open_cue,
        "subject_long_cue": subject_long_cue,
        "closed_numeric_cue": closed_numeric_cue,
        "route_long": route_long,
        "conservative_long": conservative_long,
    }


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


def pct(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else 0.0


def run_audit() -> dict[str, Any]:
    import pandas as pd

    data = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    data = data.rename(columns={"query": "question", "answer": "reference_answer"})
    data = data.dropna(subset=["question", "reference_answer"]).copy()

    rows = int(len(data))
    result: dict[str, Any] = {
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
    }

    ref_tokens: list[int] = []
    route_tokens: defaultdict[str, list[int]] = defaultdict(list)
    feature_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    route_long_by_ref_band: Counter[str] = Counter()
    all_ref_band: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()

    for _, row in data.iterrows():
        question = str(row["question"])
        reference = str(row["reference_answer"])
        feats = question_features(question)
        rtoks = word_count(reference)
        ref_tokens.append(rtoks)

        ref_band = "ref_tiny" if rtoks <= 12 else "ref_short" if rtoks <= 60 else "ref_medium" if rtoks <= 160 else "ref_long" if rtoks <= 300 else "ref_very_long"
        all_ref_band[ref_band] += 1

        for name in ["digit", "expr", "open_cue", "subject_long_cue", "closed_numeric_cue"]:
            feature_counts[name] += int(feats[name])

        for route_name in ["route_long", "conservative_long"]:
            if feats[route_name]:
                route_counts[route_name] += 1
                route_tokens[route_name].append(rtoks)
                if route_name == "route_long":
                    route_long_by_ref_band[ref_band] += 1

        combo = "|".join(
            [
                "open" if feats["open_cue"] else "no_open",
                "subject" if feats["subject_long_cue"] else "no_subject",
                "numeric" if feats["digit"] else "nonnum",
                "expr" if feats["expr"] else "noexpr",
                "q_long" if feats["chars"] >= 180 else "q_shortish",
                "route" if feats["route_long"] else "fallback",
            ]
        )
        combo_counts[combo] += 1

    ref_long_160 = sum(1 for value in ref_tokens if value >= 160)
    ref_long_240 = sum(1 for value in ref_tokens if value >= 240)
    routed = route_counts["route_long"]
    conservative = route_counts["conservative_long"]
    routed_ref_long_160 = sum(1 for value in route_tokens["route_long"] if value >= 160)
    routed_ref_long_240 = sum(1 for value in route_tokens["route_long"] if value >= 240)

    route_share = pct(routed, rows)
    conservative_share = pct(conservative, rows)
    precision_160 = pct(routed_ref_long_160, routed)
    capture_160 = pct(routed_ref_long_160, ref_long_160)
    precision_240 = pct(routed_ref_long_240, routed)
    capture_240 = pct(routed_ref_long_240, ref_long_240)

    gate_pass = bool(
        route_share >= 0.05
        and routed <= int(rows * 0.45)
        and precision_160 >= 0.50
        and capture_160 >= 0.08
    )

    result.update(
        {
            "reference_token_stats": stats(ref_tokens),
            "feature_counts": dict(feature_counts),
            "reference_length_bands": dict(all_ref_band),
            "routes": {
                "route_long": {
                    "rows": int(routed),
                    "share": route_share,
                    "reference_token_stats": stats(route_tokens["route_long"]),
                    "ref_ge_160": int(routed_ref_long_160),
                    "ref_ge_160_precision": precision_160,
                    "ref_ge_160_capture": capture_160,
                    "ref_ge_240": int(routed_ref_long_240),
                    "ref_ge_240_precision": precision_240,
                    "ref_ge_240_capture": capture_240,
                    "reference_length_bands": dict(route_long_by_ref_band),
                },
                "conservative_long": {
                    "rows": int(conservative),
                    "share": conservative_share,
                    "reference_token_stats": stats(route_tokens["conservative_long"]),
                },
            },
            "top_feature_combos": [
                {"combo": combo, "rows": int(count), "share": pct(count, rows)}
                for combo, count in combo_counts.most_common(20)
            ],
            "gate": {
                "s2_gate1_pass": gate_pass,
                "min_route_share_5pct": route_share >= 0.05,
                "max_route_share_45pct": routed <= int(rows * 0.45),
                "precision_ref_ge_160_at_least_50pct": precision_160 >= 0.50,
                "capture_ref_ge_160_at_least_8pct": capture_160 >= 0.08,
                "decision_recommendation": "MUTATE" if gate_pass else "KILL",
                "next_if_pass": "Run paired C111 adaptive max_tokens route vs C111 baseline on 512 rows.",
            },
        }
    )
    return result


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    gate = summary.get("gate", {})
    report = f"""# C269 Adaptive-Length Gate Audit

## Objective
- Start S2 category-adaptive length targeting.
- Zero-GPU remote-only audit of an inference-visible long-answer route.
- Return aggregate metrics only; no raw prompts, references, row ids, outputs, datasets, weights, or adapters.

## Result
- status: `{summary.get("status")}`
- decision recommendation: `{gate.get("decision_recommendation")}`
- rows: `{summary.get("rows")}`

## Gate
`{json.dumps(gate, ensure_ascii=False)}`

## Reference Length
`{json.dumps(summary.get("reference_token_stats", {}), ensure_ascii=False)}`

## Feature Counts
`{json.dumps(summary.get("feature_counts", {}), ensure_ascii=False)}`

## Reference Length Bands
`{json.dumps(summary.get("reference_length_bands", {}), ensure_ascii=False)}`

## Route Metrics
`{json.dumps(summary.get("routes", {}), ensure_ascii=False)}`

## Top Feature Combos
`{json.dumps(summary.get("top_feature_combos", []), ensure_ascii=False)}`

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
        except Exception as exc:  # pragma: no cover - artifact path for remote diagnostics
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
