from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "C324"
EXPERIMENT_SLUG = "C324_compressed_reference_sft_gate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C324_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C324 compressed-reference SFT target gate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-tokens", type=int, default=160)
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
    values = sorted(values)

    def q(p: float) -> float:
        if len(values) == 1:
            return float(values[0])
        pos = (len(values) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return float(values[lo])
        return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))

    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "std_over_mean": std / mean if mean else None,
        "min": min(values),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": max(values),
    }


def compressed_token_len(text: str, target_tokens: int) -> tuple[int, bool]:
    text = str(text).strip()
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= target_tokens:
        return len(tokens), False
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    kept: list[str] = []
    kept_len = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = word_count(sentence)
        if kept and kept_len + sentence_len > target_tokens:
            break
        if not kept and sentence_len > target_tokens:
            return target_tokens, True
        kept.append(sentence)
        kept_len += sentence_len
    if kept:
        return kept_len, True
    return target_tokens, True


def run_audit(target_tokens: int) -> dict[str, Any]:
    import pandas as pd

    data = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    data = data.rename(columns={"query": "question", "answer": "reference_answer"})
    data = data.dropna(subset=["question", "reference_answer"]).copy()

    original_lengths: list[int] = []
    compressed_lengths: list[int] = []
    changed = 0
    severe_compression = 0
    too_short_after = 0
    cap_risk_after = 0
    multiline_after = 0
    list_like_after = 0

    for _, row in data.iterrows():
        reference = str(row["reference_answer"])
        original_len = word_count(reference)
        compressed_len, was_changed = compressed_token_len(reference, target_tokens)
        original_lengths.append(original_len)
        compressed_lengths.append(compressed_len)
        changed += int(was_changed)
        severe_compression += int(was_changed and compressed_len < max(20, original_len * 0.35))
        too_short_after += int(compressed_len <= 12)
        cap_risk_after += int(compressed_len > 220)
        multiline_after += int("\n" in reference and compressed_len > 20)
        list_like_after += int(bool(re.search(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", reference)) and compressed_len > 20)

    rows = int(len(data))
    gate = {
        "rows_at_least_1000": rows >= 1000,
        "compressed_p90_under_180": (stats(compressed_lengths).get("p90") or 0) <= 180,
        "too_short_rate_under_10pct": (too_short_after / rows) < 0.10 if rows else False,
        "severe_compression_rate_under_20pct": (severe_compression / rows) < 0.20 if rows else False,
        "cap_risk_under_5pct": (cap_risk_after / rows) < 0.05 if rows else False,
        "no_raw_artifact": True,
    }
    gate["decision_recommendation"] = "MUTATE" if all(gate.values()) else "KILL"

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
        "target_tokens": target_tokens,
        "original_reference_tokens": stats(original_lengths),
        "compressed_reference_tokens": stats(compressed_lengths),
        "compression_counts": {
            "changed": int(changed),
            "unchanged": int(rows - changed),
            "severe_compression": int(severe_compression),
            "too_short_after": int(too_short_after),
            "cap_risk_after": int(cap_risk_after),
            "multiline_preserved_proxy": int(multiline_after),
            "list_like_preserved_proxy": int(list_like_after),
        },
        "recommended_next": {
            "experiment": "C325",
            "mechanism": "tiny compressed-reference SFT smoke",
            "gpu_preference": "L4_or_T4",
            "train_rows": 96,
            "validation_rows": 24,
            "target_tokens": target_tokens,
            "validation_max_new_tokens": 224,
            "kill_gate": "kill unless both ref-in-output and output-in-ref improve without cap/invalid/repetition worsening",
        },
        "gate": gate,
    }


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    report = f"""# C324 Compressed-Reference SFT Gate

## Objective
- Continue the S3/global distribution-alignment route after full-reference SFT cap saturation.
- Audit sentence-bounded compressed reference targets without returning raw task data.
- No leaderboard submission, model load, training, adapter, raw prompt, raw reference, row id, or output artifact.

## Result
- status: `{summary.get("status")}`
- decision recommendation: `{summary.get("gate", {}).get("decision_recommendation")}`
- rows: `{summary.get("rows")}`
- target tokens: `{summary.get("target_tokens")}`

## Gate
`{json.dumps(summary.get("gate", {}), ensure_ascii=False)}`

## Reference Lengths
- original: `{json.dumps(summary.get("original_reference_tokens", {}), ensure_ascii=False)}`
- compressed: `{json.dumps(summary.get("compressed_reference_tokens", {}), ensure_ascii=False)}`

## Compression Counts
`{json.dumps(summary.get("compression_counts", {}), ensure_ascii=False)}`

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
            "target_tokens": args.target_tokens,
            "gate": {"decision_recommendation": "DRY_RUN"},
        }
    else:
        try:
            summary = run_audit(args.target_tokens)
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
