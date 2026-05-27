from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "C271"
EXPERIMENT_SLUG = "C271_full_reference_sft_gate_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C271_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C271 S3 full-reference SFT gate audit.")
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


def run_audit() -> dict[str, Any]:
    import pandas as pd

    data = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    data = data.rename(columns={"query": "question", "answer": "reference_answer"})
    data = data.dropna(subset=["question", "reference_answer"]).copy()

    question_tokens: list[int] = []
    reference_tokens: list[int] = []
    combined_tokens: list[int] = []
    script_counts: Counter[str] = Counter()
    length_bands: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()

    for _, row in data.iterrows():
        question = str(row["question"])
        reference = str(row["reference_answer"])
        qtoks = word_count(question)
        rtoks = word_count(reference)
        question_tokens.append(qtoks)
        reference_tokens.append(rtoks)
        combined_tokens.append(qtoks + rtoks)

        cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in question)
        lat = sum("a" <= ch.lower() <= "z" for ch in question)
        script_counts["cyrillic" if cyr > lat else "latin" if lat > cyr else "mixed_or_symbolic"] += 1
        length_bands["tiny"] += int(rtoks <= 12)
        length_bands["short"] += int(12 < rtoks <= 60)
        length_bands["medium"] += int(60 < rtoks <= 160)
        length_bands["long"] += int(160 < rtoks <= 300)
        length_bands["very_long"] += int(rtoks > 300)
        structure_counts["multiline"] += int("\n" in reference)
        structure_counts["list_like"] += int(bool(re.search(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", reference)))
        structure_counts["equation_like"] += int(bool(re.search(r"[=<>]|(?:\d\s*[+*/:×-]\s*\d)", reference)))

    rows = int(len(data))
    train_rows = int(rows * 0.8)
    val_rows = rows - train_rows
    estimated_adapter_mb = {
        "rank_16": 180,
        "rank_32": 360,
        "note": "Order-of-magnitude LoRA-only estimate for Qwen3-8B target modules; excludes base weights already packaged.",
    }
    gate = {
        "rows_at_least_1000": rows >= 1000,
        "validation_rows_at_least_500": val_rows >= 500,
        "combined_p90_under_900_tokens": (stats(combined_tokens).get("p90") or 0) < 900,
        "estimated_adapter_fits_package_budget": estimated_adapter_mb["rank_32"] < 1024,
        "no_raw_artifact": True,
    }
    gate["s3_gate1_pass"] = all(gate.values())
    gate["decision_recommendation"] = "MUTATE" if gate["s3_gate1_pass"] else "KILL"

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
        "planned_split": {"train_rows": train_rows, "validation_rows": val_rows},
        "question_token_stats": stats(question_tokens),
        "reference_token_stats": stats(reference_tokens),
        "combined_token_stats": stats(combined_tokens),
        "script_counts": dict(script_counts),
        "reference_length_bands": dict(length_bands),
        "reference_structure_counts": dict(structure_counts),
        "recommended_sft_smoke": {
            "training_target": "full_reference_answer",
            "rank": 16,
            "epochs": 1,
            "learning_rate": "2e-4",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "forbidden_modules": ["embed_tokens", "lm_head"],
            "validation_rows": 96,
            "artifact_policy": "aggregate-only; delete adapter scratch; return no weights/raw rows",
        },
        "estimated_adapter_mb": estimated_adapter_mb,
        "gate": gate,
    }


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    report = f"""# C271 Full-Reference SFT Gate Audit

## Objective
- Start S3 full-reference SFT with a zero-GPU remote-only feasibility gate.
- Audit training-table scale, length budget, split feasibility, and adapter package feasibility.
- Return aggregate metrics only; no raw prompts, references, outputs, row ids, datasets, model weights, or adapters.

## Result
- status: `{summary.get("status")}`
- decision recommendation: `{summary.get("gate", {}).get("decision_recommendation")}`
- rows: `{summary.get("rows")}`
- planned split: `{summary.get("planned_split")}`

## Gate
`{json.dumps(summary.get("gate", {}), ensure_ascii=False)}`

## Token Stats
- question: `{json.dumps(summary.get("question_token_stats", {}), ensure_ascii=False)}`
- reference: `{json.dumps(summary.get("reference_token_stats", {}), ensure_ascii=False)}`
- combined: `{json.dumps(summary.get("combined_token_stats", {}), ensure_ascii=False)}`

## Aggregate Structure
- scripts: `{json.dumps(summary.get("script_counts", {}), ensure_ascii=False)}`
- reference length bands: `{json.dumps(summary.get("reference_length_bands", {}), ensure_ascii=False)}`
- reference structure counts: `{json.dumps(summary.get("reference_structure_counts", {}), ensure_ascii=False)}`

## Recommended SFT Smoke
`{json.dumps(summary.get("recommended_sft_smoke", {}), ensure_ascii=False)}`

## Package Estimate
`{json.dumps(summary.get("estimated_adapter_mb", {}), ensure_ascii=False)}`

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
        except Exception as exc:  # pragma: no cover
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
