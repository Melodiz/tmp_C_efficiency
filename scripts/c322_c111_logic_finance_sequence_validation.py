from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as io
import c258_c111_family_stratified_validation as base


EXPERIMENT_ID = "C322"
EXPERIMENT_SLUG = "C322_c111_logic_finance_sequence_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C322_artifacts"
DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_SEED = 322

FAMILY_PATTERNS = {
    "logic_table": [
        r"\b(?:truth table|logic|logical|boolean|statement)\b",
        r"(?:таблиц[аы]?\s+истин|логик|булев|высказыван|истинн|ложн)",
    ],
    "finance_percent": [
        r"\b(?:percent|discount|interest|price|cost|currency|dollar|euro)\b",
        r"(?:процент|скидк|стоимост|цена|рубл|доллар|евро|валют|прибыл|процентн)",
    ],
    "base_number_system": [
        r"\b(?:binary|hexadecimal|octal|base\s*\d+|number system)\b",
        r"(?:двоичн|шестнадцатеричн|восьмеричн|систем[аеы]?\s+счислен|основани[ея]\s+\d+)",
    ],
    "sequence_progression": [
        r"\b(?:sequence|series|progression|next term|arithmetic progression|geometric progression)\b",
        r"(?:последовательн|прогресси|следующ(?:ее|ий|ая)?\s+числ|член\s+последовательности|ряд\s+чисел)",
    ],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C322 C111 logic/finance/sequence family validation.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def configure_base() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    base.DEFAULT_SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE
    base.DEFAULT_SEED = DEFAULT_SEED
    base.FAMILY_PATTERNS = FAMILY_PATTERNS


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C322 C111 Logic/Finance/Sequence Validation",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Measure proven C111 quality and visible failures for the remaining smaller C257 coherent families before any port.",
        "- Return only aggregate metrics; no raw prompts, references, outputs, row ids, datasets, weights, or adapter files.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- imports: `{summary.get('imports')}`",
        "",
        "## Sample",
        f"`{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Overall Quality",
        f"`{summary.get('overall_quality')}`",
        "",
        "## Overall Validity",
        f"`{summary.get('overall_validity')}`",
        "",
        "## Family Counts",
        f"`{summary.get('family_counts')}`",
        "",
        "## Weak Family Summary",
        f"`{summary.get('weak_family_summary')}`",
        "",
        "## Family Quality",
        f"`{summary.get('family_quality')}`",
        "",
        "## Family Validity",
        f"`{summary.get('family_validity')}`",
        "",
        "## Family Categories",
        f"`{summary.get('family_categories')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- outputs returned: `{summary.get('outputs_returned')}`",
        f"- model weights returned: `{summary.get('model_weights_returned')}`",
        f"- training started: `{summary.get('training_started')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    configure_base()
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    summary = base.run_validation(args)
    summary["reason"] = "C111 logic/finance/sequence/base-number family validation completed."
    io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    io.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
