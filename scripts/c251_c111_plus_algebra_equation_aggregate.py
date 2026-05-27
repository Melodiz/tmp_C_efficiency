from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import c201_c111_vs_current_stack_aggregate as rollback
import c243_c111_plus_formulaic_aggregate as base

_BASE_RUN_VALIDATION = base.run_validation


def c111_plus_algebra_stack(current: Any, c111: Any, question: str, answer: str) -> tuple[str, str]:
    c111_answer, c111_handler = rollback.c111_stack(c111, question, answer)
    if c111_handler != "fallback_model":
        return c111_answer, c111_handler
    algebra = current.algebra_equation_answer(question)
    if algebra is not None:
        return algebra, "algebra_equation"
    return c111_answer, "fallback_model"


def run_validation(args: Any) -> dict[str, Any]:
    base.c111_plus_formulaic_stack = c111_plus_algebra_stack
    summary = _BASE_RUN_VALIDATION(args)
    summary["experiment_id"] = base.EXPERIMENT_ID
    summary["experiment_slug"] = base.EXPERIMENT_SLUG
    summary["mechanism"] = "C111 plus only algebra_equation_answer on C111 fallback rows"
    if summary.get("reason") == "C111 plus isolated formulaic solver aggregate completed.":
        summary["reason"] = "C111 plus isolated algebra/equation solver aggregate completed."
    if "formulaic_quality" in summary:
        summary["algebra_quality"] = summary["formulaic_quality"]
    if "delta_formulaic_minus_c111" in summary:
        summary["delta_algebra_minus_c111"] = summary["delta_formulaic_minus_c111"]
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C251 C111 Plus Isolated Algebra/Equation Solver Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Base Validity",
        f"`{summary.get('base_validity')}`",
        "",
        "## Quality",
        f"- C111 stack: `{summary.get('c111_quality')}`",
        f"- C111 plus algebra: `{summary.get('algebra_quality')}`",
        f"- delta algebra minus C111: `{summary.get('delta_algebra_minus_c111')}`",
        "",
        "## Changed Rows",
        f"`{summary.get('changed_rows')}`",
        "",
        "## Handler Counts",
        f"`{summary.get('handler_counts')}`",
        "",
        "## Changed Slices",
        f"- by category: `{summary.get('changed_by_category')}`",
        f"- by bucket: `{summary.get('changed_by_bucket')}`",
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
    base.EXPERIMENT_ID = "C251"
    base.EXPERIMENT_SLUG = "C251_c111_plus_algebra_equation_aggregate"
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C251_artifacts"
    base.c111_plus_formulaic_stack = c111_plus_algebra_stack
    base.run_validation = run_validation
    base.write_report = write_report
    return base.run(argv)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
