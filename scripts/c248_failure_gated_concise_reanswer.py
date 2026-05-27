from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import c071_probe as probe
import c246_failure_gated_same_model_512 as base


REPAIR_PREFIX = (
    "Answer the task directly and briefly. Preserve the language of the task. "
    "Do not explain or show reasoning. Give only the final answer."
)
_BASE_RUN_VALIDATION = base.run_validation


def repair_prompt(tokenizer: Any, question: str) -> str:
    return probe.apply_user_only_template(tokenizer, question, True, REPAIR_PREFIX)


def run_validation(args: Any) -> dict[str, Any]:
    summary = _BASE_RUN_VALIDATION(args)
    summary["experiment_id"] = base.EXPERIMENT_ID
    summary["experiment_slug"] = base.EXPERIMENT_SLUG
    summary["route"] = (
        "Use same-model concise re-answer prompt only when C111 output hits max_tokens "
        "or repetition-loop flags."
    )
    if summary.get("reason") == "Failure-gated same-model 512 fallback aggregate completed.":
        summary["reason"] = "Failure-gated same-model concise re-answer aggregate completed."
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C248 Failure-Gated Same-Model Concise Re-Answer",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- route: {summary.get('route')}",
        f"- repair prefix: `{REPAIR_PREFIX}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Route Counts",
        f"`{summary.get('route_counts')}`",
        "",
        "## Selected Minus Baseline",
        f"`{summary.get('delta_selected_minus_baseline_all')}`",
        "",
        "## Routed Rows: Repair Minus Baseline",
        f"`{summary.get('delta_fallback_minus_baseline_routed_only')}`",
        "",
        "## Baseline All",
        f"`{summary.get('baseline_all')}`",
        "",
        "## Selected All",
        f"`{summary.get('selected_all')}`",
        "",
        "## Routed Baseline",
        f"`{summary.get('baseline_routed_only')}`",
        "",
        "## Routed Repair",
        f"`{summary.get('fallback_512_routed_only')}`",
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
    base.EXPERIMENT_ID = "C248"
    base.EXPERIMENT_SLUG = "C248_failure_gated_concise_reanswer"
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C248_artifacts"
    base.FALLBACK_MAX_TOKENS = 320
    base.FALLBACK_PREFIX_OVERRIDE = REPAIR_PREFIX
    base.run_validation = run_validation
    base.write_report = write_report
    return base.run(argv)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
