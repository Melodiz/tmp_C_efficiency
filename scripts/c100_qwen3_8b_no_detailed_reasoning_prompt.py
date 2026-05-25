from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c096_qwen3_8b_thinking_final_only_prompt as c096


EXPERIMENT_ID = "C100"
EXPERIMENT_SLUG = "C100_qwen3_8b_no_detailed_reasoning_prompt"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C100_artifacts"
NO_DETAILED_REASONING_PREFIX = (
    "Ответь на языке задания. Не показывай подробный ход решения. "
    "Для выбора, пропуска, числа, арифметики и перевода единиц дай только итоговый ответ. "
    "Для текста или объяснения дай краткий ответ: 2-5 предложений или короткий список. "
    "Не повторяй условие."
)


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no no-detailed-reasoning prompt evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C100 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic output validity failed."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if int(validity.get("max_token_hit_rows") or 0) > 5:
        return "KILL", "The prompt did not control truncation versus C093/C090."
    if int(validity.get("repetition_loop_suspected_rows") or 0) > 3:
        return "KILL", "The prompt increased repetition risk."
    return "MUTATE", "The prompt passed validity gates; row-level review is needed before hard-audit validation."


def write_report(report_path: Path, metrics: dict[str, Any], args: Any, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C100 Qwen3-8B No-Detailed-Reasoning Prompt Report",
        "",
        "## Objective",
        "- ID: C100",
        "- Mechanism: prompt-only concise/no-detailed-reasoning instruction with Qwen thinking disabled.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- prefix: `{NO_DETAILED_REASONING_PREFIX}`",
        "- `enable_thinking=False` is passed to the chat template.",
        "- No deterministic handlers, retrieval, cache, SFT, LoRA, model/backend change, or sampling change.",
        "",
        "## Results",
        "| status | rows | max-token hits | thinking traces | empty answers | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {cap_hits} | {thinking} | {empty} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            empty=validity.get("empty_answer_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Decision recommendation",
        "",
        rec,
        "",
        "## Strongest reason against recommendation",
        f"- {reason}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c096.EXPERIMENT_ID = EXPERIMENT_ID
    c096.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c096.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c096.THINKING_FINAL_ONLY_PREFIX = NO_DETAILED_REASONING_PREFIX
    c096.OMIT_ENABLE_THINKING_FALSE = False
    c096.MECHANISM_ID = "no_detailed_reasoning_prompt_thinking_disabled"
    c096.recommendation = recommendation
    c096.write_report = write_report
    return c096.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
