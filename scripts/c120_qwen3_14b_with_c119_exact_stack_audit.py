from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

import c107_qwen3_14b_awq_c104_handlers_hard_audit as c107


EXPERIMENT_ID = "C120"
EXPERIMENT_SLUG = "C120_qwen3_14b_with_c119_exact_stack_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C120_artifacts"


def load_final_solution_module() -> Any:
    solution_path = Path(__file__).resolve().parents[1] / "simple_solution" / "solution.py"
    spec = importlib.util.spec_from_file_location("c120_final_solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load final solution module from {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_stage(rows: list[dict[str, Any]], stage_name: str, func: Any, stats: dict[str, Any]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    applied: list[int] = []
    for row in rows:
        updated = dict(row)
        replacement = func(str(updated.get("question", "")), str(updated.get("answer", "")))
        if replacement is not None and replacement != updated.get("answer"):
            updated["answer"] = replacement
            applied.append(int(updated.get("rid") or updated.get("row_id") or -1))
            stack = dict(updated.get("c119_exact_stack") or {})
            stack[stage_name] = {"applied": True}
            updated["c119_exact_stack"] = stack
        processed.append(updated)
    stats[stage_name] = {"applied_count": len(applied), "applied_row_ids": applied}
    return processed


def apply_c119_exact_stack(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solution = load_final_solution_module()
    stats: dict[str, Any] = {}

    current = apply_stage(
        rows,
        "expression_substitution",
        lambda question, answer: solution.expression_substitution_answer(question),
        stats,
    )
    current = apply_stage(
        current,
        "exact_numeric",
        lambda question, answer: solution.exact_numeric_answer(question),
        stats,
    )
    current = apply_stage(
        current,
        "chemistry_stoichiometry",
        lambda question, answer: solution.chemistry_stoichiometry_answer(question),
        stats,
    )
    current = apply_stage(
        current,
        "formulaic_math_physics",
        lambda question, answer: solution.formulaic_math_physics_answer(question),
        stats,
    )
    current = apply_stage(
        current,
        "comma_repetition_dedup",
        lambda question, answer: solution.dedup_comma_loop(answer),
        stats,
    )
    current = apply_stage(
        current,
        "strict_english_cleanup",
        lambda question, answer: solution.cleanup_english_cloze_answer(question, answer),
        stats,
    )
    current = apply_stage(
        current,
        "quantity_conversion",
        lambda question, answer: solution.quantity_conversion_answer(question),
        stats,
    )
    current = apply_stage(
        current,
        "km_meters_guard",
        lambda question, answer: solution.km_meters_answer(question),
        stats,
    )

    stats["total_unique_rows_changed"] = len(
        {
            rid
            for stage in stats.values()
            if isinstance(stage, dict)
            for rid in stage.get("applied_row_ids", [])
        }
    )
    return current, stats


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no hard-audit evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C120 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after applying the current exact stack."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if int(validity.get("max_token_hit_rows") or 0) > 4:
        return "KILL", "Hard-audit truncation risk is too high for a near-limit 14B package."
    if int(validity.get("repetition_loop_suspected_rows") or 0) > 2:
        return "KILL", "Repetition risk remains too high."
    return "MUTATE", "14B plus the C119 exact stack passed validity gates; estimate broad quality gain before any package smoke."


def write_report(report_path: Path, metrics: dict[str, Any], args: Any, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    handlers = metrics.get("handler_stats") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C120 Qwen3-14B-AWQ With C119 Exact Stack Audit Report",
        "",
        "## Objective",
        "- ID: C120",
        "- Mechanism: validate Qwen3-14B-AWQ on hard-audit, then apply the current C119 final exact-solver stack.",
        "- Leaderboard submission: NO.",
        "",
        "## Results",
        "| status | rows | max-token hits | thinking traces | empty answers | repetition suspects | projected 4000q min | exact-stack changed rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {cap_hits} | {thinking} | {empty} | {repetition} | {projected} | {changed} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            empty=validity.get("empty_answer_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
            changed=handlers.get("total_unique_rows_changed", "n/a"),
        ),
        "",
        "## Exact-Stack Coverage",
    ]
    for key in [
        "expression_substitution",
        "exact_numeric",
        "chemistry_stoichiometry",
        "formulaic_math_physics",
        "comma_repetition_dedup",
        "strict_english_cleanup",
        "quantity_conversion",
        "km_meters_guard",
    ]:
        item = handlers.get(key) or {}
        lines.append(f"- {key}: `{item.get('applied_row_ids', [])}`")
    lines.extend(
        [
            "",
            "## Decision recommendation",
            "",
            rec,
            "",
            "## Strongest reason against recommendation",
            f"- {reason}",
            "",
            "## Expected-gain note",
            "- Do not build or flag a submission zip unless row-level review supports a likely public gain above the user's +3 to +5 point threshold over C111 74.7.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_experiment() -> None:
    c107.EXPERIMENT_ID = EXPERIMENT_ID
    c107.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c107.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c107.apply_c104_handlers = apply_c119_exact_stack
    c107.recommendation = recommendation
    c107.write_report = write_report


def run(argv: Sequence[str] | None = None) -> int:
    configure_experiment()
    return c107.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
