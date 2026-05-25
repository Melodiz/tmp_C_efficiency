from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088
import c131_russian_morph_grammar_final_smoke as c131


EXPERIMENT_ID = "C135"
EXPERIMENT_SLUG = "C135_calculator_written_arithmetic_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C135_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C135 Calculator / Written-Arithmetic Final Smoke Report",
        "",
        "## Objective",
        "- ID: C135",
        "- Mechanism: final-entrypoint smoke for strict calculator/written-arithmetic templates.",
        "- Leaderboard submission: NO.",
        "- Model weights and Python packages were used only inside the remote runtime and are not included in this artifact.",
        "",
        "## Results",
        "| status | rows | return code | runtime s | output rows | checks passed | weights GB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {returncode} | {runtime:.2f} | {output_rows} | {checks_passed}/{checks_total} | {weights_gb:.2f} |".format(
            status=summary.get("status"),
            rows=len(c088.SMOKE_ROWS),
            returncode=summary.get("returncode"),
            runtime=float(summary.get("runtime_s") or 0),
            output_rows=summary.get("output_rows"),
            checks_passed=checks.get("passed", 0),
            checks_total=checks.get("total", 0),
            weights_gb=(summary.get("weights_size_bytes") or 0) / (1024**3),
        ),
        "",
        "## Checks",
    ]
    for item in checks.get("items", []):
        lines.append(
            "- rid `{rid}` expected contains `{expected}`, exact `{exact}`, forbids `{forbidden}`: `{passed}`".format(
                rid=item.get("rid"),
                expected=item.get("expected_contains"),
                exact=item.get("expected_exact"),
                forbidden=item.get("forbidden_contains", []),
                passed=item.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision recommendation",
            "",
            summary.get("decision_recommendation", "REVIEW"),
            "",
            "## Strongest reason against recommendation",
            f"- {summary.get('reason', 'Review smoke outputs and expected-gain estimate before packaging.')}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    original_run = c088.run
    try:
        c088.run = lambda argv=None: 0
        c131.run([])
    finally:
        c088.run = original_run
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.write_report = write_report
    c088.SMOKE_ROWS.extend(
        [
            {
                "rid": 2984,
                "question": "5842 разделить на 834 в столбик",
                "expected_contains": "7 (остаток 4)",
                "expected_exact": "7 (остаток 4)\n\nИтоговый ответ: 7 (остаток 4)",
            },
            {
                "rid": 7383,
                "question": "45060 разделить на 72 в столбик",
                "expected_contains": "625 (остаток 60)",
                "expected_exact": "625 (остаток 60)\n\nИтоговый ответ: 625 (остаток 60)",
            },
            {
                "rid": 9508,
                "question": "как решить столбиком 84:21",
                "expected_contains": "4",
                "expected_exact": "4\n\nИтоговый ответ: 4",
            },
            {
                "rid": 3384,
                "question": "380 × 17 столбиком",
                "expected_contains": "6460",
                "expected_exact": "6460\n\nИтоговый ответ: 6460",
            },
            {
                "rid": 9054,
                "question": "680 умножить на 145 столбиком",
                "expected_contains": "98600",
                "expected_exact": "98600\n\nИтоговый ответ: 98600",
            },
            {
                "rid": 1086,
                "question": "Arctg 3",
                "expected_contains": "1,249",
                "expected_exact": "1,249\n\nИтоговый ответ: 1,249",
            },
            {
                "rid": 5767,
                "question": "арктангенс 0,6",
                "expected_contains": "0,5404",
                "expected_exact": "0,5404\n\nИтоговый ответ: 0,5404",
            },
            {
                "rid": 8583,
                "question": "арктангенс 3,7",
                "expected_contains": "1,3068",
                "expected_exact": "1,3068\n\nИтоговый ответ: 1,3068",
            },
            {
                "rid": 4679,
                "question": "синус 0,8",
                "expected_contains": "0,7174",
                "expected_exact": "0,7174\n\nИтоговый ответ: 0,7174",
            },
            {
                "rid": 7744,
                "question": "синус 420",
                "expected_contains": "√3/2",
                "expected_exact": "√3/2\n\nИтоговый ответ: √3/2",
            },
        ]
    )
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
