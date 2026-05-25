from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088
import c135_calculator_written_arithmetic_final_smoke as c135


EXPERIMENT_ID = "C140"
EXPERIMENT_SLUG = "C140_algebra_equation_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C140_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C140 Algebra / Equation Final Smoke Report",
        "",
        "## Objective",
        "- ID: C140",
        "- Mechanism: final-entrypoint smoke for strict algebra/equation exact solver.",
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
        c135.run([])
    finally:
        c088.run = original_run
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.write_report = write_report
    c088.SMOKE_ROWS.extend(
        [
            {
                "rid": 165,
                "question": "Найдите все числа, кратные 17, которые являются решением неравенства: 34 > X > 12.",
                "expected_contains": "17",
                "expected_exact": "17\n\nИтоговый ответ: 17",
                "forbidden_contains": ["34"],
            },
            {
                "rid": 571,
                "question": "5x²-6x+2=0 решить уравнение",
                "expected_contains": "x1 = (3+i)/5",
                "expected_exact": "x1 = (3+i)/5, x2 = (3-i)/5\n\nИтоговый ответ: x1 = (3+i)/5, x2 = (3-i)/5",
            },
            {
                "rid": 3397,
                "question": "3. Решить систему линейных уравнений методом обратной матрицы: x₁ + 3x₂ − 2x₃ = 4, 3x₁ − x₂ + x₃ = 6, 2x₁ + x₂ + 2x₃ = 8.",
                "expected_contains": "x1 = 52/25",
                "expected_exact": "x1 = 52/25, x2 = 36/25, x3 = 6/5\n\nИтоговый ответ: x1 = 52/25, x2 = 36/25, x3 = 6/5",
            },
            {
                "rid": 3734,
                "question": "-xy(x^2-y) умножить",
                "expected_contains": "-x^3y + xy^2",
                "expected_exact": "-x^3y + xy^2\n\nИтоговый ответ: -x^3y + xy^2",
            },
            {
                "rid": 7901,
                "question": "решите уравнение: 100x^2-160x+63=0",
                "expected_contains": "x1 = 7/10",
                "expected_exact": "x1 = 7/10, x2 = 9/10\n\nИтоговый ответ: x1 = 7/10, x2 = 9/10",
            },
            {
                "rid": 8043,
                "question": "Реши пропорцию. $x : \\frac {8} {15} = \\frac {20} {144} : \\frac {20} {60}$ Запиши ответ в виде несократимой дроби, используя символ «/».",
                "expected_contains": "2/9",
                "expected_exact": "2/9\n\nИтоговый ответ: 2/9",
            },
        ]
    )
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
