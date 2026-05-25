from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088
import c140_algebra_equation_final_smoke as c140


EXPERIMENT_ID = "C156"
EXPERIMENT_SLUG = "C156_geometry_exact_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C156_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C156 Geometry Exact-Formula Final Smoke Report",
        "",
        "## Objective",
        "- ID: C156",
        "- Mechanism: final-entrypoint smoke for strict geometry exact-formula handlers.",
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
        c140.run([])
    finally:
        c088.run = original_run
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.write_report = write_report
    c088.SMOKE_ROWS.extend(
        [
            {
                "rid": 411,
                "question": "Найдите длину дуг, на которые разбивается окружность двумя радиусами, если угол между ними равен 45°, а радиус окружности равен 8 см.",
                "expected_contains": "2π см и 14π см",
                "expected_exact": "2π см и 14π см\n\nИтоговый ответ: 2π см и 14π см",
            },
            {
                "rid": 3816,
                "question": "Сторона правильного треугольника равна √5. Найдите радиус окружности, вписанной в этот треугольник.",
                "expected_contains": "√15/6",
                "expected_exact": "√15/6\n\nИтоговый ответ: √15/6",
            },
            {
                "rid": 6812,
                "question": "9. MN — диаметр окружности с центром K, L — точка этой окружности. Найдите периметр MNKL, если известно, что MN = 18, ML = 14.",
                "expected_contains": "50",
                "expected_exact": "50\n\nИтоговый ответ: 50",
            },
        ]
    )
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
