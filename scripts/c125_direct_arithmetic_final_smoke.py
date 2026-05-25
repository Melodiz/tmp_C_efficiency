from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088
import c123_structured_school_task_final_smoke as c123


EXPERIMENT_ID = "C125"
EXPERIMENT_SLUG = "C125_direct_arithmetic_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C125_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C125 Direct Arithmetic Final Smoke Report",
        "",
        "## Objective",
        "- ID: C125",
        "- Mechanism: final-entrypoint smoke for C123 plus strict direct arithmetic parser.",
        "- Leaderboard submission: NO.",
        "- Model weights were downloaded only inside the remote runtime and are not included in this artifact.",
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


def populate_c123_rows() -> None:
    original_run = c088.run
    try:
        c088.run = lambda argv=None: 0
        c123.run([])
    finally:
        c088.run = original_run


def run(argv: Sequence[str] | None = None) -> int:
    populate_c123_rows()
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.write_report = write_report
    c088.SMOKE_ROWS.extend(
        [
            {"rid": 1414, "question": "15 - 2,43", "expected_contains": "12,57", "expected_exact": "12,57\n\nИтоговый ответ: 12,57"},
            {"rid": 2987, "question": "1543 + 3286", "expected_contains": "4829", "expected_exact": "4829\n\nИтоговый ответ: 4829"},
            {"rid": 3509, "question": "36 * 15", "expected_contains": "540", "expected_exact": "540\n\nИтоговый ответ: 540"},
            {"rid": 4092, "question": "-7+15,3", "expected_contains": "8,3", "expected_exact": "8,3\n\nИтоговый ответ: 8,3"},
            {"rid": 4496, "question": "900-356", "expected_contains": "544", "expected_exact": "544\n\nИтоговый ответ: 544"},
            {"rid": 5370, "question": "56-24", "expected_contains": "32", "expected_exact": "32\n\nИтоговый ответ: 32"},
            {"rid": 6308, "question": "750-42873", "expected_contains": "-42123", "expected_exact": "-42123\n\nИтоговый ответ: -42123"},
            {"rid": 7011, "question": "450-230", "expected_contains": "220", "expected_exact": "220\n\nИтоговый ответ: 220"},
            {"rid": 7185, "question": "37+15", "expected_contains": "52", "expected_exact": "52\n\nИтоговый ответ: 52"},
            {"rid": 7359, "question": "18+45", "expected_contains": "63", "expected_exact": "63\n\nИтоговый ответ: 63"},
            {"rid": 9168, "question": "2,74-98", "expected_contains": "-95,26", "expected_exact": "-95,26\n\nИтоговый ответ: -95,26"},
            {"rid": 70, "question": "18 на 3", "expected_contains": "6", "expected_exact": "6\n\nИтоговый ответ: 6"},
            {"rid": 7000, "question": "420 на 75", "expected_contains": "5,6", "expected_exact": "5,6\n\nИтоговый ответ: 5,6"},
        ]
    )
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
