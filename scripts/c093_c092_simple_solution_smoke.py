from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088


EXPERIMENT_ID = "C093"
EXPERIMENT_SLUG = "C093_c092_simple_solution_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C093_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C093 C092 Simple-Solution Smoke Report",
        "",
        "## Objective",
        "- ID: C093",
        "- Mechanism: final-entrypoint smoke for the C092 candidate in `simple_solution/solution.py`.",
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
            f"- {summary.get('reason', 'Review smoke outputs before packaging.')}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.SMOKE_ROWS = [
        {
            "rid": 8295,
            "question": "Найди значение выражения $(x + y)^2 + 5x^2 - 2x - 2(x + y) + 5$ при $x = 4$, $y = 2$.",
            "expected_contains": "101",
        },
        {
            "rid": 4242,
            "question": "Amazingly, many of the houses __________________ several centuries ago! BUILD",
            "expected_contains": "were built",
            "expected_exact": "were built",
            "forbidden_contains": ["Ответ", "были"],
        },
        {
            "rid": 2506,
            "question": "Задание 4. Выберите один из нескольких вариантов. Choose the correct answer. Some teenagers believe that getting a job is ＿ (challenging) than studying at college.",
            "expected_contains": "more challenging",
            "expected_exact": "more challenging",
            "forbidden_contains": ["Ответ", "слож"],
        },
        {
            "rid": 5782,
            "question": "составить слова из букв слова «брелок»",
            "expected_contains": "Итоговый ответ",
        },
    ]
    c088.write_report = write_report
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
