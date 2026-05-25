from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088


EXPERIMENT_ID = "C113"
EXPERIMENT_SLUG = "C113_numeric_exact_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C113_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C113 Numeric Exact Final Smoke Report",
        "",
        "## Objective",
        "- ID: C113",
        "- Mechanism: final-entrypoint smoke for C111 plus the C112 strict numeric exact solver.",
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
    c088.write_report = write_report
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
        {
            "rid": 4637,
            "question": "45 километров 70 метров это сколько метров",
            "expected_contains": "45070 метров",
            "expected_exact": "45070 метров\n\nИтоговый ответ: 45070 метров",
        },
        {
            "rid": 6234,
            "question": "450 дециметров квадратных переведи в квадратные метры",
            "expected_contains": "4,5 м²",
            "expected_exact": "4,5 м²\n\nИтоговый ответ: 4,5 м²",
        },
        {
            "rid": 6615,
            "question": "ундециллион — это сколько триллионов",
            "expected_contains": "10^24 триллионов",
            "expected_exact": "10^24 триллионов\n\nИтоговый ответ: 10^24 триллионов",
        },
        {
            "rid": 8978,
            "question": "Переведи 3\u202f568,4217 га в квадратные километры.",
            "expected_contains": "35,684217 км²",
            "expected_exact": "35,684217 км²\n\nИтоговый ответ: 35,684217 км²",
        },
        {
            "rid": 7900,
            "question": "перевести 2345 кв. см в метры",
            "expected_contains": "0,2345 м²",
            "expected_exact": "0,2345 м²\n\nИтоговый ответ: 0,2345 м²",
        },
        {
            "rid": 7401,
            "question": "18 процентов от 350000",
            "expected_contains": "63000",
            "expected_exact": "63000\n\nИтоговый ответ: 63000",
        },
        {
            "rid": 8180,
            "question": "выполните сложение в двоичной системе счисления 10101100+101101",
            "expected_contains": "11011001",
            "expected_exact": "11011001\n\nИтоговый ответ: 11011001",
        },
        {
            "rid": 7156,
            "question": "выполни перевод из восьмеричной системы счисления в десятичную: 72₈ и 236₈",
            "expected_contains": "58 и 158",
            "expected_exact": "58 и 158\n\nИтоговый ответ: 58 и 158",
        },
        {
            "rid": 9992,
            "question": "3 5/20 переводится в десятичную",
            "expected_contains": "3,25",
            "expected_exact": "3,25\n\nИтоговый ответ: 3,25",
        },
        {
            "rid": 2508,
            "question": "420+8%",
            "expected_contains": "453,6",
            "expected_exact": "453,6\n\nИтоговый ответ: 453,6",
        },
    ]
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
