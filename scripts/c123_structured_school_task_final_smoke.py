from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088
import c119_formulaic_math_physics_final_smoke as c119


EXPERIMENT_ID = "C123"
EXPERIMENT_SLUG = "C123_structured_school_task_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C123_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C123 Structured School-Task Final Smoke Report",
        "",
        "## Objective",
        "- ID: C123",
        "- Mechanism: final-entrypoint smoke for C119 plus strict structured school-task solver expansion.",
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


def populate_c119_rows() -> None:
    original_run = c088.run
    try:
        c088.run = lambda argv=None: 0
        c119.run([])
    finally:
        c088.run = original_run


def run(argv: Sequence[str] | None = None) -> int:
    populate_c119_rows()
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.write_report = write_report
    c088.SMOKE_ROWS.extend(
        [
            {
                "rid": 8559,
                "question": "2 м 3 дм сколько дм",
                "expected_contains": "23 дм",
                "expected_exact": "23 дм\n\nИтоговый ответ: 23 дм",
            },
            {
                "rid": 9451,
                "question": "сколько литров в 3 кубических метрах",
                "expected_contains": "3000 литров",
                "expected_exact": "3000 литров\n\nИтоговый ответ: 3000 литров",
            },
            {
                "rid": 6051,
                "question": "сколько граммов в 2 тоннах, представь ответ в виде таблицы",
                "expected_contains": "2000000 граммов",
                "expected_exact": "2000000 граммов\n\nИтоговый ответ: 2000000 граммов",
            },
            {
                "rid": 8889,
                "question": "переведите в радианную меру угла 45 120 300",
                "expected_contains": "π/4, 2π/3, 5π/3",
                "expected_exact": "π/4, 2π/3, 5π/3\n\nИтоговый ответ: π/4, 2π/3, 5π/3",
            },
            {
                "rid": 9176,
                "question": "2024 в римских цифрах",
                "expected_contains": "MMXXIV",
                "expected_exact": "MMXXIV\n\nИтоговый ответ: MMXXIV",
                "forbidden_contains": ["ММ", "І", "Ү"],
            },
            {
                "rid": 5082,
                "question": "Периметр равнобедренного треугольника составляет 64 см, при этом основание превышает боковую сторону на 8 см. Найдите длину боковой стороны.",
                "expected_contains": "56/3 см",
                "expected_exact": "56/3 см\n\nИтоговый ответ: 56/3 см",
            },
            {
                "rid": 842,
                "question": "Задача. На концах невесомого рычага действуют силы 60 и 300 Н. Расстояние от точки опоры до меньшей силы равно 0,08 м. Определи длину плеча большей силы, если рычаг находится в равновесии.",
                "expected_contains": "0,016 м",
                "expected_exact": "0,016 м\n\nИтоговый ответ: 0,016 м",
            },
            {
                "rid": 4802,
                "question": "В задании 1.19.8. Взяли некоторое количество досок и распилили их. Всего выполнили 7 поперечных распилов, в результате получилось 15 кусков. Сколько досок взяли изначально?",
                "expected_contains": "8",
                "expected_exact": "8\n\nИтоговый ответ: 8",
            },
            {
                "rid": 7817,
                "question": "Игорь не очень любит бегать, но решил делать утренние пробежки для здоровья. Вероятность того, что Игорь на пробежке преодолеет больше 400 метров, равна 0,7, а вероятность того, что он пробежит более 900 метров, равна 0,35. Какова вероятность, что Игорь пробежит более 400 метров, но не более 900 метров?",
                "expected_contains": "0,35",
                "expected_exact": "0,35\n\nИтоговый ответ: 0,35",
            },
        ]
    )
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
