from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088
import c116_chemistry_stoichiometry_final_smoke as c116


EXPERIMENT_ID = "C119"
EXPERIMENT_SLUG = "C119_formulaic_math_physics_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C119_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C119 Formulaic Math/Physics Final Smoke Report",
        "",
        "## Objective",
        "- ID: C119",
        "- Mechanism: final-entrypoint smoke for C116 plus strict formulaic math/physics exact solver.",
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


def populate_c116_rows() -> None:
    original_run = c088.run
    try:
        c088.run = lambda argv=None: 0
        c116.run([])
    finally:
        c088.run = original_run


def run(argv: Sequence[str] | None = None) -> int:
    populate_c116_rows()
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.write_report = write_report
    c088.SMOKE_ROWS.extend(
        [
            {
                "rid": 5447,
                "question": "Найдите площадь боковой поверхности конуса, если образующая конуса равна 10 см, а диаметр основания — 8 см. Ответ: (запишите число без единиц измерения) см.",
                "expected_contains": "40π",
                "expected_exact": "40π\n\nИтоговый ответ: 40π",
            },
            {
                "rid": 8323,
                "question": "диагональ квадрата равна 64, чему равна площадь квадрата?",
                "expected_contains": "2048",
                "expected_exact": "2048\n\nИтоговый ответ: 2048",
            },
            {
                "rid": 3968,
                "question": "сколько десятков в числе 723000",
                "expected_contains": "72300",
                "expected_exact": "72300\n\nИтоговый ответ: 72300",
            },
            {
                "rid": 969,
                "question": "15 от 3000000 это сколько",
                "expected_contains": "450000",
                "expected_exact": "450000\n\nИтоговый ответ: 450000",
            },
            {
                "rid": 503,
                "question": "Сколько льда при 0 °C расплавится, если ему передать количество теплоты, которое выделится при конденсации водяного пара массой 12 кг и температурой 100 °C при нормальном атмосферном давлении?",
                "expected_contains": "81,2 кг",
                "expected_exact": "81,2 кг\n\nИтоговый ответ: 81,2 кг",
            },
            {
                "rid": 4498,
                "question": "2. Какое давление сжатого воздуха, находящегося в баллоне объёмом 30 л при 20 °C, если масса воздуха 3 кг?",
                "expected_contains": "8,4 МПа",
                "expected_exact": "8,4 МПа\n\nИтоговый ответ: 8,4 МПа",
            },
            {
                "rid": 1881,
                "question": "Электрический кипятильник рассчитан на 230 В и силу тока 4 А. Какова мощность тока в кипятильнике?",
                "expected_contains": "920 Вт",
                "expected_exact": "920 Вт\n\nИтоговый ответ: 920 Вт",
            },
            {
                "rid": 5669,
                "question": "Какова скорость света в алмазе, если его показатель преломления равен 2,42?",
                "expected_contains": "1,24 × 10^8 м/с",
                "expected_exact": "1,24 × 10^8 м/с\n\nИтоговый ответ: 1,24 × 10^8 м/с",
            },
            {
                "rid": 2061,
                "question": "8. В случайном эксперименте симметричную монету подбрасывают дважды. Найдите вероятность того, что орёл выпадет ровно один раз.",
                "expected_contains": "1/2",
                "expected_exact": "1/2\n\nИтоговый ответ: 1/2",
            },
            {
                "rid": 8601,
                "question": "найдите делимое, если неполное частное 42, делитель 15 и остаток 0",
                "expected_contains": "630",
                "expected_exact": "630\n\nИтоговый ответ: 630",
            },
            {
                "rid": 7558,
                "question": "Участок земли для строительства базы отдыха имеет форму прямоугольника со сторонами 60 м и 40 м. Одна из длинных сторон участка расположена вдоль озера, а остальные три стороны необходимо обнести забором. Определите длину забора в метрах.",
                "expected_contains": "140",
                "expected_exact": "140\n\nИтоговый ответ: 140",
            },
            {
                "rid": 945,
                "question": "Правильная четырёхугольная призма описана около цилиндра, радиус основания которого равен 3. Площадь боковой поверхности призмы равна 72. Найдите высоту цилиндра.",
                "expected_contains": "3",
                "expected_exact": "3\n\nИтоговый ответ: 3",
            },
            {
                "rid": 6735,
                "question": "Правильная четырёхугольная призма описана около цилиндра, радиус основания которого равен 2. Площадь боковой поверхности призмы равна 36. Найдите высоту цилиндра.",
                "expected_contains": "2,25",
                "expected_exact": "2,25\n\nИтоговый ответ: 2,25",
            },
            {
                "rid": 7052,
                "question": "Катеты прямоугольного треугольника 30 и 40. Найдите высоту, проведённую к гипотенузе. Ответ округлите до сотых.",
                "expected_contains": "24",
                "expected_exact": "24\n\nИтоговый ответ: 24",
            },
            {
                "rid": 8852,
                "question": "задачи по теме молярный объём 8 класс: какой объём занимают 4 моля кислорода?",
                "expected_contains": "89,6 л",
                "expected_exact": "89,6 л\n\nИтоговый ответ: 89,6 л",
            },
        ]
    )
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
