from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import c088_simple_solution_candidate_smoke as c088
import c125_direct_arithmetic_final_smoke as c125


EXPERIMENT_ID = "C131"
EXPERIMENT_SLUG = "C131_russian_morph_grammar_final_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C131_artifacts"


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C131 Russian Morphology / Grammar Final Smoke Report",
        "",
        "## Objective",
        "- ID: C131",
        "- Mechanism: final-entrypoint smoke for strict Russian morphology/grammar templates using `pymorphy3`.",
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


def install_morph_dependency() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "pymorphy3==2.0.6",
            "pymorphy3-dicts-ru",
            "razdel==0.5.0",
        ]
    )


def run(argv: Sequence[str] | None = None) -> int:
    original_run = c088.run
    try:
        c088.run = lambda argv=None: 0
        c125.run([])
    finally:
        c088.run = original_run
    install_morph_dependency()
    c088.EXPERIMENT_ID = EXPERIMENT_ID
    c088.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c088.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c088.write_report = write_report
    c088.SMOKE_ROWS.extend(
        [
            {
                "rid": 2020,
                "question": "определи падеж слова «радость»",
                "expected_contains": "Без контекста нельзя",
                "forbidden_contains": ["предложный падеж\n\nИтоговый ответ: предложный падеж"],
            },
            {
                "rid": 6217,
                "question": "определи падеж слова «гибкость»",
                "expected_contains": "Без контекста нельзя",
                "forbidden_contains": ["именительный падеж\n\nИтоговый ответ: именительный падеж"],
            },
            {
                "rid": 5282,
                "question": "определи тип связи в словосочетании «пошёл вперёд»",
                "expected_contains": "примыкание",
                "expected_exact": "примыкание\n\nИтоговый ответ: примыкание",
            },
            {
                "rid": 7597,
                "question": "определите вид односоставного предложения: «принеси-ка книгу»",
                "expected_contains": "определённо-личное",
                "expected_exact": "определённо-личное\n\nИтоговый ответ: определённо-личное",
            },
            {
                "rid": 8685,
                "question": "Определи тип связи в словосочетании «расцвела вишня»",
                "expected_contains": "согласование",
                "expected_exact": "согласование\n\nИтоговый ответ: согласование",
            },
            {
                "rid": 6294,
                "question": "склонение и падеж слова «тележкой»",
                "expected_contains": "творительный падеж",
                "forbidden_contains": ["Итоговый ответ: **тележкой**"],
            },
            {
                "rid": 736,
                "question": "сделай морфологический разбор слова «заставили»",
                "expected_contains": "прошедшее время",
                "forbidden_contains": ["второе лицо", "повелительное"],
            },
            {
                "rid": 1547,
                "question": "морфологический разбор глагола «охватывал»",
                "expected_contains": "мужской род",
                "forbidden_contains": ["1-е лицо", "мн. ч.", "3-е лицо"],
            },
            {
                "rid": 5089,
                "question": "морфологический разбор слова «загадочном»",
                "expected_contains": "предложный падеж",
                "forbidden_contains": ["именительный падеж"],
            },
            {
                "rid": 7522,
                "question": "морфологический разбор слова «десятидневная»",
                "expected_contains": "прилагательное",
                "forbidden_contains": ["настоящее время", "несовершенный вид"],
            },
            {
                "rid": 8835,
                "question": "морфологический разбор слова «кочующих»",
                "expected_contains": "причастие",
                "forbidden_contains": ["прилагательное, сравнительная"],
            },
            {
                "rid": 9834,
                "question": "морфологический разбор слова «рука»",
                "expected_contains": "1-е склонение",
                "forbidden_contains": ["неизменяемое"],
            },
            {
                "rid": 6332,
                "question": "морфологический разбор слова «живой»",
                "expected_contains": "живой",
                "forbidden_contains": ["женский род, родительный падеж"],
            },
        ]
    )
    return c088.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
