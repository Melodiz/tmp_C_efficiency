from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as io
import c258_c111_family_stratified_validation as base


EXPERIMENT_ID = "C263"
EXPERIMENT_SLUG = "C263_c111_datetime_subfamily_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C263_artifacts"
DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_SEED = 263

SUBFAMILY_PATTERNS = {
    "weekday_calendar": [
        r"(?:день\s+недел|понедельник|вторник|среда|четверг|пятниц|суббот|воскресен|weekday|day\s+of\s+week)",
    ],
    "date_arithmetic": [
        r"(?:через|назад|спустя|после|до|прибав|вычт|сколько\s+дней|days?\s+(?:after|before)|add\s+\d+\s+days?)",
        r"(?:\d{1,2}[./-]\d{1,2}|\d{4}|январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр|date)",
    ],
    "time_arithmetic": [
        r"(?:час|минут|секунд|сутк|time|hour|minute|second)",
        r"(?:через|назад|спустя|после|до|прибав|вычт|сколько|duration|elapsed|after|before)",
    ],
    "time_conversion": [
        r"(?:час|минут|секунд|сутк|time|hour|minute|second)",
        r"(?:перевед|сколько|convert|in\s+seconds|in\s+minutes|in\s+hours)",
    ],
    "age_year": [
        r"(?:возраст|лет\s+будет|родил|birth|born|age|years?\s+old)",
    ],
    "month_day_count": [
        r"(?:месяц|январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр|month)",
        r"(?:сколько|количеств|days?|дней|суток)",
    ],
    "leap_year": [
        r"(?:високос|leap\s+year|366|29\s+феврал)",
    ],
    "clock_angle": [
        r"(?:стрелк|циферблат|clock\s+hand|часов[аы]?[я]?\s+стрелк|минутн[а-я]*\s+стрелк)",
    ],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C263 C111 date/time/calendar subfamily validation.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "zip": out_dir.with_suffix(".zip"),
    }


def configure_base() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    base.DEFAULT_SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE
    base.DEFAULT_SEED = DEFAULT_SEED
    base.FAMILY_PATTERNS = SUBFAMILY_PATTERNS


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C263 C111 Date/Time/Calendar Subfamily Validation",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Measure proven C111 quality and visible failures by C262 date/time/calendar subfamily before any solver port.",
        "- Return only aggregate metrics; no raw prompts, references, outputs, row ids, datasets, weights, or adapter files.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- imports: `{summary.get('imports')}`",
        "",
        "## Sample",
        f"`{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Tokens",
        f"`{summary.get('tokens')}`",
        "",
        "## Overall Quality",
        f"`{summary.get('overall_quality')}`",
        "",
        "## Overall Validity",
        f"`{summary.get('overall_validity')}`",
        "",
        "## Subfamily Counts",
        f"`{summary.get('family_counts')}`",
        "",
        "## Weak Subfamily Summary",
        f"`{summary.get('weak_family_summary')}`",
        "",
        "## Subfamily Quality",
        f"`{summary.get('family_quality')}`",
        "",
        "## Subfamily Validity",
        f"`{summary.get('family_validity')}`",
        "",
        "## Subfamily Handlers",
        f"`{summary.get('family_handlers')}`",
        "",
        "## Subfamily Buckets",
        f"`{summary.get('family_buckets')}`",
        "",
        "## Subfamily Categories",
        f"`{summary.get('family_categories')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- outputs returned: `{summary.get('outputs_returned')}`",
        f"- model weights returned: `{summary.get('model_weights_returned')}`",
        f"- training started: `{summary.get('training_started')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    configure_base()
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    summary = base.run_validation(args)
    summary["reason"] = "C111 date/time/calendar subfamily aggregate validation completed."
    io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    io.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
