from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Sequence

import c285_open_ended_fuller_prompt_route as base


EXPERIMENT_ID = "C292"
EXPERIMENT_SLUG = "C292_math_scratchpad_route"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C292_artifacts"

MATH_SCRATCHPAD_PREFIX = (
    "Реши задачу внимательно, используя черновик только внутри рассуждения. "
    "В ответе выведи только итоговый числовой, формульный или символьный ответ. "
    "Сохрани единицы измерения, если они нужны."
)

MATH_ROUTE = re.compile(
    r"\d|[=+\-*/^√]|\\b(sin|cos|tg|ctg|tan|log|sqrt)\\b|"
    r"реши|решить|вычисл|найд[и]?|чему равн|сколько|уравнен|"
    r"площад|периметр|об[ъь]ем|радиус|диаметр|угол|координат|"
    r"дроб|процент|%|км|см|мм|метр|литр|грамм|тонн|скорост|масса",
    re.IGNORECASE,
)

OPEN_OR_LANGUAGE_GUARD = re.compile(
    r"перевед|translate|сочинен|эссе|объясн|почему|расскаж|опиши|"
    r"напиши текст|падеж|склонен|спряж|част[ьи] речи|морфолог|"
    r"граммат|литератур|истори|биолог|географ",
    re.IGNORECASE,
)


def route_prefix(question: str, c111_prefix: str) -> tuple[str, str]:
    text = str(question)
    if MATH_ROUTE.search(text) and not OPEN_OR_LANGUAGE_GUARD.search(text):
        return "math_scratchpad", MATH_SCRATCHPAD_PREFIX
    return "c111_default", c111_prefix


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# C292 Math-Only Scratchpad Route Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Input Tokens",
        f"`{summary.get('tokens')}`",
        "",
        "## Delta Math Scratchpad Route Minus C111",
        f"`{summary.get('delta_variant_minus_c111')}`",
        "",
        "## C111 Prefix Control",
        f"`{summary.get('control_c111_prefix')}`",
        "",
        "## Math Scratchpad Route Variant",
        f"`{summary.get('variant_open_ended_fuller_route')}`",
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
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    base.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    base.route_prefix = route_prefix

    args = base.parse_args(argv)
    paths = base.artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)

    summary = base.run_validation(args)
    summary.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_slug": EXPERIMENT_SLUG,
            "reason": "C111 stack paired math-only scratchpad route aggregate completed.",
            "mechanism": "question-text route formulaic/numeric closed rows to compact scratchpad math prefix",
        }
    )
    base.io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    base.agg.base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
