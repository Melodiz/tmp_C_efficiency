from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Sequence

import c285_open_ended_fuller_prompt_route as base
import c292_math_scratchpad_route as c292


EXPERIMENT_ID = "C293"
EXPERIMENT_SLUG = "C293_strict_math_scratchpad_route"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C293_artifacts"

STRICT_SYMBOLIC = re.compile(r"(\d+\s*[=+\-*/^]\s*\d+|[=+\-*/^]\s*[a-zа-я]|\b\d+/\d+\b)", re.IGNORECASE)
STRICT_ARITH_WORDS = re.compile(
    r"дроб|сократ|числител|знаменател|процент|%|"
    r"сложи|сложите|вычти|вычтите|умнож|раздел|подели|"
    r"вычисл|чему равн|найд[и]? значение|реши уравнен",
    re.IGNORECASE,
)
STRICT_UNIT_ARITH = re.compile(
    r"\d+\s*(?:км|см|мм|м|метр|литр|л|грамм|г|кг|тонн|мин|сек|час)",
    re.IGNORECASE,
)
STRICT_GUARD = re.compile(
    r"перевед|translate|сочинен|эссе|объясн|почему|расскаж|опиши|"
    r"напиши текст|падеж|склонен|спряж|част[ьи] речи|морфолог|граммат|"
    r"литератур|истори|биолог|географ|хими|молекул|атом|элемент|реакц|"
    r"площад|периметр|радиус|диаметр|угол|координат|геометр",
    re.IGNORECASE,
)


def route_prefix(question: str, c111_prefix: str) -> tuple[str, str]:
    text = str(question)
    if STRICT_GUARD.search(text):
        return "c111_default", c111_prefix
    if STRICT_SYMBOLIC.search(text) or STRICT_ARITH_WORDS.search(text) or STRICT_UNIT_ARITH.search(text):
        return "strict_math_scratchpad", c292.MATH_SCRATCHPAD_PREFIX
    return "c111_default", c111_prefix


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# C293 Strict Math Scratchpad Route Aggregate",
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
        "## Delta Strict Math Scratchpad Route Minus C111",
        f"`{summary.get('delta_variant_minus_c111')}`",
        "",
        "## C111 Prefix Control",
        f"`{summary.get('control_c111_prefix')}`",
        "",
        "## Strict Math Scratchpad Route Variant",
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
            "reason": "C111 stack paired strict math scratchpad route aggregate completed.",
            "mechanism": "strict arithmetic/fraction/symbolic-expression route to compact scratchpad math prefix",
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
