from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c309_qwen3_4b_2507_native_prompt as c309


EXPERIMENT_ID = "C318"
EXPERIMENT_SLUG = "C318_qwen3_30b_a3b_fp8_native_prompt"
VARIANT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_PACKAGE_METADATA = {
    "variant_selected_files_gb": 31.195,
    "metadata_source": "Hugging Face API files_metadata check from controller on 2026-05-28",
    "package_risk": "large remote model diagnostic; not a submission candidate without separate packaging/runtime review",
}
ORIGINAL_WRITE_REPORT = c309.write_report


def write_report(path: Path, summary: dict) -> None:
    ORIGINAL_WRITE_REPORT(path, summary)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "# C309 Qwen3-4B-Instruct-2507-FP8 Native Prompt",
        "# C318 Qwen3-30B-A3B-Instruct-2507-FP8 Native Prompt",
        1,
    )
    text = text.replace("## Delta 4B Native Minus 8B", "## Delta 30B-A3B Native Minus 8B")
    text = text.replace("## Variant 4B Native Prompt", "## Variant 30B-A3B Native Prompt")
    path.write_text(text, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c309.EXPERIMENT_ID = EXPERIMENT_ID
    c309.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c309.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C318_artifacts"
    c309.VARIANT_MODEL_ID = VARIANT_MODEL_ID
    c309.MODEL_PACKAGE_METADATA = MODEL_PACKAGE_METADATA
    c309.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--sample-size", "128"),
        ("--seed", "318"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c309.run(forwarded)


if __name__ == "__main__":
    raise SystemExit(run())
