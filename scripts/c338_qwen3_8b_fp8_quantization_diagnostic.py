from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c307_qwen3_4b_2507_recommended_sampling as base


EXPERIMENT_ID = "C338"
EXPERIMENT_SLUG = "C338_qwen3_8b_fp8_quantization_diagnostic"
VARIANT_MODEL_ID = "Qwen/Qwen3-8B-FP8"
MODEL_PACKAGE_METADATA = {
    "baseline_model": "Qwen/Qwen3-8B-AWQ",
    "variant_model": VARIANT_MODEL_ID,
    "metadata_source": "Hugging Face API files_metadata check from controller on 2026-05-28",
    "baseline_selected_files_gb": 6.115,
    "variant_selected_files_gb": 9.453,
    "status": "accessible and ungated; same-base FP8 quantization diagnostic with package-plausible size",
}
GREEDY_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
}
ORIGINAL_WRITE_REPORT = base.write_report


def write_report(path: Path, summary: dict) -> None:
    ORIGINAL_WRITE_REPORT(path, summary)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "# C307 Qwen3-4B-Instruct-2507-FP8 Recommended Sampling",
        "# C338 Qwen3-8B-FP8 Quantization Diagnostic",
        1,
    )
    text = text.replace("## Delta 4B Recommended Minus 8B", "## Delta 8B-FP8 Minus 8B-AWQ")
    text = text.replace("## Variant 4B Recommended", "## Variant 8B-FP8")
    text = text.replace("Recommended sampling 4B-2507", "Qwen3-8B-FP8 greedy")
    path.write_text(text, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C338_artifacts"
    base.VARIANT_MODEL_ID = VARIANT_MODEL_ID
    base.MODEL_PACKAGE_METADATA = MODEL_PACKAGE_METADATA
    base.RECOMMENDED_SAMPLING = GREEDY_SAMPLING
    base.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--sample-size", "256"),
        ("--seed", "338"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return base.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
