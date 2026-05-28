from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c307_qwen3_4b_2507_recommended_sampling as c307


EXPERIMENT_ID = "C317"
EXPERIMENT_SLUG = "C317_qwen3_30b_a3b_fp8_greedy"
VARIANT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_PACKAGE_METADATA = {
    "variant_selected_files_gb": 31.195,
    "metadata_source": "Hugging Face API files_metadata check from controller on 2026-05-28",
    "package_risk": "large remote model diagnostic; not a submission candidate without separate packaging/runtime review",
}
GREEDY_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
}
ORIGINAL_WRITE_REPORT = c307.write_report


def write_report(path: Path, summary: dict) -> None:
    ORIGINAL_WRITE_REPORT(path, summary)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "# C307 Qwen3-4B-Instruct-2507-FP8 Recommended Sampling",
        "# C317 Qwen3-30B-A3B-Instruct-2507-FP8 Greedy Diagnostic",
        1,
    )
    text = text.replace("## Delta 4B Recommended Minus 8B", "## Delta 30B-A3B Greedy Minus 8B")
    text = text.replace("## Variant 4B Recommended", "## Variant 30B-A3B Greedy")
    path.write_text(text, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c307.EXPERIMENT_ID = EXPERIMENT_ID
    c307.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c307.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C317_artifacts"
    c307.VARIANT_MODEL_ID = VARIANT_MODEL_ID
    c307.MODEL_PACKAGE_METADATA = MODEL_PACKAGE_METADATA
    c307.RECOMMENDED_SAMPLING = GREEDY_SAMPLING
    c307.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--sample-size", "128"),
        ("--seed", "317"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c307.run(forwarded)


if __name__ == "__main__":
    raise SystemExit(run())
