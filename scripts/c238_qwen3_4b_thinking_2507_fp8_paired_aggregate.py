from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c218_qwen3_4b_2507_fp8_paired_aggregate as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C238"
    base.EXPERIMENT_SLUG = "C238_qwen3_4b_thinking_2507_fp8_paired_aggregate"
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C238_artifacts"
    base.VARIANT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507-FP8"
    base.MODEL_PACKAGE_METADATA = {
        "baseline_selected_files_gb": 6.115,
        "variant_selected_files_gb": 5.190,
        "metadata_source": "Hugging Face API tree expand=true, checked 2026-05-27",
    }
    return base.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
