from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c227_phi4_mini_paired_aggregate as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C287"
    base.EXPERIMENT_SLUG = "C287_qwen3_8b_dense_paired_aggregate"
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C287_artifacts"
    base.VARIANT_MODEL_ID = "Qwen/Qwen3-8B"
    base.MODEL_PACKAGE_METADATA = {
        "baseline_model": "Qwen/Qwen3-8B-AWQ",
        "variant_model": "Qwen/Qwen3-8B",
        "metadata_source": "Hugging Face public metadata with file sizes, 2026-05-27",
        "baseline_selected_files_gb": 6.11,
        "variant_selected_files_gb": 16.40,
        "status": "accessible and ungated; same-base dense serving diagnostic with package/VRAM risk",
    }
    forwarded = list(argv or [])
    if "--sample-size" not in forwarded:
        forwarded.extend(["--sample-size", "256"])
    if "--seed" not in forwarded:
        forwarded.extend(["--seed", "287"])
    return base.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
