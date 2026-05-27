from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c227_phi4_mini_paired_aggregate as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C281"
    base.EXPERIMENT_SLUG = "C281_mistral7b_awq_paired_aggregate"
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C281_artifacts"
    base.VARIANT_MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    base.MODEL_PACKAGE_METADATA = {
        "baseline_model": "Qwen/Qwen3-8B-AWQ",
        "variant_model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "metadata_source": "Hugging Face public API access check, 2026-05-27",
        "status": "public accessible; exploratory package-plausible AWQ model-family diagnostic",
    }
    forwarded = list(argv or [])
    if "--sample-size" not in forwarded:
        forwarded.extend(["--sample-size", "256"])
    if "--seed" not in forwarded:
        forwarded.extend(["--seed", "281"])
    return base.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
