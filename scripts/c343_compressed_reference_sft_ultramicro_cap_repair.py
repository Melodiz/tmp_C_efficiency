from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c341_compressed_reference_sft_light_bootstrap_smoke as c341


EXPERIMENT_ID = "C343"
EXPERIMENT_SLUG = "C343_compressed_reference_sft_ultramicro_cap_repair"


def run(argv: Sequence[str] | None = None) -> int:
    c341.EXPERIMENT_ID = EXPERIMENT_ID
    c341.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c341.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C343_artifacts"
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "24"),
        ("--val-rows", "8"),
        ("--steps", "6"),
        ("--max-new-tokens", "320"),
        ("--seed", "343"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c341.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
