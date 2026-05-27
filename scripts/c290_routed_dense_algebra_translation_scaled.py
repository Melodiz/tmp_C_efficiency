from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c289_routed_dense_algebra_translation as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C290"
    base.EXPERIMENT_SLUG = "C290_routed_dense_algebra_translation_scaled"
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C290_artifacts"
    forwarded = list(argv or [])
    if "--sample-size" not in forwarded:
        forwarded.extend(["--sample-size", "2000"])
    if "--seed" not in forwarded:
        forwarded.extend(["--seed", "290"])
    return base.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
