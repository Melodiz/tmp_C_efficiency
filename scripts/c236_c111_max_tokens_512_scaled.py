from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c235_c111_max_tokens_512 as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C236"
    base.EXPERIMENT_SLUG = "C236_c111_max_tokens_512_scaled"
    base.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C236_artifacts"

    forwarded = list(argv or [])
    if "--sample-size" not in forwarded:
        forwarded.extend(["--sample-size", "2000"])
    if "--seed" not in forwarded:
        forwarded.extend(["--seed", "236"])
    return base.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
