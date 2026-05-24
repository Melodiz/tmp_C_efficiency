from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c075_deterministic_guard as guard


guard.EXPERIMENT_ID = "C077"
guard.EXPERIMENT_SLUG = "C077_slash_fraction_guard_abstention"
guard.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C077_artifacts"


def run(argv: Sequence[str] | None = None) -> int:
    forwarded = ["--sample-source", "locked_val"]
    if argv:
        forwarded.extend(argv)
    return guard.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
