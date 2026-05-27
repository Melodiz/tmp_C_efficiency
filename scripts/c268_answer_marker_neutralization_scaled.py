from __future__ import annotations

import sys
from typing import Sequence

import c267_answer_marker_neutralization as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C268"
    base.EXPERIMENT_SLUG = "C268_answer_marker_neutralization_scaled"
    base.DEFAULT_OUT_DIR = base.Path("artifacts") / "tmp" / "C268_artifacts"

    forwarded = list(argv or [])
    if "--sample-size" not in forwarded:
        forwarded.extend(["--sample-size", "2000"])
    if "--seed" not in forwarded:
        forwarded.extend(["--seed", "268"])
    return base.run(forwarded)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
