from __future__ import annotations

import sys
from typing import Sequence

import c195_direct_probe_aggregate_validation as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C196"
    base.EXPERIMENT_SLUG = "C196_current_stack_scaled_aggregate_validation"
    base.DEFAULT_OUT_DIR = base.Path("artifacts") / "tmp" / "C196_artifacts"
    base.DEFAULT_SAMPLE_SIZE = 256
    base.DEFAULT_SEED = 196

    forwarded = list(argv or [])
    if "--sample-size" not in forwarded:
        forwarded.extend(["--sample-size", str(base.DEFAULT_SAMPLE_SIZE)])
    if "--seed" not in forwarded:
        forwarded.extend(["--seed", str(base.DEFAULT_SEED)])
    return base.run(forwarded)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
