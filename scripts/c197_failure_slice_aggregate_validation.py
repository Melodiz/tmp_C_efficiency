from __future__ import annotations

import sys
from typing import Sequence

import c195_direct_probe_aggregate_validation as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C197"
    base.EXPERIMENT_SLUG = "C197_failure_slice_aggregate_validation"
    base.DEFAULT_OUT_DIR = base.Path("artifacts") / "tmp" / "C197_artifacts"
    base.DEFAULT_SAMPLE_SIZE = 512
    base.DEFAULT_SEED = 197

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
