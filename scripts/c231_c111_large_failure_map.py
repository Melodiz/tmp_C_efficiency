from __future__ import annotations

import sys
from typing import Sequence

import c195_direct_probe_aggregate_validation as base


def run(argv: Sequence[str] | None = None) -> int:
    base.EXPERIMENT_ID = "C231"
    base.EXPERIMENT_SLUG = "C231_c111_large_failure_map"
    base.DEFAULT_OUT_DIR = base.Path("artifacts") / "tmp" / "C231_artifacts"
    base.DEFAULT_SAMPLE_SIZE = 2048
    base.DEFAULT_SEED = 231

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
