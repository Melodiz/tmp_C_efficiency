from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c273_min_full_reference_sft_unblock as c273


EXPERIMENT_ID = "C274"
EXPERIMENT_SLUG = "C274_full_reference_sft_validation_cap_repair"


def run(argv: Sequence[str] | None = None) -> int:
    c273.c177.EXPERIMENT_ID = EXPERIMENT_ID
    c273.c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c273.c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C274_artifacts"
    c273.c177.DEFAULT_TARGET_DIR = Path("/content/c274_train_site")
    c273.c177.REMOTE_ADAPTER_DIR = Path("/content/c274_adapter_scratch")
    c273.c177.task_probe_source = c273.task_probe_source
    c273.c177.write_report = c273.c178.write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "32"),
        ("--val-rows", "32"),
        ("--steps", "8"),
        ("--max-seq-len", "384"),
        ("--max-new-tokens", "320"),
        ("--seed", "274"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c273.c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
