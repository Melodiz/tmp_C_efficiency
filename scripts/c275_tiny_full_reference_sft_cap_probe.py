from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c274_full_reference_sft_validation_cap_repair as c274


EXPERIMENT_ID = "C275"
EXPERIMENT_SLUG = "C275_tiny_full_reference_sft_cap_probe"


def run(argv: Sequence[str] | None = None) -> int:
    c274.c273.c177.EXPERIMENT_ID = EXPERIMENT_ID
    c274.c273.c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c274.c273.c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C275_artifacts"
    c274.c273.c177.DEFAULT_TARGET_DIR = Path("/content/c275_train_site")
    c274.c273.c177.REMOTE_ADAPTER_DIR = Path("/content/c275_adapter_scratch")
    c274.c273.c177.task_probe_source = c274.c273.task_probe_source
    c274.c273.c177.write_report = c274.c273.c178.write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "32"),
        ("--val-rows", "8"),
        ("--steps", "8"),
        ("--max-seq-len", "384"),
        ("--max-new-tokens", "320"),
        ("--seed", "275"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c274.c273.c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
