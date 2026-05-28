from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c299_anchor_mixed_sft_smoke as c299


EXPERIMENT_ID = "C300"
EXPERIMENT_SLUG = "C300_anchor_mixed_sft_micro_retry"


def write_report(path: Path, summary: dict) -> None:
    c299.write_report(path, summary)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("# C299 Anchor-Mixed SFT Smoke", "# C300 Anchor-Mixed SFT Micro Retry", 1), encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C300_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c300_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c300_adapter_scratch")
    c177.task_probe_source = c299.task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "24"),
        ("--val-rows", "16"),
        ("--steps", "8"),
        ("--max-seq-len", "512"),
        ("--max-new-tokens", "192"),
        ("--seed", "300"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
