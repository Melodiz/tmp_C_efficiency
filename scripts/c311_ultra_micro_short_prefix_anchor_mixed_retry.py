from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c301_short_prefix_anchor_mixed_micro as c301


EXPERIMENT_ID = "C311"
EXPERIMENT_SLUG = "C311_ultra_micro_short_prefix_anchor_mixed_retry"


def write_report(path: Path, summary: dict) -> None:
    c301.write_report(path, summary)
    text = path.read_text(encoding="utf-8")
    text = text.replace("# C301 Short-Prefix Anchor-Mixed Micro", "# C311 Ultra-Micro Short-Prefix Anchor-Mixed Retry", 1)
    path.write_text(text, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C311_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c311_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c311_adapter_scratch")
    c177.task_probe_source = c301.task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "8"),
        ("--val-rows", "4"),
        ("--steps", "2"),
        ("--max-seq-len", "512"),
        ("--max-new-tokens", "320"),
        ("--seed", "311"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


if __name__ == "__main__":
    raise SystemExit(run())
