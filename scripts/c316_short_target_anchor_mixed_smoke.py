from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c301_short_prefix_anchor_mixed_micro as c301


EXPERIMENT_ID = "C316"
EXPERIMENT_SLUG = "C316_short_target_anchor_mixed_smoke"


def task_probe_source(
    model_id: str,
    train_rows: int,
    val_rows: int,
    steps: int,
    max_seq_len: int,
    max_new_tokens: int,
    seed: int,
) -> str:
    source = c301.task_probe_source(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, seed)
    source = source.replace('data["answer_len"] <= 3600', 'data["answer_len"] <= 240')
    return source


def write_report(path: Path, summary: dict) -> None:
    c301.write_report(path, summary)
    text = path.read_text(encoding="utf-8")
    text = text.replace("# C301 Short-Prefix Anchor-Mixed Micro", "# C316 Short-Target Anchor-Mixed Smoke", 1)
    path.write_text(text, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C316_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c316_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c316_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "24"),
        ("--val-rows", "16"),
        ("--steps", "6"),
        ("--max-seq-len", "512"),
        ("--max-new-tokens", "160"),
        ("--seed", "316"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


if __name__ == "__main__":
    raise SystemExit(run())
