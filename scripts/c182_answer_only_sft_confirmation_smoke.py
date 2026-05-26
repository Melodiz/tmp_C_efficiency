from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c181_answer_only_tiny_sft_smoke as c181


EXPERIMENT_ID = "C182"
EXPERIMENT_SLUG = "C182_answer_only_sft_confirmation_smoke"


def task_probe_source(model_id: str, train_rows: int, val_rows: int, steps: int, max_seq_len: int, max_new_tokens: int, seed: int) -> str:
    source = c181.task_probe_source(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, 181)
    return source.replace("random_state=181", f"random_state={seed}")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C182_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c182_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c182_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = c181.write_report
    forwarded = list(argv or [])
    for flag, value in (("--train-rows", "32"), ("--val-rows", "32"), ("--steps", "16"), ("--max-new-tokens", "24"), ("--seed", "182")):
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
