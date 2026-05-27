from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c178_sft_aggregate_metric_cap_diagnostic as c178
import c272_full_reference_sft_smoke as c272


EXPERIMENT_ID = "C273"
EXPERIMENT_SLUG = "C273_min_full_reference_sft_unblock"


def task_probe_source(
    model_id: str,
    train_rows: int,
    val_rows: int,
    steps: int,
    max_seq_len: int,
    max_new_tokens: int,
    seed: int,
) -> str:
    source = c272.task_probe_source(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, seed)
    return source.replace(
        'lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")',
        'lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")',
    )


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C273_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c273_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c273_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = c178.write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "32"),
        ("--val-rows", "32"),
        ("--steps", "8"),
        ("--max-seq-len", "384"),
        ("--max-new-tokens", "96"),
        ("--seed", "273"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
