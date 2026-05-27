from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c178_sft_aggregate_metric_cap_diagnostic as c178


EXPERIMENT_ID = "C272"
EXPERIMENT_SLUG = "C272_full_reference_sft_smoke"
FULL_REFERENCE_PREFIX = "Ответь на языке задания. Дай полный, аккуратный учебный ответ в стиле решения."


def task_probe_source(
    model_id: str,
    train_rows: int,
    val_rows: int,
    steps: int,
    max_seq_len: int,
    max_new_tokens: int,
    seed: int,
) -> str:
    source = c178.task_probe_source(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, seed)
    source = source.replace(
        'def build_messages(question, answer=None):\n'
        '    messages = [{"role": "user", "content": "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ.\\n\\n" + str(question)}]\n'
        '    if answer is not None:\n'
        '        messages.append({"role": "assistant", "content": str(answer)})\n'
        '    return messages\n',
        'def build_messages(question, answer=None):\n'
        f'    messages = [{{"role": "user", "content": "{FULL_REFERENCE_PREFIX}\\n\\n" + str(question)}}]\n'
        '    if answer is not None:\n'
        '        messages.append({"role": "assistant", "content": str(answer)})\n'
        '    return messages\n',
    )
    source = source.replace(
        '    pool = data[(data["question_len"] <= 350) & (data["answer_len"] <= 80)].copy()\n',
        '    pool = data[(data["question_len"] <= 500) & (data["answer_len"] <= 3600)].copy()\n',
    )
    source = source.replace(
        '    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")\n',
        '    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")\n',
    )
    source = source.replace(
        '    result["data_meta"] = {\n'
        '        "data_rows": int(len(data)),\n'
        '        "pool_rows": int(len(pool)),\n',
        '    result["data_meta"] = {\n'
        '        "data_rows": int(len(data)),\n'
        '        "pool_rows": int(len(pool)),\n'
        '        "training_target": "full_reference_answer",\n'
        '        "prefix": "full_reference_style",\n',
    )
    return source


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C272_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c272_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c272_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = c178.write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "192"),
        ("--val-rows", "96"),
        ("--steps", "96"),
        ("--max-seq-len", "768"),
        ("--max-new-tokens", "192"),
        ("--seed", "272"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
