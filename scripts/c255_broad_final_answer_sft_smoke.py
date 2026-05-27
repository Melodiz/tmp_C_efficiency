from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c178_sft_aggregate_metric_cap_diagnostic as c178
import c181_answer_only_tiny_sft_smoke as c181


EXPERIMENT_ID = "C255"
EXPERIMENT_SLUG = "C255_broad_final_answer_sft_smoke"


def task_probe_source(model_id: str, train_rows: int, val_rows: int, steps: int, max_seq_len: int, max_new_tokens: int, seed: int) -> str:
    source = c178.task_probe_source(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, seed)
    source = source.replace(
        'def build_messages(question, answer=None):\n'
        '    messages = [{"role": "user", "content": "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ.\\n\\n" + str(question)}]\n'
        '    if answer is not None:\n'
        '        messages.append({"role": "assistant", "content": str(answer)})\n'
        '    return messages\n',
        'def build_messages(question, answer=None):\n'
        '    messages = [{"role": "user", "content": "Ответь кратко и точно на языке задания. Не повторяй условие. Дай только итоговый ответ.\\n\\n" + str(question)}]\n'
        '    if answer is not None:\n'
        '        messages.append({"role": "assistant", "content": str(answer)})\n'
        '    return messages\n'
        '\n'
        'def final_answer_target(value):\n'
        '    text = str(value).strip()\n'
        '    text = re.sub(r"^(ответ|итоговый ответ|итог|answer|final answer)\\s*[:：-]\\s*", "", text, flags=re.IGNORECASE).strip()\n'
        '    lines = [part.strip() for part in text.splitlines() if part.strip()]\n'
        '    if not lines:\n'
        '        return None, "empty"\n'
        '    cleaned = []\n'
        '    for line in lines:\n'
        '        line = re.sub(r"^(ответ|итоговый ответ|итог|answer|final answer)\\s*[:：-]\\s*", "", line, flags=re.IGNORECASE).strip()\n'
        '        line = line.strip(" .;:")\n'
        '        if not line:\n'
        '            continue\n'
        '        if len(line) <= 100 and len(line.split()) <= 18:\n'
        '            cleaned.append(line)\n'
        '    if not cleaned:\n'
        '        return None, "no_short_line"\n'
        '    return cleaned[-1], "ok"\n',
    )
    source = source.replace(
        '    data["question_len"] = data["question"].astype(str).str.len()\n'
        '    data["answer_len"] = data["reference_answer"].astype(str).str.len()\n'
        '    pool = data[(data["question_len"] <= 350) & (data["answer_len"] <= 80)].copy()\n'
        f'    selected = pool.sample({train_rows + val_rows}, random_state={seed}).to_dict(orient="records")\n',
        '    data["question_len"] = data["question"].astype(str).str.len()\n'
        '    target_pairs = data["reference_answer"].map(final_answer_target)\n'
        '    data["final_answer_target"] = target_pairs.map(lambda item: item[0])\n'
        '    data["target_reject_reason"] = target_pairs.map(lambda item: item[1])\n'
        '    data["answer_len"] = data["final_answer_target"].fillna("").astype(str).str.len()\n'
        '    target_rejection_counts = dict(Counter(data["target_reject_reason"].astype(str)))\n'
        '    pool = data[(data["question_len"] <= 350) & (data["target_reject_reason"] == "ok")].copy()\n'
        '    pool["reference_answer"] = pool["final_answer_target"]\n'
        f'    selected = pool.sample({train_rows + val_rows}, random_state={seed}).to_dict(orient="records")\n',
    )
    source = source.replace(
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        '        "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),\n',
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        '        "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),\n'
        '        "target_rejection_counts": target_rejection_counts,\n',
    )
    return source


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C255_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c255_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c255_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = c181.write_report
    forwarded = list(argv or [])
    for flag, value in (("--train-rows", "96"), ("--val-rows", "96"), ("--steps", "48"), ("--max-new-tokens", "24"), ("--seed", "255")):
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
