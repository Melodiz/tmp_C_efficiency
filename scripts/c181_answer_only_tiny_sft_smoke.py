from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c178_sft_aggregate_metric_cap_diagnostic as c178


EXPERIMENT_ID = "C181"
EXPERIMENT_SLUG = "C181_answer_only_tiny_sft_smoke"


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
        'def answer_only_target(value):\n'
        '    text = str(value).strip()\n'
        '    text = re.sub(r"^(ответ|итог|answer)\\s*[:：-]\\s*", "", text, flags=re.IGNORECASE).strip()\n'
        '    lines = [part.strip() for part in text.splitlines() if part.strip()]\n'
        '    if not lines:\n'
        '        return None, "empty"\n'
        '    target = lines[0]\n'
        '    if len(lines) > 1:\n'
        '        return None, "multiline"\n'
        '    if len(target) > 80:\n'
        '        return None, "long"\n'
        '    if len(target.split()) > 14:\n'
        '        return None, "essay_like"\n'
        '    return target, "ok"\n',
    )
    source = source.replace(
        '    data["question_len"] = data["question"].astype(str).str.len()\n'
        '    data["answer_len"] = data["reference_answer"].astype(str).str.len()\n'
        '    pool = data[(data["question_len"] <= 350) & (data["answer_len"] <= 80)].copy()\n'
        '    selected = pool.sample(64, random_state=181).to_dict(orient="records")\n',
        '    data["question_len"] = data["question"].astype(str).str.len()\n'
        '    target_pairs = data["reference_answer"].map(answer_only_target)\n'
        '    data["answer_only_target"] = target_pairs.map(lambda item: item[0])\n'
        '    data["target_reject_reason"] = target_pairs.map(lambda item: item[1])\n'
        '    data["answer_len"] = data["answer_only_target"].fillna("").astype(str).str.len()\n'
        '    target_rejection_counts = dict(Counter(data["target_reject_reason"].astype(str)))\n'
        '    pool = data[(data["question_len"] <= 350) & (data["target_reject_reason"] == "ok")].copy()\n'
        '    pool["reference_answer"] = pool["answer_only_target"]\n'
        '    selected = pool.sample(64, random_state=181).to_dict(orient="records")\n',
    )
    source = source.replace(
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        '        "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),\n',
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        '        "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),\n'
        '        "target_rejection_counts": target_rejection_counts,\n',
    )
    return source


def write_report(path: Path, summary: dict) -> None:
    data_meta = summary.get("data_meta") or {}
    train = summary.get("train") or {}
    val = summary.get("validation") or {}
    runtime = summary.get("remote_runtime") or {}
    probe = summary.get("probe") or {}
    lines = [
        "# C181 Answer-Only Tiny SFT Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Answer-only training target formulation.",
        "- Aggregate diagnostics only; no raw task text, targets, outputs, row ids, model weights, or adapter weights returned.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- install return code: `{summary.get('install_returncode')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        "",
        "## Data",
        f"- remote data read: `{probe.get('raw_task_data_read_remote_only')}`",
        f"- pool rows: `{data_meta.get('pool_rows')}`",
        f"- train rows: `{data_meta.get('train_rows')}`",
        f"- val rows: `{data_meta.get('val_rows')}`",
        f"- train/val overlap rows: `{data_meta.get('train_val_overlap_rows')}`",
        f"- target rejection counts: `{data_meta.get('target_rejection_counts')}`",
        f"- val shape counts: `{data_meta.get('val_shape_counts')}`",
        "",
        "## Train",
        f"- steps: `{train.get('steps')}`",
        f"- losses: `{train.get('losses')}`",
        f"- loss finite: `{train.get('loss_finite')}`",
        f"- adapter scratch deleted: `{probe.get('adapter_scratch_deleted')}`",
        "",
        "## Validation",
        f"- val rows: `{val.get('val_rows')}`",
        f"- base exact / LoRA exact: `{val.get('base_exact_count')}` / `{val.get('lora_exact_count')}`",
        f"- base ref-in-output / LoRA ref-in-output: `{val.get('base_ref_in_output_count')}` / `{val.get('lora_ref_in_output_count')}`",
        f"- base final-line exact / LoRA final-line exact: `{val.get('base_final_exact_count')}` / `{val.get('lora_final_exact_count')}`",
        f"- changed output count: `{val.get('changed_output_count')}`",
        f"- base-only exact / LoRA-only exact: `{val.get('base_only_exact_count')}` / `{val.get('lora_only_exact_count')}`",
        f"- base invalid/cap hit: `{val.get('base_invalid_count')}` / `{val.get('base_cap_hit_count')}`",
        f"- LoRA invalid/cap hit: `{val.get('lora_invalid_count')}` / `{val.get('lora_cap_hit_count')}`",
        f"- base avg new tokens: `{val.get('base_avg_new_tokens')}`",
        f"- LoRA avg new tokens: `{val.get('lora_avg_new_tokens')}`",
        f"- pair shape buckets: `{val.get('pair_shape_buckets')}`",
        "",
        "## Runtime",
        f"- remote seconds: `{runtime.get('total_seconds')}`",
        f"- vram after load MB: `{runtime.get('vram_after_load_mb')}`",
        "",
        "## Hygiene",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
        f"- error: `{probe.get('error')}`" if probe.get("error") else "- error: none",
        "",
        "## Next",
        "Scale only if LoRA shows aggregate exact/final-line wins or a large cap-hit reduction without regressions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C181_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c181_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c181_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    for flag, value in (("--train-rows", "32"), ("--val-rows", "32"), ("--steps", "16"), ("--max-new-tokens", "24"), ("--seed", "181")):
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
