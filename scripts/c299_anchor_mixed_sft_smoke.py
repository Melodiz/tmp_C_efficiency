from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c178_sft_aggregate_metric_cap_diagnostic as c178


EXPERIMENT_ID = "C299"
EXPERIMENT_SLUG = "C299_anchor_mixed_sft_smoke"
ANCHOR_PREFIX = "Ответь на языке задания. Дай полный, аккуратный учебный ответ в стиле решения."


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
        'def invalid_output(text):\n'
        '    text = str(text).strip()\n'
        '    return (not text) or len(text) > 700 or text.count("\\n") > 10\n',
        'def invalid_output(text):\n'
        '    text = str(text).strip()\n'
        '    return (not text) or len(text) > 700 or text.count("\\n") > 10\n'
        '\n'
        'def repetition_loop(text):\n'
        '    compact = re.sub(r"\\s+", " ", str(text).strip().lower())\n'
        '    if not compact:\n'
        '        return False\n'
        '    chunks = re.findall(r"[\\wа-яА-ЯёЁ+\\-*/=,.%]+", compact)\n'
        '    if len(chunks) >= 18 and len(set(chunks[-12:])) <= 3:\n'
        '        return True\n'
        '    return bool(re.search(r"(.{12,80})\\1\\1", compact))\n'
        '\n'
        'def anchor_ok(text, new_tokens, reference):\n'
        '    text = str(text).strip()\n'
        f'    if invalid_output(text) or repetition_loop(text) or new_tokens >= {max_new_tokens}:\n'
        '        return False\n'
        '    ref_tokens = max(1, len(str(reference).split()))\n'
        '    out_tokens = max(1, len(text.split()))\n'
        '    ratio = out_tokens / ref_tokens\n'
        '    return 0.15 <= ratio <= 6.0 and 4 <= out_tokens <= 220\n',
    )
    source = source.replace(
        'def build_messages(question, answer=None):\n'
        '    messages = [{"role": "user", "content": "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ.\\n\\n" + str(question)}]\n'
        '    if answer is not None:\n'
        '        messages.append({"role": "assistant", "content": str(answer)})\n'
        '    return messages\n',
        'def build_messages(question, answer=None):\n'
        f'    messages = [{{"role": "user", "content": "{ANCHOR_PREFIX}\\n\\n" + str(question)}}]\n'
        '    if answer is not None:\n'
        '        messages.append({"role": "assistant", "content": str(answer)})\n'
        '    return messages\n',
    )
    source = source.replace(
        '        for key, value in (("exact", exact), ("ref_in_output", ref_in_output), ("output_in_ref", output_in_ref), ("final_exact", final_exact), ("invalid", invalid), ("cap_hit", cap_hit)):\n'
        '            stats[key] += int(value)\n'
        '            bucket_stats[key] += int(value)\n',
        '        repeat = repetition_loop(text)\n'
        '        for key, value in (("exact", exact), ("ref_in_output", ref_in_output), ("output_in_ref", output_in_ref), ("final_exact", final_exact), ("invalid", invalid), ("cap_hit", cap_hit), ("repetition", repeat)):\n'
        '            stats[key] += int(value)\n'
        '            bucket_stats[key] += int(value)\n',
    )
    source = source.replace(
        'outputs.append({"norm": norm_text, "exact": exact, "ref_in_output": ref_in_output, "output_in_ref": output_in_ref, "final_exact": final_exact, "invalid": invalid, "cap_hit": cap_hit, "new_tokens": new_tokens, "bucket": bucket})',
        'outputs.append({"norm": norm_text, "exact": exact, "ref_in_output": ref_in_output, "output_in_ref": output_in_ref, "final_exact": final_exact, "invalid": invalid, "cap_hit": cap_hit, "repetition": repeat, "new_tokens": new_tokens, "bucket": bucket})',
    )
    source = source.replace(
        '    pool = data[(data["question_len"] <= 350) & (data["answer_len"] <= 80)].copy()\n'
        f'    selected = pool.sample({train_rows + val_rows}, random_state={seed}).to_dict(orient="records")\n'
        f'    train = selected[:{train_rows}]\n'
        f'    val = selected[{train_rows}:{train_rows + val_rows}]\n',
        '    pool = data[(data["question_len"] <= 500) & (data["answer_len"] <= 3600)].copy()\n'
        f'    selected = pool.sample(min({train_rows + val_rows}, len(pool)), random_state={seed}).to_dict(orient="records")\n'
        f'    train_candidates = selected[:{train_rows}]\n'
        f'    val = selected[{train_rows}:{train_rows + val_rows}]\n'
        '    train = []\n'
        '    anchor_candidates = []\n',
    )
    source = source.replace(
        '        "train_rows": len(train),\n',
        '        "train_rows": len(train_candidates),\n'
        '        "mixed_train_rows": 0,\n'
        '        "anchor_train_rows": 0,\n'
        '        "full_reference_train_rows": 0,\n',
    )
    source = source.replace(
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n',
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train_candidates)),\n',
    )
    source = source.replace(
        '    model.eval()\n'
        '    base_stats, base_buckets, base_outputs = evaluate(model, tokenizer, val)\n'
        '\n'
        '    model = prepare_model_for_kbit_training(model)\n',
        '    model.eval()\n'
        '    base_stats, base_buckets, base_outputs = evaluate(model, tokenizer, val)\n'
        '    for row in train_candidates:\n'
        '        prompt = tokenizer.apply_chat_template(build_messages(row["question"]), tokenize=False, add_generation_prompt=True)\n'
        f'        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length={max_seq_len}).to(model.device)\n'
        f'        output_ids = model.generate(**inputs, max_new_tokens={max_new_tokens}, do_sample=False, pad_token_id=tokenizer.eos_token_id)\n'
        '        gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]\n'
        '        new_tokens = int(gen_ids.shape[-1])\n'
        '        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()\n'
        '        if anchor_ok(text, new_tokens, row["reference_answer"]):\n'
        '            copied = dict(row)\n'
        '            copied["training_target"] = text\n'
        '            copied["target_kind"] = "base_anchor"\n'
        '            anchor_candidates.append(copied)\n'
        '    min_anchor_rows = max(1, len(train_candidates) // 10)\n'
        '    if len(anchor_candidates) < min_anchor_rows:\n'
        '        result["data_meta"]["anchor_candidate_rows"] = len(anchor_candidates)\n'
        '        result["data_meta"]["anchor_pool_gate_failed"] = True\n'
        '        result["status"] = "completed"\n'
        '        result["runtime"] = {"total_seconds": time.time() - start, "vram_before_load_mb": before_load, "vram_after_load_mb": after_load, "vram_after_cleanup_mb": gpu_memory_mb()}\n'
        '        print(json.dumps(result, ensure_ascii=False, indent=2))\n'
        '        raise SystemExit(0)\n'
        '    anchor_rows = max(1, len(train_candidates) // 4)\n'
        '    anchor_train = anchor_candidates[:anchor_rows]\n'
        '    anchor_source_ids = set(r["row_id"] for r in anchor_train)\n'
        '    full_train = []\n'
        '    for row in train_candidates:\n'
        '        if row["row_id"] in anchor_source_ids:\n'
        '            continue\n'
        '        copied = dict(row)\n'
        '        copied["training_target"] = copied["reference_answer"]\n'
        '        copied["target_kind"] = "full_reference"\n'
        '        full_train.append(copied)\n'
        '        if len(full_train) >= max(1, len(train_candidates) - len(anchor_train)):\n'
        '            break\n'
        '    train = full_train + anchor_train\n'
        f'    random.Random({seed}).shuffle(train)\n'
        '    result["data_meta"]["mixed_train_rows"] = len(train)\n'
        '    result["data_meta"]["anchor_candidate_rows"] = len(anchor_candidates)\n'
        '    result["data_meta"]["anchor_train_rows"] = len(anchor_train)\n'
        '    result["data_meta"]["full_reference_train_rows"] = len(full_train)\n'
        '    result["data_meta"]["anchor_pool_gate_failed"] = False\n'
        '\n'
        '    model = prepare_model_for_kbit_training(model)\n',
    )
    source = source.replace(
        '        full = tokenizer.apply_chat_template(build_messages(row["question"], row["reference_answer"]), tokenize=False)\n',
        '        full = tokenizer.apply_chat_template(build_messages(row["question"], row["training_target"]), tokenize=False)\n',
    )
    source = source.replace(
        '        "base_cap_hit_count": int(base_stats.get("cap_hit", 0)),\n'
        '        "lora_cap_hit_count": int(lora_stats.get("cap_hit", 0)),\n',
        '        "base_cap_hit_count": int(base_stats.get("cap_hit", 0)),\n'
        '        "lora_cap_hit_count": int(lora_stats.get("cap_hit", 0)),\n'
        '        "base_repetition_count": int(base_stats.get("repetition", 0)),\n'
        '        "lora_repetition_count": int(lora_stats.get("repetition", 0)),\n',
    )
    return source


def write_report(path: Path, summary: dict) -> None:
    data_meta = summary.get("data_meta") or {}
    train = summary.get("train") or {}
    val = summary.get("validation") or {}
    runtime = summary.get("remote_runtime") or {}
    probe = summary.get("probe") or {}
    lines = [
        "# C299 Anchor-Mixed SFT Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- One mechanism: q/v rank-8 LoRA trained on mixed full-reference plus valid base-output anchor targets.",
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
        f"- train candidate rows: `{data_meta.get('train_rows')}`",
        f"- mixed train rows: `{data_meta.get('mixed_train_rows')}`",
        f"- anchor candidates / anchor train rows: `{data_meta.get('anchor_candidate_rows')}` / `{data_meta.get('anchor_train_rows')}`",
        f"- full-reference train rows: `{data_meta.get('full_reference_train_rows')}`",
        f"- anchor pool gate failed: `{data_meta.get('anchor_pool_gate_failed')}`",
        f"- val rows: `{data_meta.get('val_rows')}`",
        f"- train/val overlap rows: `{data_meta.get('train_val_overlap_rows')}`",
        f"- val shape counts: `{data_meta.get('val_shape_counts')}`",
        "",
        "## Train",
        f"- steps: `{train.get('steps')}`",
        f"- losses: `{train.get('losses')}`",
        f"- loss finite: `{train.get('loss_finite')}`",
        f"- trainable params: `{train.get('trainable_params')}`",
        f"- adapter scratch deleted: `{probe.get('adapter_scratch_deleted')}`",
        "",
        "## Validation",
        f"- val rows: `{val.get('val_rows')}`",
        f"- base exact / LoRA exact: `{val.get('base_exact_count')}` / `{val.get('lora_exact_count')}`",
        f"- base ref-in-output / LoRA ref-in-output: `{val.get('base_ref_in_output_count')}` / `{val.get('lora_ref_in_output_count')}`",
        f"- base output-in-ref / LoRA output-in-ref: `{val.get('base_output_in_ref_count')}` / `{val.get('lora_output_in_ref_count')}`",
        f"- base final-line exact / LoRA final-line exact: `{val.get('base_final_exact_count')}` / `{val.get('lora_final_exact_count')}`",
        f"- changed output count: `{val.get('changed_output_count')}`",
        f"- base invalid/cap/repetition: `{val.get('base_invalid_count')}` / `{val.get('base_cap_hit_count')}` / `{val.get('base_repetition_count')}`",
        f"- LoRA invalid/cap/repetition: `{val.get('lora_invalid_count')}` / `{val.get('lora_cap_hit_count')}` / `{val.get('lora_repetition_count')}`",
        f"- base avg new tokens: `{val.get('base_avg_new_tokens')}`",
        f"- LoRA avg new tokens: `{val.get('lora_avg_new_tokens')}`",
        f"- pair shape buckets: `{val.get('pair_shape_buckets')}`",
        "",
        "## Runtime",
        f"- remote seconds: `{runtime.get('total_seconds')}`",
        f"- vram after load MB: `{runtime.get('vram_after_load_mb')}`",
        f"- vram after cleanup MB: `{runtime.get('vram_after_cleanup_mb')}`",
        "",
        "## Hygiene",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
        f"- error: `{probe.get('error')}`" if probe.get("error") else "- error: none",
        "",
        "## Next",
        "Kill unless cap/invalid improve versus the full-reference SFT failure pattern and at least one containment metric improves without the other falling.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C299_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c299_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c299_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "64"),
        ("--val-rows", "64"),
        ("--steps", "32"),
        ("--max-seq-len", "768"),
        ("--max-new-tokens", "320"),
        ("--seed", "299"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
