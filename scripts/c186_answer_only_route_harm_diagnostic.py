from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c181_answer_only_tiny_sft_smoke as c181


EXPERIMENT_ID = "C186"
EXPERIMENT_SLUG = "C186_answer_only_route_harm_diagnostic"


def task_probe_source(model_id: str, train_rows: int, val_rows: int, steps: int, max_seq_len: int, max_new_tokens: int, seed: int) -> str:
    source = c181.task_probe_source(model_id, 32, 32, steps, max_seq_len, max_new_tokens, 181)
    source = source.replace(
        '    data["question_len"] = data["question"].astype(str).str.len()\n'
        '    target_pairs = data["reference_answer"].map(answer_only_target)\n'
        '    data["answer_only_target"] = target_pairs.map(lambda item: item[0])\n'
        '    data["target_reject_reason"] = target_pairs.map(lambda item: item[1])\n'
        '    data["answer_len"] = data["answer_only_target"].fillna("").astype(str).str.len()\n'
        '    target_rejection_counts = dict(Counter(data["target_reject_reason"].astype(str)))\n'
        '    pool = data[(data["question_len"] <= 350) & (data["target_reject_reason"] == "ok")].copy()\n'
        '    pool["reference_answer"] = pool["answer_only_target"]\n'
        '    selected = pool.sample(64, random_state=181).to_dict(orient="records")\n'
        '    train = selected[:32]\n'
        '    val = selected[32:64]\n'
        '    result["raw_task_data_read_remote_only"] = True\n'
        '    result["data_meta"] = {\n'
        '        "data_rows": int(len(data)),\n'
        '        "pool_rows": int(len(pool)),\n'
        '        "train_rows": len(train),\n'
        '        "val_rows": len(val),\n'
        '        "train_val_overlap_rows": len(set(r["row_id"] for r in train) & set(r["row_id"] for r in val)),\n'
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        '        "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),\n'
        '        "target_rejection_counts": target_rejection_counts,\n'
        '    }\n',
        f'    data["question_len"] = data["question"].astype(str).str.len()\n'
        f'    target_pairs = data["reference_answer"].map(answer_only_target)\n'
        f'    data["answer_only_target"] = target_pairs.map(lambda item: item[0])\n'
        f'    data["target_reject_reason"] = target_pairs.map(lambda item: item[1])\n'
        f'    data["answer_len"] = data["answer_only_target"].fillna("").astype(str).str.len()\n'
        f'    target_rejection_counts = dict(Counter(data["target_reject_reason"].astype(str)))\n'
        f'    ok_pool = data[(data["question_len"] <= 350) & (data["target_reject_reason"] == "ok")].copy()\n'
        f'    ok_pool["reference_answer"] = ok_pool["answer_only_target"]\n'
        f'    train_df = ok_pool.sample({train_rows}, random_state={seed})\n'
        f'    train = train_df.to_dict(orient="records")\n'
        f'    train_ids = set(train_df["row_id"].tolist())\n'
        f'    ok_val_df = ok_pool[~ok_pool["row_id"].isin(train_ids)].sample(min({val_rows}, max(0, len(ok_pool) - len(train_df))), random_state={seed + 1})\n'
        f'    multiline_df = data[(data["question_len"] <= 350) & (data["target_reject_reason"] == "multiline")].sample(n=min(48, int((data["question_len"] <= 350).mul(data["target_reject_reason"] == "multiline").sum())), random_state={seed + 2})\n'
        f'    long_df = data[(data["question_len"] <= 350) & (data["target_reject_reason"].isin(["long", "essay_like"]))].sample(n=min(32, int(((data["question_len"] <= 350) & (data["target_reject_reason"].isin(["long", "essay_like"]))).sum())), random_state={seed + 3})\n'
        f'    val_strata = {{\n'
        f'        "ok_short": ok_val_df.to_dict(orient="records"),\n'
        f'        "rejected_multiline": multiline_df.to_dict(orient="records"),\n'
        f'        "rejected_long_or_essay": long_df.to_dict(orient="records"),\n'
        f'    }}\n'
        f'    val = [row for rows in val_strata.values() for row in rows]\n'
        f'    result["raw_task_data_read_remote_only"] = True\n'
        f'    result["data_meta"] = {{\n'
        f'        "data_rows": int(len(data)),\n'
        f'        "pool_rows": int(len(ok_pool)),\n'
        f'        "train_rows": len(train),\n'
        f'        "val_rows": len(val),\n'
        f'        "strata_rows": {{name: len(rows) for name, rows in val_strata.items()}},\n'
        f'        "train_val_overlap_rows": len(set(r["row_id"] for r in train) & set(r["row_id"] for r in val)),\n'
        f'        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        f'        "val_shape_counts": {{name: dict(Counter(shape_bucket(r) for r in rows)) for name, rows in val_strata.items()}},\n'
        f'        "target_rejection_counts": target_rejection_counts,\n'
        f'    }}\n',
    )
    source = source.replace("random_state=181", f"random_state={seed}")
    source = source.replace(
        "    base_stats, base_buckets, base_outputs = evaluate(model, tokenizer, val)\n",
        '    base_eval_by_stratum = {}\n'
        '    for name, rows in val_strata.items():\n'
        '        stats, buckets, outputs = evaluate(model, tokenizer, rows)\n'
        '        base_eval_by_stratum[name] = {"stats": stats, "buckets": buckets, "outputs": outputs}\n',
    )
    lora_start = source.index("    lora_stats, lora_buckets, lora_outputs = evaluate(model, tokenizer, val)\n")
    adapter_start = source.index("    adapter_dir.mkdir", lora_start)
    lora_block = (
        '    lora_eval_by_stratum = {}\n'
        '    pair_by_stratum = {}\n'
        '    for name, rows in val_strata.items():\n'
        '        stats, buckets, outputs = evaluate(model, tokenizer, rows)\n'
        '        lora_eval_by_stratum[name] = {"stats": stats, "buckets": buckets, "outputs": outputs}\n'
        '        pair_stats = Counter()\n'
        '        pair_buckets = {}\n'
        '        for base_out, lora_out in zip(base_eval_by_stratum[name]["outputs"], outputs):\n'
        '            changed = base_out["norm"] != lora_out["norm"]\n'
        '            both_exact = base_out["exact"] and lora_out["exact"]\n'
        '            base_only = base_out["exact"] and not lora_out["exact"]\n'
        '            lora_only = lora_out["exact"] and not base_out["exact"]\n'
        '            both_wrong_same = (not base_out["exact"]) and (not lora_out["exact"]) and (not changed)\n'
        '            both_wrong_changed = (not base_out["exact"]) and (not lora_out["exact"]) and changed\n'
        '            for key, value in (\n'
        '                ("changed_output_count", changed),\n'
        '                ("both_exact_count", both_exact),\n'
        '                ("base_only_exact_count", base_only),\n'
        '                ("lora_only_exact_count", lora_only),\n'
        '                ("both_wrong_same_count", both_wrong_same),\n'
        '                ("both_wrong_changed_count", both_wrong_changed),\n'
        '            ):\n'
        '                pair_stats[key] += int(value)\n'
        '            bucket = base_out["bucket"]\n'
        '            b = pair_buckets.setdefault(bucket, Counter())\n'
        '            b["rows"] += 1\n'
        '            b["changed_output_count"] += int(changed)\n'
        '            b["base_only_exact_count"] += int(base_only)\n'
        '            b["lora_only_exact_count"] += int(lora_only)\n'
        '        pair_by_stratum[name] = {"stats": dict(pair_stats), "buckets": {k: dict(v) for k, v in pair_buckets.items()}}\n\n'
    )
    source = source[:lora_start] + lora_block + source[adapter_start:]
    val_start = source.index('    result["validation"] = {\n')
    cleanup_start = source.index("    del model", val_start)
    validation_block = (
        '    strata_validation = {}\n'
        '    for name, rows in val_strata.items():\n'
        '        base_stats = base_eval_by_stratum[name]["stats"]\n'
        '        lora_stats = lora_eval_by_stratum[name]["stats"]\n'
        '        pair_stats = pair_by_stratum[name]["stats"]\n'
        '        strata_validation[name] = {\n'
        '            "val_rows": len(rows),\n'
        '            "base_exact_count": int(base_stats.get("exact", 0)),\n'
        '            "lora_exact_count": int(lora_stats.get("exact", 0)),\n'
        '            "both_exact_count": int(pair_stats.get("both_exact_count", 0)),\n'
        '            "base_only_exact_count": int(pair_stats.get("base_only_exact_count", 0)),\n'
        '            "lora_only_exact_count": int(pair_stats.get("lora_only_exact_count", 0)),\n'
        '            "both_wrong_same_count": int(pair_stats.get("both_wrong_same_count", 0)),\n'
        '            "both_wrong_changed_count": int(pair_stats.get("both_wrong_changed_count", 0)),\n'
        '            "changed_output_count": int(pair_stats.get("changed_output_count", 0)),\n'
        '            "base_ref_in_output_count": int(base_stats.get("ref_in_output", 0)),\n'
        '            "lora_ref_in_output_count": int(lora_stats.get("ref_in_output", 0)),\n'
        '            "base_output_in_ref_count": int(base_stats.get("output_in_ref", 0)),\n'
        '            "lora_output_in_ref_count": int(lora_stats.get("output_in_ref", 0)),\n'
        '            "base_final_exact_count": int(base_stats.get("final_exact", 0)),\n'
        '            "lora_final_exact_count": int(lora_stats.get("final_exact", 0)),\n'
        '            "base_invalid_count": int(base_stats.get("invalid", 0)),\n'
        '            "lora_invalid_count": int(lora_stats.get("invalid", 0)),\n'
        '            "base_cap_hit_count": int(base_stats.get("cap_hit", 0)),\n'
        '            "lora_cap_hit_count": int(lora_stats.get("cap_hit", 0)),\n'
        '            "base_avg_new_tokens": float(base_stats.get("avg_new_tokens_x1000", 0) / 1000.0),\n'
        '            "lora_avg_new_tokens": float(lora_stats.get("avg_new_tokens_x1000", 0) / 1000.0),\n'
        '            "base_shape_buckets": base_eval_by_stratum[name]["buckets"],\n'
        '            "lora_shape_buckets": lora_eval_by_stratum[name]["buckets"],\n'
        '            "pair_shape_buckets": pair_by_stratum[name]["buckets"],\n'
        '        }\n'
        '    result["validation"] = {"total_val_rows": len(val), "strata": strata_validation}\n'
    )
    source = source[:val_start] + validation_block + source[cleanup_start:]
    return source


def write_report(path: Path, summary: dict) -> None:
    data_meta = summary.get("data_meta") or {}
    train = summary.get("train") or {}
    val = summary.get("validation") or {}
    runtime = summary.get("remote_runtime") or {}
    probe = summary.get("probe") or {}
    lines = [
        "# C186 Answer-Only Route/Harm Diagnostic",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Same answer-only training mechanism as C184.",
        "- Aggregate strata diagnostics only; no raw task text, targets, outputs, row ids, model weights, or adapter weights returned.",
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
        f"- total val rows: `{data_meta.get('val_rows')}`",
        f"- strata rows: `{data_meta.get('strata_rows')}`",
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
        "## Validation Strata",
    ]
    for name, stats in (val.get("strata") or {}).items():
        lines.extend(
            [
                f"### {name}",
                f"- val rows: `{stats.get('val_rows')}`",
                f"- base exact / LoRA exact: `{stats.get('base_exact_count')}` / `{stats.get('lora_exact_count')}`",
                f"- base ref-in-output / LoRA ref-in-output: `{stats.get('base_ref_in_output_count')}` / `{stats.get('lora_ref_in_output_count')}`",
                f"- base final-line exact / LoRA final-line exact: `{stats.get('base_final_exact_count')}` / `{stats.get('lora_final_exact_count')}`",
                f"- changed output count: `{stats.get('changed_output_count')}`",
                f"- base-only exact / LoRA-only exact: `{stats.get('base_only_exact_count')}` / `{stats.get('lora_only_exact_count')}`",
                f"- base invalid/cap hit: `{stats.get('base_invalid_count')}` / `{stats.get('base_cap_hit_count')}`",
                f"- LoRA invalid/cap hit: `{stats.get('lora_invalid_count')}` / `{stats.get('lora_cap_hit_count')}`",
                f"- base avg new tokens: `{stats.get('base_avg_new_tokens')}`",
                f"- LoRA avg new tokens: `{stats.get('lora_avg_new_tokens')}`",
                f"- pair shape buckets: `{stats.get('pair_shape_buckets')}`",
                "",
            ]
        )
    lines.extend(
        [
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
            "Proceed only if the answer-only signal repeats on compatible rows without broad harm on rejected strata.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C186_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c186_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c186_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    for flag, value in (("--train-rows", "96"), ("--val-rows", "96"), ("--steps", "48"), ("--max-new-tokens", "24"), ("--seed", "186")):
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
