from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c181_answer_only_tiny_sft_smoke as c181


EXPERIMENT_ID = "C207"
EXPERIMENT_SLUG = "C207_routed_answer_only_adapter_diagnostic"


def task_probe_source(model_id: str, train_rows: int, val_rows: int, steps: int, max_seq_len: int, max_new_tokens: int, seed: int) -> str:
    source = c181.task_probe_source(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, 181)
    source = source.replace("random_state=181", f"random_state={seed}")
    source = source.replace(
        '        def invalid_output(text):\n'
        '            text = str(text).strip()\n'
        '            return (not text) or len(text) > 700 or text.count("\\n") > 10\n',
        '        def invalid_output(text):\n'
        '            text = str(text).strip()\n'
        '            return (not text) or len(text) > 700 or text.count("\\n") > 10\n'
        '\n'
        '        def repetition_loop(text):\n'
        '            compact = re.sub(r"\\s+", " ", str(text).strip().lower())\n'
        '            if not compact:\n'
        '                return False\n'
        '            chunks = re.findall(r"[\\wа-яА-ЯёЁ+\\-*/=,.%]+", compact)\n'
        '            if len(chunks) >= 18 and len(set(chunks[-12:])) <= 3:\n'
        '                return True\n'
        '            return bool(re.search(r"(.{12,80})\\1\\1", compact))\n',
    )
    source = source.replace(
        '                invalid = invalid_output(text)\n'
        '                cap_hit = new_tokens >= {max_new_tokens}\n'
        '                bucket = shape_bucket(row)\n'
        '                bucket_stats = buckets.setdefault(bucket, Counter())\n'
        '                for key, value in (("exact", exact), ("ref_in_output", ref_in_output), ("output_in_ref", output_in_ref), ("final_exact", final_exact), ("invalid", invalid), ("cap_hit", cap_hit)):\n'
        '                    stats[key] += int(value)\n'
        '                    bucket_stats[key] += int(value)\n'
        '                stats["rows"] += 1\n'
        '                bucket_stats["rows"] += 1\n'
        '                outputs.append({"norm": norm_text, "exact": exact, "ref_in_output": ref_in_output, "output_in_ref": output_in_ref, "final_exact": final_exact, "invalid": invalid, "cap_hit": cap_hit, "new_tokens": new_tokens, "bucket": bucket})\n',
        '                invalid = invalid_output(text)\n'
        '                cap_hit = new_tokens >= {max_new_tokens}\n'
        '                repeat = repetition_loop(text)\n'
        '                bucket = shape_bucket(row)\n'
        '                bucket_stats = buckets.setdefault(bucket, Counter())\n'
        '                for key, value in (("exact", exact), ("ref_in_output", ref_in_output), ("output_in_ref", output_in_ref), ("final_exact", final_exact), ("invalid", invalid), ("cap_hit", cap_hit), ("repetition", repeat)):\n'
        '                    stats[key] += int(value)\n'
        '                    bucket_stats[key] += int(value)\n'
        '                stats["rows"] += 1\n'
        '                bucket_stats["rows"] += 1\n'
        '                outputs.append({"norm": norm_text, "exact": exact, "ref_in_output": ref_in_output, "output_in_ref": output_in_ref, "final_exact": final_exact, "invalid": invalid, "cap_hit": cap_hit, "repetition": repeat, "new_tokens": new_tokens, "bucket": bucket})\n',
    )
    source = source.replace(
        '            pool = data[(data["question_len"] <= 350) & (data["target_reject_reason"] == "ok")].copy()\n'
        '            pool["reference_answer"] = pool["answer_only_target"]\n'
        '            selected = pool.sample(64, random_state=181).to_dict(orient="records")\n'
        f'            train = selected[:{train_rows}]\n'
        f'            val = selected[{train_rows}:{train_rows + val_rows}]\n',
        '            ok_pool = data[(data["question_len"] <= 350) & (data["target_reject_reason"] == "ok")].copy()\n'
        '            train_df = ok_pool.sample(min({train_rows}, len(ok_pool)), random_state={seed})\n'
        '            train_df = train_df.copy()\n'
        '            train_df["reference_answer"] = train_df["answer_only_target"]\n'
        '            train = train_df.to_dict(orient="records")\n'
        '            train_ids = set(train_df["row_id"].tolist())\n'
        '            val_pool = data[(data["question_len"] <= 500) & (~data["row_id"].isin(train_ids))].copy()\n'
        '            val_df = val_pool.sample(min({val_rows}, len(val_pool)), random_state={val_seed})\n'
        '            val = val_df.to_dict(orient="records")\n'.format(train_rows=train_rows, val_rows=val_rows, seed=seed, val_seed=seed + 1),
    )
    source = source.replace('"pool_rows": int(len(pool)),', '"pool_rows": int(len(ok_pool)), "val_pool_rows": int(len(val_pool)),')
    source = source.replace(
        '            pair_stats = Counter()\n'
        '            pair_buckets = {}\n'
        '            for base_out, lora_out in pairs:\n',
        '            pair_stats = Counter()\n'
        '            routed_stats = Counter()\n'
        '            pair_buckets = {}\n'
        '            routed_buckets = {}\n'
        '            for base_out, lora_out in pairs:\n',
    )
    source = source.replace(
        '                bucket = base_out["bucket"]\n'
        '                b = pair_buckets.setdefault(bucket, Counter())\n'
        '                b["rows"] += 1\n'
        '                b["changed_output_count"] += int(changed)\n'
        '                b["base_only_exact_count"] += int(base_only)\n'
        '                b["lora_only_exact_count"] += int(lora_only)\n',
        '                route = bool(base_out.get("cap_hit") or base_out.get("invalid") or base_out.get("repetition"))\n'
        '                accept = bool(route and not lora_out.get("invalid") and not lora_out.get("repetition") and lora_out.get("new_tokens", 10**9) < base_out.get("new_tokens", 0))\n'
        '                chosen = lora_out if accept else base_out\n'
        '                for key in ("exact", "ref_in_output", "output_in_ref", "final_exact", "invalid", "cap_hit", "repetition"):\n'
        '                    routed_stats[key] += int(chosen.get(key, False))\n'
        '                routed_stats["rows"] += 1\n'
        '                routed_stats["route_attempts"] += int(route)\n'
        '                routed_stats["route_accepts"] += int(accept)\n'
        '                routed_stats["route_rejects"] += int(route and not accept)\n'
        '                routed_stats["changed_from_base"] += int(chosen["norm"] != base_out["norm"])\n'
        '                bucket = base_out["bucket"]\n'
        '                b = pair_buckets.setdefault(bucket, Counter())\n'
        '                b["rows"] += 1\n'
        '                b["changed_output_count"] += int(changed)\n'
        '                b["base_only_exact_count"] += int(base_only)\n'
        '                b["lora_only_exact_count"] += int(lora_only)\n'
        '                rb = routed_buckets.setdefault(bucket, Counter())\n'
        '                rb["rows"] += 1\n'
        '                rb["route_attempts"] += int(route)\n'
        '                rb["route_accepts"] += int(accept)\n'
        '                rb["changed_from_base"] += int(chosen["norm"] != base_out["norm"])\n'
        '                rb["routed_exact"] += int(chosen.get("exact", False))\n',
    )
    source = source.replace(
        '                "base_invalid_count": int(base_stats.get("invalid", 0)),\n'
        '                "lora_invalid_count": int(lora_stats.get("invalid", 0)),\n'
        '                "base_cap_hit_count": int(base_stats.get("cap_hit", 0)),\n'
        '                "lora_cap_hit_count": int(lora_stats.get("cap_hit", 0)),\n',
        '                "base_invalid_count": int(base_stats.get("invalid", 0)),\n'
        '                "lora_invalid_count": int(lora_stats.get("invalid", 0)),\n'
        '                "base_cap_hit_count": int(base_stats.get("cap_hit", 0)),\n'
        '                "lora_cap_hit_count": int(lora_stats.get("cap_hit", 0)),\n'
        '                "base_repetition_count": int(base_stats.get("repetition", 0)),\n'
        '                "lora_repetition_count": int(lora_stats.get("repetition", 0)),\n'
        '                "routed_exact_count": int(routed_stats.get("exact", 0)),\n'
        '                "routed_ref_in_output_count": int(routed_stats.get("ref_in_output", 0)),\n'
        '                "routed_output_in_ref_count": int(routed_stats.get("output_in_ref", 0)),\n'
        '                "routed_final_exact_count": int(routed_stats.get("final_exact", 0)),\n'
        '                "routed_invalid_count": int(routed_stats.get("invalid", 0)),\n'
        '                "routed_cap_hit_count": int(routed_stats.get("cap_hit", 0)),\n'
        '                "routed_repetition_count": int(routed_stats.get("repetition", 0)),\n'
        '                "route_attempt_count": int(routed_stats.get("route_attempts", 0)),\n'
        '                "route_accept_count": int(routed_stats.get("route_accepts", 0)),\n'
        '                "route_reject_count": int(routed_stats.get("route_rejects", 0)),\n'
        '                "routed_changed_from_base_count": int(routed_stats.get("changed_from_base", 0)),\n',
    )
    source = source.replace(
        '                "pair_shape_buckets": {k: dict(v) for k, v in pair_buckets.items()},\n',
        '                "pair_shape_buckets": {k: dict(v) for k, v in pair_buckets.items()},\n'
        '                "routed_shape_buckets": {k: dict(v) for k, v in routed_buckets.items()},\n',
    )
    return source


def write_report(path: Path, summary: dict) -> None:
    data_meta = summary.get("data_meta") or {}
    train = summary.get("train") or {}
    val = summary.get("validation") or {}
    runtime = summary.get("remote_runtime") or {}
    probe = summary.get("probe") or {}
    lines = [
        "# C207 Routed Answer-Only Adapter Diagnostic",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- One mechanism: second-pass answer-only adapter only for base-output failure flags.",
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
        f"- answer-only train pool rows: `{data_meta.get('pool_rows')}`",
        f"- validation pool rows: `{data_meta.get('val_pool_rows')}`",
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
        f"- trainable params: `{train.get('trainable_params')}`",
        f"- adapter scratch deleted: `{probe.get('adapter_scratch_deleted')}`",
        "",
        "## Validation",
        f"- val rows: `{val.get('val_rows')}`",
        f"- route attempts / accepts / rejects: `{val.get('route_attempt_count')}` / `{val.get('route_accept_count')}` / `{val.get('route_reject_count')}`",
        f"- base exact / LoRA exact / routed exact: `{val.get('base_exact_count')}` / `{val.get('lora_exact_count')}` / `{val.get('routed_exact_count')}`",
        f"- base ref-in-output / LoRA ref-in-output / routed ref-in-output: `{val.get('base_ref_in_output_count')}` / `{val.get('lora_ref_in_output_count')}` / `{val.get('routed_ref_in_output_count')}`",
        f"- base output-in-ref / LoRA output-in-ref / routed output-in-ref: `{val.get('base_output_in_ref_count')}` / `{val.get('lora_output_in_ref_count')}` / `{val.get('routed_output_in_ref_count')}`",
        f"- base final-line exact / LoRA final-line exact / routed final-line exact: `{val.get('base_final_exact_count')}` / `{val.get('lora_final_exact_count')}` / `{val.get('routed_final_exact_count')}`",
        f"- base invalid/cap/repetition: `{val.get('base_invalid_count')}` / `{val.get('base_cap_hit_count')}` / `{val.get('base_repetition_count')}`",
        f"- LoRA invalid/cap/repetition: `{val.get('lora_invalid_count')}` / `{val.get('lora_cap_hit_count')}` / `{val.get('lora_repetition_count')}`",
        f"- routed invalid/cap/repetition: `{val.get('routed_invalid_count')}` / `{val.get('routed_cap_hit_count')}` / `{val.get('routed_repetition_count')}`",
        f"- routed changed from base: `{val.get('routed_changed_from_base_count')}`",
        f"- base avg new tokens: `{val.get('base_avg_new_tokens')}`",
        f"- LoRA avg new tokens: `{val.get('lora_avg_new_tokens')}`",
        f"- routed shape buckets: `{val.get('routed_shape_buckets')}`",
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
        "Proceed only if routed policy improves failure counts without hurting aggregate quality proxies.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C207_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c207_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c207_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    for flag, value in (("--train-rows", "96"), ("--val-rows", "128"), ("--steps", "48"), ("--max-new-tokens", "24"), ("--seed", "207")):
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
