from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177


EXPERIMENT_ID = "C178"
EXPERIMENT_SLUG = "C178_sft_aggregate_metric_cap_diagnostic"
ORIGINAL_TASK_PROBE_SOURCE = c177.task_probe_source


def task_probe_source(model_id: str, train_rows: int, val_rows: int, steps: int, max_seq_len: int, max_new_tokens: int, seed: int) -> str:
    source = ORIGINAL_TASK_PROBE_SOURCE(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, seed)
    source = source.replace(
        'exact = normalize(text) == normalize(row["reference_answer"])\n'
        '                invalid = invalid_output(text)\n'
        '                cap_hit = new_tokens >= {max_new_tokens}\n',
        'norm_text = normalize(text)\n'
        '                norm_ref = normalize(row["reference_answer"])\n'
        '                lines = [part for part in str(text).splitlines() if part.strip()]\n'
        '                final_norm = normalize(lines[-1] if lines else text)\n'
        '                exact = norm_text == norm_ref\n'
        '                ref_in_output = bool(norm_ref) and norm_ref in norm_text\n'
        '                output_in_ref = bool(norm_text) and norm_text in norm_ref\n'
        '                final_exact = final_norm == norm_ref\n'
        '                invalid = invalid_output(text)\n'
        '                cap_hit = new_tokens >= {max_new_tokens}\n',
    )
    source = source.replace(
        'for key, value in (("exact", exact), ("invalid", invalid), ("cap_hit", cap_hit)):\n'
        '                    stats[key] += int(value)\n'
        '                    bucket_stats[key] += int(value)\n',
        'for key, value in (("exact", exact), ("ref_in_output", ref_in_output), ("output_in_ref", output_in_ref), ("final_exact", final_exact), ("invalid", invalid), ("cap_hit", cap_hit)):\n'
        '                    stats[key] += int(value)\n'
        '                    bucket_stats[key] += int(value)\n',
    )
    source = source.replace(
        'outputs.append({"norm": normalize(text), "exact": exact, "invalid": invalid, "cap_hit": cap_hit, "new_tokens": new_tokens, "bucket": bucket})',
        'outputs.append({"norm": norm_text, "exact": exact, "ref_in_output": ref_in_output, "output_in_ref": output_in_ref, "final_exact": final_exact, "invalid": invalid, "cap_hit": cap_hit, "new_tokens": new_tokens, "bucket": bucket})',
    )
    source = source.replace(
        '"base_invalid_count": int(base_stats.get("invalid", 0)),\n'
        '                "lora_invalid_count": int(lora_stats.get("invalid", 0)),\n',
        '"base_ref_in_output_count": int(base_stats.get("ref_in_output", 0)),\n'
        '                "lora_ref_in_output_count": int(lora_stats.get("ref_in_output", 0)),\n'
        '                "base_output_in_ref_count": int(base_stats.get("output_in_ref", 0)),\n'
        '                "lora_output_in_ref_count": int(lora_stats.get("output_in_ref", 0)),\n'
        '                "base_final_exact_count": int(base_stats.get("final_exact", 0)),\n'
        '                "lora_final_exact_count": int(lora_stats.get("final_exact", 0)),\n'
        '                "base_invalid_count": int(base_stats.get("invalid", 0)),\n'
        '                "lora_invalid_count": int(lora_stats.get("invalid", 0)),\n',
    )
    return source


def write_report(path: Path, summary: dict) -> None:
    data_meta = summary.get("data_meta") or {}
    train = summary.get("train") or {}
    val = summary.get("validation") or {}
    runtime = summary.get("remote_runtime") or {}
    probe = summary.get("probe") or {}
    lines = [
        "# C178 SFT Aggregate Metric/Cap Diagnostic",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Same tiny training mechanism as C177.",
        "- Aggregate diagnostics only; no raw task text, outputs, row ids, model weights, or adapter weights returned.",
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
        f"- data rows: `{data_meta.get('data_rows')}`",
        f"- pool rows: `{data_meta.get('pool_rows')}`",
        f"- train rows: `{data_meta.get('train_rows')}`",
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
        f"- base invalid/cap hit: `{val.get('base_invalid_count')}` / `{val.get('base_cap_hit_count')}`",
        f"- LoRA invalid/cap hit: `{val.get('lora_invalid_count')}` / `{val.get('lora_cap_hit_count')}`",
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
        "If containment/final-line metrics remain zero or cap-dominated, kill or park tiny SFT rather than scaling it.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C178_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c178_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c178_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = write_report
    return c177.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
