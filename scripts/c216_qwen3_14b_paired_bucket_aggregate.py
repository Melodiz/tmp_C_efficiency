from __future__ import annotations

import argparse
import gc
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202


EXPERIMENT_ID = "C216"
EXPERIMENT_SLUG = "C216_qwen3_14b_paired_bucket_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C216_artifacts"
BASELINE_MODEL_ID = "Qwen/Qwen3-8B-AWQ"
VARIANT_MODEL_ID = "Qwen/Qwen3-14B-AWQ"
DELTA_METRICS = (
    "exact",
    "final_line_exact",
    "ref_in_output",
    "output_in_ref",
    "hit_max_tokens",
    "repetition_loop",
    "avg_output_tokens",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C216 paired Qwen3-8B vs Qwen3-14B aggregate diagnostic.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=216)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "zip": out_dir.with_suffix(".zip"),
    }


def _metric_value(summary: dict[str, Any], metric: str) -> float:
    if metric in ("hit_max_tokens", "repetition_loop"):
        return float((summary.get("validity") or {}).get(metric, 0))
    if metric == "avg_output_tokens":
        return float((summary.get("tokens") or {}).get(metric, 0.0))
    return float((summary.get("quality") or summary).get(metric, 0))


def overall_delta(variant: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {metric: _metric_value(variant, metric) - _metric_value(baseline, metric) for metric in DELTA_METRICS}


def keyed_delta(variant: dict[str, Any], baseline: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    variant_groups = variant.get(key) or {}
    baseline_groups = baseline.get(key) or {}
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(set(variant_groups) | set(baseline_groups)):
        v = variant_groups.get(name) or {}
        b = baseline_groups.get(name) or {}
        rows = int(max(v.get("rows", 0), b.get("rows", 0)))
        deltas = {
            "exact": int(v.get("exact", 0)) - int(b.get("exact", 0)),
            "final_line_exact": int(v.get("final_line_exact", 0)) - int(b.get("final_line_exact", 0)),
            "ref_in_output": int(v.get("ref_in_output", 0)) - int(b.get("ref_in_output", 0)),
            "output_in_ref": int(v.get("output_in_ref", 0)) - int(b.get("output_in_ref", 0)),
        }
        out[name] = {"rows": rows, **deltas}
    return out


def run_model(model_id: str, c111: Any, rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in prompts]

    startup_t0 = time.perf_counter()
    llm = LLM(
        model=model_id,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=c111.MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=seed,
        trust_remote_code=False,
    )
    startup_s = time.perf_counter() - startup_t0
    sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    generation_t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling)
    generation_s = time.perf_counter() - generation_t0
    result = c202.summarize_rows(c111, tokenizer, rows, outputs)
    runtime = {
        "startup_s": startup_s,
        "generation_s": generation_s,
        "avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens)),
    }

    del outputs
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return result, runtime


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "c111_commit": rollback.C111_COMMIT,
        "baseline_model_id": BASELINE_MODEL_ID,
        "variant_model_id": VARIANT_MODEL_ID,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    baseline, baseline_runtime = run_model(BASELINE_MODEL_ID, c111, rows, args.seed)
    variant, variant_runtime = run_model(VARIANT_MODEL_ID, c111, rows, args.seed)
    sampler.stop()

    total_generation_s = baseline_runtime["generation_s"] + variant_runtime["generation_s"]
    total_startup_s = baseline_runtime["startup_s"] + variant_runtime["startup_s"]
    projected_total_4000_s = total_startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "Paired Qwen3-8B-AWQ vs Qwen3-14B-AWQ aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "baseline": baseline_runtime,
                "variant": variant_runtime,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "baseline_8b": baseline,
            "variant_14b": variant,
            "delta_14b_minus_8b": overall_delta(variant, baseline),
            "delta_by_category": keyed_delta(variant, baseline, "by_category"),
            "delta_by_bucket": keyed_delta(variant, baseline, "top_buckets"),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C216 Qwen3-14B Paired Bucket/Category Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- baseline model: `{summary.get('baseline_model_id')}`",
        f"- variant model: `{summary.get('variant_model_id')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Delta 14B Minus 8B",
        f"`{summary.get('delta_14b_minus_8b')}`",
        "",
        "## Delta By Category",
        f"`{summary.get('delta_by_category')}`",
        "",
        "## Delta By Bucket",
        f"`{summary.get('delta_by_bucket')}`",
        "",
        "## Baseline 8B",
        f"`{summary.get('baseline_8b')}`",
        "",
        "## Variant 14B",
        f"`{summary.get('variant_14b')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- outputs returned: `{summary.get('outputs_returned')}`",
        f"- model weights returned: `{summary.get('model_weights_returned')}`",
        f"- training started: `{summary.get('training_started')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    summary = run_validation(args)
    io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    agg.base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
