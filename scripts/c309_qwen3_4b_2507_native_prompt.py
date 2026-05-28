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
import c216_qwen3_14b_paired_bucket_aggregate as paired


EXPERIMENT_ID = "C309"
EXPERIMENT_SLUG = "C309_qwen3_4b_2507_native_prompt"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C309_artifacts"
BASELINE_MODEL_ID = "Qwen/Qwen3-8B-AWQ"
VARIANT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507-FP8"
MODEL_PACKAGE_METADATA = {
    "baseline_selected_files_gb": 6.115,
    "variant_selected_files_gb": 5.206,
    "metadata_source": "C218 Hugging Face API metadata, checked 2026-05-27",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C309 Qwen3-4B-2507-FP8 native-prompt aggregate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=309)
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


def run_variant_native_prompt(
    model_id: str, c111: Any, rows: list[dict[str, Any]], seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, None) for row in rows]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in prompts]

    startup_t0 = time.perf_counter()
    llm = LLM(
        model=model_id,
        dtype="float16",
        quantization="fp8",
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
        "prompt": "native_question_only_enable_thinking_false",
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
        "model_package_metadata": MODEL_PACKAGE_METADATA,
        "variant_prompt": "native question only, no C111 short prefix",
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    baseline, baseline_runtime = paired.run_model(BASELINE_MODEL_ID, c111, rows, args.seed, "awq_marlin")
    variant, variant_runtime = run_variant_native_prompt(VARIANT_MODEL_ID, c111, rows, args.seed)
    sampler.stop()

    total_generation_s = baseline_runtime["generation_s"] + variant_runtime["generation_s"]
    total_startup_s = baseline_runtime["startup_s"] + variant_runtime["startup_s"]
    projected_total_4000_s = total_startup_s + (total_generation_s / max(1, len(rows))) * 4000
    delta = paired.overall_delta(variant, baseline)
    gate = {
        "exact_nonnegative": delta.get("exact", 0) >= 0,
        "ref_in_output_nonnegative": delta.get("ref_in_output", 0) >= 0,
        "output_in_ref_nonnegative": delta.get("output_in_ref", 0) >= 0,
        "one_containment_positive": delta.get("ref_in_output", 0) > 0 or delta.get("output_in_ref", 0) > 0,
        "hit_max_tokens_not_worse": delta.get("hit_max_tokens", 0) <= 0,
        "repetition_not_worse": delta.get("repetition_loop", 0) <= 0,
        "runtime_under_12_min": projected_total_4000_s < 720,
    }
    gate["pass"] = bool(
        gate["ref_in_output_nonnegative"]
        and gate["output_in_ref_nonnegative"]
        and gate["one_containment_positive"]
        and gate["hit_max_tokens_not_worse"]
        and gate["runtime_under_12_min"]
    )
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if gate["pass"] else "KILL",
            "reason": "Native-prompt 4B-2507 cleared local gates." if gate["pass"] else "Native-prompt 4B-2507 did not clear local gates.",
            "raw_task_data_read_remote_only": True,
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
            "baseline_8b_awq_c111_prefix": baseline,
            "variant_4b_2507_fp8_native_prompt": variant,
            "delta_4b_native_minus_8b": delta,
            "delta_by_category": paired.keyed_delta(variant, baseline, "by_category"),
            "delta_by_bucket": paired.keyed_delta(variant, baseline, "top_buckets"),
            "gate": gate,
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C309 Qwen3-4B-Instruct-2507-FP8 Native Prompt",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- baseline model: `{summary.get('baseline_model_id')}`",
        f"- variant model: `{summary.get('variant_model_id')}`",
        f"- variant prompt: `{summary.get('variant_prompt')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Package Metadata",
        f"`{summary.get('model_package_metadata')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Gate",
        f"`{summary.get('gate')}`",
        "",
        "## Delta 4B Native Minus 8B",
        f"`{summary.get('delta_4b_native_minus_8b')}`",
        "",
        "## Delta By Category",
        f"`{summary.get('delta_by_category')}`",
        "",
        "## Delta By Bucket",
        f"`{summary.get('delta_by_bucket')}`",
        "",
        "## Baseline 8B",
        f"`{summary.get('baseline_8b_awq_c111_prefix')}`",
        "",
        "## Variant 4B Native Prompt",
        f"`{summary.get('variant_4b_2507_fp8_native_prompt')}`",
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


if __name__ == "__main__":
    raise SystemExit(run())
