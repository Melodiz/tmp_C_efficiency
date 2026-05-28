from __future__ import annotations

import argparse
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
import c270_adaptive_length_paired_aggregate as c270


EXPERIMENT_ID = "C335"
EXPERIMENT_SLUG = "C335_long_route_adaptive_mild_repetition_penalty"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C335_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
ROUTE_MAX_TOKENS = 512
REPETITION_PENALTY = 1.03


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C335 C269 long-route 512-token mild repetition-penalty decode.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=335)
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


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "prompts_returned": False,
        "references_returned": False,
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "c111_commit": rollback.C111_COMMIT,
        "model_id": MODEL_ID,
        "mechanism": "C269 long-answer route uses C111 prompt with max_tokens=512 and repetition_penalty=1.03; non-routed rows preserve C111 decode.",
        "route_max_tokens": ROUTE_MAX_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    routed_indices = c270.route_indices(rows)
    routed_set = set(routed_indices)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]
    routed_prompts = [prompts[idx] for idx in routed_indices]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in prompts]

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    startup_t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=c111.MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=args.seed,
        trust_remote_code=False,
    )
    startup_s = time.perf_counter() - startup_t0
    baseline_sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    routed_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=ROUTE_MAX_TOKENS,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=REPETITION_PENALTY,
    )

    baseline_t0 = time.perf_counter()
    baseline_outputs = llm.generate(prompts, sampling_params=baseline_sampling)
    baseline_generation_s = time.perf_counter() - baseline_t0

    routed_t0 = time.perf_counter()
    routed_outputs = llm.generate(routed_prompts, sampling_params=routed_sampling) if routed_prompts else []
    routed_generation_s = time.perf_counter() - routed_t0
    sampler.stop()

    variant_outputs = list(baseline_outputs)
    for idx, output in zip(routed_indices, routed_outputs):
        variant_outputs[idx] = output

    baseline_caps = [c111.MAX_NEW_TOKENS for _ in rows]
    variant_caps = [ROUTE_MAX_TOKENS if idx in routed_set else c111.MAX_NEW_TOKENS for idx in range(len(rows))]
    baseline = c270.summarize_rows_with_caps(c111, tokenizer, rows, baseline_outputs, baseline_caps)
    variant = c270.summarize_rows_with_caps(c111, tokenizer, rows, variant_outputs, variant_caps)
    routed_rows = [rows[idx] for idx in routed_indices]
    routed_baseline_outputs = [baseline_outputs[idx] for idx in routed_indices]
    routed_variant_outputs = [variant_outputs[idx] for idx in routed_indices]
    routed_baseline = (
        c270.summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_baseline_outputs, [c111.MAX_NEW_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )
    routed_variant = (
        c270.summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_variant_outputs, [ROUTE_MAX_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )
    delta = c270.delta(variant, baseline)
    routed_delta = c270.delta(routed_variant, routed_baseline) if routed_rows else {}
    total_generation_s = baseline_generation_s + routed_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    gate = {
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
        and gate["repetition_not_worse"]
        and gate["runtime_under_12_min"]
    )
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if gate["pass"] else "KILL",
            "reason": "Long-route adaptive mild repetition penalty cleared local gate." if gate["pass"] else "Long-route adaptive mild repetition penalty failed local gate.",
            "raw_task_data_read_remote_only": True,
            "model_loaded": True,
            "sample_meta": {
                "rows": len(rows),
                "routed_rows": len(routed_indices),
                "routed_share": len(routed_indices) / max(1, len(rows)),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "baseline_generation_s": baseline_generation_s,
                "routed_generation_s": routed_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {"avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens))},
            "baseline_c111": baseline,
            "variant_long_route_decode": variant,
            "delta_variant_minus_c111": delta,
            "routed_baseline_c111": routed_baseline,
            "routed_variant_long_route_decode": routed_variant,
            "routed_delta_variant_minus_c111": routed_delta,
            "gate": gate,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C335 Long-Route Adaptive Mild Repetition Penalty",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Gate",
        f"`{summary.get('gate')}`",
        "",
        "## Delta Variant Minus C111",
        f"`{summary.get('delta_variant_minus_c111')}`",
        "",
        "## Routed Delta Variant Minus C111",
        f"`{summary.get('routed_delta_variant_minus_c111')}`",
        "",
        "## Baseline C111",
        f"`{summary.get('baseline_c111')}`",
        "",
        "## Variant Long-Route Decode",
        f"`{summary.get('variant_long_route_decode')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- prompts returned: `{summary.get('prompts_returned')}`",
        f"- references returned: `{summary.get('references_returned')}`",
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
