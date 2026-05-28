from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c201_c111_vs_current_stack_aggregate as rollback
import c216_qwen3_14b_paired_bucket_aggregate as paired
import c246_failure_gated_same_model_512 as c246


EXPERIMENT_ID = "C315"
EXPERIMENT_SLUG = "C315_failure_gated_512_repetition_penalty"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C315_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
FALLBACK_MAX_TOKENS = 512
REPETITION_PENALTY = 1.08


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C315 route visible C111 failures to 512-token repetition-penalty fallback.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=315)
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
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "c111_commit": rollback.C111_COMMIT,
        "model_id": MODEL_ID,
        "route": "Use 512-token repetition-penalty fallback only when C111 320-token output hits max tokens or repetition-loop flags; accept only if fallback is visibly valid.",
        "fallback_sampling": {
            "max_tokens": FALLBACK_MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": REPETITION_PENALTY,
        },
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]
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
    baseline_t0 = time.perf_counter()
    baseline_outputs = llm.generate(prompts, sampling_params=baseline_sampling)
    baseline_generation_s = time.perf_counter() - baseline_t0
    baseline_eval = [
        c246.row_eval(c111, tokenizer, row, out, c111.MAX_NEW_TOKENS) for row, out in zip(rows, baseline_outputs)
    ]
    routed_indices = [
        idx
        for idx, item in enumerate(baseline_eval)
        if item["flags"]["hit_max_tokens"] or item["flags"]["repetition_loop"]
    ]
    routed_prompts = [prompts[idx] for idx in routed_indices]
    routed_rows = [rows[idx] for idx in routed_indices]
    fallback_generation_s = 0.0
    fallback_eval: list[dict[str, Any]] = []
    if routed_prompts:
        fallback_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=FALLBACK_MAX_TOKENS,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=REPETITION_PENALTY,
        )
        fallback_t0 = time.perf_counter()
        fallback_outputs = llm.generate(routed_prompts, sampling_params=fallback_sampling)
        fallback_generation_s = time.perf_counter() - fallback_t0
        fallback_eval = [
            c246.row_eval(c111, tokenizer, row, out, FALLBACK_MAX_TOKENS)
            for row, out in zip(routed_rows, fallback_outputs)
        ]
    sampler.stop()

    selected_eval = list(baseline_eval)
    route_counts: Counter[str] = Counter()
    for idx, fallback_item in zip(routed_indices, fallback_eval):
        base_item = baseline_eval[idx]
        use_fallback = c246.invalid(base_item) and not c246.invalid(fallback_item)
        selected_eval[idx] = fallback_item if use_fallback else base_item
        route_counts["accepted_fallback"] += int(use_fallback)
        route_counts["rejected_fallback"] += int(not use_fallback)
        route_counts["fallback_invalid"] += int(c246.invalid(fallback_item))
        route_counts["fallback_hit_max_tokens"] += int(fallback_item["flags"]["hit_max_tokens"])
        route_counts["fallback_repetition_loop"] += int(fallback_item["flags"]["repetition_loop"])
    route_counts["rows"] = len(rows)
    route_counts["routed_rows"] = len(routed_rows)

    baseline_all = c246.quality_table(rows, baseline_eval)
    selected_all = c246.quality_table(rows, selected_eval)
    baseline_routed = c246.quality_table(routed_rows, [baseline_eval[idx] for idx in routed_indices])
    fallback_routed = c246.quality_table(routed_rows, fallback_eval)
    delta = paired.overall_delta(selected_all, baseline_all)
    projected_total_4000_s = startup_s + ((baseline_generation_s + fallback_generation_s) / max(1, len(rows))) * 4000
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
            "reason": "Failure-gated 512 repetition-penalty fallback cleared local gate." if gate["pass"] else "Failure-gated 512 repetition-penalty fallback did not clear local gate.",
            "raw_task_data_read_remote_only": True,
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "baseline_generation_s": baseline_generation_s,
                "fallback_generation_s": fallback_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {"avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens))},
            "route_counts": {k: int(v) for k, v in route_counts.items()},
            "baseline_all": baseline_all,
            "selected_all": selected_all,
            "delta_selected_minus_baseline_all": delta,
            "baseline_routed_only": baseline_routed,
            "fallback_routed_only": fallback_routed,
            "delta_fallback_minus_baseline_routed_only": paired.overall_delta(fallback_routed, baseline_routed),
            "delta_by_category_selected": paired.keyed_delta(selected_all, baseline_all, "by_category"),
            "delta_by_bucket_selected": paired.keyed_delta(selected_all, baseline_all, "top_buckets"),
            "gate": gate,
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C315 Failure-Gated 512 Repetition-Penalty Fallback",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- route: {summary.get('route')}",
        f"- fallback sampling: `{summary.get('fallback_sampling')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Gate",
        f"`{summary.get('gate')}`",
        "",
        "## Route Counts",
        f"`{summary.get('route_counts')}`",
        "",
        "## Selected Minus Baseline",
        f"`{summary.get('delta_selected_minus_baseline_all')}`",
        "",
        "## Routed Rows: Fallback Minus Baseline",
        f"`{summary.get('delta_fallback_minus_baseline_routed_only')}`",
        "",
        "## Delta By Category Selected",
        f"`{summary.get('delta_by_category_selected')}`",
        "",
        "## Delta By Bucket Selected",
        f"`{summary.get('delta_by_bucket_selected')}`",
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
