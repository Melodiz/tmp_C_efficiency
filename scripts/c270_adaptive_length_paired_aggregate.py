from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202
import c269_adaptive_length_gate_audit as gate


EXPERIMENT_ID = "C270"
EXPERIMENT_SLUG = "C270_adaptive_length_paired_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C270_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
VARIANT_LONG_MAX_TOKENS = 768


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C270 C111 adaptive max_tokens paired aggregate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=270)
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


def delta(variant: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact": variant["quality"].get("exact", 0) - baseline["quality"].get("exact", 0),
        "final_line_exact": variant["quality"].get("final_line_exact", 0)
        - baseline["quality"].get("final_line_exact", 0),
        "ref_in_output": variant["quality"].get("ref_in_output", 0)
        - baseline["quality"].get("ref_in_output", 0),
        "output_in_ref": variant["quality"].get("output_in_ref", 0)
        - baseline["quality"].get("output_in_ref", 0),
        "hit_max_tokens": variant["validity"].get("hit_max_tokens", 0)
        - baseline["validity"].get("hit_max_tokens", 0),
        "repetition_loop": variant["validity"].get("repetition_loop", 0)
        - baseline["validity"].get("repetition_loop", 0),
        "empty": variant["validity"].get("empty", 0) - baseline["validity"].get("empty", 0),
        "avg_output_tokens": variant["tokens"].get("avg_output_tokens", 0.0)
        - baseline["tokens"].get("avg_output_tokens", 0.0),
    }


def route_indices(rows: list[dict[str, Any]]) -> list[int]:
    return [idx for idx, row in enumerate(rows) if gate.question_features(str(row["question"]))["route_long"]]


def summarize_rows_with_caps(
    solution: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    outputs: Any,
    max_tokens_by_row: list[int],
) -> dict[str, Any]:
    quality: Counter[str] = Counter()
    validity: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handlers: Counter[str] = Counter()
    output_tokens: list[int] = []

    for row, out, row_max_tokens in zip(rows, outputs, max_tokens_by_row):
        completion = out.outputs[0]
        base_answer = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(base_answer).input_ids)
        output_tokens.append(out_tokens)

        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(question)
        final, handler_name = rollback.c111_stack(solution, question, base_answer)
        flags = c202.retry_base.valid_flags(base_answer, out_tokens, row_max_tokens)

        handlers[handler_name] += 1
        c202.retry_base.quality_update(quality, final, reference)
        c202.retry_base.quality_update(by_category[category], final, reference)
        c202.retry_base.quality_update(by_bucket[bucket], final, reference)
        validity["rows"] += 1
        validity["empty"] += int(flags["empty"])
        validity["thinking"] += int(flags["thinking"])
        validity["hit_max_tokens"] += int(flags["hit_max_tokens"])
        validity["repetition_loop"] += int(flags["repetition_loop"])
        validity["deterministic_first_fire"] += int(handler_name != "fallback_model")

    return {
        "quality": agg.rates({"overall": quality})["overall"],
        "validity": {k: int(v) for k, v in validity.items()},
        "handler_counts": {k: int(v) for k, v in handlers.items()},
        "tokens": {
            "avg_output_tokens": sum(output_tokens) / max(1, len(output_tokens)),
            "max_output_tokens": max(output_tokens) if output_tokens else None,
        },
        "by_category": dict(sorted(agg.rates(by_category).items())),
        "by_bucket": dict(sorted(agg.rates(by_bucket).items())),
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
        "baseline_max_tokens": None,
        "variant_long_max_tokens": VARIANT_LONG_MAX_TOKENS,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    routed_indices = route_indices(rows)
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
    long_sampling = SamplingParams(temperature=0.0, max_tokens=VARIANT_LONG_MAX_TOKENS, top_p=1.0, top_k=-1)

    baseline_t0 = time.perf_counter()
    baseline_outputs = llm.generate(prompts, sampling_params=baseline_sampling)
    baseline_generation_s = time.perf_counter() - baseline_t0

    routed_t0 = time.perf_counter()
    routed_outputs = llm.generate(routed_prompts, sampling_params=long_sampling) if routed_prompts else []
    routed_generation_s = time.perf_counter() - routed_t0
    sampler.stop()

    variant_outputs = list(baseline_outputs)
    for idx, output in zip(routed_indices, routed_outputs):
        variant_outputs[idx] = output

    original_max_tokens = c111.MAX_NEW_TOKENS
    baseline_caps = [original_max_tokens for _ in rows]
    variant_caps = [VARIANT_LONG_MAX_TOKENS if idx in set(routed_indices) else original_max_tokens for idx in range(len(rows))]
    baseline = summarize_rows_with_caps(c111, tokenizer, rows, baseline_outputs, baseline_caps)
    variant = summarize_rows_with_caps(c111, tokenizer, rows, variant_outputs, variant_caps)
    routed_rows = [rows[idx] for idx in routed_indices]
    routed_baseline_outputs = [baseline_outputs[idx] for idx in routed_indices]
    routed_variant_outputs = [variant_outputs[idx] for idx in routed_indices]
    routed_baseline = (
        summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_baseline_outputs, [original_max_tokens] * len(routed_rows))
        if routed_rows
        else {}
    )
    routed_variant = (
        summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_variant_outputs, [VARIANT_LONG_MAX_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )

    total_generation_s = baseline_generation_s + routed_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    overall_delta = delta(variant, baseline)
    routed_delta = delta(routed_variant, routed_baseline) if routed_indices else {}
    success_gate = {
        "routed_share": len(routed_indices) / max(1, len(rows)),
        "cap_hits_drop_at_least_15": overall_delta.get("hit_max_tokens", 0) <= -15,
        "ref_in_output_improves_at_least_3": overall_delta.get("ref_in_output", 0) >= 3,
        "output_in_ref_nonnegative": overall_delta.get("output_in_ref", 0) >= 0,
        "repetition_not_increased": overall_delta.get("repetition_loop", 0) <= 0,
        "runtime_under_12_min": projected_total_4000_s < 720,
    }
    success_gate["s2_gate2_pass"] = bool(
        success_gate["cap_hits_drop_at_least_15"]
        and success_gate["ref_in_output_improves_at_least_3"]
        and success_gate["output_in_ref_nonnegative"]
        and success_gate["repetition_not_increased"]
        and success_gate["runtime_under_12_min"]
    )

    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if success_gate["s2_gate2_pass"] else "KILL",
            "reason": "C111 adaptive max_tokens route paired aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "model_loaded": True,
            "baseline_max_tokens": original_max_tokens,
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
            "baseline_320": baseline,
            "variant_adaptive": variant,
            "delta_adaptive_minus_320": overall_delta,
            "routed_baseline_320": routed_baseline,
            "routed_variant_768": routed_variant,
            "routed_delta_768_minus_320": routed_delta,
            "success_gate": success_gate,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C270 Adaptive-Length Paired Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- baseline max tokens: `{summary.get('baseline_max_tokens')}`",
        f"- routed long max tokens: `{summary.get('variant_long_max_tokens')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Success Gate",
        f"`{summary.get('success_gate')}`",
        "",
        "## Delta Adaptive Minus 320",
        f"`{summary.get('delta_adaptive_minus_320')}`",
        "",
        "## Routed Delta 768 Minus 320",
        f"`{summary.get('routed_delta_768_minus_320')}`",
        "",
        "## Baseline 320",
        f"`{summary.get('baseline_320')}`",
        "",
        "## Variant Adaptive",
        f"`{summary.get('variant_adaptive')}`",
        "",
        "## Routed Baseline 320",
        f"`{summary.get('routed_baseline_320')}`",
        "",
        "## Routed Variant 768",
        f"`{summary.get('routed_variant_768')}`",
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
