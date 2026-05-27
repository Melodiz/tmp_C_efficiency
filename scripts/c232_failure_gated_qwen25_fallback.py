from __future__ import annotations

import argparse
import gc
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
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c216_qwen3_14b_paired_bucket_aggregate as paired
import c229_qwen25_7b_awq_paired_aggregate as c229


EXPERIMENT_ID = "C232"
EXPERIMENT_SLUG = "C232_failure_gated_qwen25_fallback"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C232_artifacts"
BASELINE_MODEL_ID = c229.BASELINE_MODEL_ID
VARIANT_MODEL_ID = c229.VARIANT_MODEL_ID
MODEL_PACKAGE_METADATA = c229.MODEL_PACKAGE_METADATA


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C232 failure-gated Qwen2.5 fallback over C111.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=232)
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


def _generate(
    model_id: str,
    c111: Any,
    rows: list[dict[str, Any]],
    seed: int,
    quantization: str = "awq_marlin",
) -> tuple[Any, Any, dict[str, Any]]:
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in prompts]

    startup_t0 = time.perf_counter()
    llm = LLM(
        model=model_id,
        dtype="float16",
        quantization=quantization,
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
    runtime = {
        "startup_s": startup_s,
        "generation_s": generation_s,
        "avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens)),
        "rows": len(rows),
    }

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return tokenizer, outputs, runtime


def _row_eval(solution: Any, tokenizer: Any, row: dict[str, Any], out: Any) -> dict[str, Any]:
    completion = out.outputs[0]
    base_answer = completion.text.strip()
    token_ids = getattr(completion, "token_ids", None)
    out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(base_answer).input_ids)
    final, handler = rollback.c111_stack(solution, str(row["question"]), base_answer)
    flags = retry_base.valid_flags(base_answer, out_tokens, solution.MAX_NEW_TOKENS)
    return {
        "final": final,
        "handler": handler,
        "base_answer": base_answer,
        "out_tokens": out_tokens,
        "flags": flags,
    }


def _quality_table(rows: list[dict[str, Any]], evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    quality: Counter[str] = Counter()
    validity: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handlers: Counter[str] = Counter()
    output_tokens: list[int] = []
    for row, item in zip(rows, evaluated):
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(str(row["question"]))
        retry_base.quality_update(quality, item["final"], reference)
        retry_base.quality_update(by_category[category], item["final"], reference)
        retry_base.quality_update(by_bucket[bucket], item["final"], reference)
        flags = item["flags"]
        validity["rows"] += 1
        validity["empty"] += int(flags["empty"])
        validity["thinking"] += int(flags["thinking"])
        validity["hit_max_tokens"] += int(flags["hit_max_tokens"])
        validity["repetition_loop"] += int(flags["repetition_loop"])
        validity["deterministic_first_fire"] += int(item["handler"] != "fallback_model")
        handlers[item["handler"]] += 1
        output_tokens.append(int(item["out_tokens"]))
    return {
        "quality": agg.rates({"overall": quality})["overall"],
        "validity": {k: int(v) for k, v in validity.items()},
        "handler_counts": {k: int(v) for k, v in handlers.items()},
        "tokens": {
            "avg_output_tokens": sum(output_tokens) / max(1, len(output_tokens)),
            "max_output_tokens": max(output_tokens) if output_tokens else None,
        },
        "by_category": dict(sorted(agg.rates(by_category).items())),
        "top_buckets": dict(sorted(agg.rates(by_bucket).items(), key=lambda kv: -kv[1].get("rows", 0))[:20]),
    }


def _delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return paired.overall_delta(a, b)


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
        "route": "Use Qwen2.5-7B-AWQ only when C111 base output hits max_tokens or repetition-loop flags.",
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    baseline_tokenizer, baseline_outputs, baseline_runtime = _generate(BASELINE_MODEL_ID, c111, rows, args.seed)
    baseline_eval = [_row_eval(c111, baseline_tokenizer, row, out) for row, out in zip(rows, baseline_outputs)]
    routed_indices = [
        idx
        for idx, item in enumerate(baseline_eval)
        if item["flags"]["hit_max_tokens"] or item["flags"]["repetition_loop"]
    ]
    routed_rows = [rows[idx] for idx in routed_indices]

    variant_runtime: dict[str, Any] = {"startup_s": 0.0, "generation_s": 0.0, "avg_input_tokens": 0.0, "rows": 0}
    variant_eval: list[dict[str, Any]] = []
    if routed_rows:
        variant_tokenizer, variant_outputs, variant_runtime = _generate(VARIANT_MODEL_ID, c111, routed_rows, args.seed)
        variant_eval = [_row_eval(c111, variant_tokenizer, row, out) for row, out in zip(routed_rows, variant_outputs)]
        del variant_outputs
    sampler.stop()

    selected_eval = list(baseline_eval)
    for idx, item in zip(routed_indices, variant_eval):
        selected_eval[idx] = item

    baseline_all = _quality_table(rows, baseline_eval)
    selected_all = _quality_table(rows, selected_eval)
    baseline_routed = _quality_table(routed_rows, [baseline_eval[idx] for idx in routed_indices])
    variant_routed = _quality_table(routed_rows, variant_eval)

    total_generation_s = baseline_runtime["generation_s"] + variant_runtime["generation_s"]
    total_startup_s = baseline_runtime["startup_s"] + variant_runtime["startup_s"]
    projected_total_4000_s = total_startup_s + (total_generation_s / max(1, len(rows))) * 4000

    del baseline_outputs
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "Failure-gated Qwen2.5 fallback aggregate completed.",
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
                "variant_routed": variant_runtime,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "route_counts": {
                "rows": len(rows),
                "routed_rows": len(routed_rows),
                "routed_rate": len(routed_rows) / max(1, len(rows)),
            },
            "baseline_all": baseline_all,
            "selected_all": selected_all,
            "delta_selected_minus_baseline_all": _delta(selected_all, baseline_all),
            "baseline_routed_only": baseline_routed,
            "variant_routed_only": variant_routed,
            "delta_variant_minus_baseline_routed_only": _delta(variant_routed, baseline_routed),
            "delta_by_category_selected": paired.keyed_delta(selected_all, baseline_all, "by_category"),
            "delta_by_bucket_selected": paired.keyed_delta(selected_all, baseline_all, "top_buckets"),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C232 Failure-Gated Qwen2.5 Fallback",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- baseline model: `{summary.get('baseline_model_id')}`",
        f"- fallback model: `{summary.get('variant_model_id')}`",
        f"- route: {summary.get('route')}",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Package Metadata",
        f"`{summary.get('model_package_metadata')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Route Counts",
        f"`{summary.get('route_counts')}`",
        "",
        "## Selected Minus Baseline",
        f"`{summary.get('delta_selected_minus_baseline_all')}`",
        "",
        "## Routed Rows: Fallback Minus Baseline",
        f"`{summary.get('delta_variant_minus_baseline_routed_only')}`",
        "",
        "## Baseline All",
        f"`{summary.get('baseline_all')}`",
        "",
        "## Selected All",
        f"`{summary.get('selected_all')}`",
        "",
        "## Routed Baseline",
        f"`{summary.get('baseline_routed_only')}`",
        "",
        "## Routed Fallback",
        f"`{summary.get('variant_routed_only')}`",
        "",
        "## Delta By Category",
        f"`{summary.get('delta_by_category_selected')}`",
        "",
        "## Delta By Bucket",
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


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
