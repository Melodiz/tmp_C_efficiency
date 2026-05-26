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
import c198_targeted_failure_retry as retry_base


EXPERIMENT_ID = "C199"
EXPERIMENT_SLUG = "C199_answer_first_prompt_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C199_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
VARIANT_PREFIX = "Сначала напиши итоговый ответ одной строкой. Затем, только если нужно, дай очень краткое пояснение. Не повторяй условие."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C199 paired aggregate prompt comparison.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=199)
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


def summarize_rows(solution: Any, tokenizer: Any, rows: list[dict[str, Any]], outputs: Any) -> dict[str, Any]:
    quality: Counter[str] = Counter()
    validity: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handlers: Counter[str] = Counter()
    output_tokens: list[int] = []
    for row, out in zip(rows, outputs):
        completion = out.outputs[0]
        answer = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(answer).input_ids)
        output_tokens.append(out_tokens)
        handler, final = agg.first_handler(solution, str(row["question"]), answer)
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(str(row["question"]))
        flags = retry_base.valid_flags(answer, out_tokens, solution.MAX_NEW_TOKENS)
        handlers[handler] += 1
        retry_base.quality_update(quality, final, reference)
        retry_base.quality_update(by_category[category], final, reference)
        retry_base.quality_update(by_bucket[bucket], final, reference)
        validity["rows"] += 1
        validity["empty"] += int(flags["empty"])
        validity["thinking"] += int(flags["thinking"])
        validity["hit_max_tokens"] += int(flags["hit_max_tokens"])
        validity["repetition_loop"] += int(flags["repetition_loop"])
        validity["deterministic_first_fire"] += int(handler != "fallback_model")
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
        "variant_prefix": VARIANT_PREFIX,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    agg.install_final_path_dependencies()
    solution = retry_base.load_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    current_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, solution.USER_PREFIX) for row in rows
    ]
    variant_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, VARIANT_PREFIX) for row in rows
    ]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in current_prompts]

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    startup_t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=solution.MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=args.seed,
        trust_remote_code=False,
    )
    startup_s = time.perf_counter() - startup_t0
    sampling = SamplingParams(temperature=0.0, max_tokens=solution.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    current_t0 = time.perf_counter()
    current_outputs = llm.generate(current_prompts, sampling_params=sampling)
    current_generation_s = time.perf_counter() - current_t0
    variant_t0 = time.perf_counter()
    variant_outputs = llm.generate(variant_prompts, sampling_params=sampling)
    variant_generation_s = time.perf_counter() - variant_t0
    sampler.stop()

    current = summarize_rows(solution, tokenizer, rows, current_outputs)
    variant = summarize_rows(solution, tokenizer, rows, variant_outputs)
    total_generation_s = current_generation_s + variant_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE",
            "reason": "Paired answer-first prompt aggregate validation completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"solution": "ok", "pymorphy_available": bool(solution.get_morph_analyzer())},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "current_generation_s": current_generation_s,
                "variant_generation_s": variant_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {"avg_input_tokens_current": sum(input_tokens) / max(1, len(input_tokens))},
            "current": current,
            "variant": variant,
            "delta": {
                "exact": variant["quality"].get("exact", 0) - current["quality"].get("exact", 0),
                "final_line_exact": variant["quality"].get("final_line_exact", 0)
                - current["quality"].get("final_line_exact", 0),
                "ref_in_output": variant["quality"].get("ref_in_output", 0)
                - current["quality"].get("ref_in_output", 0),
                "output_in_ref": variant["quality"].get("output_in_ref", 0)
                - current["quality"].get("output_in_ref", 0),
                "hit_max_tokens": variant["validity"].get("hit_max_tokens", 0)
                - current["validity"].get("hit_max_tokens", 0),
                "repetition_loop": variant["validity"].get("repetition_loop", 0)
                - current["validity"].get("repetition_loop", 0),
                "avg_output_tokens": variant["tokens"].get("avg_output_tokens", 0)
                - current["tokens"].get("avg_output_tokens", 0),
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C199 Answer-First Prompt Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- sample: `{summary.get('sample_meta')}`",
        f"- variant prefix: `{summary.get('variant_prefix')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Delta",
        f"`{summary.get('delta')}`",
        "",
        "## Current",
        f"`{summary.get('current')}`",
        "",
        "## Variant",
        f"`{summary.get('variant')}`",
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
