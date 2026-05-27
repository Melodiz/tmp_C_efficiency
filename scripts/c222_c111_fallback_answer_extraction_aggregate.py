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
import c201_c111_vs_current_stack_aggregate as rollback
import c220_paired_answer_judge_selector_aggregate as judge_base


EXPERIMENT_ID = "C222"
EXPERIMENT_SLUG = "C222_c111_fallback_answer_extraction_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C222_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
EXTRACT_PREFIX = (
    "Extract the final answer from the previous solution. Reply only with the final answer, "
    "without explanation. Preserve the task language and required units."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C222 extract final answers from C111 fallback outputs.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=222)
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


def control_records(solution: Any, tokenizer: Any, rows: list[dict[str, Any]], outputs: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row, out in zip(rows, outputs):
        completion = out.outputs[0]
        raw_answer = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(raw_answer).input_ids)
        final, handler = rollback.c111_stack(solution, str(row["question"]), raw_answer)
        flags = retry_base.valid_flags(raw_answer, out_tokens, solution.MAX_NEW_TOKENS)
        records.append(
            {
                "raw_answer": raw_answer,
                "final": final,
                "handler": handler,
                "out_tokens": out_tokens,
                "flags": flags,
            }
        )
    return records


def extracted_records(
    solution: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    extraction_outputs: Any,
    fallback_indices: list[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = [dict(record) for record in controls]
    stats: Counter[str] = Counter()
    for idx, out in zip(fallback_indices, extraction_outputs):
        completion = out.outputs[0]
        extracted = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(extracted).input_ids)
        final, handler = rollback.c111_stack(solution, str(rows[idx]["question"]), extracted)
        flags = retry_base.valid_flags(extracted, out_tokens, solution.MAX_NEW_TOKENS)
        if flags["empty"]:
            stats["empty_extraction_default_control"] += 1
            continue
        stats["extracted_rows"] += 1
        selected[idx] = {
            "final": final,
            "handler": handler,
            "out_tokens": out_tokens,
            "flags": flags,
        }
    return selected, {k: int(v) for k, v in stats.items()}


def extraction_prompt(question: str, raw_answer: str) -> str:
    return (
        f"Task:\n{question}\n\n"
        f"Previous solution:\n{raw_answer}\n\n"
        "Final answer only:"
    )


def summarize_records(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    quality: Counter[str] = Counter()
    validity: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handlers: Counter[str] = Counter()
    output_tokens: list[int] = []

    for row, record in zip(rows, records):
        final = str(record["final"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(str(row["question"]))
        flags = record["flags"]
        handler = str(record["handler"])

        handlers[handler] += 1
        output_tokens.append(int(record["out_tokens"]))
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
        "c111_commit": rollback.C111_COMMIT,
        "mechanism": "same-model final-answer extraction on C111 fallback outputs only",
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    control_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows
    ]
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

    control_t0 = time.perf_counter()
    control_outputs = llm.generate(
        control_prompts,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1),
    )
    control_generation_s = time.perf_counter() - control_t0
    controls = control_records(c111, tokenizer, rows, control_outputs)
    fallback_indices = [i for i, record in enumerate(controls) if record["handler"] == "fallback_model"]
    extraction_prompts = [
        probe.apply_user_only_template(
            tokenizer,
            extraction_prompt(str(rows[idx]["question"]), str(controls[idx]["raw_answer"])),
            True,
            EXTRACT_PREFIX,
        )
        for idx in fallback_indices
    ]
    extraction_t0 = time.perf_counter()
    extraction_outputs = llm.generate(
        extraction_prompts,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=96, top_p=1.0, top_k=-1),
    )
    extraction_generation_s = time.perf_counter() - extraction_t0
    sampler.stop()

    selected, extraction_stats = extracted_records(c111, tokenizer, rows, controls, extraction_outputs, fallback_indices)
    control_summary = summarize_records(rows, controls)
    selected_summary = summarize_records(rows, selected)
    total_generation_s = control_generation_s + extraction_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000

    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "C111 fallback answer-extraction aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "control_generation_s": control_generation_s,
                "extraction_generation_s": extraction_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "extraction_stats": {
                **extraction_stats,
                "fallback_rows": len(fallback_indices),
                "preserved_deterministic_rows": len(rows) - len(fallback_indices),
            },
            "control_c111": control_summary,
            "extracted_selected": selected_summary,
            "delta_extracted_minus_control": judge_base.delta(selected_summary, control_summary),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C222 C111 Fallback Answer-Extraction Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Extraction Stats",
        f"`{summary.get('extraction_stats')}`",
        "",
        "## Delta Extracted Minus Control",
        f"`{summary.get('delta_extracted_minus_control')}`",
        "",
        "## C111 Control",
        f"`{summary.get('control_c111')}`",
        "",
        "## Extracted/Selected Output",
        f"`{summary.get('extracted_selected')}`",
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
