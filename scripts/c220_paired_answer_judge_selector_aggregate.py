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
import c211_c111_task_conditional_prompt_aggregate as task_conditional


EXPERIMENT_ID = "C220"
EXPERIMENT_SLUG = "C220_paired_answer_judge_selector_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C220_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
JUDGE_PREFIX = (
    "Choose the better final answer for the task. Reply with exactly one letter: A or B. "
    "Prefer A if both answers look equally valid or if you are unsure."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C220 judge-select C111 vs C211 candidate answers.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=220)
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


def candidate_records(solution: Any, tokenizer: Any, rows: list[dict[str, Any]], outputs: Any) -> list[dict[str, Any]]:
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
                "final": final,
                "handler": handler,
                "out_tokens": out_tokens,
                "flags": flags,
            }
        )
    return records


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


def delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "exact": left["quality"].get("exact", 0) - right["quality"].get("exact", 0),
        "final_line_exact": left["quality"].get("final_line_exact", 0)
        - right["quality"].get("final_line_exact", 0),
        "ref_in_output": left["quality"].get("ref_in_output", 0) - right["quality"].get("ref_in_output", 0),
        "output_in_ref": left["quality"].get("output_in_ref", 0) - right["quality"].get("output_in_ref", 0),
        "hit_max_tokens": left["validity"].get("hit_max_tokens", 0)
        - right["validity"].get("hit_max_tokens", 0),
        "repetition_loop": left["validity"].get("repetition_loop", 0)
        - right["validity"].get("repetition_loop", 0),
        "avg_output_tokens": left["tokens"].get("avg_output_tokens", 0)
        - right["tokens"].get("avg_output_tokens", 0),
    }


def judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return (
        f"Task:\n{question}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Which answer is better? Reply with exactly A or B."
    )


def parse_choice(text: str) -> tuple[str, bool]:
    cleaned = text.strip().upper()
    if cleaned.startswith("B"):
        return "B", True
    if cleaned.startswith("A"):
        return "A", True
    return "A", False


def selected_records(
    rows: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
    judge_outputs: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    choice_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for control, variant, out in zip(control_records, variant_records, judge_outputs):
        text = out.outputs[0].text
        choice, valid = parse_choice(text)
        if not valid:
            choice_counts["invalid_default_a"] += 1
        choice_counts[choice] += 1
        selected.append(variant if choice == "B" else control)
    return selected, {k: int(v) for k, v in choice_counts.items()}


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
        "mechanism": "same-model A/B judge selector between C111 and C211 candidate final answers",
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
    route_counts: Counter[str] = Counter()
    variant_prompts = []
    for row in rows:
        route, prefix = task_conditional.route_prefix(str(row["question"]), c111.USER_PREFIX)
        route_counts[route] += 1
        variant_prompts.append(probe.apply_user_only_template(tokenizer, str(row["question"]), True, prefix))

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
    candidate_sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)

    control_t0 = time.perf_counter()
    control_outputs = llm.generate(control_prompts, sampling_params=candidate_sampling)
    control_generation_s = time.perf_counter() - control_t0
    variant_t0 = time.perf_counter()
    variant_outputs = llm.generate(variant_prompts, sampling_params=candidate_sampling)
    variant_generation_s = time.perf_counter() - variant_t0

    control_records = candidate_records(c111, tokenizer, rows, control_outputs)
    variant_records = candidate_records(c111, tokenizer, rows, variant_outputs)
    judge_prompts = [
        probe.apply_user_only_template(
            tokenizer,
            judge_prompt(str(row["question"]), str(control["final"]), str(variant["final"])),
            True,
            JUDGE_PREFIX,
        )
        for row, control, variant in zip(rows, control_records, variant_records)
    ]
    judge_t0 = time.perf_counter()
    judge_outputs = llm.generate(judge_prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=4, top_p=1.0, top_k=-1))
    judge_generation_s = time.perf_counter() - judge_t0
    sampler.stop()

    selected, judge_counts = selected_records(rows, control_records, variant_records, judge_outputs)
    control = summarize_records(rows, control_records)
    variant = summarize_records(rows, variant_records)
    selected_summary = summarize_records(rows, selected)

    total_generation_s = control_generation_s + variant_generation_s + judge_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "C111-vs-C211 same-model judge selector aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
                "route_counts": {k: int(v) for k, v in route_counts.items()},
            },
            "runtime": {
                "startup_s": startup_s,
                "control_generation_s": control_generation_s,
                "variant_generation_s": variant_generation_s,
                "judge_generation_s": judge_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "judge_counts": judge_counts,
            "control_c111": control,
            "variant_task_conditional": variant,
            "selected_by_judge": selected_summary,
            "delta_variant_minus_control": delta(variant, control),
            "delta_selected_minus_control": delta(selected_summary, control),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C220 Paired Answer-Judge Selector Aggregate",
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
        "## Judge Counts",
        f"`{summary.get('judge_counts')}`",
        "",
        "## Delta Variant Minus Control",
        f"`{summary.get('delta_variant_minus_control')}`",
        "",
        "## Delta Selected Minus Control",
        f"`{summary.get('delta_selected_minus_control')}`",
        "",
        "## C111 Control",
        f"`{summary.get('control_c111')}`",
        "",
        "## Task-Conditional Variant",
        f"`{summary.get('variant_task_conditional')}`",
        "",
        "## Judge-Selected Output",
        f"`{summary.get('selected_by_judge')}`",
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
