from __future__ import annotations

import argparse
import os
import re
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
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202


EXPERIMENT_ID = "C234"
EXPERIMENT_SLUG = "C234_semantic_proxy_calibration"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C234_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
COMPARATOR_PREFIX = c202.VARIANT_PREFIX


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C234 reference-aware semantic proxy calibration.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=234)
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


def _completion_text(out: Any) -> str:
    return str(out.outputs[0].text).strip()


def _truncate(text: str, limit: int = 1200) -> str:
    value = str(text).strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"


def judge_prompt(question: str, reference: str, answer: str) -> str:
    return (
        "Ты строгий проверяющий ответа школьной задачи. "
        "Сравни ответ ученика с эталонным ответом с учетом смысла, чисел, единиц и языка. "
        "Если ответ ученика по сути принимаетcя как правильный, выведи только 1. "
        "Если он неверный, неполный, противоречит эталону или не отвечает на вопрос, выведи только 0.\n\n"
        f"Задание:\n{_truncate(question)}\n\n"
        f"Эталон:\n{_truncate(reference)}\n\n"
        f"Ответ ученика:\n{_truncate(answer)}\n\n"
        "Оценка:"
    )


def judge_pass(text: str) -> bool:
    cleaned = str(text).strip()
    match = re.search(r"[01]", cleaned)
    return bool(match and match.group(0) == "1")


def semantic_rates(rows: list[dict[str, Any]], pass_bits: list[bool]) -> dict[str, Any]:
    overall: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row, bit in zip(rows, pass_bits):
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(str(row["question"]))
        for counter in (overall, by_category[category], by_bucket[bucket]):
            counter["rows"] += 1
            counter["semantic_pass"] += int(bit)

    def convert(table: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, counts in table.items():
            rows_count = int(counts.get("rows", 0))
            item = {name: int(value) for name, value in counts.items()}
            item["semantic_pass_rate"] = item.get("semantic_pass", 0) / rows_count if rows_count else 0.0
            out[str(key)] = item
        return out

    return {
        "overall": convert({"overall": overall})["overall"],
        "by_category": dict(sorted(convert(by_category).items())),
        "top_buckets": dict(sorted(convert(by_bucket).items(), key=lambda kv: -kv[1].get("rows", 0))[:20]),
    }


def semantic_delta(variant: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    v = variant["overall"]
    b = baseline["overall"]
    return {
        "semantic_pass": int(v.get("semantic_pass", 0)) - int(b.get("semantic_pass", 0)),
        "semantic_pass_rate": float(v.get("semantic_pass_rate", 0.0)) - float(b.get("semantic_pass_rate", 0.0)),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": False,
        "raw_examples_returned": False,
        "judge_rationales_returned": False,
        "row_ids_returned": False,
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "c111_commit": rollback.C111_COMMIT,
        "model_id": MODEL_ID,
        "comparator": "C202 no-detailed-reasoning prefix",
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    c111_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows
    ]
    comparator_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, COMPARATOR_PREFIX) for row in rows
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
    answer_sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    judge_sampling = SamplingParams(temperature=0.0, max_tokens=4, top_p=1.0, top_k=-1)

    c111_t0 = time.perf_counter()
    c111_outputs = llm.generate(c111_prompts, sampling_params=answer_sampling)
    c111_generation_s = time.perf_counter() - c111_t0
    comparator_t0 = time.perf_counter()
    comparator_outputs = llm.generate(comparator_prompts, sampling_params=answer_sampling)
    comparator_generation_s = time.perf_counter() - comparator_t0

    c111_answers = [_completion_text(out) for out in c111_outputs]
    comparator_answers = [_completion_text(out) for out in comparator_outputs]
    c111_judge_prompts = [
        judge_prompt(str(row["question"]), str(row.get("reference_answer", "")), answer)
        for row, answer in zip(rows, c111_answers)
    ]
    comparator_judge_prompts = [
        judge_prompt(str(row["question"]), str(row.get("reference_answer", "")), answer)
        for row, answer in zip(rows, comparator_answers)
    ]

    judge_t0 = time.perf_counter()
    c111_judge_outputs = llm.generate(c111_judge_prompts, sampling_params=judge_sampling)
    comparator_judge_outputs = llm.generate(comparator_judge_prompts, sampling_params=judge_sampling)
    judge_generation_s = time.perf_counter() - judge_t0
    sampler.stop()

    c111_string = c202.summarize_rows(c111, tokenizer, rows, c111_outputs)
    comparator_string = c202.summarize_rows(c111, tokenizer, rows, comparator_outputs)
    c111_pass = [judge_pass(_completion_text(out)) for out in c111_judge_outputs]
    comparator_pass = [judge_pass(_completion_text(out)) for out in comparator_judge_outputs]
    c111_semantic = semantic_rates(rows, c111_pass)
    comparator_semantic = semantic_rates(rows, comparator_pass)

    total_generation_s = c111_generation_s + comparator_generation_s + judge_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    string_delta = {
        "exact": comparator_string["quality"].get("exact", 0) - c111_string["quality"].get("exact", 0),
        "final_line_exact": comparator_string["quality"].get("final_line_exact", 0)
        - c111_string["quality"].get("final_line_exact", 0),
        "ref_in_output": comparator_string["quality"].get("ref_in_output", 0)
        - c111_string["quality"].get("ref_in_output", 0),
        "output_in_ref": comparator_string["quality"].get("output_in_ref", 0)
        - c111_string["quality"].get("output_in_ref", 0),
        "hit_max_tokens": comparator_string["validity"].get("hit_max_tokens", 0)
        - c111_string["validity"].get("hit_max_tokens", 0),
        "repetition_loop": comparator_string["validity"].get("repetition_loop", 0)
        - c111_string["validity"].get("repetition_loop", 0),
        "avg_output_tokens": comparator_string["tokens"].get("avg_output_tokens", 0.0)
        - c111_string["tokens"].get("avg_output_tokens", 0.0),
    }

    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "Reference-aware semantic proxy calibration completed.",
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
                "c111_generation_s": c111_generation_s,
                "comparator_generation_s": comparator_generation_s,
                "judge_generation_s": judge_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "string_proxy": {
                "c111": c111_string,
                "comparator": comparator_string,
                "delta_comparator_minus_c111": string_delta,
            },
            "semantic_proxy": {
                "c111": c111_semantic,
                "comparator": comparator_semantic,
                "delta_comparator_minus_c111": semantic_delta(comparator_semantic, c111_semantic),
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C234 Reference-Aware Semantic Proxy Calibration",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- comparator: {summary.get('comparator')}",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## String Proxy Delta",
        f"`{(summary.get('string_proxy') or {}).get('delta_comparator_minus_c111')}`",
        "",
        "## Semantic Proxy Delta",
        f"`{(summary.get('semantic_proxy') or {}).get('delta_comparator_minus_c111')}`",
        "",
        "## C111 Semantic",
        f"`{(summary.get('semantic_proxy') or {}).get('c111')}`",
        "",
        "## Comparator Semantic",
        f"`{(summary.get('semantic_proxy') or {}).get('comparator')}`",
        "",
        "## C111 String",
        f"`{(summary.get('string_proxy') or {}).get('c111')}`",
        "",
        "## Comparator String",
        f"`{(summary.get('string_proxy') or {}).get('comparator')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- judge rationales returned: `{summary.get('judge_rationales_returned')}`",
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
