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
import c293_strict_math_scratchpad_route as c293


EXPERIMENT_ID = "C294"
EXPERIMENT_SLUG = "C294_strict_scratchpad_extraction"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C294_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"

ANSWER_MARKER = re.compile(r"(?:ответ|итог|результат)\s*[:：]\s*", re.IGNORECASE)
TOKEN_CANDIDATE = re.compile(
    r"[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+(?:[.,]\d+)?)?(?:\s*%|\s*[a-zA-Zа-яА-Я°]+)?|"
    r"[a-zA-Zа-яА-Я]\s*=\s*[-+]?\d+(?:[.,]\d+)?|"
    r"[-+]?\d+(?:[.,]\d+)?\s*[+\-*/^]\s*[-+]?\d+(?:[.,]\d+)?"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C294 strict math scratchpad route with final-answer extraction.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=294)
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


def extract_final_answer(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return raw
    marker_parts = ANSWER_MARKER.split(raw)
    if len(marker_parts) > 1:
        tail = marker_parts[-1].strip()
    else:
        lines = [line.strip(" \t-•") for line in raw.splitlines() if line.strip()]
        numeric_lines = [line for line in lines if TOKEN_CANDIDATE.search(line)]
        tail = numeric_lines[-1] if numeric_lines else lines[-1] if lines else raw
    tail = re.split(r"(?:\n|[;。])", tail.strip())[0].strip()
    candidates = TOKEN_CANDIDATE.findall(tail)
    if candidates:
        return candidates[-1].strip(" .,!?:;")
    return tail[:120].strip(" .,!?:;")


def summarize_rows(
    solution: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    outputs: Any,
    routes: list[str] | None = None,
) -> dict[str, Any]:
    quality: Counter[str] = Counter()
    validity: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handlers: Counter[str] = Counter()
    output_tokens: list[int] = []
    extraction_count = 0

    for idx, (row, out) in enumerate(zip(rows, outputs)):
        completion = out.outputs[0]
        raw_answer = completion.text.strip()
        route = routes[idx] if routes else "control"
        answer = extract_final_answer(raw_answer) if route == "strict_math_scratchpad" else raw_answer
        extraction_count += int(answer != raw_answer)
        out_tokens = len(tokenizer(answer).input_ids)
        output_tokens.append(out_tokens)

        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(question)
        final, handler = rollback.c111_stack(solution, question, answer)
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
        "extraction_count": extraction_count,
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
        "mechanism": "C293 strict math scratchpad route plus final numeric/symbolic answer extraction",
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
    routes: list[str] = []
    variant_prompts = []
    for row in rows:
        route, prefix = c293.route_prefix(str(row["question"]), c111.USER_PREFIX)
        routes.append(route)
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
    sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    control_t0 = time.perf_counter()
    control_outputs = llm.generate(control_prompts, sampling_params=sampling)
    control_generation_s = time.perf_counter() - control_t0
    variant_t0 = time.perf_counter()
    variant_outputs = llm.generate(variant_prompts, sampling_params=sampling)
    variant_generation_s = time.perf_counter() - variant_t0
    sampler.stop()

    control = summarize_rows(c111, tokenizer, rows, control_outputs)
    variant = summarize_rows(c111, tokenizer, rows, variant_outputs, routes)
    total_generation_s = control_generation_s + variant_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE",
            "reason": "C293 strict scratchpad route plus extraction aggregate completed.",
            "raw_task_data_read_remote_only": True,
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
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "control_c111_prefix": control,
            "variant_strict_scratchpad_extracted": variant,
            "delta_variant_minus_c111": {
                "exact": variant["quality"].get("exact", 0) - control["quality"].get("exact", 0),
                "final_line_exact": variant["quality"].get("final_line_exact", 0)
                - control["quality"].get("final_line_exact", 0),
                "ref_in_output": variant["quality"].get("ref_in_output", 0)
                - control["quality"].get("ref_in_output", 0),
                "output_in_ref": variant["quality"].get("output_in_ref", 0)
                - control["quality"].get("output_in_ref", 0),
                "hit_max_tokens": variant["validity"].get("hit_max_tokens", 0)
                - control["validity"].get("hit_max_tokens", 0),
                "repetition_loop": variant["validity"].get("repetition_loop", 0)
                - control["validity"].get("repetition_loop", 0),
                "avg_output_tokens": variant["tokens"].get("avg_output_tokens", 0)
                - control["tokens"].get("avg_output_tokens", 0),
            },
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C294 Strict Scratchpad Route Final-Answer Extraction Aggregate",
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
        "## Delta Extracted Strict Scratchpad Route Minus C111",
        f"`{summary.get('delta_variant_minus_c111')}`",
        "",
        "## C111 Prefix Control",
        f"`{summary.get('control_c111_prefix')}`",
        "",
        "## Extracted Strict Scratchpad Variant",
        f"`{summary.get('variant_strict_scratchpad_extracted')}`",
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
