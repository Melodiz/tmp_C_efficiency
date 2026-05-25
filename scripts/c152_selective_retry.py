from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c072_output_control as base
import c100_qwen3_8b_no_detailed_reasoning_prompt as c100


EXPERIMENT_ID = "C152"
EXPERIMENT_SLUG = "C152_selective_cap_loop_retry"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C152_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
BASE_PREFIX = "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ."
RETRY_PREFIX = c100.NO_DETAILED_REASONING_PREFIX


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


@lru_cache(maxsize=1)
def load_final_solution_module() -> Any:
    try:
        return importlib.import_module("simple_solution.solution")
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        module_path = repo_root / "simple_solution" / "solution.py"
        spec = importlib.util.spec_from_file_location("c152_final_solution", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["c152_final_solution"] = module
        spec.loader.exec_module(module)
        return module


def final_stack_answer(question: str, answer: str) -> tuple[str, str | None]:
    final = load_final_solution_module()
    handlers = [
        ("expression_substitution", lambda: final.expression_substitution_answer(question)),
        ("algebra_equation", lambda: final.algebra_equation_answer(question)),
        ("exact_numeric", lambda: final.exact_numeric_answer(question)),
        ("direct_arithmetic", lambda: final.direct_arithmetic_answer(question)),
        ("chemistry_stoichiometry", lambda: final.chemistry_stoichiometry_answer(question)),
        ("formulaic_math_physics", lambda: final.formulaic_math_physics_answer(question)),
        ("structured_school_task", lambda: final.structured_school_task_answer(question)),
        ("calculator_written_arithmetic", lambda: final.calculator_written_arithmetic_answer(question)),
        ("russian_morph_grammar", lambda: final.russian_morph_grammar_answer(question)),
        ("comma_loop_dedup", lambda: final.dedup_comma_loop(answer)),
        ("english_cloze_cleanup", lambda: final.cleanup_english_cloze_answer(question, answer)),
        ("quantity_conversion", lambda: final.quantity_conversion_answer(question)),
        ("km_meters", lambda: final.km_meters_answer(question)),
    ]
    for name, handler in handlers:
        fixed = handler()
        if fixed is not None:
            return fixed, name
    return answer, None


def output_row(
    run_id: str,
    sample_index: int,
    row: dict[str, Any],
    prompt: str | None,
    prompt_tokens: int | None,
    answer: str,
    output_tokens: int,
    finish_reason: str | None,
    stop_reason: str | None,
    base_answer: str | None = None,
    retry_answer: str | None = None,
    retried: bool = False,
    retry_reason: str | None = None,
    exact_handler: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "candidate": "qwen3-8b-awq",
        "model_ref": MODEL_ID,
        "sample_index": sample_index,
        "rid": int(row["row_id"]),
        "row_id": int(row["row_id"]),
        "category": row.get("category"),
        "question": row["question"],
        "reference_answer": row.get("reference_answer"),
        "prompt": prompt,
        "input_tokens": prompt_tokens,
        "answer": answer,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "has_thinking_trace": "<think" in answer or "</think>" in answer,
        "hit_max_tokens": output_tokens >= 320,
        "repetition_loop_suspected": probe.has_repetition_loop(answer),
        "selective_retry": {
            "retried": retried,
            "reason": retry_reason,
            "base_answer": base_answer,
            "retry_answer": retry_answer,
        },
        "exact_stack_handler": exact_handler,
    }


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no selective-retry evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    retry = metrics.get("selective_retry") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C152 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after selective retry."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Retry overhead misses the 12 minute runtime gate."
    if int(retry.get("retried_rows") or 0) > 20:
        return "KILL", "Retry trigger fired too broadly."
    if int(validity.get("max_token_hit_rows") or 0) > 2:
        return "KILL", "Selective retry did not control capped outputs enough."
    return "MUTATE", "Selective retry is mechanically feasible; row-level review is needed before any final-solution port."


def build_metrics(summary: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    validity = summary.get("validity") or {}
    runtime = summary.get("runtime") or {}
    sample_rows = int(runtime.get("sample_rows") or validity.get("jsonl_rows") or 0)
    projected = runtime.get("projected_total_4000_s")
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "status": summary.get("status"),
        "candidate": summary.get("candidate"),
        "model_ref": summary.get("model_ref"),
        "summary_path": str(paths["summary"]),
        "outputs_path": str(paths["outputs"]),
        "sample_rows": sample_rows,
        "runtime": runtime,
        "tokens": summary.get("tokens"),
        "validity": validity,
        "selective_retry": summary.get("selective_retry"),
        "exact_stack": summary.get("exact_stack"),
        "environment": summary.get("environment"),
        "hf_metadata": summary.get("hf_metadata"),
        "rates": {
            "projected_total_4000_min": projected / 60 if isinstance(projected, (int, float)) else None,
            "retry_rate": (summary.get("selective_retry") or {}).get("retried_rows", 0) / sample_rows
            if sample_rows
            else None,
            "max_token_hit_rate": int(validity.get("max_token_hit_rows") or 0) / sample_rows if sample_rows else None,
            "repetition_loop_suspected_rate": int(validity.get("repetition_loop_suspected_rows") or 0) / sample_rows
            if sample_rows
            else None,
        },
    }


def create_dry_run(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_c152_dry_run"
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "run_id": run_id,
        "status": "dry_run",
        "candidate": "qwen3-8b-awq",
        "model_ref": MODEL_ID,
        "config": {
            "sample_source": args.sample_source,
            "sample_size_requested": args.sample_size,
            "base_prefix": BASE_PREFIX,
            "retry_prefix": RETRY_PREFIX,
            "mechanism": "retry_only_first_pass_max_token_or_repetition_rows",
        },
        "runtime": {"sample_rows": 0},
        "validity": {
            "jsonl_rows": 0,
            "one_answer_per_input": None,
            "thinking_trace_rows": 0,
            "max_token_hit_rows": 0,
            "empty_answer_rows": 0,
            "repetition_loop_suspected_rows": 0,
        },
        "selective_retry": {"retried_rows": 0, "retried_row_ids": []},
        "paths": {k: str(v) for k, v in run_paths.items() if k in {"summary", "metrics", "outputs", "log"}},
    }
    base.append_jsonl(run_paths["outputs"], [])
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].write_text("dry_run=true\n", encoding="utf-8")
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    run_id = f"{base.utc_stamp()}_qwen3-8b-awq_{args.sample_size}"
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "samples": paths["results_dir"] / f"{run_id}.samples.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    sample_df = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    sample_rows = sample_df.to_dict(orient="records")
    base.append_jsonl(run_paths["samples"], sample_rows)

    tokenizer_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer_s = time.perf_counter() - tokenizer_t0

    base_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, BASE_PREFIX) for row in sample_rows
    ]
    input_token_counts = [len(tokenizer(prompt).input_ids) for prompt in base_prompts]

    sampler = probe.GpuMemorySampler(interval_s=args.gpu_sample_interval)
    sampler.start()
    startup_t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tokenizer_mode="auto",
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    startup_s = time.perf_counter() - startup_t0
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, top_p=1.0, top_k=-1)

    base_t0 = time.perf_counter()
    base_outputs = llm.generate(base_prompts, sampling_params=sampling)
    base_generation_s = time.perf_counter() - base_t0

    retry_indices: list[int] = []
    base_answers: list[str] = []
    base_output_tokens: list[int] = []
    base_finish_reasons: list[str | None] = []
    for out in base_outputs:
        completion = out.outputs[0]
        answer = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        output_tokens = len(token_ids) if token_ids is not None else len(tokenizer(answer).input_ids)
        idx = len(base_answers)
        if output_tokens >= args.max_tokens or probe.has_repetition_loop(answer):
            retry_indices.append(idx)
        base_answers.append(answer)
        base_output_tokens.append(output_tokens)
        base_finish_reasons.append(getattr(completion, "finish_reason", None))

    retry_answers: dict[int, tuple[str, int, str | None, str | None]] = {}
    retry_generation_s = 0.0
    if retry_indices:
        retry_prompts = [
            probe.apply_user_only_template(tokenizer, str(sample_rows[i]["question"]), True, RETRY_PREFIX)
            for i in retry_indices
        ]
        retry_t0 = time.perf_counter()
        retry_outputs = llm.generate(retry_prompts, sampling_params=sampling)
        retry_generation_s = time.perf_counter() - retry_t0
        for row_index, out in zip(retry_indices, retry_outputs):
            completion = out.outputs[0]
            answer = completion.text.strip()
            token_ids = getattr(completion, "token_ids", None)
            output_tokens = len(token_ids) if token_ids is not None else len(tokenizer(answer).input_ids)
            retry_answers[row_index] = (
                answer,
                output_tokens,
                getattr(completion, "finish_reason", None),
                getattr(completion, "stop_reason", None),
            )
    sampler.stop()

    result_rows: list[dict[str, Any]] = []
    exact_handler_counts: dict[str, int] = {}
    for i, row in enumerate(sample_rows):
        retried = i in retry_answers
        raw_answer = retry_answers[i][0] if retried else base_answers[i]
        raw_tokens = retry_answers[i][1] if retried else base_output_tokens[i]
        finish_reason = retry_answers[i][2] if retried else base_finish_reasons[i]
        stop_reason = retry_answers[i][3] if retried else None
        final_answer, handler = final_stack_answer(str(row["question"]), raw_answer)
        if handler:
            exact_handler_counts[handler] = exact_handler_counts.get(handler, 0) + 1
        final_tokens = len(tokenizer(final_answer).input_ids)
        retry_reason = None
        if retried:
            base_cap = base_output_tokens[i] >= args.max_tokens
            base_rep = probe.has_repetition_loop(base_answers[i])
            retry_reason = "+".join(part for part, yes in [("max_token", base_cap), ("repetition", base_rep)] if yes)
        result_rows.append(
            output_row(
                run_id=run_id,
                sample_index=i,
                row=row,
                prompt=None,
                prompt_tokens=input_token_counts[i],
                answer=final_answer,
                output_tokens=final_tokens,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                base_answer=base_answers[i] if retried else None,
                retry_answer=retry_answers[i][0] if retried else None,
                retried=retried,
                retry_reason=retry_reason,
                exact_handler=handler,
            )
        )

    base.append_jsonl(run_paths["outputs"], result_rows)
    total_generation_s = base_generation_s + retry_generation_s
    projected_generation_4000_s = (total_generation_s / max(1, len(sample_rows))) * 4000
    projected_total_4000_s = startup_s + projected_generation_4000_s
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "run_id": run_id,
        "status": "completed",
        "candidate": "qwen3-8b-awq",
        "model_ref": MODEL_ID,
        "model_source": "huggingface",
        "hf_metadata": None if args.skip_hf_metadata else probe.hf_metadata(MODEL_ID),
        "config": {
            "sample_source": args.sample_source,
            "sample_size_requested": args.sample_size,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "dtype": "float16",
            "quantization": "awq_marlin",
            "base_prefix": BASE_PREFIX,
            "retry_prefix": RETRY_PREFIX,
            "mechanism": "retry_only_first_pass_max_token_or_repetition_rows",
            "model_backend_sampling_handlers_changed": False,
        },
        "environment": probe.environment_snapshot(),
        "sample": {
            "rows": len(sample_rows),
            "category_counts": sample_df["category"].value_counts().sort_index().to_dict()
            if "category" in sample_df
            else {},
        },
        "runtime": {
            "tokenizer_load_s": tokenizer_s,
            "startup_s": startup_s,
            "base_generation_s": base_generation_s,
            "retry_generation_s": retry_generation_s,
            "generation_s": total_generation_s,
            "sample_rows": len(sample_rows),
            "projected_generation_4000_s": projected_generation_4000_s,
            "projected_total_4000_s": projected_total_4000_s,
            "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
        },
        "tokens": {
            "avg_input_tokens": sum(input_token_counts) / max(1, len(input_token_counts)),
            "avg_output_tokens_after_final_stack": sum(int(row["output_tokens"]) for row in result_rows)
            / max(1, len(result_rows)),
            "max_output_tokens_after_final_stack": max((int(row["output_tokens"]) for row in result_rows), default=None),
        },
        "validity": {
            "jsonl_rows": len(result_rows),
            "one_answer_per_input": len(result_rows) == len(sample_rows),
            "thinking_trace_rows": sum(1 for row in result_rows if row["has_thinking_trace"]),
            "max_token_hit_rows": sum(1 for row in result_rows if row["hit_max_tokens"]),
            "empty_answer_rows": sum(1 for row in result_rows if not str(row["answer"]).strip()),
            "repetition_loop_suspected_rows": sum(1 for row in result_rows if row["repetition_loop_suspected"]),
        },
        "selective_retry": {
            "retried_rows": len(retry_indices),
            "retried_row_ids": [int(sample_rows[i]["row_id"]) for i in retry_indices],
            "retry_rate": len(retry_indices) / max(1, len(sample_rows)),
        },
        "exact_stack": {"handler_counts": dict(sorted(exact_handler_counts.items()))},
        "gpu_memory_samples": sampler.samples[-20:],
        "paths": {
            "summary": str(run_paths["summary"]),
            "metrics": str(run_paths["metrics"]),
            "outputs": str(run_paths["outputs"]),
            "samples": str(run_paths["samples"]),
            "log": str(run_paths["log"]),
        },
    }
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    rec, reason = recommendation(metrics, dry_run=False)
    run_paths["log"].write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"summary={run_paths['summary']}",
                f"outputs={run_paths['outputs']}",
                f"status={summary.get('status')}",
                f"retried_rows={len(retry_indices)}",
                f"recommendation={rec}",
                f"reason={reason}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    retry = metrics.get("selective_retry") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C152 Selective Cap/Loop Retry Feasibility Report",
        "",
        "## Objective",
        "- ID: C152",
        "- Mechanism: retry only first-pass max-token/repetition rows with the C100 no-detailed-reasoning prompt.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- base prefix: `{BASE_PREFIX}`",
        f"- retry prefix: `{RETRY_PREFIX}`",
        "- model/backend/sampling: Qwen3-8B-AWQ, awq_marlin, greedy, max_tokens=320.",
        "- deterministic exact stack: current `simple_solution.solution` handlers applied after first-pass/retry answer.",
        "",
        "## Results",
        "| status | rows | retry fires | max-token hits | thinking traces | empty answers | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {fires} | {cap_hits} | {thinking} | {empty} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            fires=retry.get("retried_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            empty=validity.get("empty_answer_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Retry Coverage",
        f"- retried row ids: `{retry.get('retried_row_ids', [])}`",
        f"- exact-stack handler counts: `{(metrics.get('exact_stack') or {}).get('handler_counts', {})}`",
        "",
        "## Decision recommendation",
        "",
        rec,
        "",
        "## Strongest reason against recommendation",
        f"- {reason}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C152 selective cap/loop retry feasibility.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="hard_audit")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    archived_previous = base.prepare_out_dir(out_dir)
    paths = artifact_paths(out_dir)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    record = create_dry_run(paths, args) if args.dry_run else create_gpu_artifacts(paths, args)
    write_report(paths["report"], record["metrics"], args, dry_run=args.dry_run)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "created_utc": base.utc_stamp(),
        "dry_run": args.dry_run,
        "out_dir": str(paths["out_dir"]),
        "zip_path": str(paths["zip"]),
        "archived_previous_out_dir": str(archived_previous) if archived_previous else None,
        "runs": [
            {
                "run_id": record.get("run_id"),
                "summary_path": str(record["paths"]["summary"]),
                "metrics_path": str(record["paths"]["metrics"]),
                "outputs_path": str(record["paths"]["outputs"]),
                "log_path": str(record["paths"]["log"]),
                "status": record["summary"].get("status"),
            }
        ],
    }
    base.write_json(paths["out_dir"] / "artifact_manifest.json", manifest)
    zip_path = base.make_zip(paths["out_dir"])
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "packaged",
                "dry_run": args.dry_run,
                "out_dir": str(paths["out_dir"]),
                "zip_path": str(zip_path),
                "report": str(paths["report"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
