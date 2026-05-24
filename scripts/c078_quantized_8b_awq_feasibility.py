from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import c071_probe as probe
import c072_output_control as base
import c073_short_prefix_output_control as c073


EXPERIMENT_ID = "C078"
EXPERIMENT_SLUG = "C078_qwen3_8b_awq_feasibility"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C078_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
QUANTIZATION = "AWQ"
SOURCE_EXPERIMENT_ID = "C073"
RUNNER_SCRIPT = "scripts/c078_quantized_8b_awq_feasibility.py"
SAMPLING_CHANGED_FROM_C073 = False
MECHANISM_DESCRIPTION = "switch only the model path to `Qwen/Qwen3-8B-AWQ` with AWQ quantization."


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


def projected_min(runtime: dict[str, Any]) -> float | None:
    projected = runtime.get("projected_total_4000_s")
    if isinstance(projected, (int, float)):
        return projected / 60
    return None


def hf_size_gb(hf_metadata: dict[str, Any] | None) -> float | None:
    if not isinstance(hf_metadata, dict):
        return None
    size = hf_metadata.get("total_file_size_bytes")
    if isinstance(size, (int, float)):
        return size / 1_000_000_000
    return None


def build_metrics(summary: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    validity = summary.get("validity") or {}
    runtime = summary.get("runtime") or {}
    sample_rows = int(runtime.get("sample_rows") or validity.get("jsonl_rows") or 0)
    max_token_hit_rows = int(validity.get("max_token_hit_rows") or 0)
    empty_answer_rows = int(validity.get("empty_answer_rows") or 0)
    thinking_trace_rows = int(validity.get("thinking_trace_rows") or 0)
    repetition_rows = int(validity.get("repetition_loop_suspected_rows") or 0)
    size_gb = hf_size_gb(summary.get("hf_metadata"))

    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "status": summary.get("status"),
        "candidate": summary.get("candidate"),
        "model_ref": summary.get("model_ref"),
        "summary_path": str(paths["summary"]),
        "outputs_path": str(paths["outputs"]),
        "log_path": str(paths["log"]),
        "model_change": {
            "source": "C073_short_prefix_320",
            "model_ref": MODEL_ID,
            "quantization": QUANTIZATION,
            "only_mechanism_change": True,
            "deterministic_guard_retrieval_cache_sft_lora": False,
        },
        "sample_rows": sample_rows,
        "runtime": runtime,
        "tokens": summary.get("tokens"),
        "validity": validity,
        "environment": summary.get("environment"),
        "hf_metadata": summary.get("hf_metadata"),
        "rates": {
            "max_token_hit_rate": max_token_hit_rows / sample_rows if sample_rows else None,
            "empty_answer_rate": empty_answer_rows / sample_rows if sample_rows else None,
            "thinking_trace_rate": thinking_trace_rows / sample_rows if sample_rows else None,
            "repetition_loop_suspected_rate": repetition_rows / sample_rows if sample_rows else None,
            "projected_total_4000_min": projected_min(runtime),
        },
        "feasibility": {
            "hf_file_size_gb": size_gb,
            "peak_vram_used_mb_nvidia_smi": runtime.get("peak_vram_used_mb_nvidia_smi"),
            "package_size_under_10gb_estimate": None if size_gb is None else size_gb < 10,
        },
        "basic_validity_pass": bool(
            summary.get("status") == "completed"
            and validity.get("one_answer_per_input") is True
            and thinking_trace_rows == 0
            and empty_answer_rows == 0
        ),
    }


def decision_recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no model load, runtime, or quality evidence was produced."
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The quantized 8B runner did not complete; inspect the error summary/log before mutating."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    feasibility = metrics.get("feasibility") or {}
    projected = rates.get("projected_total_4000_min")
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed for the quantized 8B run."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "MUTATE", "The model loaded but missed the runtime gate; one inference-trick follow-up may be justified."
    if feasibility.get("package_size_under_10gb_estimate") is False:
        return "KILL", "The observed HF file size is incompatible with the final package limit."
    if (validity.get("repetition_loop_suspected_rows") or 0) > 10:
        return "MUTATE", "The model is feasible but repetition is high; Qwen non-thinking sampling can be tested separately."
    return "MUTATE", "The quantized 8B path is feasible enough for qualitative review and one narrow follow-up."


def write_c078_summary_fields(summary: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    source_experiment = summary.get("experiment_id")
    if source_experiment and source_experiment != EXPERIMENT_ID:
        summary["source_experiment_id"] = source_experiment
    summary["experiment_id"] = EXPERIMENT_ID
    summary["experiment_slug"] = EXPERIMENT_SLUG
    summary["candidate"] = "qwen3-8b-awq"
    summary["model_ref"] = MODEL_ID
    config = summary.setdefault("config", {})
    config.update(
        {
            "c078_mechanism": "quantized_8b_model_feasibility",
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "source_c073_variant": "short_prefix_320",
            "model_changed_from_c073": True,
            "quantization": args.quantization,
            "model_prompt_sampling_changed_from_c073": SAMPLING_CHANGED_FROM_C073,
            "short_user_prefix": c073.SHORT_USER_PREFIX,
            "system_prompt": False,
            "deterministic_guard_retrieval_cache_rag_sft_lora": False,
        }
    )
    summary["paths"] = {
        "summary": str(paths["summary"]),
        "metrics": str(paths["metrics"]),
        "outputs": str(paths["outputs"]),
        "log": str(paths["log"]),
    }
    return summary


def issue_examples(outputs_path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not outputs_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with outputs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("hit_max_tokens") or row.get("repetition_loop_suspected") or row.get("has_thinking_trace"):
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation, reason = decision_recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    runtime = metrics.get("runtime") or {}
    rates = metrics.get("rates") or {}
    feasibility = metrics.get("feasibility") or {}
    environment = metrics.get("environment") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    size_gb = feasibility.get("hf_file_size_gb")
    size_text = f"{size_gb:.2f} GB" if isinstance(size_gb, (int, float)) else "unknown"
    examples = issue_examples(Path(metrics.get("outputs_path") or ""))

    lines = [
        f"# {EXPERIMENT_ID} Qwen3-8B-AWQ Quantized Feasibility Report",
        "",
        "## Objective",
        f"- ID: {EXPERIMENT_ID}",
        f"- Mechanism: {MECHANISM_DESCRIPTION}",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python {RUNNER_SCRIPT} --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- model id: `{MODEL_ID}`",
        f"- quantization: `{args.quantization}`",
        f"- max tokens: `{args.max_tokens}`",
        f"- max model len: `{args.max_model_len}`",
        f"- temperature/top_p/top_k: `{args.temperature}` / `{args.top_p}` / `{args.top_k}`",
        f"- dry run: `{dry_run}`",
        f"- short user prefix: `{c073.SHORT_USER_PREFIX}`",
        "- forbidden methods: no deterministic guard, retrieval/RAG, cache, SFT, LoRA, system prompt, or new prompt text.",
        "",
        "## Environment",
    ]
    if environment:
        lines.extend(
            [
                f"- torch: `{environment.get('torch', 'unknown')}`",
                f"- cuda available: `{environment.get('cuda_available', 'unknown')}`",
                f"- cuda version: `{environment.get('cuda_version', 'unknown')}`",
                f"- vLLM: `{environment.get('vllm', 'unknown')}`",
                f"- transformers: `{environment.get('transformers', 'unknown')}`",
                f"- GPUs: `{environment.get('gpus', environment.get('gpu_count', 'unknown'))}`",
            ]
        )
    else:
        lines.append("- Not captured in dry run or before probe startup.")

    lines.extend(
        [
            "",
            "## Results",
            "| status | rows | max-token hits | thinking traces | repetition suspects | projected 4000q min | peak VRAM MB |",
            "|---|---:|---:|---:|---:|---:|---:|",
            "| {status} | {rows} | {cap_hits} | {thinking} | {repetition} | {projected} | {vram} |".format(
                status=metrics.get("status"),
                rows=metrics.get("sample_rows", 0),
                cap_hits=validity.get("max_token_hit_rows", "n/a"),
                thinking=validity.get("thinking_trace_rows", "n/a"),
                repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
                projected=projected_text,
                vram=runtime.get("peak_vram_used_mb_nvidia_smi", "n/a"),
            ),
            "",
            "## Package/Offline Feasibility",
            f"- HF file size observed: `{size_text}`",
            f"- package size under 10 GB estimate: `{feasibility.get('package_size_under_10gb_estimate')}`",
            "- Offline packaging was not performed by this runner.",
            "",
            "## Comparison Baselines",
            "- C073 Qwen3-4B short-prefix hard-audit: 9/200 cap hits, 2/200 repetition suspects, projected 8.04 min.",
            "- C076/C077 held-out guard checks used Qwen3-4B plus deterministic postprocessing; C078 uses no guard.",
            "- C000 public baseline remains 46.00 and is still the only leaderboard reference.",
            "",
            "## Issue Examples",
        ]
    )
    if examples:
        for row in examples:
            answer = str(row.get("answer", "")).replace("\n", "\\n")
            if len(answer) > 220:
                answer = answer[:217] + "..."
            lines.append(
                f"- row `{row.get('row_id')}` `{row.get('category')}`: cap={row.get('hit_max_tokens')} rep={row.get('repetition_loop_suspected')} answer=`{answer}`"
            )
    else:
        lines.append("- No max-token/repetition/thinking examples were selected.")

    lines.extend(
        [
            "",
            "## Artifact Layout",
            f"- report: `reports/{EXPERIMENT_SLUG}_report.md`",
            f"- results: `results/{EXPERIMENT_ID}/*.summary.json`, `*.metrics.json`, `*.outputs.jsonl`",
            f"- logs: `logs/{EXPERIMENT_ID}/*.log`",
            "",
            "## Decision recommendation",
            "",
            recommendation,
            "",
            "## Strongest reason against recommendation",
            f"- {reason}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_dry_run_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_qwen3-8b-awq_dry_run"
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
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "user_message_only": True,
            "short_user_prefix": c073.SHORT_USER_PREFIX,
            "enable_thinking_false_in_template": True,
            "deterministic_guard_retrieval_cache_rag_sft_lora": False,
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
        "paths": {
            "summary": str(run_paths["summary"]),
            "metrics": str(run_paths["metrics"]),
            "outputs": str(run_paths["outputs"]),
            "log": str(run_paths["log"]),
        },
    }
    base.append_jsonl(run_paths["outputs"], [])
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].parent.mkdir(parents=True, exist_ok=True)
    run_paths["log"].write_text("dry_run=true\nNo GPU experiment was executed.\n", encoding="utf-8")
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    run_args = SimpleNamespace(
        candidate="qwen3-8b-awq",
        model_id=MODEL_ID,
        baseline_local_path=str(probe.BASELINE_LOCAL_PATH),
        sample_source=args.sample_source,
        sample_size=args.sample_size,
        output_dir=str(paths["results_dir"]),
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        dtype=args.dtype,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_sample_interval=args.gpu_sample_interval,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        no_enable_thinking_false=False,
        user_prefix=c073.SHORT_USER_PREFIX,
        skip_hf_metadata=args.skip_hf_metadata,
        save_prompts=False,
        dry_run=False,
        no_fail=True,
    )
    summary = probe.run_probe(run_args)
    summary_path = Path((summary.get("paths") or {}).get("summary", ""))
    outputs_path = Path((summary.get("paths") or {}).get("outputs", ""))
    log_path = paths["logs_dir"] / f"{summary.get('run_id', base.utc_stamp())}_qwen3-8b-awq.log"
    run_paths = {
        **paths,
        "summary": summary_path,
        "outputs": outputs_path,
        "metrics": summary_path.with_name(summary_path.name.replace(".summary.json", ".metrics.json")),
        "log": log_path,
    }
    summary = write_c078_summary_fields(summary, run_paths, args)
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"model_id={MODEL_ID}",
                f"quantization={args.quantization}",
                f"summary={run_paths['summary']}",
                f"outputs={run_paths['outputs']}",
                f"status={summary.get('status')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": summary.get("run_id"), "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C078 Qwen3-8B-AWQ quantized feasibility probe.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=QUANTIZATION)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create the artifact layout without loading a model.")
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

    started = time.perf_counter()
    if args.dry_run:
        record = create_dry_run_artifacts(paths, args)
    else:
        record = create_gpu_artifacts(paths, args)
        record["summary"].setdefault("runtime", {})["c078_wrapper_s"] = time.perf_counter() - started
        base.write_json(record["paths"]["summary"], record["summary"])
        record["metrics"] = build_metrics(record["summary"], record["paths"])
        base.write_json(record["paths"]["metrics"], record["metrics"])

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
                "metrics": record["metrics"],
            }
        ],
    }
    base.write_json(out_dir / "artifact_manifest.json", manifest)
    zip_path = base.make_zip(out_dir)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "packaged",
                "dry_run": args.dry_run,
                "out_dir": str(out_dir),
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
