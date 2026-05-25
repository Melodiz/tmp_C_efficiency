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
import c078_quantized_8b_awq_feasibility as c078


EXPERIMENT_ID = "C096"
EXPERIMENT_SLUG = "C096_qwen3_8b_thinking_final_only_prompt"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C096_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
THINKING_FINAL_ONLY_PREFIX = (
    "Реши задачу внимательно. Думай при необходимости, но в ответе выведи только итоговый ответ "
    "без объяснений. Сохрани язык задания."
)
OMIT_ENABLE_THINKING_FALSE = True
MECHANISM_ID = "thinking_final_only_prompt"


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


def build_metrics(summary: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    metrics = c078.build_metrics(summary, paths)
    metrics["experiment_id"] = EXPERIMENT_ID
    metrics["experiment_slug"] = EXPERIMENT_SLUG
    metrics["prompt_change"] = {
        "source": "C093/C082 language-preserving prefix",
        "mechanism": "enable_qwen_thinking_template_with_final_only_prefix",
        "user_prefix": THINKING_FINAL_ONLY_PREFIX,
        "enable_thinking_false_in_template": not OMIT_ENABLE_THINKING_FALSE,
        "model_backend_sampling_changed": False,
        "deterministic_handlers_added": False,
    }
    return metrics


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no prompt evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C096 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("empty_answer_rows"):
        return "KILL", "Basic output validity failed."
    if validity.get("thinking_trace_rows"):
        return "KILL", "Thinking traces leaked into answers."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if int(validity.get("max_token_hit_rows") or 0) > 8:
        return "KILL", "The thinking prompt increased truncation risk versus the C093 branch."
    if int(validity.get("repetition_loop_suspected_rows") or 0) > 4:
        return "KILL", "The thinking prompt increased repetition risk versus the C093 branch."
    return "MUTATE", "The thinking/final-only prompt passed validity gates; row-level review and hard-audit validation are needed."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C096 Qwen3-8B Thinking Final-Only Prompt Report",
        "",
        "## Objective",
        "- ID: C096",
        "- Mechanism: prompt-only test enabling the Qwen thinking chat-template path while asking for final answer only.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- prefix: `{THINKING_FINAL_ONLY_PREFIX}`",
        f"- `enable_thinking=False` passed to chat template: `{not OMIT_ENABLE_THINKING_FALSE}`.",
        "- No deterministic handlers, retrieval, cache, SFT, LoRA, model/backend change, or sampling change.",
        "",
        "## Results",
        "| status | rows | max-token hits | thinking traces | empty answers | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {cap_hits} | {thinking} | {empty} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            empty=validity.get("empty_answer_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
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


def create_dry_run(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_c096_dry_run"
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
            "user_prefix": THINKING_FINAL_ONLY_PREFIX,
            "enable_thinking_false_in_template": not OMIT_ENABLE_THINKING_FALSE,
            "c096_mechanism": MECHANISM_ID,
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
    run_paths["log"].write_text("dry_run=true\n", encoding="utf-8")
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
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        dtype="float16",
        quantization="awq_marlin",
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_sample_interval=args.gpu_sample_interval,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        no_enable_thinking_false=OMIT_ENABLE_THINKING_FALSE,
        user_prefix=THINKING_FINAL_ONLY_PREFIX,
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
            "c096_mechanism": MECHANISM_ID,
            "source_experiment_id": "C093_prompt_baseline",
            "model_backend_sampling_changed_from_c093": False,
            "deterministic_handlers_added": False,
            "user_prefix": THINKING_FINAL_ONLY_PREFIX,
            "enable_thinking_false_in_template": not OMIT_ENABLE_THINKING_FALSE,
        }
    )
    summary["paths"] = {
        "summary": str(run_paths["summary"]),
        "outputs": str(run_paths["outputs"]),
        "metrics": str(run_paths["metrics"]),
        "log": str(run_paths["log"]),
    }
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"summary={run_paths['summary']}",
                f"outputs={run_paths['outputs']}",
                f"status={summary.get('status')}",
                f"recommendation={recommendation(metrics, False)[0]}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": summary.get("run_id"), "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C096 Qwen3-8B thinking final-only prompt.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create artifact layout without a model run.")
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
    record = create_dry_run(paths, args) if args.dry_run else create_gpu_artifacts(paths, args)
    if not args.dry_run:
        record["summary"].setdefault("runtime", {})["c096_wrapper_s"] = time.perf_counter() - started
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
