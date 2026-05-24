from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import c072_output_control as base
import c075_deterministic_guard as guard_base
import c084_c083_hard_audit_validation as c084


EXPERIMENT_ID = "C086"
EXPERIMENT_SLUG = "C086_c084_repetition_list_dedup"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C086_artifacts"
SOURCE_EXPERIMENT_ID = "C084"


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_item(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().strip(" .;:!?")).replace("ё", "е")


def dedup_comma_loop(answer: str) -> str | None:
    if "," not in answer or not guard_base.has_repetition_loop(answer):
        return None
    compact = " ".join(answer.replace("\n", " ").split())
    prefix = ""
    body = compact
    if ":" in compact[:80]:
        prefix, body = compact.split(":", 1)
        prefix = prefix.strip() + ": "

    raw_items = [item.strip(" .;:!?") for item in body.split(",")]
    items = [item for item in raw_items if item]
    if len(items) < 12:
        return None

    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = normalize_item(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    if len(unique) < 2 or len(unique) / len(items) > 0.45:
        return None

    answer_list = ", ".join(unique[:30])
    if not answer_list:
        return None
    first_line = f"{prefix}{answer_list}".strip()
    return f"{first_line}\n\nИтоговый ответ: {answer_list}"


def apply_postprocess(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processed_rows: list[dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    applied_row_ids: list[int] = []

    for row in rows:
        new_row = dict(row)
        fixed = dedup_comma_loop(str(row.get("answer", "")))
        if fixed is not None:
            row_id = int(row.get("row_id", -1))
            by_kind["comma_repetition_dedup"] += 1
            applied_row_ids.append(row_id)
            new_row["source_c084_answer"] = row.get("answer")
            new_row["answer"] = fixed
            new_row["output_tokens"] = guard_base.estimate_output_tokens(fixed)
            new_row["finish_reason"] = "deterministic_postprocess"
            new_row["stop_reason"] = "comma_repetition_dedup"
            new_row["hit_max_tokens"] = False
            new_row["repetition_loop_suspected"] = False
            new_row["deterministic_postprocess"] = {
                "applied": True,
                "kind": "comma_repetition_dedup",
            }
        else:
            new_row["deterministic_postprocess"] = {"applied": False}
            new_row["repetition_loop_suspected"] = guard_base.has_repetition_loop(str(new_row.get("answer", "")))
        processed_rows.append(new_row)

    return processed_rows, {
        "applied_rows": len(applied_row_ids),
        "applied_row_ids": applied_row_ids,
        "by_kind": dict(sorted(by_kind.items())),
        "abstains_unless_obvious_comma_repetition_loop": True,
    }


def build_summary(
    source_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    post_stats: dict[str, Any],
    paths: dict[str, Path],
    source_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    output_token_counts = [int(row.get("output_tokens") or 0) for row in rows]
    summary = dict(source_summary)
    source_experiment = summary.get("experiment_id")
    if source_experiment and source_experiment != EXPERIMENT_ID:
        summary["source_experiment_id"] = source_experiment
    summary["experiment_id"] = EXPERIMENT_ID
    summary["experiment_slug"] = EXPERIMENT_SLUG
    summary["status"] = "completed"
    summary["source_c084_manifest"] = source_manifest
    config = summary.setdefault("config", {})
    config.update(
        {
            "c086_mechanism": "deterministic_comma_repetition_dedup",
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "model_backend_prompt_sampling_guard_changed_from_c084": False,
            "router_retrieval_cache_sft_lora": False,
        }
    )
    runtime = summary.setdefault("runtime", {})
    runtime["sample_rows"] = len(rows)
    summary["tokens"] = {
        **(summary.get("tokens") or {}),
        "avg_output_tokens_after_postprocess": sum(output_token_counts) / max(1, len(output_token_counts)),
        "max_output_tokens_after_postprocess": max(output_token_counts) if output_token_counts else None,
        "total_output_tokens_after_postprocess": sum(output_token_counts),
    }
    summary["validity"] = {
        "jsonl_rows": len(rows),
        "one_answer_per_input": len(rows) == int((summary.get("sample") or {}).get("rows") or len(rows)),
        "thinking_trace_rows": sum(1 for row in rows if "<think" in str(row.get("answer", "")) or "</think>" in str(row.get("answer", ""))),
        "max_token_hit_rows": sum(1 for row in rows if row.get("hit_max_tokens")),
        "empty_answer_rows": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
        "repetition_loop_suspected_rows": sum(1 for row in rows if row.get("repetition_loop_suspected")),
    }
    summary["deterministic_postprocess"] = post_stats
    summary["paths"] = {
        "summary": str(paths["summary"]),
        "metrics": str(paths["metrics"]),
        "outputs": str(paths["outputs"]),
        "log": str(paths["log"]),
    }
    return summary


def build_metrics(summary: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    validity = summary.get("validity") or {}
    runtime = summary.get("runtime") or {}
    sample_rows = int(runtime.get("sample_rows") or validity.get("jsonl_rows") or 0)
    max_token_hit_rows = int(validity.get("max_token_hit_rows") or 0)
    empty_answer_rows = int(validity.get("empty_answer_rows") or 0)
    thinking_trace_rows = int(validity.get("thinking_trace_rows") or 0)
    repetition_rows = int(validity.get("repetition_loop_suspected_rows") or 0)
    projected = runtime.get("projected_total_4000_s")
    post_stats = summary.get("deterministic_postprocess") or {}
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "status": summary.get("status"),
        "candidate": summary.get("candidate"),
        "model_ref": summary.get("model_ref"),
        "summary_path": str(paths["summary"]),
        "outputs_path": str(paths["outputs"]),
        "source": SOURCE_EXPERIMENT_ID,
        "deterministic_postprocess": post_stats,
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
            "postprocess_fire_rate": post_stats.get("applied_rows", 0) / sample_rows if sample_rows else None,
            "projected_total_4000_min": projected / 60 if projected is not None else None,
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
        return "INVESTIGATE", "Dry run only; no model or postprocess evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    post_stats = metrics.get("deterministic_postprocess") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C086 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after repetition postprocess."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if post_stats.get("applied_rows", 0) == 0:
        return "KILL", "The repetition postprocess did not fire on the known loop row."
    return "MUTATE", "The known comma-list repetition loop was postprocessed; remaining cap/quality risks need review."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation, reason = decision_recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    post_stats = metrics.get("deterministic_postprocess") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C086 C084 Repetition List Dedup Report",
        "",
        "## Objective",
        "- ID: C086",
        "- Mechanism: deterministic postprocess for obvious comma-separated repetition loops on top of unchanged C084.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python scripts/c086_c084_repetition_list_dedup.py --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- dry run: `{dry_run}`",
        "- C084 model/backend/prefix/sampling/guard changed: `False`.",
        "- forbidden methods: no retrieval/RAG, cache, SFT, LoRA, system prompt, sampling change, or broad solver.",
        "",
        "## Results",
        "| status | rows | postprocess fires | max-token hits | thinking traces | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {fires} | {cap_hits} | {thinking} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            fires=post_stats.get("applied_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Postprocess Coverage",
        f"- applied rows: `{post_stats.get('applied_row_ids', [])}`",
        f"- by kind: `{post_stats.get('by_kind', {})}`",
        "",
        "## Remaining Known Risk",
        "- C086 does not address complex algebra truncation such as row 3970.",
        "- C086 does not address broader grammar, geometry, chemistry, or essay quality.",
        "- Offline packaging was not performed by this runner.",
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
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_dry_run(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_c086_dry_run"
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
        "model_ref": "Qwen/Qwen3-8B-AWQ",
        "config": {
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "sample_source": args.sample_source,
            "sample_size_requested": args.sample_size,
            "c086_mechanism": "deterministic_comma_repetition_dedup",
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
        "deterministic_postprocess": {
            "applied_rows": 0,
            "applied_row_ids": [],
            "by_kind": {},
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


def run_source_c084(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c084_{base.utc_stamp()}"
    if source_out.exists():
        shutil.rmtree(source_out)
    forwarded = [
        "--out",
        str(source_out),
        "--sample-source",
        args.sample_source,
        "--sample-size",
        str(args.sample_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
    ]
    if args.skip_hf_metadata:
        forwarded.append("--skip-hf-metadata")
    if args.trust_remote_code:
        forwarded.append("--trust-remote-code")
    code = c084.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C084 source runner failed with exit {code}")
    return source_out


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_out = run_source_c084(paths["out_dir"], args)
    source_manifest = base.read_json(source_out / "artifact_manifest.json")
    source_run = source_manifest["runs"][0]
    source_summary = base.read_json(Path(source_run["summary_path"]))
    source_outputs = read_jsonl(Path(source_run["outputs_path"]))
    processed_rows, post_stats = apply_postprocess(source_outputs)

    run_id = str(source_summary.get("run_id") or f"{base.utc_stamp()}_c086")
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = build_summary(source_summary, processed_rows, post_stats, run_paths, source_manifest)
    summary.setdefault("runtime", {})["c086_postprocess_s"] = time.perf_counter() - started
    base.append_jsonl(run_paths["outputs"], processed_rows)
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].parent.mkdir(parents=True, exist_ok=True)
    run_paths["log"].write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"source_out={source_out}",
                f"source_summary={source_run.get('summary_path')}",
                f"postprocess_applied_rows={post_stats['applied_rows']}",
                f"postprocess_by_kind={post_stats['by_kind']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C086 repetition-list dedup over unchanged C084.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="hard_audit")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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

    if args.dry_run:
        run_record = create_dry_run(paths, args)
    else:
        run_record = create_gpu_artifacts(paths, args)

    write_report(paths["report"], run_record["metrics"], args, dry_run=args.dry_run)
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
                "run_id": run_record["run_id"],
                "summary_path": str(run_record["paths"]["summary"]),
                "metrics_path": str(run_record["paths"]["metrics"]),
                "outputs_path": str(run_record["paths"]["outputs"]),
                "log_path": str(run_record["paths"]["log"]),
                "metrics": run_record["metrics"],
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
