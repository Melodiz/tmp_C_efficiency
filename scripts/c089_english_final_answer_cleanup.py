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
import c087_c086_locked_val_validation as c087


EXPERIMENT_ID = "C089"
EXPERIMENT_SLUG = "C089_english_final_answer_cleanup"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C089_artifacts"
SOURCE_EXPERIMENT_ID = "C087"


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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_english_prompt(question: str) -> bool:
    latin = len(re.findall(r"[A-Za-z]", question))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", question))
    return latin >= 10 and latin > cyrillic * 2


def cleanup_english_answer(question: str, answer: str) -> str | None:
    if not is_english_prompt(question):
        return None
    if not re.search(r"[А-Яа-яЁё]", answer) or not re.search(r"ответ\s*:", answer, flags=re.IGNORECASE):
        return None

    before_marker = re.split(r"\*{0,2}\s*Ответ\s*:\s*\*{0,2}", answer, maxsplit=1, flags=re.IGNORECASE)[0]
    first_lines = [line.strip(" *") for line in before_marker.splitlines() if line.strip(" *")]
    if not first_lines:
        return None
    first = first_lines[0].strip()
    if not re.search(r"[A-Za-z]", first) or re.search(r"[А-Яа-яЁё]", first):
        return None
    if len(first.split()) > 12:
        return None
    return first


def apply_cleanup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processed = []
    by_kind: Counter[str] = Counter()
    applied_row_ids: list[int] = []
    for row in rows:
        new_row = dict(row)
        fixed = cleanup_english_answer(str(row.get("question", "")), str(row.get("answer", "")))
        if fixed is not None:
            row_id = int(row.get("row_id", row.get("rid", -1)))
            by_kind["english_russian_answer_tail_strip"] += 1
            applied_row_ids.append(row_id)
            new_row["source_c087_answer"] = row.get("answer")
            new_row["answer"] = fixed
            new_row["output_tokens"] = guard_base.estimate_output_tokens(fixed)
            new_row["finish_reason"] = "deterministic_postprocess"
            new_row["stop_reason"] = "english_russian_answer_tail_strip"
            new_row["hit_max_tokens"] = False
            new_row["repetition_loop_suspected"] = False
            new_row["english_cleanup"] = {"applied": True, "kind": "english_russian_answer_tail_strip"}
        else:
            new_row["english_cleanup"] = {"applied": False}
            new_row["repetition_loop_suspected"] = guard_base.has_repetition_loop(str(new_row.get("answer", "")))
        processed.append(new_row)
    return processed, {
        "applied_rows": len(applied_row_ids),
        "applied_row_ids": applied_row_ids,
        "by_kind": dict(sorted(by_kind.items())),
        "abstains_unless_english_prompt_with_russian_answer_tail": True,
    }


def build_summary(source_summary: dict[str, Any], rows: list[dict[str, Any]], stats: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    output_token_counts = [int(row.get("output_tokens") or 0) for row in rows]
    summary = dict(source_summary)
    source_experiment = summary.get("experiment_id")
    if source_experiment and source_experiment != EXPERIMENT_ID:
        summary["source_experiment_id"] = source_experiment
    summary["experiment_id"] = EXPERIMENT_ID
    summary["experiment_slug"] = EXPERIMENT_SLUG
    summary["status"] = "completed"
    config = summary.setdefault("config", {})
    config.update(
        {
            "c089_mechanism": "deterministic_english_final_answer_cleanup",
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "model_prompt_sampling_guard_repetition_changed_from_c087": False,
        }
    )
    runtime = summary.setdefault("runtime", {})
    runtime["sample_rows"] = len(rows)
    summary["tokens"] = {
        **(summary.get("tokens") or {}),
        "avg_output_tokens_after_english_cleanup": sum(output_token_counts) / max(1, len(output_token_counts)),
        "max_output_tokens_after_english_cleanup": max(output_token_counts) if output_token_counts else None,
        "total_output_tokens_after_english_cleanup": sum(output_token_counts),
    }
    summary["validity"] = {
        "jsonl_rows": len(rows),
        "one_answer_per_input": len(rows) == int((summary.get("sample") or {}).get("rows") or len(rows)),
        "thinking_trace_rows": sum(1 for row in rows if "<think" in str(row.get("answer", "")) or "</think>" in str(row.get("answer", ""))),
        "max_token_hit_rows": sum(1 for row in rows if row.get("hit_max_tokens")),
        "empty_answer_rows": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
        "repetition_loop_suspected_rows": sum(1 for row in rows if row.get("repetition_loop_suspected")),
    }
    summary["english_cleanup"] = stats
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
    projected = runtime.get("projected_total_4000_s")
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "status": summary.get("status"),
        "candidate": summary.get("candidate"),
        "model_ref": summary.get("model_ref"),
        "summary_path": str(paths["summary"]),
        "outputs_path": str(paths["outputs"]),
        "source": SOURCE_EXPERIMENT_ID,
        "english_cleanup": summary.get("english_cleanup"),
        "sample_rows": sample_rows,
        "runtime": runtime,
        "tokens": summary.get("tokens"),
        "validity": validity,
        "rates": {
            "max_token_hit_rate": int(validity.get("max_token_hit_rows") or 0) / sample_rows if sample_rows else None,
            "empty_answer_rate": int(validity.get("empty_answer_rows") or 0) / sample_rows if sample_rows else None,
            "thinking_trace_rate": int(validity.get("thinking_trace_rows") or 0) / sample_rows if sample_rows else None,
            "repetition_loop_suspected_rate": int(validity.get("repetition_loop_suspected_rows") or 0) / sample_rows if sample_rows else None,
            "english_cleanup_fire_rate": (summary.get("english_cleanup") or {}).get("applied_rows", 0) / sample_rows if sample_rows else None,
            "projected_total_4000_min": projected / 60 if projected is not None else None,
        },
        "basic_validity_pass": bool(
            summary.get("status") == "completed"
            and validity.get("one_answer_per_input") is True
            and not validity.get("thinking_trace_rows")
            and not validity.get("empty_answer_rows")
        ),
    }


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no cleanup evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C089 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after English cleanup."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if cleanup.get("applied_rows", 0) == 0:
        return "KILL", "The cleanup did not fire on the known English leakage row."
    if cleanup.get("applied_rows", 0) > 5:
        return "KILL", "The cleanup fired too broadly on held-out English rows."
    return "MUTATE", "English/Russian final-answer leakage was cleaned sparsely; hard-audit validation is needed."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C089 English Final-Answer Cleanup Report",
        "",
        "## Objective",
        "- ID: C089",
        "- Mechanism: deterministic cleanup for English prompts with Russian `Ответ:` translation tails.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python scripts/c089_english_final_answer_cleanup.py --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- dry run: `{dry_run}`",
        "- C087 model/prompt/sampling/guards/repetition postprocess changed: `False`.",
        "",
        "## Results",
        "| status | rows | cleanup fires | max-token hits | thinking traces | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {fires} | {cap_hits} | {thinking} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            fires=cleanup.get("applied_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Cleanup Coverage",
        f"- applied rows: `{cleanup.get('applied_row_ids', [])}`",
        f"- by kind: `{cleanup.get('by_kind', {})}`",
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
    run_id = f"{base.utc_stamp()}_c089_dry_run"
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
        "runtime": {"sample_rows": 0},
        "validity": {"jsonl_rows": 0, "one_answer_per_input": None, "thinking_trace_rows": 0, "max_token_hit_rows": 0, "empty_answer_rows": 0, "repetition_loop_suspected_rows": 0},
        "english_cleanup": {"applied_rows": 0, "applied_row_ids": [], "by_kind": {}},
        "paths": {"summary": str(run_paths["summary"]), "metrics": str(run_paths["metrics"]), "outputs": str(run_paths["outputs"]), "log": str(run_paths["log"])},
    }
    base.append_jsonl(run_paths["outputs"], [])
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].parent.mkdir(parents=True, exist_ok=True)
    run_paths["log"].write_text("dry_run=true\nNo GPU experiment was executed.\n", encoding="utf-8")
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def run_source_c087(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c087_{base.utc_stamp()}"
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
    code = c087.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C087 source runner failed with exit {code}")
    return source_out


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_out = run_source_c087(paths["out_dir"], args)
    source_manifest = base.read_json(source_out / "artifact_manifest.json")
    source_run = source_manifest["runs"][0]
    source_summary = base.read_json(Path(source_run["summary_path"]))
    source_outputs = read_jsonl(Path(source_run["outputs_path"]))
    processed_rows, stats = apply_cleanup(source_outputs)
    run_id = str(source_summary.get("run_id") or f"{base.utc_stamp()}_c089")
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = build_summary(source_summary, processed_rows, stats, run_paths)
    summary.setdefault("runtime", {})["c089_postprocess_s"] = time.perf_counter() - started
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
                f"cleanup_applied_rows={stats['applied_rows']}",
                f"cleanup_by_kind={stats['by_kind']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C089 English final-answer cleanup over unchanged C087.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-hf-metadata", action="store_true")
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
    run_record = create_dry_run(paths, args) if args.dry_run else create_gpu_artifacts(paths, args)
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
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "packaged", "dry_run": args.dry_run, "out_dir": str(out_dir), "zip_path": str(zip_path), "report": str(paths["report"])}, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
