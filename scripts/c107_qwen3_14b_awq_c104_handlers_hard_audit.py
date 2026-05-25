from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import c072_output_control as base
import c083_qwen3_8b_expression_substitution_guard as c083
import c086_c084_repetition_list_dedup as c086
import c089_english_final_answer_cleanup as c089
import c090_strict_english_cloze_cleanup as c090
import c094_km_meters_guard as c094
import c106_qwen3_14b_awq_feasibility as c106


EXPERIMENT_ID = "C107"
EXPERIMENT_SLUG = "C107_qwen3_14b_awq_c104_handlers_hard_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C107_artifacts"


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


def run_source_c106(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c106_{base.utc_stamp()}"
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
    code = c106.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C106 source runner failed with exit {code}")
    return source_out


def apply_c104_handlers(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expr_rows, expr_stats = c083.apply_guard(rows)
    dedup_rows, dedup_stats = c086.apply_postprocess(expr_rows)
    c089.cleanup_english_answer = c090.strict_cleanup_english_answer
    english_rows, english_stats = c089.apply_cleanup(dedup_rows)
    final_rows, km_stats = c094.apply_guard(english_rows)
    return final_rows, {
        "expression_guard": expr_stats,
        "comma_repetition_dedup": dedup_stats,
        "strict_english_cleanup": english_stats,
        "km_meters_guard": km_stats,
    }


def build_summary(source_summary: dict[str, Any], rows: list[dict[str, Any]], handler_stats: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
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
            "c107_mechanism": "qwen3_14b_awq_with_existing_c104_handlers",
            "source_experiment_id": "C106",
            "new_handlers_added": False,
            "handlers_match_c104_public_best": True,
            "sample_source": "hard_audit",
        }
    )
    runtime = summary.setdefault("runtime", {})
    runtime["sample_rows"] = len(rows)
    summary["tokens"] = {
        **(summary.get("tokens") or {}),
        "avg_output_tokens_after_c104_handlers": sum(output_token_counts) / max(1, len(output_token_counts)),
        "max_output_tokens_after_c104_handlers": max(output_token_counts) if output_token_counts else None,
        "total_output_tokens_after_c104_handlers": sum(output_token_counts),
    }
    summary["validity"] = {
        "jsonl_rows": len(rows),
        "one_answer_per_input": len(rows) == int((summary.get("sample") or {}).get("rows") or len(rows)),
        "thinking_trace_rows": sum(1 for row in rows if "<think" in str(row.get("answer", "")) or "</think>" in str(row.get("answer", ""))),
        "max_token_hit_rows": sum(1 for row in rows if row.get("hit_max_tokens")),
        "empty_answer_rows": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
        "repetition_loop_suspected_rows": sum(1 for row in rows if row.get("repetition_loop_suspected")),
    }
    summary["handler_stats"] = handler_stats
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
        "sample_rows": sample_rows,
        "summary_path": str(paths["summary"]),
        "outputs_path": str(paths["outputs"]),
        "runtime": runtime,
        "tokens": summary.get("tokens"),
        "validity": validity,
        "handler_stats": summary.get("handler_stats"),
        "rates": {
            "max_token_hit_rate": int(validity.get("max_token_hit_rows") or 0) / sample_rows if sample_rows else None,
            "empty_answer_rate": int(validity.get("empty_answer_rows") or 0) / sample_rows if sample_rows else None,
            "thinking_trace_rate": int(validity.get("thinking_trace_rows") or 0) / sample_rows if sample_rows else None,
            "repetition_loop_suspected_rate": int(validity.get("repetition_loop_suspected_rows") or 0) / sample_rows if sample_rows else None,
            "projected_total_4000_min": projected / 60 if projected is not None else None,
        },
    }


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no hard-audit evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C107 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after applying the existing C104 handlers."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if int(validity.get("max_token_hit_rows") or 0) > 3:
        return "KILL", "Hard-audit truncation risk is too high versus C104/C092."
    if int(validity.get("repetition_loop_suspected_rows") or 0) > 2:
        return "KILL", "Repetition risk remains too high."
    return "MUTATE", "14B plus unchanged C104 handlers passed hard-audit validity gates; row-level review and package-size smoke are needed."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    handlers = metrics.get("handler_stats") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C107 Qwen3-14B-AWQ With C104 Handlers Hard-Audit Report",
        "",
        "## Objective",
        "- ID: C107",
        "- Mechanism: validate Qwen3-14B-AWQ on hard-audit with only the existing C104 public-best handler stack.",
        "- Leaderboard submission: NO.",
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
        "## Handler Coverage",
        f"- expression guard: `{(handlers.get('expression_guard') or {}).get('applied_row_ids', [])}`",
        f"- comma dedup: `{(handlers.get('comma_repetition_dedup') or {}).get('applied_row_ids', [])}`",
        f"- strict English cleanup: `{(handlers.get('strict_english_cleanup') or {}).get('applied_row_ids', [])}`",
        f"- km/meters guard: `{(handlers.get('km_meters_guard') or {}).get('applied_row_ids', [])}`",
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


def create_dry_run(paths: dict[str, Path]) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_c107_dry_run"
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
        "runtime": {"sample_rows": 0},
        "validity": {"jsonl_rows": 0, "one_answer_per_input": None, "thinking_trace_rows": 0, "max_token_hit_rows": 0, "empty_answer_rows": 0, "repetition_loop_suspected_rows": 0},
        "handler_stats": {},
    }
    base.append_jsonl(run_paths["outputs"], [])
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].write_text("dry_run=true\n", encoding="utf-8")
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_out = run_source_c106(paths["out_dir"], args)
    source_manifest = base.read_json(source_out / "artifact_manifest.json")
    source_run = source_manifest["runs"][0]
    source_summary = base.read_json(Path(source_run["summary_path"]))
    source_outputs = read_jsonl(Path(source_run["outputs_path"]))
    processed_rows, handler_stats = apply_c104_handlers(source_outputs)
    run_id = str(source_summary.get("run_id") or f"{base.utc_stamp()}_c107")
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = build_summary(source_summary, processed_rows, handler_stats, run_paths)
    summary.setdefault("runtime", {})["c107_wrapper_s"] = time.perf_counter() - started
    base.append_jsonl(run_paths["outputs"], processed_rows)
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"source_out={source_out}",
                f"handler_stats={json.dumps(handler_stats, ensure_ascii=False)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C107 14B-AWQ plus unchanged C104 handlers on hard-audit.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="hard_audit")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create artifact layout without a model run.")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    archived_previous = base.prepare_out_dir(out_dir)
    paths = artifact_paths(out_dir)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    run_record = create_dry_run(paths) if args.dry_run else create_gpu_artifacts(paths, args)
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
                "status": run_record["summary"].get("status"),
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
