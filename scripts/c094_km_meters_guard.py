from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

import c072_output_control as base
import c075_deterministic_guard as guard_base
import c092_true_c090_hard_audit_validation as c092


EXPERIMENT_ID = "C094"
EXPERIMENT_SLUG = "C094_km_meters_guard"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C094_artifacts"
NUMBER_RE = r"[+-]?\d+(?:[,.]\d+)?"


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
                import json

                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    normalized = text.lower().replace("\u202f", " ").replace("\xa0", " ").replace("−", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def parse_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ".").replace("−", "-"))
    except (InvalidOperation, ValueError):
        return None


def format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f").rstrip("0").rstrip(".").replace(".", ",")


def km_meters_guard(question: str) -> dict[str, str] | None:
    q = normalize_text(question)
    match = re.fullmatch(
        rf"({NUMBER_RE})\s+километр(?:ов|а)?\s+({NUMBER_RE})\s+метр(?:ов|а)?\s+.*сколько\s+метр(?:ов)?",
        q,
    )
    if not match:
        return None
    km = parse_decimal(match.group(1))
    meters = parse_decimal(match.group(2))
    if km is None or meters is None:
        return None
    value = format_decimal(km * Decimal(1000) + meters)
    answer = f"{value} метров\n\nИтоговый ответ: {value} метров"
    return {"kind": "km_m_to_m", "answer": answer, "value": f"{value} метров"}


def apply_guard(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processed = []
    by_kind: Counter[str] = Counter()
    applied_row_ids: list[int] = []
    for row in rows:
        new_row = dict(row)
        result = km_meters_guard(str(row.get("question", "")))
        if result is not None:
            row_id = int(row.get("row_id", row.get("rid", -1)))
            by_kind[result["kind"]] += 1
            applied_row_ids.append(row_id)
            new_row["source_c092_answer"] = row.get("answer")
            new_row["answer"] = result["answer"]
            new_row["output_tokens"] = guard_base.estimate_output_tokens(result["answer"])
            new_row["finish_reason"] = "deterministic_guard"
            new_row["stop_reason"] = result["kind"]
            new_row["hit_max_tokens"] = False
            new_row["repetition_loop_suspected"] = False
            new_row["km_meters_guard"] = {"applied": True, "kind": result["kind"], "value": result["value"]}
        else:
            new_row["km_meters_guard"] = {"applied": False}
            new_row["repetition_loop_suspected"] = guard_base.has_repetition_loop(str(new_row.get("answer", "")))
        processed.append(new_row)
    return processed, {
        "applied_rows": len(applied_row_ids),
        "applied_row_ids": applied_row_ids,
        "by_kind": dict(sorted(by_kind.items())),
        "abstains_unless_explicit_km_m_to_meters": True,
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
            "c094_mechanism": "deterministic_km_m_to_meters_guard",
            "source_experiment_id": "C092",
            "model_prompt_sampling_other_guards_changed_from_c092": False,
        }
    )
    runtime = summary.setdefault("runtime", {})
    runtime["sample_rows"] = len(rows)
    summary["tokens"] = {
        **(summary.get("tokens") or {}),
        "avg_output_tokens_after_c094_guard": sum(output_token_counts) / max(1, len(output_token_counts)),
        "max_output_tokens_after_c094_guard": max(output_token_counts) if output_token_counts else None,
        "total_output_tokens_after_c094_guard": sum(output_token_counts),
    }
    summary["validity"] = {
        "jsonl_rows": len(rows),
        "one_answer_per_input": len(rows) == int((summary.get("sample") or {}).get("rows") or len(rows)),
        "thinking_trace_rows": sum(1 for row in rows if "<think" in str(row.get("answer", "")) or "</think>" in str(row.get("answer", ""))),
        "max_token_hit_rows": sum(1 for row in rows if row.get("hit_max_tokens")),
        "empty_answer_rows": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
        "repetition_loop_suspected_rows": sum(1 for row in rows if row.get("repetition_loop_suspected")),
    }
    summary["km_meters_guard"] = stats
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
        "source": "C092",
        "km_meters_guard": summary.get("km_meters_guard"),
        "sample_rows": sample_rows,
        "runtime": runtime,
        "tokens": summary.get("tokens"),
        "validity": validity,
        "rates": {
            "max_token_hit_rate": int(validity.get("max_token_hit_rows") or 0) / sample_rows if sample_rows else None,
            "empty_answer_rate": int(validity.get("empty_answer_rows") or 0) / sample_rows if sample_rows else None,
            "thinking_trace_rate": int(validity.get("thinking_trace_rows") or 0) / sample_rows if sample_rows else None,
            "repetition_loop_suspected_rate": int(validity.get("repetition_loop_suspected_rows") or 0) / sample_rows if sample_rows else None,
            "guard_fire_rate": (summary.get("km_meters_guard") or {}).get("applied_rows", 0) / sample_rows if sample_rows else None,
            "projected_total_4000_min": projected / 60 if projected is not None else None,
        },
    }


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no guard evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    guard = metrics.get("km_meters_guard") or {}
    projected = rates.get("projected_total_4000_min")
    fires = int(guard.get("applied_rows") or 0)
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C094 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after km/meters guard."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if 4637 not in (guard.get("applied_row_ids") or []):
        return "KILL", "The guard did not fire on the observed km/meters miss row 4637."
    if fires > 4:
        return "KILL", "The km/meters guard fired more broadly than expected."
    return "MUTATE", "The km/meters guard corrected the observed miss sparsely; held-out validation is needed."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    guard = metrics.get("km_meters_guard") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C094 Km/Meters Guard Report",
        "",
        "## Objective",
        "- ID: C094",
        "- Mechanism: narrow deterministic guard for explicit `X километров Y метров ... сколько метров` prompts.",
        "- Leaderboard submission: NO.",
        "",
        "## Results",
        "| status | rows | guard fires | max-token hits | thinking traces | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {fires} | {cap_hits} | {thinking} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            fires=guard.get("applied_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Guard Coverage",
        f"- applied rows: `{guard.get('applied_row_ids', [])}`",
        f"- by kind: `{guard.get('by_kind', {})}`",
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


def run_source_c092(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c092_{base.utc_stamp()}"
    forwarded = [
        "--out",
        str(source_out),
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
    code = c092.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C092 source runner failed with exit {code}")
    return source_out


def create_dry_run(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_c094_dry_run"
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "run_id": run_id,
        "status": "dry_run",
        "runtime": {"sample_rows": 0},
        "validity": {"jsonl_rows": 0, "one_answer_per_input": None, "thinking_trace_rows": 0, "max_token_hit_rows": 0, "empty_answer_rows": 0, "repetition_loop_suspected_rows": 0},
        "km_meters_guard": {"applied_rows": 0, "applied_row_ids": [], "by_kind": {}},
    }
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["summary"], summary)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["outputs"].write_text("", encoding="utf-8")
    run_paths["log"].write_text("dry_run=true\n", encoding="utf-8")
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_out = run_source_c092(paths["out_dir"], args)
    source_manifest = base.read_json(source_out / "artifact_manifest.json")
    source_run = source_manifest["runs"][0]
    source_summary = base.read_json(Path(source_run["summary_path"]))
    source_outputs = read_jsonl(Path(source_run["outputs_path"]))
    processed_rows, stats = apply_guard(source_outputs)
    run_id = str(source_summary.get("run_id") or f"{base.utc_stamp()}_c094")
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = build_summary(source_summary, processed_rows, stats, run_paths)
    summary.setdefault("runtime", {})["c094_postprocess_s"] = time.perf_counter() - started
    base.append_jsonl(run_paths["outputs"], processed_rows)
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"source_out={source_out}",
                f"guard_applied_rows={stats['applied_rows']}",
                f"guard_by_kind={stats['by_kind']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C094 km/meters guard over unchanged C092.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
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
                "status": run_record["summary"].get("status"),
            }
        ],
    }
    base.write_json(out_dir / "artifact_manifest.json", manifest)
    zip_path = base.make_zip(out_dir)
    import json

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
