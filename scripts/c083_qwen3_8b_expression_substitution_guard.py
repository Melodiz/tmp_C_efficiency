from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import c072_output_control as base
import c075_deterministic_guard as guard_base
import c082_qwen3_8b_language_preserving_prefix as c082


EXPERIMENT_ID = "C083"
EXPERIMENT_SLUG = "C083_qwen3_8b_expression_substitution_guard"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C083_artifacts"
SOURCE_EXPERIMENT_ID = "C082"
RUNNER_SCRIPT = "scripts/c083_qwen3_8b_expression_substitution_guard.py"
NUMBER_RE = r"[+-]?\d+(?:[,.]\d+)?"


class UnsafeExpression(ValueError):
    pass


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


def parse_fraction(raw: str) -> Fraction:
    return Fraction(raw.replace(",", ".").replace("−", "-"))


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def normalize_expression(raw: str) -> str | None:
    expr = raw.strip().strip("$").strip()
    expr = expr.replace("\\cdot", "*").replace("×", "*").replace("−", "-")
    expr = expr.replace("{", "(").replace("}", ")")
    expr = re.sub(r"\s+", "", expr)
    if not expr or len(expr) > 160:
        return None
    if re.search(r"[^0-9A-Za-z+\-*/^().,]", expr):
        return None
    expr = expr.replace(",", ".").replace("^", "**")
    expr = re.sub(r"(\d)([A-Za-z])", r"\1*\2", expr)
    expr = re.sub(r"([A-Za-z]|\d|\))\(", r"\1*(", expr)
    expr = re.sub(r"\)([A-Za-z]|\d)", r")*\1", expr)
    return expr


def eval_ast(node: ast.AST, variables: dict[str, Fraction]) -> Fraction:
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise UnsafeExpression(f"unbound variable {node.id}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = eval_ast(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
    if isinstance(node, ast.BinOp):
        left = eval_ast(node.left, variables)
        right = eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsafeExpression("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 6:
                raise UnsafeExpression("unsafe exponent")
            return left**right.numerator
    raise UnsafeExpression(f"unsupported expression node {type(node).__name__}")


def expression_substitution_guard(question: str) -> dict[str, Any] | None:
    text = " ".join(question.replace("\u202f", " ").replace("\xa0", " ").split())
    if not re.search(r"найд[иите]\s+значение\s+выражения", text, flags=re.IGNORECASE):
        return None
    match = re.search(r"значение\s+выражения\s+(.+?)\s+при\s+(.+)", text, flags=re.IGNORECASE)
    if not match:
        return None

    expr_raw = match.group(1)
    assign_text = match.group(2)
    dollar = re.search(r"\$(.+?)\$", expr_raw)
    if dollar:
        expr_raw = dollar.group(1)

    assignments = {
        name: parse_fraction(value)
        for name, value in re.findall(r"\b([A-Za-z])\s*=\s*(" + NUMBER_RE + r")", assign_text)
    }
    if not assignments:
        return None

    expr = normalize_expression(expr_raw)
    if expr is None:
        return None
    used_names = set(re.findall(r"[A-Za-z]", expr))
    if not used_names or not used_names.issubset(assignments):
        return None

    try:
        tree = ast.parse(expr, mode="eval")
        value = eval_ast(tree, assignments)
    except (SyntaxError, UnsafeExpression, ValueError, ZeroDivisionError):
        return None

    answer = format_fraction(value)
    return {
        "kind": "expression_substitution",
        "answer": f"{answer}\n\nИтоговый ответ: {answer}",
        "value": answer,
        "expression": expr,
        "assignments": {key: format_fraction(val) for key, val in assignments.items()},
    }


def apply_guard(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    guarded_rows: list[dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    applied_row_ids: list[int] = []
    corrected_known: dict[str, str] = {}

    for row in rows:
        new_row = dict(row)
        row_id = int(row.get("row_id", -1))
        result = expression_substitution_guard(str(row.get("question", "")))
        if result is not None:
            by_kind[str(result["kind"])] += 1
            applied_row_ids.append(row_id)
            if row_id == 8295:
                corrected_known[str(row_id)] = str(result["value"])
            new_row["source_c082_answer"] = row.get("answer")
            new_row["answer"] = result["answer"]
            new_row["output_tokens"] = guard_base.estimate_output_tokens(result["answer"])
            new_row["finish_reason"] = "deterministic_guard"
            new_row["stop_reason"] = str(result["kind"])
            new_row["hit_max_tokens"] = False
            new_row["repetition_loop_suspected"] = False
            new_row["deterministic_guard"] = {
                "applied": True,
                "kind": result["kind"],
                "value": result["value"],
                "expression": result["expression"],
                "assignments": result["assignments"],
            }
        else:
            new_row["deterministic_guard"] = {"applied": False}
            new_row["repetition_loop_suspected"] = guard_base.has_repetition_loop(str(new_row.get("answer", "")))
        guarded_rows.append(new_row)

    return guarded_rows, {
        "applied_rows": len(applied_row_ids),
        "applied_row_ids": applied_row_ids,
        "by_kind": dict(sorted(by_kind.items())),
        "target_known_miss": {"8295": "101"},
        "corrected_known_misses": corrected_known,
        "abstains_unless_explicit_expression_assignments": True,
    }


def build_summary(
    source_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    guard_stats: dict[str, Any],
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
    summary["source_c082_manifest"] = source_manifest
    config = summary.setdefault("config", {})
    config.update(
        {
            "c083_mechanism": "deterministic_expression_substitution_guard",
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "model_backend_prompt_sampling_changed_from_c082": False,
            "router_retrieval_cache_sft_lora": False,
        }
    )
    runtime = summary.setdefault("runtime", {})
    runtime["sample_rows"] = len(rows)
    summary["tokens"] = {
        **(summary.get("tokens") or {}),
        "avg_output_tokens_after_guard": sum(output_token_counts) / max(1, len(output_token_counts)),
        "max_output_tokens_after_guard": max(output_token_counts) if output_token_counts else None,
        "total_output_tokens_after_guard": sum(output_token_counts),
    }
    summary["validity"] = {
        "jsonl_rows": len(rows),
        "one_answer_per_input": len(rows) == int((summary.get("sample") or {}).get("rows") or len(rows)),
        "thinking_trace_rows": sum(1 for row in rows if "<think" in str(row.get("answer", "")) or "</think>" in str(row.get("answer", ""))),
        "max_token_hit_rows": sum(1 for row in rows if row.get("hit_max_tokens")),
        "empty_answer_rows": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
        "repetition_loop_suspected_rows": sum(1 for row in rows if row.get("repetition_loop_suspected")),
    }
    summary["deterministic_guard"] = guard_stats
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
    guard_stats = summary.get("deterministic_guard") or {}
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "status": summary.get("status"),
        "candidate": summary.get("candidate"),
        "model_ref": summary.get("model_ref"),
        "summary_path": str(paths["summary"]),
        "outputs_path": str(paths["outputs"]),
        "source": SOURCE_EXPERIMENT_ID,
        "deterministic_guard": guard_stats,
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
            "guard_fire_rate": guard_stats.get("applied_rows", 0) / sample_rows if sample_rows else None,
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
        return "INVESTIGATE", "Dry run only; no model or guard evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    guard_stats = metrics.get("deterministic_guard") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", f"The {EXPERIMENT_ID} runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after expression-substitution guard application."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if guard_stats.get("corrected_known_misses", {}).get("8295") != "101":
        return "MUTATE", "The guard did not correct the observed C082 expression-substitution regression."
    return "MUTATE", "The target exact-expression miss is corrected; English leakage and broader reasoning risks remain."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation, reason = decision_recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    guard_stats = metrics.get("deterministic_guard") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        f"# {EXPERIMENT_ID} Qwen3-8B-AWQ Expression Substitution Guard Report",
        "",
        "## Objective",
        f"- ID: {EXPERIMENT_ID}",
        "- Mechanism: high-confidence deterministic expression-substitution guard on top of unchanged C082.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python {RUNNER_SCRIPT} --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- dry run: `{dry_run}`",
        "- C082 model/backend/prefix/sampling changed: `False`.",
        "- forbidden methods: no retrieval/RAG, cache, SFT, LoRA, system prompt, sampling change, or broad solver.",
        "",
        "## Results",
        "| status | rows | guard fires | max-token hits | thinking traces | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {fires} | {cap_hits} | {thinking} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            fires=guard_stats.get("applied_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Guard Coverage",
        f"- applied rows: `{guard_stats.get('applied_row_ids', [])}`",
        f"- by kind: `{guard_stats.get('by_kind', {})}`",
        f"- corrected known C082 miss: `{guard_stats.get('corrected_known_misses', {})}`",
        "",
        "## Remaining Known Risk",
        f"- {EXPERIMENT_ID} does not address English final-answer language leakage such as row 4242.",
        f"- {EXPERIMENT_ID} does not address open-ended reasoning, geometry, chemistry, or essay quality.",
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
    run_id = f"{base.utc_stamp()}_c083_dry_run"
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
            "c083_mechanism": "deterministic_expression_substitution_guard",
            "model_backend_prompt_sampling_changed_from_c082": False,
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
        "deterministic_guard": {
            "applied_rows": 0,
            "applied_row_ids": [],
            "by_kind": {},
            "target_known_miss": {"8295": "101"},
            "corrected_known_misses": {},
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


def run_source_c082(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c082_{base.utc_stamp()}"
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
    code = c082.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C082 source runner failed with exit {code}")
    return source_out


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_out = run_source_c082(paths["out_dir"], args)
    source_manifest_path = source_out / "artifact_manifest.json"
    source_manifest = base.read_json(source_manifest_path)
    source_run = source_manifest["runs"][0]
    source_summary = base.read_json(Path(source_run["summary_path"]))
    source_outputs = read_jsonl(Path(source_run["outputs_path"]))
    guarded_rows, guard_stats = apply_guard(source_outputs)

    run_id = str(source_summary.get("run_id") or f"{base.utc_stamp()}_c083")
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = build_summary(source_summary, guarded_rows, guard_stats, run_paths, source_manifest)
    summary.setdefault("runtime", {})["c083_postprocess_s"] = time.perf_counter() - started
    base.append_jsonl(run_paths["outputs"], guarded_rows)
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
                f"guard_applied_rows={guard_stats['applied_rows']}",
                f"guard_by_kind={guard_stats['by_kind']}",
                f"corrected_known_misses={guard_stats['corrected_known_misses']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C083 expression-substitution guard over unchanged C082.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
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
