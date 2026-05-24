from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import c072_output_control as base


EXPERIMENT_ID = "C073"
EXPERIMENT_SLUG = "C073_qwen3_4b_short_prefix_output_control"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C073_artifacts"
SHORT_USER_PREFIX = "Ответь кратко и точно. Не повторяй условие. В конце дай итоговый ответ."
VARIANT_MAX_TOKENS = {
    "short_prefix_320": 320,
    "short_prefix_384": 384,
}
BASELINE_COMPARISONS = {
    "C071_raw_384": {
        "max_tokens": 384,
        "sample_rows": 200,
        "max_token_hit_rows": 112,
        "repetition_loop_suspected_rows": 37,
        "projected_total_4000_min": 10.53,
    },
    "C072_cap_only_320": {
        "max_tokens": 320,
        "sample_rows": 200,
        "max_token_hit_rows": 125,
        "repetition_loop_suspected_rows": 27,
        "projected_total_4000_min": 8.58,
    },
}


def parse_variants(value: str) -> list[tuple[str, int]]:
    variants: list[tuple[str, int]] = []
    for raw in value.split(","):
        name = raw.strip()
        if not name:
            continue
        if name.isdigit():
            name = f"short_prefix_{name}"
        if name not in VARIANT_MAX_TOKENS:
            allowed = ", ".join(VARIANT_MAX_TOKENS)
            raise argparse.ArgumentTypeError(f"unknown variant {raw!r}; expected one of: {allowed}")
        variants.append((name, VARIANT_MAX_TOKENS[name]))
    if not variants:
        raise argparse.ArgumentTypeError("at least one C073 variant is required")
    return variants


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


def completed_metric_value(metrics: dict[str, Any], key_path: Sequence[str]) -> Any:
    value: Any = metrics
    for key in key_path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def build_metrics(
    summary: dict[str, Any],
    summary_path: Path,
    outputs_path: Path | None,
    variant_name: str,
    max_tokens: int,
) -> dict[str, Any]:
    validity = summary.get("validity") or {}
    runtime = summary.get("runtime") or {}
    tokens = summary.get("tokens") or {}
    sample_rows = int(runtime.get("sample_rows") or validity.get("jsonl_rows") or summary.get("sample", {}).get("rows") or 0)
    max_token_hit_rows = int(validity.get("max_token_hit_rows") or 0)
    empty_answer_rows = int(validity.get("empty_answer_rows") or 0)
    thinking_trace_rows = int(validity.get("thinking_trace_rows") or 0)
    repetition_rows = int(validity.get("repetition_loop_suspected_rows") or 0)

    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "candidate": summary.get("candidate"),
        "model_ref": summary.get("model_ref"),
        "summary_path": str(summary_path),
        "outputs_path": str(outputs_path) if outputs_path else None,
        "output_control": {
            "mechanism": "short_user_prefix",
            "variant": variant_name,
            "max_tokens": max_tokens,
            "prompt_changed": True,
            "user_prefix": SHORT_USER_PREFIX,
            "system_prompt": False,
            "router_retrieval_cache_handlers_sft_lora": False,
        },
        "comparison_baselines": BASELINE_COMPARISONS,
        "sample_rows": sample_rows,
        "runtime": runtime,
        "tokens": tokens,
        "validity": validity,
        "environment": summary.get("environment"),
        "hf_metadata": summary.get("hf_metadata"),
        "rates": {
            "max_token_hit_rate": max_token_hit_rows / sample_rows if sample_rows else None,
            "empty_answer_rate": empty_answer_rows / sample_rows if sample_rows else None,
            "thinking_trace_rate": thinking_trace_rows / sample_rows if sample_rows else None,
            "repetition_loop_suspected_rate": repetition_rows / sample_rows if sample_rows else None,
            "projected_total_4000_min": runtime.get("projected_total_4000_s") / 60
            if runtime.get("projected_total_4000_s") is not None
            else None,
        },
        "basic_validity_pass": bool(
            summary.get("status") == "completed"
            and validity.get("one_answer_per_input") is True
            and thinking_trace_rows == 0
            and empty_answer_rows == 0
        ),
    }


def write_c073_summary_fields(summary: dict[str, Any], variant_name: str, max_tokens: int) -> dict[str, Any]:
    source_experiment_id = summary.get("experiment_id")
    if source_experiment_id and source_experiment_id != EXPERIMENT_ID:
        summary["source_experiment_id"] = source_experiment_id
    summary["experiment_id"] = EXPERIMENT_ID
    summary["experiment_slug"] = EXPERIMENT_SLUG
    config = summary.setdefault("config", {})
    config.update(
        {
            "c073_variant": variant_name,
            "short_user_prefix": SHORT_USER_PREFIX,
            "max_tokens": max_tokens,
            "prompt_changed": True,
            "system_prompt": False,
            "router_retrieval_cache_handlers_sft_lora": False,
        }
    )
    return summary


def truncate_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def read_examples(outputs_path: str | None, limit: int = 5) -> list[dict[str, Any]]:
    if not outputs_path:
        return []
    path = Path(outputs_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return []

    issue_rows = [
        row
        for row in rows
        if row.get("hit_max_tokens") or row.get("repetition_loop_suspected") or row.get("has_thinking_trace")
    ]
    selected: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in issue_rows[: max(1, limit // 2)] + rows:
        key = row.get("row_id", row.get("sample_index"))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def decision_recommendation(run_records: list[dict[str, Any]], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no model quality or runtime evidence was produced."
    completed = [record for record in run_records if (record.get("metrics") or {}).get("status") == "completed"]
    if not completed:
        return "INVESTIGATE", "No completed C073 model run was recorded."

    primary = next((record for record in completed if record.get("variant") == "short_prefix_320"), completed[0])
    metrics = primary.get("metrics") or {}
    rates = metrics.get("rates") or {}
    validity = metrics.get("validity") or {}
    projected_min = rates.get("projected_total_4000_min")
    cap_rate = rates.get("max_token_hit_rate")
    cap_hits = validity.get("max_token_hit_rows")
    repetition = validity.get("repetition_loop_suspected_rows") or 0
    thinking = validity.get("thinking_trace_rows") or 0
    empty = validity.get("empty_answer_rows") or 0

    if thinking or empty:
        return "KILL", "The primary run produced thinking traces or empty answers."
    if isinstance(projected_min, (int, float)) and projected_min >= 12:
        return "KILL", "The primary run misses the 12 minute projected runtime gate."
    if isinstance(cap_rate, (int, float)) and cap_rate < 0.25:
        return "MUTATE", "The primary run passes the cap-hit gate; packaging and qualitative regression review remain."
    if isinstance(cap_hits, int) and cap_hits < 100 and repetition < 27:
        return "MUTATE", "The primary run is still above the cap-hit gate but improves sharply enough to inspect one narrow follow-up."
    return "KILL", "The primary run does not reduce truncation enough versus C071/C072."


def write_report(report_path: Path, run_records: list[dict[str, Any]], args: argparse.Namespace, dry_run: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    first_metrics = next((record.get("metrics") for record in run_records if record.get("metrics")), {})
    environment = first_metrics.get("environment") or {}
    hf_metadata = first_metrics.get("hf_metadata") or {}
    total_file_size = hf_metadata.get("total_file_size_bytes") if isinstance(hf_metadata, dict) else None
    total_file_size_gb = total_file_size / 1_000_000_000 if isinstance(total_file_size, (int, float)) else None
    recommendation, reason = decision_recommendation(run_records, dry_run)

    lines = [
        "# C073 Qwen3-4B Short-Prefix Output Control Report",
        "",
        "## Objective",
        "- ID: C073",
        "- Mechanism: short user-prefix output control.",
        "- Leaderboard submission: NO.",
        "- Production inference behavior changed: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python scripts/c073_short_prefix_output_control.py --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- variants: `{','.join(name for name, _ in args.variants)}`",
        f"- dry run: `{dry_run}`",
        f"- short user prefix: `{SHORT_USER_PREFIX}`",
        "- prompt shape: user-message-only chat template inherited from `scripts/c071_probe.py`.",
        "- forbidden methods: no system prompt, router, retrieval, exact cache, deterministic handlers, SFT, or LoRA.",
        "",
        "## Environment",
    ]
    if environment:
        lines.extend(
            [
                f"- python: `{environment.get('python', 'unknown')}`",
                f"- platform: `{environment.get('platform', 'unknown')}`",
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
        ]
    )
    if not run_records:
        lines.append("- No run records were produced.")
    else:
        lines.extend(
            [
                "| variant | max_tokens | status | rows | max-token hits | thinking traces | repetition suspects | projected 4000q min | log |",
                "|---|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for record in run_records:
            metrics = record.get("metrics") or {}
            validity = metrics.get("validity") or {}
            rates = metrics.get("rates") or {}
            projected = rates.get("projected_total_4000_min")
            projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
            lines.append(
                "| {variant} | {max_tokens} | {status} | {rows} | {cap_hits} | {thinking} | {repetition} | {projected} | `{log}` |".format(
                    variant=record.get("variant"),
                    max_tokens=record.get("max_tokens"),
                    status=metrics.get("status", record.get("status")),
                    rows=metrics.get("sample_rows", 0),
                    cap_hits=validity.get("max_token_hit_rows", "n/a"),
                    thinking=validity.get("thinking_trace_rows", "n/a"),
                    repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
                    projected=projected_text,
                    log=record.get("log_path"),
                )
            )

    lines.extend(
        [
            "",
            "## Comparison Anchors",
            "- C071 raw 384: 112/200 max-token hits, 37/200 repetition suspects, projected 10.53 min.",
            "- C072 cap-only 320: 125/200 max-token hits, 27/200 repetition suspects, projected 8.58 min.",
            "- C073 success gate: under 12 min and below 25% max-token hits, or sharply lower truncation with strong qualitative evidence.",
            "",
            "## Qualitative Examples",
        ]
    )
    example_count = 0
    for record in run_records:
        examples = read_examples(record.get("outputs_path"), limit=5)
        if not examples:
            continue
        lines.append(f"### {record.get('variant')}")
        for row in examples:
            flags = []
            if row.get("hit_max_tokens"):
                flags.append("hit_max_tokens")
            if row.get("repetition_loop_suspected"):
                flags.append("repetition_suspect")
            if row.get("has_thinking_trace"):
                flags.append("thinking_trace")
            flag_text = ", ".join(flags) if flags else "no automatic flags"
            lines.extend(
                [
                    f"- row_id `{row.get('row_id')}` category `{row.get('category')}` flags `{flag_text}`",
                    f"  - question: {truncate_text(row.get('question'))}",
                    f"  - answer: {truncate_text(row.get('answer'))}",
                ]
            )
            example_count += 1
    if example_count == 0:
        lines.append("- No qualitative examples are available in this artifact.")

    lines.extend(
        [
            "",
            "## Package Feasibility",
            f"- HF file size observed: `{total_file_size_gb:.2f} GB`"
            if total_file_size_gb is not None
            else "- HF file size observed: `unknown in this artifact`",
            "- Offline packaging was not performed by this C073 runner.",
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


def create_dry_run_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    run_records: list[dict[str, Any]] = []
    for variant_name, max_tokens in args.variants:
        run_id = f"{base.utc_stamp()}_qwen3-4b_dry_run_{variant_name}"
        summary_path = paths["results_dir"] / f"{run_id}.summary.json"
        metrics_path = paths["results_dir"] / f"{run_id}.metrics.json"
        outputs_path = paths["results_dir"] / f"{run_id}.outputs.jsonl"
        log_path = paths["logs_dir"] / f"{run_id}.log"
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "experiment_slug": EXPERIMENT_SLUG,
            "run_id": run_id,
            "status": "dry_run",
            "candidate": "qwen3-4b",
            "model_ref": "Qwen/Qwen3-4B-Instruct-2507",
            "config": {
                "c073_variant": variant_name,
                "sample_source": args.sample_source,
                "sample_size_requested": args.sample_size,
                "max_model_len": args.max_model_len,
                "max_tokens": max_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "dtype": args.dtype,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "user_message_only": True,
                "short_user_prefix": SHORT_USER_PREFIX,
                "prompt_changed": True,
                "system_prompt": False,
                "router_retrieval_cache_handlers_sft_lora": False,
            },
            "paths": {
                "summary": str(summary_path),
                "metrics": str(metrics_path),
                "outputs": str(outputs_path),
                "log": str(log_path),
            },
            "validity": {
                "jsonl_rows": 0,
                "one_answer_per_input": None,
                "thinking_trace_rows": 0,
                "max_token_hit_rows": 0,
                "empty_answer_rows": 0,
                "repetition_loop_suspected_rows": 0,
            },
            "runtime": {"sample_rows": 0},
        }
        base.write_json(summary_path, summary)
        base.append_jsonl(outputs_path, [])
        metrics = build_metrics(summary, summary_path, outputs_path, variant_name, max_tokens)
        base.write_json(metrics_path, metrics)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("dry_run=true\nNo GPU experiment was executed.\n", encoding="utf-8")
        run_records.append(
            {
                "variant": variant_name,
                "max_tokens": max_tokens,
                "status": "dry_run",
                "summary_path": str(summary_path),
                "metrics_path": str(metrics_path),
                "outputs_path": str(outputs_path),
                "log_path": str(log_path),
                "metrics": metrics,
            }
        )
        time.sleep(1)
    return run_records


def run_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    run_records: list[dict[str, Any]] = []
    c071_script = base.repo_root() / "scripts" / "c071_probe.py"
    for variant_name, max_tokens in args.variants:
        stamp = base.utc_stamp()
        log_path = paths["logs_dir"] / f"{stamp}_qwen3-4b_{variant_name}.log"
        existing = set(paths["results_dir"].glob("*.summary.json"))
        command = [
            sys.executable,
            str(c071_script),
            "--candidate",
            "qwen3-4b",
            "--sample-source",
            args.sample_source,
            "--sample-size",
            str(args.sample_size),
            "--output-dir",
            str(paths["results_dir"]),
            "--max-model-len",
            str(args.max_model_len),
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--dtype",
            args.dtype,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--seed",
            str(args.seed),
            "--user-prefix",
            SHORT_USER_PREFIX,
            "--no-fail",
        ]
        if args.skip_hf_metadata:
            command.append("--skip-hf-metadata")
        if args.trust_remote_code:
            command.append("--trust-remote-code")

        returncode = base.run_subprocess(command, log_path)
        summary_path = base.find_new_summary(paths["results_dir"], existing)
        if summary_path is None:
            run_id = f"{stamp}_qwen3-4b_{variant_name}_wrapper_error"
            summary_path = paths["results_dir"] / f"{run_id}.summary.json"
            outputs_path = paths["results_dir"] / f"{run_id}.outputs.jsonl"
            summary = {
                "experiment_id": EXPERIMENT_ID,
                "experiment_slug": EXPERIMENT_SLUG,
                "run_id": run_id,
                "status": "wrapper_error",
                "candidate": "qwen3-4b",
                "model_ref": "Qwen/Qwen3-4B-Instruct-2507",
                "returncode": returncode,
                "config": {
                    "c073_variant": variant_name,
                    "short_user_prefix": SHORT_USER_PREFIX,
                    "max_tokens": max_tokens,
                    "prompt_changed": True,
                    "system_prompt": False,
                    "router_retrieval_cache_handlers_sft_lora": False,
                },
                "paths": {"summary": str(summary_path), "outputs": str(outputs_path), "log": str(log_path)},
            }
            base.write_json(summary_path, summary)
            base.append_jsonl(outputs_path, [])
        else:
            summary = base.read_json(summary_path)
            outputs_path = base.ensure_outputs_path(summary, summary_path, paths["results_dir"])
            write_c073_summary_fields(summary, variant_name, max_tokens)
            base.write_json(summary_path, summary)

        metrics = build_metrics(summary, summary_path, outputs_path, variant_name, max_tokens)
        metrics_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".metrics.json"))
        base.write_json(metrics_path, metrics)
        run_records.append(
            {
                "variant": variant_name,
                "max_tokens": max_tokens,
                "status": summary.get("status"),
                "returncode": returncode,
                "summary_path": str(summary_path),
                "metrics_path": str(metrics_path),
                "outputs_path": str(outputs_path) if outputs_path else None,
                "log_path": str(log_path),
                "metrics": metrics,
            }
        )
    return run_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C073 Qwen3-4B short-prefix output-control experiment.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="hard_audit")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--variants", type=parse_variants, default=parse_variants("short_prefix_320"))
    parser.add_argument("--variant", default=None, help="Optional single variant alias, for example short_prefix_384.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create the artifact layout without loading a model.")
    args = parser.parse_args(argv)
    if args.variant:
        args.variants = parse_variants(args.variant)
    return args


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
        run_records = create_dry_run_artifacts(paths, args)
    else:
        run_records = run_gpu_artifacts(paths, args)

    write_report(paths["report"], run_records, args, dry_run=args.dry_run)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "created_utc": base.utc_stamp(),
        "dry_run": args.dry_run,
        "out_dir": str(paths["out_dir"]),
        "zip_path": str(paths["zip"]),
        "archived_previous_out_dir": str(archived_previous) if archived_previous else None,
        "runs": run_records,
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
