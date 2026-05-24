from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "C072"
EXPERIMENT_SLUG = "C072_qwen3_4b_output_control"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C072_artifacts"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_variants(value: str) -> list[int]:
    variants: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("max-token variants must be positive integers")
        variants.append(parsed)
    if not variants:
        raise argparse.ArgumentTypeError("at least one max-token variant is required")
    return variants


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


def prepare_out_dir(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None
    if not out_dir.is_dir():
        raise ValueError(f"--out must be a directory path, got existing file: {out_dir}")
    if not any(out_dir.iterdir()):
        return None

    managed_markers = [
        out_dir / "artifact_manifest.json",
        out_dir / "reports",
        out_dir / "results",
        out_dir / "logs",
    ]
    if not any(path.exists() for path in managed_markers):
        raise ValueError(f"Refusing to reuse non-empty unmanaged artifact directory: {out_dir}")

    archive_root = out_dir.parent / f"{out_dir.name}_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / utc_stamp()
    shutil.move(str(out_dir), str(archive_path))
    return archive_path


def ensure_outputs_path(summary: dict[str, Any], summary_path: Path, results_dir: Path) -> Path:
    outputs_path_raw = (summary.get("paths") or {}).get("outputs")
    if outputs_path_raw:
        outputs_path = Path(outputs_path_raw)
    else:
        outputs_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".outputs.jsonl"))
        summary.setdefault("paths", {})["outputs"] = str(outputs_path)
    if not outputs_path.is_absolute():
        outputs_path = repo_root() / outputs_path
    if results_dir.is_absolute() and results_dir in outputs_path.parents:
        outputs_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        outputs_path = results_dir / outputs_path.name
        summary["paths"]["outputs"] = str(outputs_path)
    if not outputs_path.exists():
        append_jsonl(outputs_path, [])
    return outputs_path


def write_log_header(log_path: Path, command: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"started_utc={utc_stamp()}",
                f"cwd={Path.cwd()}",
                "command=" + " ".join(command),
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_subprocess(command: list[str], log_path: Path) -> int:
    write_log_header(log_path, command)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=repo_root(), stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\nfinished_utc={utc_stamp()}\nreturncode={completed.returncode}\n")
    return completed.returncode


def find_new_summary(results_dir: Path, existing: set[Path]) -> Path | None:
    candidates = [path for path in results_dir.glob("*.summary.json") if path not in existing]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_metrics(summary: dict[str, Any], summary_path: Path, outputs_path: Path | None, max_tokens: int) -> dict[str, Any]:
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
            "mechanism": "max_tokens_cap",
            "max_tokens": max_tokens,
            "prompt_changed": False,
            "system_prompt": False,
            "router_retrieval_cache_handlers_sft_lora": False,
        },
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


def write_report(report_path: Path, run_records: list[dict[str, Any]], args: argparse.Namespace, dry_run: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    first_metrics = next((record.get("metrics") for record in run_records if record.get("metrics")), {})
    environment = first_metrics.get("environment") or {}
    hf_metadata = first_metrics.get("hf_metadata") or {}
    total_file_size = hf_metadata.get("total_file_size_bytes") if isinstance(hf_metadata, dict) else None
    total_file_size_gb = total_file_size / 1_000_000_000 if isinstance(total_file_size, (int, float)) else None
    lines = [
        "# C072 Qwen3-4B Output Control Report",
        "",
        "## Objective",
        "- ID: C072",
        "- Mechanism: output length control via `max_tokens` only.",
        "- Leaderboard submission: NO.",
        "- Production inference behavior changed: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python scripts/c072_output_control.py --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- max-token variants: `{','.join(str(v) for v in args.max_token_variants)}`",
        f"- dry run: `{dry_run}`",
        "- prompt shape: user-message-only chat template inherited from `scripts/c071_probe.py`.",
        "- forbidden methods: no router, retrieval, exact cache, deterministic handlers, SFT, LoRA, or prompt changes.",
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
                "| max_tokens | status | rows | max-token hits | thinking traces | repetition suspects | projected 4000q min | log |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for record in run_records:
            metrics = record.get("metrics") or {}
            validity = metrics.get("validity") or {}
            rates = metrics.get("rates") or {}
            projected = rates.get("projected_total_4000_min")
            projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
            lines.append(
                "| {max_tokens} | {status} | {rows} | {cap_hits} | {thinking} | {repetition} | {projected} | `{log}` |".format(
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
            "## Package feasibility",
            f"- HF file size observed: `{total_file_size_gb:.2f} GB`"
            if total_file_size_gb is not None
            else "- HF file size observed: `unknown in this artifact`",
            "- Offline packaging was not performed by this C072 runner.",
            "",
            "## Artifact layout",
            f"- report: `reports/{EXPERIMENT_SLUG}_report.md`",
            f"- results: `results/{EXPERIMENT_ID}/*.summary.json`, `*.metrics.json`, `*.outputs.jsonl`",
            f"- logs: `logs/{EXPERIMENT_ID}/*.log`",
            "",
            "## Recommendation",
            "- Wrapper recommendation: REVIEW ARTIFACTS. Compare cap-hit rate, repetition suspects, projected runtime, and qualitative JSONL examples before deciding whether C072 should mutate, package, or stop.",
            "",
            "## Strongest reason against recommendation",
            "- The wrapper does not perform qualitative grading; manual review of the returned JSONL outputs is still required.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(out_dir: Path) -> Path:
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    return zip_path


def create_dry_run_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    run_records: list[dict[str, Any]] = []
    for max_tokens in args.max_token_variants:
        run_id = f"{utc_stamp()}_qwen3-4b_dry_run_mt{max_tokens}"
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
                "sample_source": args.sample_source,
                "sample_size_requested": args.sample_size,
                "max_model_len": args.max_model_len,
                "max_tokens": max_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "dtype": args.dtype,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "user_message_only": True,
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
        write_json(summary_path, summary)
        append_jsonl(outputs_path, [])
        metrics = build_metrics(summary, summary_path, outputs_path, max_tokens)
        write_json(metrics_path, metrics)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("dry_run=true\nNo GPU experiment was executed.\n", encoding="utf-8")
        run_records.append(
            {
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
    c071_script = repo_root() / "scripts" / "c071_probe.py"
    for max_tokens in args.max_token_variants:
        stamp = utc_stamp()
        log_path = paths["logs_dir"] / f"{stamp}_qwen3-4b_mt{max_tokens}.log"
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
            "--no-fail",
        ]
        if args.skip_hf_metadata:
            command.append("--skip-hf-metadata")
        if args.trust_remote_code:
            command.append("--trust-remote-code")

        returncode = run_subprocess(command, log_path)
        summary_path = find_new_summary(paths["results_dir"], existing)
        if summary_path is None:
            run_id = f"{stamp}_qwen3-4b_mt{max_tokens}_wrapper_error"
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
                "paths": {"summary": str(summary_path), "outputs": str(outputs_path), "log": str(log_path)},
            }
            write_json(summary_path, summary)
            append_jsonl(outputs_path, [])
        else:
            summary = read_json(summary_path)
            outputs_path = ensure_outputs_path(summary, summary_path, paths["results_dir"])
            write_json(summary_path, summary)

        metrics = build_metrics(summary, summary_path, outputs_path, max_tokens)
        metrics_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".metrics.json"))
        write_json(metrics_path, metrics)
        run_records.append(
            {
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
    parser = argparse.ArgumentParser(description="Run C072 Qwen3-4B output-control experiment and package artifacts.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="hard_audit")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-token-variants", type=parse_variants, default=parse_variants("256,320"))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--dtype", default="bfloat16")
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
    archived_previous = prepare_out_dir(out_dir)
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
        "created_utc": utc_stamp(),
        "dry_run": args.dry_run,
        "out_dir": str(paths["out_dir"]),
        "zip_path": str(paths["zip"]),
        "archived_previous_out_dir": str(archived_previous) if archived_previous else None,
        "runs": run_records,
    }
    write_json(out_dir / "artifact_manifest.json", manifest)
    zip_path = make_zip(out_dir)
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
