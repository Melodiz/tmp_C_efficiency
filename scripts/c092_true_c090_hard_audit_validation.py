from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Sequence

import c086_c084_repetition_list_dedup as c086
import c089_english_final_answer_cleanup as c089
import c090_strict_english_cloze_cleanup as c090
import c072_output_control as base


EXPERIMENT_ID = "C092"
EXPERIMENT_SLUG = "C092_true_c090_hard_audit_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C092_artifacts"


def run_source_c086_hard_audit(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c086_{base.utc_stamp()}"
    if source_out.exists():
        shutil.rmtree(source_out)
    forwarded = [
        "--out",
        str(source_out),
        "--sample-source",
        "hard_audit",
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
    code = c086.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C086 hard-audit source runner failed with exit {code}")
    return source_out


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no true hard-audit validation evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    fires = int(cleanup.get("applied_rows") or 0)
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C092 validation runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed in true hard-audit validation."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if fires > 6:
        return "KILL", "Strict cleanup fired too broadly on hard-audit."
    return "MERGE", "Strict cleanup stayed sparse on true hard-audit; consider porting it into the saved candidate."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C092 True C090 Hard-Audit Validation Report",
        "",
        "## Objective",
        "- ID: C092",
        "- Mechanism: validation-only true hard-audit run for C090 strict English cloze cleanup.",
        "- Leaderboard submission: NO.",
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


def run(argv: Sequence[str] | None = None) -> int:
    c089.EXPERIMENT_ID = EXPERIMENT_ID
    c089.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c089.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c089.SOURCE_EXPERIMENT_ID = "C086"
    c089.cleanup_english_answer = c090.strict_cleanup_english_answer
    c089.run_source_c087 = run_source_c086_hard_audit
    c089.recommendation = recommendation
    c089.write_report = write_report
    return c089.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
