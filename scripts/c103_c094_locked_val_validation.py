from __future__ import annotations

import argparse
from pathlib import Path

import c090_strict_english_cloze_cleanup as c090
import c094_km_meters_guard as c094


EXPERIMENT_ID = "C103"
EXPERIMENT_SLUG = "C103_c094_locked_val_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C103_artifacts"


def run_source_c090_locked_val(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c090_{c094.base.utc_stamp()}"
    forwarded = [
        "--out",
        str(source_out),
        "--sample-source",
        "locked_val",
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
    code = c090.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C090 locked-val source runner failed with exit {code}")
    return source_out


def recommendation(metrics, dry_run):
    if dry_run:
        return "INVESTIGATE", "Dry run only; no locked-val guard evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    guard = metrics.get("km_meters_guard") or {}
    projected = rates.get("projected_total_4000_min")
    fires = int(guard.get("applied_rows") or 0)
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C103 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after km/meters guard."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if fires > 2:
        return "KILL", "The km/meters guard fired too broadly on held-out rows."
    if fires:
        return "MUTATE", "The km/meters guard fired on held-out rows; row-level review is required."
    return "MERGE", "The km/meters guard abstained on locked-val while preserving validity and runtime."


def write_report(report_path, metrics, args, dry_run):
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    guard = metrics.get("km_meters_guard") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C103 C094 Locked-Val Validation Report",
        "",
        "## Objective",
        "- ID: C103",
        "- Mechanism: validate the unchanged C094 km/meters guard on locked-val.",
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


def run(argv=None):
    c094.EXPERIMENT_ID = EXPERIMENT_ID
    c094.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c094.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c094.run_source_c092 = run_source_c090_locked_val
    c094.recommendation = recommendation
    c094.write_report = write_report
    return c094.run(argv)


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
