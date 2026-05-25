from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import c086_c084_repetition_list_dedup as c086


EXPERIMENT_ID = "C087"
EXPERIMENT_SLUG = "C087_c086_locked_val_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C087_artifacts"


def _force_arg(argv: Sequence[str] | None, name: str, value: str) -> list[str]:
    args = list(argv or [])
    forced: list[str] = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item == name:
            skip_next = True
            continue
        if item.startswith(f"{name}="):
            continue
        forced.append(item)
    forced.extend([name, value])
    return forced


def decision_recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no held-out postprocess evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    post_stats = metrics.get("deterministic_postprocess") or {}
    projected = rates.get("projected_total_4000_min")
    fires = int(post_stats.get("applied_rows") or 0)
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C087 validation runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed in held-out validation."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if fires > 2:
        return "KILL", "The repetition postprocess fired too broadly on held-out rows."
    if fires:
        return "MUTATE", "Held-out postprocess fires need row-level review before accepting."
    return "MERGE", "The C086 postprocess abstained on held-out rows while preserving validity and runtime."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation, reason = decision_recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    post_stats = metrics.get("deterministic_postprocess") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C087 C086 Locked-Val Validation Report",
        "",
        "## Objective",
        "- ID: C087",
        "- Mechanism: validation-only sample switch for the C086 comma-list repetition postprocess.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python scripts/c087_c086_locked_val_validation.py --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- dry run: `{dry_run}`",
        "- C086 postprocess changed: `False`.",
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
        "- This validation does not address C086's remaining hard-audit algebra cap row.",
        "- This validation does not prove leaderboard quality or offline package readiness.",
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


def run(argv: Sequence[str] | None = None) -> int:
    c086.EXPERIMENT_ID = EXPERIMENT_ID
    c086.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c086.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c086.SOURCE_EXPERIMENT_ID = "C086"
    c086.decision_recommendation = decision_recommendation
    c086.write_report = write_report
    forwarded = _force_arg(argv, "--sample-source", "locked_val")
    return c086.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
