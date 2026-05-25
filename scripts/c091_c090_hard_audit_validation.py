from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import c090_strict_english_cloze_cleanup as c090


EXPERIMENT_ID = "C091"
EXPERIMENT_SLUG = "C091_c090_hard_audit_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C091_artifacts"


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


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no hard-audit validation evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    fires = int(cleanup.get("applied_rows") or 0)
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C091 validation runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed in hard-audit validation."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if fires > 6:
        return "KILL", "Strict cleanup fired too broadly on hard-audit."
    return "MERGE", "Strict cleanup stayed sparse on hard-audit; consider porting it into the saved candidate."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C091 C090 Hard-Audit Validation Report",
        "",
        "## Objective",
        "- ID: C091",
        "- Mechanism: validation-only sample switch for C090 strict English cloze cleanup.",
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
    c090.c089.EXPERIMENT_ID = EXPERIMENT_ID
    c090.c089.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c090.c089.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    c090.c089.SOURCE_EXPERIMENT_ID = "C090"
    c090.c089.cleanup_english_answer = c090.strict_cleanup_english_answer
    c090.c089.recommendation = recommendation
    c090.c089.write_report = write_report
    forwarded = _force_arg(argv, "--sample-source", "hard_audit")
    return c090.c089.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
