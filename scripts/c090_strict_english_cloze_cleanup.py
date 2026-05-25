from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

import c089_english_final_answer_cleanup as c089


EXPERIMENT_ID = "C090"
EXPERIMENT_SLUG = "C090_strict_english_cloze_cleanup"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C090_artifacts"


def is_cloze_prompt(question: str) -> bool:
    text = " ".join(question.split())
    if "____" in text or "＿" in text:
        return True
    if re.search(r"\b[A-Z]{2,}\b\s*(?:[.!?])?$", text):
        return True
    if re.search(r"\bchoose\b", text, flags=re.IGNORECASE) and len(text.split()) <= 18:
        return True
    return False


def strict_cleanup_english_answer(question: str, answer: str) -> str | None:
    if not c089.is_english_prompt(question) or not is_cloze_prompt(question):
        return None
    if not re.search(r"[А-Яа-яЁё]", answer) or not re.search(r"ответ\s*:", answer, flags=re.IGNORECASE):
        return None

    before_marker = re.split(r"\*{0,2}\s*Ответ\s*:\s*\*{0,2}", answer, maxsplit=1, flags=re.IGNORECASE)[0]
    first_lines = [line.strip(" *") for line in before_marker.splitlines() if line.strip(" *")]
    if len(first_lines) != 1:
        return None
    first = first_lines[0].strip()
    if not re.search(r"[A-Za-z]", first) or re.search(r"[А-Яа-яЁё]", first):
        return None
    if len(first.split()) > 5:
        return None
    return first


def recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no cleanup evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    fires = int(cleanup.get("applied_rows") or 0)
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C090 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after strict English cleanup."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if fires == 0:
        return "KILL", "The strict cleanup did not fire on the known English leakage row."
    if fires > 6:
        return "KILL", "The strict cleanup still fired too broadly on held-out rows."
    return "MUTATE", "Strict English cleanup fired sparsely; row-level review and hard-audit validation are needed."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    cleanup = metrics.get("english_cleanup") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C090 Strict English Cloze Cleanup Report",
        "",
        "## Objective",
        "- ID: C090",
        "- Mechanism: stricter deterministic cleanup for cloze/blank-style English prompts with Russian `Ответ:` tails.",
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
    c089.SOURCE_EXPERIMENT_ID = "C087"
    c089.cleanup_english_answer = strict_cleanup_english_answer
    c089.recommendation = recommendation
    c089.write_report = write_report
    return c089.run(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
