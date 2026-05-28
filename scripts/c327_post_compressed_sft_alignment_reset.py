from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "C327"
EXPERIMENT_SLUG = "C327_post_compressed_sft_alignment_reset"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C327_artifacts"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C327 post-compressed-SFT alignment reset.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "zip": out_dir.with_suffix(".zip"),
    }


def run_audit() -> dict[str, Any]:
    return {
        "status": "completed",
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "prompts_returned": False,
        "references_returned": False,
        "outputs_returned": False,
        "model_loaded": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "decision_recommendation": "MUTATE",
        "reason": (
            "SFT target audit is promising but current SFT harness is setup-fragile; "
            "choose a non-SFT, non-handler global alignment probe."
        ),
        "synthesis": {
            "c111_public": 74.7,
            "candidate_threshold": "both ref-in-output and output-in-ref +>=5 on 2000 rows with no cap/repetition increase",
            "s1": "marker neutralization safe but tiny at 2000 rows: ref flat, output-in-ref +1",
            "s2": "adaptive length without prompt/style change flat on 512 rows",
            "s3": "compressed targets pass length gate, but C325/C326 produced no quality artifact due setup/session failure",
            "parked": [
                "deterministic handler mining",
                "nearby SFT install/harness retries",
                "global shortness prompts",
                "answer extraction",
                "question keyword echoing",
            ],
        },
        "selected_next": {
            "id": "C328",
            "name": "Long-route reference-style prompt aggregate",
            "mechanism": (
                "For the C269 long-answer route only, replace C111's short prefix with a formal "
                "reference-style prefix while preserving C111 prefix elsewhere and keeping max_tokens=320."
            ),
            "why": (
                "C270 showed length alone is flat, while C265/C266 show the major remaining public-score axis "
                "is reference-style fullness/list structure. This isolates prompt/register on a broad long-answer route "
                "without adding handlers or SFT."
            ),
            "gpu_preference": "L4_or_T4",
            "gate": (
                "Kill unless ref-in-output and output-in-ref are both nonnegative, at least one improves, "
                "and cap/repetition do not worsen. Scale only if both containment proxies look plausibly positive."
            ),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C327 Post-Compressed-SFT Alignment Reset",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        "",
        "## Synthesis",
        f"`{json.dumps(summary.get('synthesis', {}), ensure_ascii=False)}`",
        "",
        "## Selected Next",
        f"`{json.dumps(summary.get('selected_next', {}), ensure_ascii=False)}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- prompts returned: `{summary.get('prompts_returned')}`",
        f"- references returned: `{summary.get('references_returned')}`",
        f"- outputs returned: `{summary.get('outputs_returned')}`",
        f"- model loaded: `{summary.get('model_loaded')}`",
        f"- training started: `{summary.get('training_started')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    summary = {"status": "dry_run", "decision_recommendation": "DRY_RUN"} if args.dry_run else run_audit()
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(paths["report"], summary)
    if paths["zip"].exists():
        paths["zip"].unlink()
    shutil.make_archive(str(paths["out_dir"]), "zip", paths["out_dir"])
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
