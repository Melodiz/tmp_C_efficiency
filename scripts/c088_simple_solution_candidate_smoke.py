from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import snapshot_download

import c072_output_control as base


EXPERIMENT_ID = "C088"
EXPERIMENT_SLUG = "C088_simple_solution_candidate_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C088_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


SMOKE_ROWS = [
    {
        "rid": 8295,
        "question": "Найди значение выражения $(x + y)^2 + 5x^2 - 2x - 2(x + y) + 5$ при $x = 4$, $y = 2$.",
        "expected_contains": "101",
    },
    {
        "rid": 4242,
        "question": "Amazingly, many of the houses __________________ several centuries ago! BUILD",
        "expected_contains": "were built",
    },
    {
        "rid": 5782,
        "question": "составить слова из букв слова «брелок»",
        "expected_contains": "Итоговый ответ",
    },
]


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


def dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary.get("checks") or {}
    lines = [
        "# C088 Simple-Solution Candidate Smoke Report",
        "",
        "## Objective",
        "- ID: C088",
        "- Mechanism: final-entrypoint smoke for the C086/C087 candidate in `simple_solution/solution.py`.",
        "- Leaderboard submission: NO.",
        "- Model weights were downloaded only inside the remote runtime and are not included in this artifact.",
        "",
        "## Results",
        "| status | rows | return code | runtime s | output rows | checks passed | weights GB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {returncode} | {runtime:.2f} | {output_rows} | {checks_passed}/{checks_total} | {weights_gb:.2f} |".format(
            status=summary.get("status"),
            rows=len(SMOKE_ROWS),
            returncode=summary.get("returncode"),
            runtime=float(summary.get("runtime_s") or 0),
            output_rows=summary.get("output_rows"),
            checks_passed=checks.get("passed", 0),
            checks_total=checks.get("total", 0),
            weights_gb=(summary.get("weights_size_bytes") or 0) / (1024**3),
        ),
        "",
        "## Checks",
    ]
    for item in checks.get("items", []):
        lines.append(
            "- rid `{rid}` expected contains `{expected}`: `{passed}`".format(
                rid=item.get("rid"),
                expected=item.get("expected_contains"),
                passed=item.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision recommendation",
            "",
            summary.get("decision_recommendation", "REVIEW"),
            "",
            "## Strongest reason against recommendation",
            f"- {summary.get('reason', 'Review smoke outputs before packaging.')}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_dry_run(paths: dict[str, Path]) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_c088_dry_run"
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "run_id": run_id,
        "status": "dry_run",
        "model_id": MODEL_ID,
        "rows": SMOKE_ROWS,
        "returncode": None,
        "runtime_s": 0,
        "output_rows": 0,
        "weights_size_bytes": 0,
        "checks": {"passed": 0, "total": len(SMOKE_ROWS), "items": []},
        "decision_recommendation": "INVESTIGATE",
        "reason": "Dry run only; final entrypoint was not executed.",
        "paths": {
            "summary": str(paths["results_dir"] / f"{run_id}.summary.json"),
            "outputs": str(paths["results_dir"] / f"{run_id}.outputs.json"),
            "stdout": str(paths["logs_dir"] / f"{run_id}.stdout.log"),
            "stderr": str(paths["logs_dir"] / f"{run_id}.stderr.log"),
        },
    }


def create_smoke(paths: dict[str, Path]) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_simple_solution_smoke"
    solution_dir = Path("simple_solution").resolve()
    weights_dir = solution_dir / "weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    snapshot_download(repo_id=MODEL_ID, local_dir=weights_dir, local_dir_use_symlinks=False)

    input_path = solution_dir / "input.pickle"
    output_path = solution_dir / "output.json"
    with input_path.open("wb") as handle:
        pickle.dump([{"rid": row["rid"], "question": row["question"]} for row in SMOKE_ROWS], handle)
    if output_path.exists():
        output_path.unlink()

    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "solution.py"],
        cwd=solution_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    )
    runtime_s = time.perf_counter() - started

    outputs: list[dict[str, Any]] = []
    if output_path.exists():
        outputs = json.loads(output_path.read_text(encoding="utf-8"))

    by_rid = {int(item.get("rid")): str(item.get("answer", "")) for item in outputs}
    check_items = []
    passed = 0
    for row in SMOKE_ROWS:
        answer = by_rid.get(int(row["rid"]), "")
        ok = row["expected_contains"].lower() in answer.lower()
        if row.get("expected_exact") is not None:
            ok = ok and answer.strip() == str(row["expected_exact"]).strip()
        for forbidden in row.get("forbidden_contains", []):
            ok = ok and forbidden.lower() not in answer.lower()
        passed += int(ok)
        check_items.append(
            {
                "rid": row["rid"],
                "expected_contains": row["expected_contains"],
                "expected_exact": row.get("expected_exact"),
                "forbidden_contains": row.get("forbidden_contains", []),
                "passed": ok,
                "answer": answer,
            }
        )

    success = proc.returncode == 0 and len(outputs) == len(SMOKE_ROWS) and passed == len(SMOKE_ROWS)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "run_id": run_id,
        "status": "completed" if success else "failed",
        "model_id": MODEL_ID,
        "commit_smoked": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip(),
        "rows": SMOKE_ROWS,
        "returncode": proc.returncode,
        "runtime_s": runtime_s,
        "output_rows": len(outputs),
        "weights_size_bytes": dir_size_bytes(weights_dir),
        "checks": {"passed": passed, "total": len(SMOKE_ROWS), "items": check_items},
        "decision_recommendation": "SUBMIT" if success else "INVESTIGATE",
        "reason": (
            "Final entrypoint smoke passed; build a real submission zip outside this controller without adding weights here."
            if success
            else "Final entrypoint smoke failed or produced unexpected outputs."
        ),
        "paths": {
            "summary": str(paths["results_dir"] / f"{run_id}.summary.json"),
            "outputs": str(paths["results_dir"] / f"{run_id}.outputs.json"),
            "stdout": str(paths["logs_dir"] / f"{run_id}.stdout.log"),
            "stderr": str(paths["logs_dir"] / f"{run_id}.stderr.log"),
        },
    }
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    (paths["logs_dir"] / f"{run_id}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (paths["logs_dir"] / f"{run_id}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the final simple_solution C086 candidate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--dry-run", action="store_true", help="Create artifact layout without downloading weights or running model.")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    archived_previous = base.prepare_out_dir(out_dir)
    paths = artifact_paths(out_dir)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    summary = create_dry_run(paths) if args.dry_run else create_smoke(paths)
    summary["archived_previous_out_dir"] = str(archived_previous) if archived_previous else None
    summary_path = Path(summary["paths"]["summary"])
    outputs_path = Path(summary["paths"]["outputs"])
    base.write_json(summary_path, summary)
    base.write_json(outputs_path, summary.get("checks", {}).get("items", []))
    write_report(paths["report"], summary)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "created_utc": base.utc_stamp(),
        "dry_run": args.dry_run,
        "out_dir": str(paths["out_dir"]),
        "zip_path": str(paths["zip"]),
        "runs": [
            {
                "run_id": summary["run_id"],
                "summary_path": str(summary_path),
                "outputs_path": str(outputs_path),
                "stdout_path": summary["paths"]["stdout"],
                "stderr_path": summary["paths"]["stderr"],
                "status": summary["status"],
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
