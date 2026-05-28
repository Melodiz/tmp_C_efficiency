from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as base
import c171_lora_training_stack_torchao_import_smoke as c171
import c177_base_vs_lora_aggregate_validation_smoke as c177
import c178_sft_aggregate_metric_cap_diagnostic as c178
import c325_compressed_reference_sft_smoke as c325


EXPERIMENT_ID = "C341"
EXPERIMENT_SLUG = "C341_compressed_reference_sft_light_bootstrap_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C341_artifacts"
MODEL_ID = c177.MODEL_ID


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C341 compressed-reference SFT smoke with light Colab bootstrap.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--train-rows", type=int, default=96)
    parser.add_argument("--val-rows", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--seed", type=int, default=341)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "probe": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_probe.json",
        "probe_log": out_dir / "logs" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_probe.log",
        "zip": out_dir.with_suffix(".zip"),
    }


def run_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "model_id": args.model_id,
        "leaderboard_submission": False,
        "raw_examples_returned": False,
        "adapter_weights_returned": False,
        "setup": "global_env_light_bootstrap_no_vllm",
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE", "runtime": {"total_seconds": 0.0}})
        return summary

    probe_code = 999
    probe_json: dict[str, Any] | None = None
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                c325.task_probe_source(
                    args.model_id,
                    args.train_rows,
                    args.val_rows,
                    args.steps,
                    args.max_seq_len,
                    args.max_new_tokens,
                    args.seed,
                ),
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=3000,
            env=os.environ.copy(),
        )
        probe_code = result.returncode
        probe_log = ((result.stdout or "") + (result.stderr or "")).strip()
    except Exception as exc:
        probe_log = f"{type(exc).__name__}: {exc}"
    paths["probe_log"].write_text(probe_log + "\n", encoding="utf-8")
    probe_json = c171.parse_probe_json(probe_log)
    if probe_json is not None:
        base.write_json(paths["probe"], probe_json)

    ok = probe_code == 0 and probe_json is not None
    val = (probe_json or {}).get("validation") or {}
    ref_delta = int(val.get("lora_ref_in_output_count") or 0) - int(val.get("base_ref_in_output_count") or 0)
    out_delta = int(val.get("lora_output_in_ref_count") or 0) - int(val.get("base_output_in_ref_count") or 0)
    cap_delta = int(val.get("lora_cap_hit_count") or 0) - int(val.get("base_cap_hit_count") or 0)
    invalid_delta = int(val.get("lora_invalid_count") or 0) - int(val.get("base_invalid_count") or 0)
    rep_delta = int(val.get("lora_repetition_suspect_count") or 0) - int(val.get("base_repetition_suspect_count") or 0)
    decision = "MUTATE" if ok and ref_delta > 0 and out_delta > 0 and cap_delta <= 0 and invalid_delta <= 0 and rep_delta <= 0 else "KILL"
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": decision if ok else "INVESTIGATE",
            "reason": "Light-bootstrap compressed-reference SFT smoke completed." if ok else "Light-bootstrap compressed-reference SFT smoke failed.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "train": (probe_json or {}).get("train"),
            "validation": val,
            "containment_deltas": {
                "ref_in_output": ref_delta,
                "output_in_ref": out_delta,
                "cap": cap_delta,
                "invalid": invalid_delta,
                "repetition": rep_delta,
            },
            "remote_runtime": (probe_json or {}).get("runtime"),
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    c178.write_report(path, summary)
    text = path.read_text(encoding="utf-8")
    title = EXPERIMENT_SLUG.replace("_", " ").title()
    text = text.replace("# C178 SFT Aggregate Metric/Cap Diagnostic", f"# {title}", 1)
    text += f"\n## Containment Deltas\n`{json.dumps(summary.get('containment_deltas', {}), ensure_ascii=False)}`\n"
    path.write_text(text, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    for key in ("reports_dir", "results_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    summary = run_smoke(args, paths)
    base.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
