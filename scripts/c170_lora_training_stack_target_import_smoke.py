from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as base


EXPERIMENT_ID = "C170"
EXPERIMENT_SLUG = "C170_lora_training_stack_target_import_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C170_artifacts"
DEFAULT_TARGET_DIR = Path("/content/c170_train_site")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C170 target-directory LoRA training-stack import smoke.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
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
        "install_log": out_dir / "logs" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_install.log",
        "probe_log": out_dir / "logs" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_probe.log",
        "zip": out_dir.with_suffix(".zip"),
    }


def run_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "objective": "setup-only target-directory LoRA training-stack import smoke",
        "leaderboard_submission": False,
        "raw_task_data_read": False,
        "qwen_weights_downloaded": False,
        "adapter_weights_created": False,
        "training_started": False,
        "target_dir": str(args.target_dir),
        "isolation": "pip --target plus PYTHONPATH",
    }
    if args.dry_run:
        summary.update(
            {
                "status": "dry_run",
                "decision_recommendation": "INVESTIGATE",
                "reason": "Dry run only; no target directory created.",
                "runtime": {"total_seconds": 0.0},
            }
        )
        return summary

    target_dir = Path(args.target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-cache-dir",
        "--upgrade",
        "--no-deps",
        "--target",
        str(target_dir),
        "bitsandbytes==0.49.2",
        "peft==0.19.1",
        "accelerate==1.13.0",
        "huggingface_hub==0.36.2",
        "transformers==4.57.6",
    ]
    install_code, install_log = base.run_cmd(install_cmd, timeout=1200)
    paths["install_log"].write_text(" ".join(install_cmd) + "\n" + install_log + "\n", encoding="utf-8")

    probe_code = 999
    probe_log = ""
    probe_json: dict[str, Any] | None = None
    if install_code == 0:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target_dir) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "-c", base.probe_source()],
                check=False,
                text=True,
                capture_output=True,
                timeout=900,
                env=env,
            )
            probe_code = result.returncode
            probe_log = ((result.stdout or "") + (result.stderr or "")).strip()
        except Exception as exc:
            probe_log = f"{type(exc).__name__}: {exc}"
        paths["probe_log"].write_text(probe_log + "\n", encoding="utf-8")
        try:
            probe_json = json.loads(probe_log)
            base.write_json(paths["probe"], probe_json)
        except Exception:
            probe_json = None
    else:
        paths["probe_log"].write_text("Probe skipped because target install failed.\n", encoding="utf-8")

    ok = install_code == 0 and probe_code == 0 and probe_json is not None
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": (
                "Target-directory training-stack imports and tiny in-memory PEFT config succeeded."
                if ok
                else "Target-directory training-stack setup/import smoke failed; keep SFT parked."
            ),
            "install_returncode": install_code,
            "probe_returncode": probe_code,
            "probe": probe_json,
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    probe = summary.get("probe") or {}
    versions = probe.get("versions") or {}
    imports = probe.get("imports") or {}
    checks = probe.get("config_checks") or {}
    lines = [
        "# C170 Target-Directory LoRA Training-Stack Import Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- No training.",
        "- No raw task data.",
        "- No Qwen weight download.",
        "- No adapter weight creation.",
        "- Avoid venv; isolate training packages with pip --target and PYTHONPATH.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- install return code: `{summary.get('install_returncode')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        f"- target dir: `{summary.get('target_dir')}`",
        "",
        "## Versions",
    ]
    for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub"]:
        lines.append(f"- {name}: import `{imports.get(name)}`, version `{versions.get(name)}`")
    lines.extend(
        [
            "",
            "## Config Checks",
            f"- bitsandbytes config: `{checks.get('bitsandbytes_config')}`",
            f"- tiny PEFT model in memory: `{checks.get('tiny_peft_model_created_in_memory')}`",
            f"- tiny trainable params: `{checks.get('tiny_trainable_params')}`",
            f"- config error: `{checks.get('error')}`" if checks.get("error") else "- config error: none",
            "",
            "## Hygiene",
            f"- raw task data read: `{summary.get('raw_task_data_read')}`",
            f"- Qwen weights downloaded: `{summary.get('qwen_weights_downloaded')}`",
            f"- adapter weights created: `{summary.get('adapter_weights_created')}`",
            f"- training started: `{summary.get('training_started')}`",
            "",
            "## Next",
        ]
    )
    if summary.get("status") == "completed":
        lines.append("Queue one tiny remote-only adapter-training smoke with strict no-artifact-return hygiene.")
    else:
        lines.append("Keep SFT/LoRA training parked; preserve C140 unless a new route appears.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
