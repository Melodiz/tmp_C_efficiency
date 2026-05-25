from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as base
import c171_lora_training_stack_torchao_import_smoke as c171


EXPERIMENT_ID = "C172"
EXPERIMENT_SLUG = "C172_synthetic_tiny_lora_training_step_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C172_artifacts"
DEFAULT_TARGET_DIR = Path("/content/c172_train_site")
REMOTE_ADAPTER_DIR = Path("/content/c172_adapter_scratch")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C172 synthetic tiny LoRA training-step smoke.")
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


def training_probe_source() -> str:
    return textwrap.dedent(
        f"""
        import json
        import shutil
        import sys
        import traceback
        from pathlib import Path

        result = {{
            "python": sys.version,
            "raw_task_data_read": False,
            "qwen_weights_downloaded": False,
            "training_started": False,
            "adapter_weights_created_remote_scratch": False,
            "adapter_weights_returned": False,
            "adapter_scratch_deleted": False,
            "imports": {{}},
            "versions": {{}},
            "train": {{}},
        }}

        modules = ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub", "torchao"]
        ok = True
        for name in modules:
            try:
                module = __import__(name)
                result["imports"][name] = "ok"
                result["versions"][name] = getattr(module, "__version__", "unknown")
            except Exception as exc:
                ok = False
                result["imports"][name] = f"{{type(exc).__name__}}: {{exc}}"

        adapter_dir = Path("{REMOTE_ADAPTER_DIR}")
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)

        try:
            import torch
            from transformers import AutoModelForCausalLM, GPT2Config
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            torch.manual_seed(172)
            config = GPT2Config(
                vocab_size=128,
                n_positions=32,
                n_embd=32,
                n_layer=1,
                n_head=4,
                bos_token_id=0,
                eos_token_id=1,
            )
            model = AutoModelForCausalLM.from_config(config)
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["c_attn"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.train()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            input_ids = torch.tensor([[5, 7, 9, 11, 13, 1]], dtype=torch.long)
            labels = input_ids.clone()
            losses = []
            result["training_started"] = True
            for _ in range(2):
                loss = model(input_ids=input_ids, labels=labels).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir)
            adapter_files = [p for p in adapter_dir.rglob("*") if p.is_file()]
            adapter_size = sum(p.stat().st_size for p in adapter_files)
            result["adapter_weights_created_remote_scratch"] = True
            result["train"] = {{
                "steps": 2,
                "losses": losses,
                "loss_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
                "adapter_file_count_before_delete": len(adapter_files),
                "adapter_size_bytes_before_delete": int(adapter_size),
            }}
            shutil.rmtree(adapter_dir)
            result["adapter_scratch_deleted"] = not adapter_dir.exists()
        except Exception as exc:
            ok = False
            result["train"]["error"] = f"{{type(exc).__name__}}: {{exc}}"
            result["train"]["traceback_tail"] = traceback.format_exc()[-1600:]
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
                result["adapter_scratch_deleted"] = not adapter_dir.exists()

        result["status"] = "completed" if ok else "failed"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if ok else 2)
        """
    ).strip()


def install_target(target_dir: Path, log_path: Path) -> int:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        "torchao==0.17.0",
    ]
    code, log = base.run_cmd(cmd, timeout=1200)
    log_path.write_text(" ".join(cmd) + "\n" + log + "\n", encoding="utf-8")
    return code


def run_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_task_data_read": False,
        "qwen_weights_downloaded": False,
        "adapter_weights_returned": False,
        "target_dir": str(args.target_dir),
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
    install_code = install_target(target_dir, paths["install_log"])
    probe_code = 999
    probe_log = ""
    probe_json: dict[str, Any] | None = None
    if install_code == 0:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target_dir) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run(
                [sys.executable, "-c", training_probe_source()],
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
        probe_json = c171.parse_probe_json(probe_log)
        if probe_json is not None:
            base.write_json(paths["probe"], probe_json)
    else:
        paths["probe_log"].write_text("Probe skipped because target install failed.\n", encoding="utf-8")

    ok = install_code == 0 and probe_code == 0 and probe_json is not None
    train = (probe_json or {}).get("train") or {}
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": (
                "Synthetic tiny LoRA training/save/delete smoke succeeded."
                if ok
                else "Synthetic tiny LoRA training-step smoke failed; keep SFT parked."
            ),
            "install_returncode": install_code,
            "probe_returncode": probe_code,
            "probe": probe_json,
            "adapter_size_bytes_before_delete": train.get("adapter_size_bytes_before_delete"),
            "adapter_scratch_deleted": (probe_json or {}).get("adapter_scratch_deleted"),
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    probe = summary.get("probe") or {}
    train = probe.get("train") or {}
    versions = probe.get("versions") or {}
    lines = [
        "# C172 Synthetic Tiny LoRA Training-Step Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- No raw task data.",
        "- No Qwen weight download.",
        "- Synthetic token ids only.",
        "- Adapter may be created only in remote scratch and deleted before artifact zipping.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- install return code: `{summary.get('install_returncode')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        "",
        "## Versions",
    ]
    for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub", "torchao"]:
        lines.append(f"- {name}: `{versions.get(name)}`")
    lines.extend(
        [
            "",
            "## Training Smoke",
            f"- steps: `{train.get('steps')}`",
            f"- losses: `{train.get('losses')}`",
            f"- loss finite: `{train.get('loss_finite')}`",
            f"- adapter file count before delete: `{train.get('adapter_file_count_before_delete')}`",
            f"- adapter size bytes before delete: `{train.get('adapter_size_bytes_before_delete')}`",
            f"- adapter scratch deleted: `{summary.get('adapter_scratch_deleted')}`",
            f"- train error: `{train.get('error')}`" if train.get("error") else "- train error: none",
            "",
            "## Hygiene",
            f"- raw task data read: `{probe.get('raw_task_data_read')}`",
            f"- Qwen weights downloaded: `{probe.get('qwen_weights_downloaded')}`",
            f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
            "",
            "## Next",
        ]
    )
    if summary.get("status") == "completed":
        lines.append("Queue a remote-only Qwen-family tiny adapter training feasibility audit before any task-data training.")
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
