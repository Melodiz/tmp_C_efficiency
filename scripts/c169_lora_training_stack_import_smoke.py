from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "C169"
EXPERIMENT_SLUG = "C169_lora_training_stack_import_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C169_artifacts"
DEFAULT_VENV_DIR = Path("/content/c169_lora_train_env")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C169 isolated LoRA training-stack import smoke.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--venv-dir", default=str(DEFAULT_VENV_DIR))
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


def run_cmd(cmd: list[str], *, timeout: int = 900) -> tuple[int, str]:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
        return result.returncode, ((result.stdout or "") + (result.stderr or "")).strip()
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_source() -> str:
    return textwrap.dedent(
        """
        import json
        import platform
        import sys
        import traceback

        result = {
            "python": sys.version,
            "platform": platform.platform(),
            "imports": {},
            "versions": {},
            "config_checks": {},
            "raw_task_data_read": False,
            "qwen_weights_downloaded": False,
            "adapter_weights_created": False,
            "training_started": False,
        }

        modules = ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub"]
        ok = True
        for name in modules:
            try:
                module = __import__(name)
                result["imports"][name] = "ok"
                result["versions"][name] = getattr(module, "__version__", "unknown")
            except Exception as exc:
                ok = False
                result["imports"][name] = f"{type(exc).__name__}: {exc}"

        try:
            import torch
            result["cuda"] = {
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2Config
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            result["config_checks"]["bitsandbytes_config"] = bool(bnb_config.load_in_4bit)

            tiny_config = GPT2Config(
                vocab_size=128,
                n_positions=32,
                n_embd=32,
                n_layer=1,
                n_head=4,
                bos_token_id=0,
                eos_token_id=1,
            )
            model = AutoModelForCausalLM.from_config(tiny_config)
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
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            result["config_checks"]["tiny_peft_model_created_in_memory"] = True
            result["config_checks"]["tiny_trainable_params"] = int(trainable)
            result["config_checks"]["automodel_import"] = AutoModelForCausalLM.__name__
            result["config_checks"]["autotokenizer_import"] = AutoTokenizer.__name__
        except Exception as exc:
            ok = False
            result["config_checks"]["error"] = f"{type(exc).__name__}: {exc}"
            result["config_checks"]["traceback_tail"] = traceback.format_exc()[-1600:]

        result["status"] = "completed" if ok else "failed"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if ok else 2)
        """
    ).strip()


def run_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "objective": "setup-only isolated LoRA training-stack import smoke",
        "leaderboard_submission": False,
        "raw_task_data_read": False,
        "qwen_weights_downloaded": False,
        "adapter_weights_created": False,
        "training_started": False,
        "venv_dir": str(args.venv_dir),
    }
    if args.dry_run:
        summary.update(
            {
                "status": "dry_run",
                "decision_recommendation": "INVESTIGATE",
                "reason": "Dry run only; no environment created.",
                "runtime": {"total_seconds": 0.0},
            }
        )
        return summary

    venv_dir = Path(args.venv_dir)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)

    create_code, create_log = run_cmd([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])
    install_cmd = [
        str(venv_dir / "bin" / "python"),
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-cache-dir",
        "--upgrade",
        "bitsandbytes==0.49.2",
        "peft==0.19.1",
        "accelerate==1.13.0",
        "huggingface_hub>=0.36.0,<1.0",
        "transformers>=4.57.0,<5.0",
    ]
    install_code, install_log = (999, "venv creation failed")
    if create_code == 0:
        install_code, install_log = run_cmd(install_cmd, timeout=1800)
    paths["install_log"].write_text(
        "CREATE VENV\n"
        + create_log
        + "\n\nINSTALL\n"
        + " ".join(install_cmd)
        + "\n"
        + install_log
        + "\n",
        encoding="utf-8",
    )

    probe_code = 999
    probe_log = ""
    probe_json: dict[str, Any] | None = None
    if create_code == 0 and install_code == 0:
        probe_cmd = [str(venv_dir / "bin" / "python"), "-c", probe_source()]
        probe_code, probe_log = run_cmd(probe_cmd, timeout=900)
        paths["probe_log"].write_text(probe_log + "\n", encoding="utf-8")
        try:
            probe_json = json.loads(probe_log)
            write_json(paths["probe"], probe_json)
        except Exception:
            probe_json = None
    else:
        paths["probe_log"].write_text("Probe skipped because setup failed.\n", encoding="utf-8")

    ok = create_code == 0 and install_code == 0 and probe_code == 0 and probe_json is not None
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": (
                "Isolated training-stack imports and tiny in-memory PEFT config succeeded."
                if ok
                else "Isolated training-stack setup/import smoke failed; keep SFT parked."
            ),
            "create_venv_returncode": create_code,
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
        "# C169 Isolated LoRA Training-Stack Import Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- No training.",
        "- No raw task data.",
        "- No Qwen weight download.",
        "- No adapter weight creation.",
        "- Isolated setup/import proof only.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- create venv return code: `{summary.get('create_venv_returncode')}`",
        f"- install return code: `{summary.get('install_returncode')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
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


def zip_artifacts(paths: dict[str, Path]) -> None:
    zip_path = paths["zip"]
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in paths["out_dir"].rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(paths["out_dir"]))


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    for key in ("reports_dir", "results_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    summary = run_smoke(args, paths)
    write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
