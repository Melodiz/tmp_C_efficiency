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


EXPERIMENT_ID = "C173"
EXPERIMENT_SLUG = "C173_qwen3_8b_synthetic_qlora_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C173_artifacts"
DEFAULT_TARGET_DIR = Path("/content/c173_train_site")
REMOTE_ADAPTER_DIR = Path("/content/c173_adapter_scratch")
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C173 Qwen3-8B synthetic QLoRA smoke.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-seq-len", type=int, default=128)
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


def qwen_probe_source(model_id: str, max_seq_len: int) -> str:
    return textwrap.dedent(
        f"""
        import gc
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time
        import traceback
        from pathlib import Path

        result = {{
            "model_id": "{model_id}",
            "raw_task_data_read": False,
            "qwen_weights_downloaded_remote_only": False,
            "adapter_weights_created_remote_scratch": False,
            "adapter_weights_returned": False,
            "adapter_scratch_deleted": False,
            "training_started": False,
            "imports": {{}},
            "versions": {{}},
            "runtime": {{}},
            "train": {{}},
        }}

        def gpu_memory_mb():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ], text=True)
                values = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
                return max(values) if values else None
            except Exception:
                return None

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

        start = time.time()
        try:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            before_load = gpu_memory_mb()
            tokenizer = AutoTokenizer.from_pretrained("{model_id}", use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                "{model_id}",
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=False,
            )
            result["qwen_weights_downloaded_remote_only"] = True
            after_load = gpu_memory_mb()
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            text = "<|im_start|>user\\nСколько будет 2+2?\\n<|im_end|>\\n<|im_start|>assistant\\n4<|im_end|>"
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length={max_seq_len})
            batch = {{k: v.to(model.device) for k, v in batch.items()}}
            labels = batch["input_ids"].clone()
            model.train()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
            result["training_started"] = True
            loss = model(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir)
            files = [p for p in adapter_dir.rglob("*") if p.is_file()]
            adapter_size = sum(p.stat().st_size for p in files)
            result["adapter_weights_created_remote_scratch"] = True
            shutil.rmtree(adapter_dir)
            result["adapter_scratch_deleted"] = not adapter_dir.exists()
            result["train"] = {{
                "steps": 1,
                "loss": loss_value,
                "loss_finite": bool(torch.isfinite(torch.tensor(loss_value))),
                "trainable_params": int(trainable),
                "adapter_file_count_before_delete": len(files),
                "adapter_size_bytes_before_delete": int(adapter_size),
            }}
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result["runtime"] = {{
                "total_seconds": time.time() - start,
                "vram_before_load_mb": before_load,
                "vram_after_load_mb": after_load,
                "vram_after_cleanup_mb": gpu_memory_mb(),
            }}
        except Exception as exc:
            ok = False
            result["train"]["error"] = f"{{type(exc).__name__}}: {{exc}}"
            result["train"]["traceback_tail"] = traceback.format_exc()[-2200:]
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
                result["adapter_scratch_deleted"] = not adapter_dir.exists()

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
        "model_id": args.model_id,
        "leaderboard_submission": False,
        "raw_task_data_read": False,
        "adapter_weights_returned": False,
        "target_dir": str(args.target_dir),
    }
    if args.dry_run:
        summary.update(
            {
                "status": "dry_run",
                "decision_recommendation": "INVESTIGATE",
                "reason": "Dry run only; no target directory or model load.",
                "runtime": {"total_seconds": 0.0},
            }
        )
        return summary

    target_dir = Path(args.target_dir)
    install_code = install_target(target_dir, paths["install_log"])
    probe_code = 999
    probe_json: dict[str, Any] | None = None
    probe_log = ""
    if install_code == 0:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target_dir) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run(
                [sys.executable, "-c", qwen_probe_source(args.model_id, args.max_seq_len)],
                check=False,
                text=True,
                capture_output=True,
                timeout=2400,
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
    runtime = (probe_json or {}).get("runtime") or {}
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": (
                "Qwen3-8B synthetic QLoRA load/train/save/delete smoke succeeded."
                if ok
                else "Qwen3-8B synthetic QLoRA smoke failed; do not start task-data SFT."
            ),
            "install_returncode": install_code,
            "probe_returncode": probe_code,
            "probe": probe_json,
            "train": train,
            "remote_runtime": runtime,
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    probe = summary.get("probe") or {}
    train = summary.get("train") or {}
    remote_runtime = summary.get("remote_runtime") or {}
    versions = (probe.get("versions") or {})
    lines = [
        "# C173 Qwen3-8B Synthetic QLoRA Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- No raw task data.",
        "- Synthetic prompt only.",
        "- Model weights remain remote on Colab.",
        "- Adapter scratch is deleted before artifact zipping.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- install return code: `{summary.get('install_returncode')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        f"- model: `{summary.get('model_id')}`",
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
            f"- loss: `{train.get('loss')}`",
            f"- loss finite: `{train.get('loss_finite')}`",
            f"- trainable params: `{train.get('trainable_params')}`",
            f"- adapter file count before delete: `{train.get('adapter_file_count_before_delete')}`",
            f"- adapter size bytes before delete: `{train.get('adapter_size_bytes_before_delete')}`",
            f"- train error: `{train.get('error')}`" if train.get("error") else "- train error: none",
            "",
            "## Runtime",
            f"- remote seconds: `{remote_runtime.get('total_seconds')}`",
            f"- vram before load MB: `{remote_runtime.get('vram_before_load_mb')}`",
            f"- vram after load MB: `{remote_runtime.get('vram_after_load_mb')}`",
            f"- vram after cleanup MB: `{remote_runtime.get('vram_after_cleanup_mb')}`",
            "",
            "## Hygiene",
            f"- raw task data read: `{probe.get('raw_task_data_read')}`",
            f"- qwen weights downloaded remote only: `{probe.get('qwen_weights_downloaded_remote_only')}`",
            f"- adapter scratch deleted: `{probe.get('adapter_scratch_deleted')}`",
            f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
            "",
            "## Next",
        ]
    )
    if summary.get("status") == "completed":
        lines.append("Queue a zero-GPU task-data hygiene and validation protocol audit before any real SFT run.")
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
