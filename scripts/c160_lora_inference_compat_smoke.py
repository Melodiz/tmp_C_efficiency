from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "C160"
EXPERIMENT_SLUG = "C160_lora_inference_compat_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C160_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
ADAPTER_ID = "Elcaida/qwen3-8bvariations_lora"
USER_PREFIX = "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C160 LoRA inference compatibility smoke.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-id", default=ADAPTER_ID)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
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
        "outputs": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_outputs.json",
        "log": out_dir / "logs" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}.log",
        "zip": out_dir.with_suffix(".zip"),
    }


def run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except Exception as exc:
        return f"ERROR: {exc}"
    text = (result.stdout or "") + (result.stderr or "")
    return text.strip()


def gpu_memory_mb() -> int | None:
    output = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    values: list[int] = []
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            values.append(int(line))
    return max(values) if values else None


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prompt(tokenizer: Any, question: str) -> str:
    content = f"{USER_PREFIX}\n\n{question}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    if args.dry_run:
        return {
            "status": "dry_run",
            "decision_recommendation": "INVESTIGATE",
            "reason": "Dry run only; no model or adapter loaded.",
        }

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "model_id": args.model_id,
        "adapter_id": args.adapter_id,
        "config": {
            "quantization": "awq_marlin",
            "dtype": "float16",
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "training": False,
            "raw_data_used": False,
            "leaderboard_submission": False,
        },
    }

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        adapter_path = Path(
            snapshot_download(
                args.adapter_id,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
            )
        )
        summary["adapter_path"] = str(adapter_path)
        summary["adapter_size_bytes"] = directory_size_bytes(adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        prompts = [
            build_prompt(tokenizer, "Сколько будет 2 + 2?"),
            build_prompt(tokenizer, "Fill the blank: The buildings ____ built in 1900."),
        ]
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            top_p=1.0,
            top_k=-1,
        )

        before_model_vram = gpu_memory_mb()
        llm = LLM(
            model=args.model_id,
            dtype="float16",
            quantization="awq_marlin",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.9,
            tokenizer_mode="auto",
            seed=0,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=64,
        )
        after_model_vram = gpu_memory_mb()

        base_start = time.time()
        base_outputs = llm.generate(prompts, sampling_params=sampling)
        base_seconds = time.time() - base_start

        lora_request = LoRARequest("c160_probe", 1, str(adapter_path))
        lora_start = time.time()
        lora_outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)
        lora_seconds = time.time() - lora_start
        after_lora_vram = gpu_memory_mb()

        outputs = {
            "base": [item.outputs[0].text.strip() for item in base_outputs],
            "lora": [item.outputs[0].text.strip() for item in lora_outputs],
        }
        write_json(paths["outputs"], outputs)

        summary.update(
            {
                "status": "completed",
                "runtime": {
                    "total_seconds": time.time() - start,
                    "base_generate_seconds": base_seconds,
                    "lora_generate_seconds": lora_seconds,
                    "before_model_vram_mb": before_model_vram,
                    "after_model_vram_mb": after_model_vram,
                    "after_lora_vram_mb": after_lora_vram,
                },
                "outputs_path": str(paths["outputs"]),
                "basic_validity": {
                    "base_rows": len(outputs["base"]),
                    "lora_rows": len(outputs["lora"]),
                    "empty_lora_outputs": sum(1 for item in outputs["lora"] if not item),
                },
                "decision_recommendation": "MUTATE",
                "reason": "LoRA adapter loaded and generated; queue a tiny adapter-training smoke.",
            }
        )
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime": {"total_seconds": time.time() - start},
                "decision_recommendation": "INVESTIGATE",
                "reason": "LoRA adapter did not load/generate on the current AWQ path; inspect whether this is an adapter/download issue or a real vLLM compatibility failure.",
            }
        )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = summary.get("runtime") or {}
    size_mb = None
    if isinstance(summary.get("adapter_size_bytes"), int):
        size_mb = summary["adapter_size_bytes"] / 1_000_000
    lines = [
        "# C160 LoRA Inference Compatibility Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- No training.",
        "- No raw train/test artifact copied back to controller.",
        "- Mechanism check only: vLLM LoRA loading on current Qwen3-8B-AWQ awq_marlin path.",
        "",
        "## Configuration",
        f"- model: `{summary.get('model_id')}`",
        f"- adapter: `{summary.get('adapter_id')}`",
        "- quantization: `awq_marlin`",
        "- dtype: `float16`",
        f"- max model len: `{(summary.get('config') or {}).get('max_model_len')}`",
        f"- adapter size MB: `{size_mb:.2f}`" if isinstance(size_mb, float) else "- adapter size MB: `unknown`",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- error: `{summary.get('error_type')}: {summary.get('error')}`" if summary.get("error") else "- error: none",
        "",
        "## Runtime",
        f"- total seconds: `{runtime.get('total_seconds')}`",
        f"- base generate seconds: `{runtime.get('base_generate_seconds')}`",
        f"- lora generate seconds: `{runtime.get('lora_generate_seconds')}`",
        f"- vram before model MB: `{runtime.get('before_model_vram_mb')}`",
        f"- vram after model MB: `{runtime.get('after_model_vram_mb')}`",
        f"- vram after lora MB: `{runtime.get('after_lora_vram_mb')}`",
        "",
        "## Next",
    ]
    if summary.get("status") == "completed":
        lines.append("Queue a tiny adapter-training smoke with strict artifact hygiene and validation gates.")
    else:
        lines.append("Investigate the adapter/download error first; kill the branch only if a valid Qwen3 adapter also fails to load.")
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
