from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


EXPERIMENT_ID = "C071_l4_vllm_model_probe"
DEFAULT_OUT_DIR = Path("results") / EXPERIMENT_ID
DATA_PATH = Path("data/dataset_ml_challenge.parquet")
HARD_AUDIT_PATH = Path("c009_artifacts/hard_audit_set.csv")
LOCKED_SPLIT_PATH = Path("c009_artifacts/locked_validation_split.csv")
BASELINE_LOCAL_PATH = Path("simple_solution/weights")

MODEL_IDS = {
    "baseline": "Qwen/Qwen3-0.6B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_command(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception:
        return None


def dir_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def environment_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "disk": run_command(["df", "-h", "."]),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": getattr(torch.version, "cuda", None),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        info["torch_probe_error"] = repr(exc)

    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as exc:
        info["transformers_error"] = repr(exc)

    try:
        import vllm

        info["vllm"] = getattr(vllm, "__version__", "unknown")
    except Exception as exc:
        info["vllm_error"] = repr(exc)

    info["nvidia_smi"] = run_command(["nvidia-smi"], timeout=20)
    return info


class GpuMemorySampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        query = [
            "nvidia-smi",
            "--query-gpu=timestamp,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            output = run_command(query, timeout=5)
            if output:
                for line in output.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        try:
                            self.samples.append(
                                {
                                    "timestamp": parts[0],
                                    "name": parts[1],
                                    "memory_used_mb": int(parts[2]),
                                    "memory_total_mb": int(parts[3]),
                                }
                            )
                        except ValueError:
                            pass
            time.sleep(self.interval_s)

    @property
    def peak_used_mb(self) -> int | None:
        if not self.samples:
            return None
        return max(int(s["memory_used_mb"]) for s in self.samples)


def resolve_model(candidate: str, model_id: str | None, baseline_local_path: Path) -> tuple[str, str]:
    if model_id:
        return model_id, "explicit"
    if candidate == "baseline" and (baseline_local_path / "config.json").exists():
        return str(baseline_local_path), "local_baseline_weights"
    return MODEL_IDS[candidate], "huggingface"


def load_sample(sample_source: str, sample_size: int, seed: int) -> pd.DataFrame:
    data = pd.read_parquet(DATA_PATH).reset_index(drop=True).reset_index(names="row_id")
    data = data.rename(columns={"query": "question", "answer": "reference_answer"})

    if sample_source == "hard_audit":
        meta = pd.read_csv(HARD_AUDIT_PATH)
        pool = meta.merge(data, on="row_id", how="left")
    elif sample_source == "locked_val":
        meta = pd.read_csv(LOCKED_SPLIT_PATH)
        pool = meta[meta["split"] == "val"].merge(data, on="row_id", how="left")
    elif sample_source == "dataset":
        pool = data.copy()
        pool["category"] = "dataset"
    else:
        raise ValueError(f"unknown sample source: {sample_source}")

    pool = pool.dropna(subset=["question"]).copy()
    pool["category"] = pool.get("category", "unknown").fillna("unknown")
    if sample_size <= 0 or sample_size >= len(pool):
        return pool.sort_values(["category", "row_id"]).reset_index(drop=True)

    categories = sorted(pool["category"].unique())
    per_category = max(1, sample_size // max(1, len(categories)))
    selected_parts: list[pd.DataFrame] = []
    for category in categories:
        group = pool[pool["category"] == category]
        selected_parts.append(group.sample(min(per_category, len(group)), random_state=seed))

    selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["row_id"])
    if len(selected) < sample_size:
        remaining = pool[~pool["row_id"].isin(set(selected["row_id"]))]
        if len(remaining):
            fill = remaining.sample(min(sample_size - len(selected), len(remaining)), random_state=seed + 1)
            selected = pd.concat([selected, fill], ignore_index=True)
    elif len(selected) > sample_size:
        selected = selected.sample(sample_size, random_state=seed + 2)

    return selected.sort_values(["category", "row_id"]).reset_index(drop=True)


def apply_user_only_template(
    tokenizer: Any,
    question: str,
    enable_thinking_false: bool,
    user_prefix: str | None = None,
) -> str:
    user_content = f"{user_prefix}\n\n{question}" if user_prefix else question
    messages = [{"role": "user", "content": user_content}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking_false:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        if "enable_thinking" in kwargs:
            kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(messages, **kwargs)
        raise


def has_repetition_loop(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        most_common = max(lines.count(line) for line in set(lines))
        if most_common >= 4:
            return True
    words = text.split()
    if len(words) >= 80:
        tail = words[-40:]
        unique_ratio = len(set(tail)) / max(1, len(tail))
        return unique_ratio < 0.25
    return False


def hf_metadata(model_id: str) -> dict[str, Any] | None:
    if Path(model_id).exists():
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True)
        total_size = sum((getattr(s, "size", None) or 0) for s in info.siblings)
        return {
            "sha": info.sha,
            "created_at": str(info.created_at),
            "last_modified": str(info.last_modified),
            "pipeline_tag": info.pipeline_tag,
            "total_file_size_bytes": total_size,
            "safetensors": [
                {"name": s.rfilename, "size": getattr(s, "size", None)}
                for s in info.siblings
                if s.rfilename.endswith(".safetensors")
            ],
        }
    except Exception as exc:
        return {"error": repr(exc)}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_ref, model_source = resolve_model(args.candidate, args.model_id, Path(args.baseline_local_path))
    run_id = f"{utc_stamp()}_{args.candidate}_{args.sample_size}"
    summary_path = out_dir / f"{run_id}.summary.json"
    outputs_path = out_dir / f"{run_id}.outputs.jsonl"
    samples_path = out_dir / f"{run_id}.samples.jsonl"

    sample_df = load_sample(args.sample_source, args.sample_size, args.seed)
    sample_rows = sample_df.to_dict(orient="records")
    append_jsonl(samples_path, sample_rows)

    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "status": "started",
        "candidate": args.candidate,
        "model_ref": model_ref,
        "model_source": model_source,
        "model_size_bytes_local": dir_size_bytes(Path(model_ref)) if Path(model_ref).exists() else None,
        "hf_metadata": None if args.skip_hf_metadata else hf_metadata(model_ref),
        "config": {
            "sample_source": args.sample_source,
            "sample_size_requested": args.sample_size,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "dtype": args.dtype,
            "quantization": getattr(args, "quantization", None),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "user_message_only": True,
            "user_prefix": args.user_prefix,
            "enable_thinking_false_in_template": not args.no_enable_thinking_false,
            "router_retrieval_cache_handlers_sft_lora": False,
        },
        "environment": environment_snapshot(),
        "paths": {
            "summary": str(summary_path),
            "outputs": str(outputs_path),
            "samples": str(samples_path),
        },
        "sample": {
            "rows": len(sample_df),
            "category_counts": sample_df["category"].value_counts().sort_index().to_dict()
            if "category" in sample_df
            else {},
        },
    }

    if args.dry_run:
        summary["status"] = "dry_run"
        write_json(summary_path, summary)
        return summary

    sampler = GpuMemorySampler(interval_s=args.gpu_sample_interval)
    try:
        from vllm import LLM, SamplingParams

        tokenizer_t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True)
        tokenizer_s = time.perf_counter() - tokenizer_t0

        prompts = [
            apply_user_only_template(
                tokenizer,
                str(row["question"]),
                enable_thinking_false=not args.no_enable_thinking_false,
                user_prefix=args.user_prefix,
            )
            for row in sample_rows
        ]
        input_token_counts = [len(tokenizer(prompt).input_ids) for prompt in prompts]

        sampler.start()
        startup_t0 = time.perf_counter()
        llm_kwargs = {
            "model": model_ref,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tokenizer_mode": "auto",
            "seed": args.seed,
            "trust_remote_code": args.trust_remote_code,
        }
        quantization = getattr(args, "quantization", None)
        if quantization:
            llm_kwargs["quantization"] = quantization
        llm = LLM(**llm_kwargs)
        startup_s = time.perf_counter() - startup_t0

        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        generation_t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling)
        generation_s = time.perf_counter() - generation_t0
        sampler.stop()

        result_rows: list[dict[str, Any]] = []
        output_token_counts: list[int] = []
        for i, (row, prompt, prompt_tokens, out) in enumerate(zip(sample_rows, prompts, input_token_counts, outputs)):
            completion = out.outputs[0]
            answer = completion.text.strip()
            token_ids = getattr(completion, "token_ids", None)
            output_tokens = len(token_ids) if token_ids is not None else len(tokenizer(answer).input_ids)
            output_token_counts.append(output_tokens)
            result_rows.append(
                {
                    "run_id": run_id,
                    "candidate": args.candidate,
                    "model_ref": model_ref,
                    "sample_index": i,
                    "rid": int(row["row_id"]),
                    "row_id": int(row["row_id"]),
                    "category": row.get("category"),
                    "question": row["question"],
                    "reference_answer": row.get("reference_answer"),
                    "prompt": prompt if args.save_prompts else None,
                    "input_tokens": prompt_tokens,
                    "answer": answer,
                    "output_tokens": output_tokens,
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    "has_thinking_trace": "<think" in answer or "</think>" in answer,
                    "hit_max_tokens": output_tokens >= args.max_tokens,
                    "repetition_loop_suspected": has_repetition_loop(answer),
                }
            )

        append_jsonl(outputs_path, result_rows)

        total_output_tokens = sum(output_token_counts)
        projected_generation_4000_s = (generation_s / max(1, len(sample_df))) * 4000
        projected_total_4000_s = startup_s + projected_generation_4000_s
        summary.update(
            {
                "status": "completed",
                "runtime": {
                    "tokenizer_load_s": tokenizer_s,
                    "startup_s": startup_s,
                    "generation_s": generation_s,
                    "sample_rows": len(sample_df),
                    "throughput_output_tokens_per_s": total_output_tokens / generation_s if generation_s else None,
                    "throughput_questions_per_s": len(sample_df) / generation_s if generation_s else None,
                    "projected_generation_4000_s": projected_generation_4000_s,
                    "projected_total_4000_s": projected_total_4000_s,
                    "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
                },
                "tokens": {
                    "avg_input_tokens": sum(input_token_counts) / max(1, len(input_token_counts)),
                    "max_input_tokens": max(input_token_counts) if input_token_counts else None,
                    "avg_output_tokens": sum(output_token_counts) / max(1, len(output_token_counts)),
                    "max_output_tokens": max(output_token_counts) if output_token_counts else None,
                    "total_output_tokens": total_output_tokens,
                },
                "validity": {
                    "jsonl_rows": len(result_rows),
                    "one_answer_per_input": len(result_rows) == len(sample_df),
                    "thinking_trace_rows": sum(1 for row in result_rows if row["has_thinking_trace"]),
                    "max_token_hit_rows": sum(1 for row in result_rows if row["hit_max_tokens"]),
                    "empty_answer_rows": sum(1 for row in result_rows if not row["answer"]),
                    "repetition_loop_suspected_rows": sum(
                        1 for row in result_rows if row["repetition_loop_suspected"]
                    ),
                },
                "gpu_memory_samples": sampler.samples[-20:],
            }
        )
        write_json(summary_path, summary)
        return summary
    except Exception as exc:
        sampler.stop()
        summary.update(
            {
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "gpu_memory_samples": sampler.samples[-20:],
                "runtime": {
                    "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
                },
            }
        )
        write_json(summary_path, summary)
        if not args.no_fail:
            raise
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C071 L4 vLLM model probe.")
    parser.add_argument("--candidate", choices=["baseline", "qwen3-4b", "qwen3-1.7b"], required=True)
    parser.add_argument("--model-id", default=None, help="Override model path or Hugging Face repo id.")
    parser.add_argument("--baseline-local-path", default=str(BASELINE_LOCAL_PATH))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="hard_audit")
    parser.add_argument("--sample-size", type=int, default=26)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None, help="Optional vLLM quantization mode, for example AWQ.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-enable-thinking-false", action="store_true")
    parser.add_argument("--user-prefix", default=None, help="Optional prefix prepended inside each user message.")
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-fail", action="store_true", help="Write error summary and exit 0.")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args()
    summary = run_probe(args)
    print(json.dumps({k: summary.get(k) for k in ["run_id", "status", "candidate", "model_ref", "paths"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
