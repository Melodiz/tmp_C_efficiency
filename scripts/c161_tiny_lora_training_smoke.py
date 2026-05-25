from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "C161"
EXPERIMENT_SLUG = "C161_tiny_lora_training_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C161_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")
TRAIN_MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
INFER_MODEL_ID = "Qwen/Qwen3-8B-AWQ"
USER_PREFIX = "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C161 tiny LoRA training smoke.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--train-model-id", default=TRAIN_MODEL_ID)
    parser.add_argument("--infer-model-id", default=INFER_MODEL_ID)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--val-rows", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=161)
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
        "zip": out_dir.with_suffix(".zip"),
    }


def run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except Exception as exc:
        return f"ERROR: {exc}"
    return ((result.stdout or "") + (result.stderr or "")).strip()


def gpu_memory_mb() -> int | None:
    output = run_cmd(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    values = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    return max(values) if values else None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().replace("ё", "е"))


def install_training_deps() -> str:
    return run_cmd(
        [
            "/usr/bin/python3",
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--force-reinstall",
            "peft==0.17.1",
            "bitsandbytes==0.49.2",
            "accelerate==1.10.1",
        ]
    )


def build_messages(question: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": f"{USER_PREFIX}\n\n{question}"}]
    if answer is not None:
        messages.append({"role": "assistant", "content": str(answer)})
    return messages


def load_short_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    import pandas as pd

    data = pd.read_parquet(DATA_PATH).reset_index(drop=True).reset_index(names="row_id")
    data = data.rename(columns={"query": "question", "answer": "reference_answer"})
    data = data.dropna(subset=["question", "reference_answer"]).copy()
    data["question_len"] = data["question"].astype(str).str.len()
    data["answer_len"] = data["reference_answer"].astype(str).str.len()
    pool = data[(data["question_len"] <= 450) & (data["answer_len"] <= 120)].copy()
    if "category" in pool:
        pool = pool.sort_values(["category", "row_id"])
        picked = []
        rng = random.Random(args.seed)
        categories = list(pool["category"].dropna().unique())
        rng.shuffle(categories)
        for category in categories:
            rows = pool[pool["category"] == category].head(3).to_dict(orient="records")
            picked.extend(rows)
            if len(picked) >= args.train_rows + args.val_rows:
                break
        selected = picked[: args.train_rows + args.val_rows]
    else:
        selected = pool.sample(args.train_rows + args.val_rows, random_state=args.seed).to_dict(orient="records")
    train = selected[: args.train_rows]
    val = selected[args.train_rows : args.train_rows + args.val_rows]
    meta = {
        "data_rows": int(len(data)),
        "pool_rows": int(len(pool)),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_category_counts": {},
        "val_category_counts": {},
    }
    if train and "category" in train[0]:
        from collections import Counter

        meta["train_category_counts"] = dict(Counter(str(row.get("category")) for row in train))
        meta["val_category_counts"] = dict(Counter(str(row.get("category")) for row in val))
    def safe_row(row: dict[str, Any]) -> dict[str, str]:
        return {
            "question": str(row["question"]),
            "reference_answer": str(row["reference_answer"]),
        }
    return [safe_row(row) for row in train], [safe_row(row) for row in val], meta


def run_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "train_model_id": args.train_model_id,
        "infer_model_id": args.infer_model_id,
        "config": {
            "train_rows": args.train_rows,
            "val_rows": args.val_rows,
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
            "steps": args.steps,
            "seed": args.seed,
            "raw_examples_returned": False,
            "adapter_weights_returned": False,
            "leaderboard_submission": False,
        },
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    adapter_dir = Path("/content/C161_tiny_adapter")
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    try:
        install_log = install_training_deps()
        summary["install_log_tail"] = install_log[-1200:]

        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        train_rows, val_rows, data_meta = load_short_rows(args)
        summary["data_meta"] = data_meta
        tokenizer = AutoTokenizer.from_pretrained(args.train_model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        before_train_vram = gpu_memory_mb()
        model = AutoModelForCausalLM.from_pretrained(
            args.train_model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)

        train_t0 = time.time()
        losses: list[float] = []
        for step in range(args.steps):
            row = train_rows[step % len(train_rows)]
            prompt = tokenizer.apply_chat_template(build_messages(row["question"]), tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(build_messages(row["question"], row["reference_answer"]), tokenize=False)
            full_ids = tokenizer(full, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(model.device)
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids
            labels = full_ids.clone()
            prompt_len = min(prompt_ids.shape[-1], labels.shape[-1])
            labels[:, :prompt_len] = -100
            loss = model(input_ids=full_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        train_seconds = time.time() - train_t0
        after_train_vram = gpu_memory_mb()
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        adapter_size = directory_size_bytes(adapter_dir)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        infer_tokenizer = AutoTokenizer.from_pretrained(args.infer_model_id, use_fast=True)
        prompts = [
            infer_tokenizer.apply_chat_template(
                build_messages(row["question"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in val_rows
        ]
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, top_p=1.0, top_k=-1)
        before_infer_vram = gpu_memory_mb()
        llm = LLM(
            model=args.infer_model_id,
            dtype="float16",
            quantization="awq_marlin",
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            tokenizer_mode="auto",
            seed=args.seed,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=16,
        )
        lora_request = LoRARequest("c161_tiny", 1, str(adapter_dir))
        gen_t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)
        gen_seconds = time.time() - gen_t0
        predictions = [item.outputs[0].text.strip() for item in outputs]
        exact = sum(normalize(pred) == normalize(row["reference_answer"]) for pred, row in zip(predictions, val_rows))
        nonempty = sum(bool(pred) for pred in predictions)
        after_infer_vram = gpu_memory_mb()

        shutil.rmtree(adapter_dir, ignore_errors=True)

        summary.update(
            {
                "status": "completed",
                "losses": losses,
                "adapter_size_bytes_remote_only": adapter_size,
                "adapter_weights_returned": False,
                "validation": {
                    "rows": len(val_rows),
                    "nonempty": nonempty,
                    "exact_match": exact,
                },
                "runtime": {
                    "total_seconds": time.time() - start,
                    "train_seconds": train_seconds,
                    "generate_seconds": gen_seconds,
                    "before_train_vram_mb": before_train_vram,
                    "after_train_vram_mb": after_train_vram,
                    "before_infer_vram_mb": before_infer_vram,
                    "after_infer_vram_mb": after_infer_vram,
                },
                "decision_recommendation": "MUTATE" if nonempty == len(val_rows) else "KILL",
                "reason": "Tiny adapter trained and loaded over AWQ path; inspect aggregate signal before scaling."
                if nonempty == len(val_rows)
                else "Tiny adapter pipeline produced empty validation outputs.",
            }
        )
    except Exception as exc:
        shutil.rmtree(adapter_dir, ignore_errors=True)
        summary.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime": {"total_seconds": time.time() - start},
                "decision_recommendation": "INVESTIGATE",
                "reason": "Training smoke failed; decide whether this is a trivial runner/dependency fix or a branch kill.",
            }
        )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = summary.get("runtime") or {}
    validation = summary.get("validation") or {}
    adapter_mb = None
    if isinstance(summary.get("adapter_size_bytes_remote_only"), int):
        adapter_mb = summary["adapter_size_bytes_remote_only"] / 1_000_000
    lines = [
        "# C161 Tiny LoRA Training Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- One tiny adapter-training smoke only.",
        "- No raw examples returned.",
        "- No adapter weights returned to the controller workspace.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- error: `{summary.get('error_type')}: {summary.get('error')}`" if summary.get("error") else "- error: none",
        "",
        "## Training",
        f"- train model: `{summary.get('train_model_id')}`",
        f"- infer model: `{summary.get('infer_model_id')}`",
        f"- train rows: `{(summary.get('config') or {}).get('train_rows')}`",
        f"- validation rows: `{(summary.get('config') or {}).get('val_rows')}`",
        f"- steps: `{(summary.get('config') or {}).get('steps')}`",
        f"- losses: `{summary.get('losses')}`",
        f"- adapter size MB remote-only: `{adapter_mb:.2f}`" if isinstance(adapter_mb, float) else "- adapter size MB remote-only: `unknown`",
        "",
        "## Validation Aggregates",
        f"- rows: `{validation.get('rows')}`",
        f"- nonempty: `{validation.get('nonempty')}`",
        f"- exact match: `{validation.get('exact_match')}`",
        "",
        "## Runtime",
        f"- total seconds: `{runtime.get('total_seconds')}`",
        f"- train seconds: `{runtime.get('train_seconds')}`",
        f"- generate seconds: `{runtime.get('generate_seconds')}`",
        f"- after train VRAM MB: `{runtime.get('after_train_vram_mb')}`",
        f"- after inference VRAM MB: `{runtime.get('after_infer_vram_mb')}`",
        "",
        "## Hygiene",
        f"- raw examples returned: `{(summary.get('config') or {}).get('raw_examples_returned')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
    ]
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
