from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as base
import c171_lora_training_stack_torchao_import_smoke as c171
from c173_qwen3_8b_synthetic_qlora_smoke import install_target


EXPERIMENT_ID = "C175"
EXPERIMENT_SLUG = "C175_remote_tiny_task_sft_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C175_artifacts"
DEFAULT_TARGET_DIR = Path("/content/c175_train_site")
REMOTE_ADAPTER_DIR = Path("/content/c175_adapter_scratch")
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")
USER_PREFIX = "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C175 remote-only tiny task-data SFT smoke.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--train-rows", type=int, default=8)
    parser.add_argument("--val-rows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=175)
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


def task_probe_source(model_id: str, train_rows: int, val_rows: int, steps: int, max_seq_len: int, max_new_tokens: int, seed: int) -> str:
    return textwrap.dedent(
        f"""
        import gc
        import json
        import os
        import random
        import re
        import shutil
        import subprocess
        import sys
        import time
        import traceback
        from collections import Counter
        from pathlib import Path

        result = {{
            "model_id": "{model_id}",
            "raw_examples_returned": False,
            "raw_task_data_read_remote_only": False,
            "adapter_weights_returned": False,
            "adapter_scratch_deleted": False,
            "training_started": False,
            "imports": {{}},
            "versions": {{}},
            "data_meta": {{}},
            "train": {{}},
            "validation": {{}},
            "runtime": {{}},
        }}

        def gpu_memory_mb():
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
                return max(vals) if vals else None
            except Exception:
                return None

        def normalize(text):
            return re.sub(r"\\s+", " ", str(text).strip().lower().replace("ё", "е"))

        def invalid_output(text):
            text = str(text).strip()
            return (not text) or len(text) > 600 or text.count("\\n") > 8

        def build_messages(question, answer=None):
            messages = [{{"role": "user", "content": "{USER_PREFIX}\\n\\n" + str(question)}}]
            if answer is not None:
                messages.append({{"role": "assistant", "content": str(answer)}})
            return messages

        modules = ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub", "torchao", "pandas"]
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
            import pandas as pd
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            data_path = Path("{DATA_PATH}")
            data = pd.read_parquet(data_path).reset_index(drop=True).reset_index(names="row_id")
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            data["question_len"] = data["question"].astype(str).str.len()
            data["answer_len"] = data["reference_answer"].astype(str).str.len()
            pool = data[(data["question_len"] <= 350) & (data["answer_len"] <= 80)].copy()
            rng = random.Random({seed})
            selected = []
            if "category" in pool.columns:
                cats = list(pool["category"].dropna().astype(str).unique())
                rng.shuffle(cats)
                for cat in cats:
                    rows = pool[pool["category"].astype(str) == cat].head(2).to_dict(orient="records")
                    selected.extend(rows)
                    if len(selected) >= {train_rows + val_rows}:
                        break
            if len(selected) < {train_rows + val_rows}:
                selected = pool.sample({train_rows + val_rows}, random_state={seed}).to_dict(orient="records")
            selected = selected[:{train_rows + val_rows}]
            train = selected[:{train_rows}]
            val = selected[{train_rows}:{train_rows + val_rows}]
            result["raw_task_data_read_remote_only"] = True
            result["data_meta"] = {{
                "data_rows": int(len(data)),
                "pool_rows": int(len(pool)),
                "train_rows": len(train),
                "val_rows": len(val),
                "train_val_overlap_rows": len(set(r["row_id"] for r in train) & set(r["row_id"] for r in val)),
                "train_category_counts": dict(Counter(str(r.get("category", "unknown")) for r in train)),
                "val_category_counts": dict(Counter(str(r.get("category", "unknown")) for r in val)),
            }}

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
            model = AutoModelForCausalLM.from_pretrained("{model_id}", quantization_config=quant_config, device_map="auto", trust_remote_code=False)
            after_load = gpu_memory_mb()
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.train()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
            losses = []
            result["training_started"] = True
            for step in range({steps}):
                row = train[step % len(train)]
                prompt = tokenizer.apply_chat_template(build_messages(row["question"]), tokenize=False, add_generation_prompt=True)
                full = tokenizer.apply_chat_template(build_messages(row["question"], row["reference_answer"]), tokenize=False)
                full_ids = tokenizer(full, return_tensors="pt", truncation=True, max_length={max_seq_len}).input_ids.to(model.device)
                prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length={max_seq_len}).input_ids
                labels = full_ids.clone()
                labels[:, :min(prompt_ids.shape[-1], labels.shape[-1])] = -100
                loss = model(input_ids=full_ids, labels=labels).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))

            model.eval()
            exact = 0
            invalid = 0
            total_new_tokens = 0
            with torch.no_grad():
                for row in val:
                    prompt = tokenizer.apply_chat_template(build_messages(row["question"]), tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length={max_seq_len}).to(model.device)
                    output_ids = model.generate(**inputs, max_new_tokens={max_new_tokens}, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                    total_new_tokens += int(gen_ids.shape[-1])
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    invalid += int(invalid_output(text))
                    exact += int(normalize(text) == normalize(row["reference_answer"]))

            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir)
            files = [p for p in adapter_dir.rglob("*") if p.is_file()]
            adapter_size = sum(p.stat().st_size for p in files)
            shutil.rmtree(adapter_dir)
            result["adapter_scratch_deleted"] = not adapter_dir.exists()
            result["train"] = {{
                "steps": {steps},
                "losses": losses,
                "loss_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
                "trainable_params": int(trainable),
                "adapter_file_count_before_delete": len(files),
                "adapter_size_bytes_before_delete": int(adapter_size),
            }}
            result["validation"] = {{
                "val_rows": len(val),
                "exact_match_count": int(exact),
                "invalid_output_count": int(invalid),
                "changed_output_rate": None,
                "exact_stack_fire_count": None,
                "fallback_rows_evaluated": len(val),
                "avg_new_tokens": float(total_new_tokens / max(1, len(val))),
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
            result["error"] = f"{{type(exc).__name__}}: {{exc}}"
            result["traceback_tail"] = traceback.format_exc()[-2400:]
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
        "raw_examples_returned": False,
        "adapter_weights_returned": False,
        "target_dir": str(args.target_dir),
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE", "runtime": {"total_seconds": 0.0}})
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
                [sys.executable, "-c", task_probe_source(args.model_id, args.train_rows, args.val_rows, args.steps, args.max_seq_len, args.max_new_tokens, args.seed)],
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
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": "Tiny task-data SFT smoke completed with aggregate-only artifact." if ok else "Tiny task-data SFT smoke failed.",
            "install_returncode": install_code,
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "train": (probe_json or {}).get("train"),
            "validation": (probe_json or {}).get("validation"),
            "remote_runtime": (probe_json or {}).get("runtime"),
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    data_meta = summary.get("data_meta") or {}
    train = summary.get("train") or {}
    val = summary.get("validation") or {}
    runtime = summary.get("remote_runtime") or {}
    probe = summary.get("probe") or {}
    lines = [
        "# C175 Remote-Only Tiny Task-Data SFT Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Aggregate metadata only.",
        "- No raw task examples in artifact.",
        "- No model or adapter weights returned.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- install return code: `{summary.get('install_returncode')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        "",
        "## Data",
        f"- remote data read: `{probe.get('raw_task_data_read_remote_only')}`",
        f"- data rows: `{data_meta.get('data_rows')}`",
        f"- pool rows: `{data_meta.get('pool_rows')}`",
        f"- train rows: `{data_meta.get('train_rows')}`",
        f"- val rows: `{data_meta.get('val_rows')}`",
        f"- train/val overlap rows: `{data_meta.get('train_val_overlap_rows')}`",
        f"- train category counts: `{data_meta.get('train_category_counts')}`",
        f"- val category counts: `{data_meta.get('val_category_counts')}`",
        "",
        "## Train",
        f"- steps: `{train.get('steps')}`",
        f"- losses: `{train.get('losses')}`",
        f"- loss finite: `{train.get('loss_finite')}`",
        f"- trainable params: `{train.get('trainable_params')}`",
        f"- adapter size bytes before delete: `{train.get('adapter_size_bytes_before_delete')}`",
        f"- adapter scratch deleted: `{probe.get('adapter_scratch_deleted')}`",
        "",
        "## Validation",
        f"- val rows: `{val.get('val_rows')}`",
        f"- exact match count: `{val.get('exact_match_count')}`",
        f"- invalid output count: `{val.get('invalid_output_count')}`",
        f"- avg new tokens: `{val.get('avg_new_tokens')}`",
        f"- exact stack fire count: `{val.get('exact_stack_fire_count')}`",
        f"- fallback rows evaluated: `{val.get('fallback_rows_evaluated')}`",
        "",
        "## Runtime",
        f"- remote seconds: `{runtime.get('total_seconds')}`",
        f"- vram after load MB: `{runtime.get('vram_after_load_mb')}`",
        f"- vram after cleanup MB: `{runtime.get('vram_after_cleanup_mb')}`",
        "",
        "## Hygiene",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
        f"- error: `{probe.get('error')}`" if probe.get("error") else "- error: none",
        "",
        "## Next",
    ]
    if summary.get("status") == "completed":
        lines.append("Review aggregate signal before any larger SFT run; do not build a submission zip from this smoke.")
    else:
        lines.append("Keep SFT parked or fix only the setup issue shown by the aggregate report.")
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
