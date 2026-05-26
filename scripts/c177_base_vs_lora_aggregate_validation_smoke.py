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
from c173_qwen3_8b_synthetic_qlora_smoke import install_target


EXPERIMENT_ID = "C177"
EXPERIMENT_SLUG = "C177_base_vs_lora_aggregate_validation_smoke"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C177_artifacts"
DEFAULT_TARGET_DIR = Path("/content/c177_train_site")
REMOTE_ADAPTER_DIR = Path("/content/c177_adapter_scratch")
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")
USER_PREFIX = "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C177 base-vs-LoRA aggregate validation smoke.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--val-rows", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=177)
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
        import random
        import re
        import shutil
        import subprocess
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
            return (not text) or len(text) > 700 or text.count("\\n") > 10

        def shape_bucket(row):
            q = str(row["question"])
            a = str(row["reference_answer"])
            cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in q)
            lat = sum("a" <= ch.lower() <= "z" for ch in q)
            if cyr > lat:
                script = "cyrillic"
            elif lat > cyr:
                script = "latin"
            else:
                script = "mixed_or_symbolic"
            def length_bucket(n):
                if n <= 80:
                    return "short"
                if n <= 180:
                    return "medium"
                return "long"
            def answer_bucket(n):
                if n <= 12:
                    return "tiny"
                if n <= 40:
                    return "short"
                return "long"
            return f"q_{{length_bucket(len(q))}}|a_{{answer_bucket(len(a))}}|{{script}}"

        def build_messages(question, answer=None):
            messages = [{{"role": "user", "content": "{USER_PREFIX}\\n\\n" + str(question)}}]
            if answer is not None:
                messages.append({{"role": "assistant", "content": str(answer)}})
            return messages

        def evaluate(model, tokenizer, rows):
            stats = Counter()
            buckets = {{}}
            outputs = []
            total_new_tokens = 0
            for row in rows:
                prompt = tokenizer.apply_chat_template(build_messages(row["question"]), tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length={max_seq_len}).to(model.device)
                output_ids = model.generate(**inputs, max_new_tokens={max_new_tokens}, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                new_tokens = int(gen_ids.shape[-1])
                total_new_tokens += new_tokens
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                exact = normalize(text) == normalize(row["reference_answer"])
                invalid = invalid_output(text)
                cap_hit = new_tokens >= {max_new_tokens}
                bucket = shape_bucket(row)
                bucket_stats = buckets.setdefault(bucket, Counter())
                for key, value in (("exact", exact), ("invalid", invalid), ("cap_hit", cap_hit)):
                    stats[key] += int(value)
                    bucket_stats[key] += int(value)
                stats["rows"] += 1
                bucket_stats["rows"] += 1
                outputs.append({{"norm": normalize(text), "exact": exact, "invalid": invalid, "cap_hit": cap_hit, "new_tokens": new_tokens, "bucket": bucket}})
            stats["avg_new_tokens_x1000"] = int(round(1000 * total_new_tokens / max(1, len(rows))))
            return dict(stats), {{k: dict(v) for k, v in buckets.items()}}, outputs

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

            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True).reset_index(names="row_id")
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            data["question_len"] = data["question"].astype(str).str.len()
            data["answer_len"] = data["reference_answer"].astype(str).str.len()
            pool = data[(data["question_len"] <= 350) & (data["answer_len"] <= 80)].copy()
            selected = pool.sample({train_rows + val_rows}, random_state={seed}).to_dict(orient="records")
            train = selected[:{train_rows}]
            val = selected[{train_rows}:{train_rows + val_rows}]
            result["raw_task_data_read_remote_only"] = True
            result["data_meta"] = {{
                "data_rows": int(len(data)),
                "pool_rows": int(len(pool)),
                "train_rows": len(train),
                "val_rows": len(val),
                "train_val_overlap_rows": len(set(r["row_id"] for r in train) & set(r["row_id"] for r in val)),
                "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),
                "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),
            }}

            before_load = gpu_memory_mb()
            tokenizer = AutoTokenizer.from_pretrained("{model_id}", use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained("{model_id}", quantization_config=quant_config, device_map="auto", trust_remote_code=False)
            after_load = gpu_memory_mb()
            model.eval()
            base_stats, base_buckets, base_outputs = evaluate(model, tokenizer, val)

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
            lora_stats, lora_buckets, lora_outputs = evaluate(model, tokenizer, val)
            pairs = list(zip(base_outputs, lora_outputs))
            pair_stats = Counter()
            pair_buckets = {{}}
            for base_out, lora_out in pairs:
                changed = base_out["norm"] != lora_out["norm"]
                both_exact = base_out["exact"] and lora_out["exact"]
                base_only = base_out["exact"] and not lora_out["exact"]
                lora_only = lora_out["exact"] and not base_out["exact"]
                both_wrong_same = (not base_out["exact"]) and (not lora_out["exact"]) and (not changed)
                both_wrong_changed = (not base_out["exact"]) and (not lora_out["exact"]) and changed
                for key, value in (
                    ("changed_output_count", changed),
                    ("both_exact_count", both_exact),
                    ("base_only_exact_count", base_only),
                    ("lora_only_exact_count", lora_only),
                    ("both_wrong_same_count", both_wrong_same),
                    ("both_wrong_changed_count", both_wrong_changed),
                ):
                    pair_stats[key] += int(value)
                bucket = base_out["bucket"]
                b = pair_buckets.setdefault(bucket, Counter())
                b["rows"] += 1
                b["changed_output_count"] += int(changed)
                b["base_only_exact_count"] += int(base_only)
                b["lora_only_exact_count"] += int(lora_only)

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
                "base_exact_count": int(base_stats.get("exact", 0)),
                "lora_exact_count": int(lora_stats.get("exact", 0)),
                "both_exact_count": int(pair_stats.get("both_exact_count", 0)),
                "base_only_exact_count": int(pair_stats.get("base_only_exact_count", 0)),
                "lora_only_exact_count": int(pair_stats.get("lora_only_exact_count", 0)),
                "both_wrong_same_count": int(pair_stats.get("both_wrong_same_count", 0)),
                "both_wrong_changed_count": int(pair_stats.get("both_wrong_changed_count", 0)),
                "changed_output_count": int(pair_stats.get("changed_output_count", 0)),
                "base_invalid_count": int(base_stats.get("invalid", 0)),
                "lora_invalid_count": int(lora_stats.get("invalid", 0)),
                "base_cap_hit_count": int(base_stats.get("cap_hit", 0)),
                "lora_cap_hit_count": int(lora_stats.get("cap_hit", 0)),
                "base_avg_new_tokens": float(base_stats.get("avg_new_tokens_x1000", 0) / 1000.0),
                "lora_avg_new_tokens": float(lora_stats.get("avg_new_tokens_x1000", 0) / 1000.0),
                "base_shape_buckets": base_buckets,
                "lora_shape_buckets": lora_buckets,
                "pair_shape_buckets": {{k: dict(v) for k, v in pair_buckets.items()}},
            }}
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result["runtime"] = {{"total_seconds": time.time() - start, "vram_before_load_mb": before_load, "vram_after_load_mb": after_load, "vram_after_cleanup_mb": gpu_memory_mb()}}
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
                timeout=3000,
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
    val = (probe_json or {}).get("validation") or {}
    lora_delta = int(val.get("lora_exact_count") or 0) - int(val.get("base_exact_count") or 0)
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": "Base-vs-LoRA aggregate validation completed." if ok else "Base-vs-LoRA validation smoke failed.",
            "install_returncode": install_code,
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "train": (probe_json or {}).get("train"),
            "validation": val,
            "lora_exact_delta": lora_delta,
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
        "# C177 Base-vs-LoRA Aggregate Validation Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Aggregate metadata only.",
        "- No raw task questions, answers, outputs, row ids, model weights, or adapter weights returned.",
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
        f"- train shape counts: `{data_meta.get('train_shape_counts')}`",
        f"- val shape counts: `{data_meta.get('val_shape_counts')}`",
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
        f"- base exact count: `{val.get('base_exact_count')}`",
        f"- LoRA exact count: `{val.get('lora_exact_count')}`",
        f"- LoRA exact delta: `{summary.get('lora_exact_delta')}`",
        f"- both exact count: `{val.get('both_exact_count')}`",
        f"- base-only exact count: `{val.get('base_only_exact_count')}`",
        f"- LoRA-only exact count: `{val.get('lora_only_exact_count')}`",
        f"- both wrong same count: `{val.get('both_wrong_same_count')}`",
        f"- both wrong changed count: `{val.get('both_wrong_changed_count')}`",
        f"- changed output count: `{val.get('changed_output_count')}`",
        f"- base invalid/cap hit: `{val.get('base_invalid_count')}` / `{val.get('base_cap_hit_count')}`",
        f"- LoRA invalid/cap hit: `{val.get('lora_invalid_count')}` / `{val.get('lora_cap_hit_count')}`",
        f"- base avg new tokens: `{val.get('base_avg_new_tokens')}`",
        f"- LoRA avg new tokens: `{val.get('lora_avg_new_tokens')}`",
        f"- pair shape buckets: `{val.get('pair_shape_buckets')}`",
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
        lines.append("Use the aggregate base-vs-LoRA delta and changed-output counts to decide whether to scale SFT or kill the branch.")
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
