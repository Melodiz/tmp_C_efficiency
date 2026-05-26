from __future__ import annotations

import argparse
import json
import os
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


EXPERIMENT_ID = "C193"
EXPERIMENT_SLUG = "C193_current_stack_aggregate_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C193_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
MORPH_PACKAGES = ("pymorphy3==2.0.6", "pymorphy3-dicts-ru", "razdel==0.5.0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C193 aggregate-only validation of the current final stack.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=193)
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


def install_final_path_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *MORPH_PACKAGES])


def probe_source(sample_size: int, seed: int) -> str:
    return textwrap.dedent(
        f"""
        import importlib.util
        import json
        import os
        import re
        import shutil
        import time
        import traceback
        from collections import Counter, defaultdict
        from pathlib import Path

        import pandas as pd
        from huggingface_hub import snapshot_download

        MODEL_ID = {MODEL_ID!r}
        SAMPLE_SIZE = {int(sample_size)}
        SEED = {int(seed)}
        DATA_PATH = Path("data/dataset_ml_challenge.parquet")

        result = {{
            "status": "failed",
            "leaderboard_submission": False,
            "raw_task_data_read_remote_only": False,
            "raw_examples_returned": False,
            "row_ids_returned": False,
            "outputs_returned": False,
            "model_loaded": False,
            "model_weights_returned": False,
            "training_started": False,
            "adapter_weights_returned": False,
            "sample_meta": {{}},
            "imports": {{}},
            "runtime": {{}},
            "quality": {{}},
            "by_category": {{}},
            "by_bucket": {{}},
            "by_target_label": {{}},
            "by_first_handler": {{}},
            "handler_counts": {{}},
            "validity": {{}},
        }}

        def compact_counter(counter):
            return {{str(k): int(v) for k, v in counter.items()}}

        def answer_only_label(value):
            text = str(value).strip()
            text = re.sub(r"^(ответ|итог|answer)\\s*[:：-]\\s*", "", text, flags=re.IGNORECASE).strip()
            lines = [part.strip() for part in text.splitlines() if part.strip()]
            if not lines:
                return "empty"
            target = lines[0]
            if len(lines) > 1:
                return "multiline"
            if len(target) > 80:
                return "long"
            if len(target.split()) > 14:
                return "essay_like"
            return "ok"

        def script_bucket(text):
            cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in str(text))
            lat = sum("a" <= ch.lower() <= "z" for ch in str(text))
            if cyr > lat:
                return "cyrillic"
            if lat > cyr:
                return "latin"
            return "mixed_or_symbolic"

        def length_bucket(text):
            n = len(str(text))
            if n <= 80:
                return "q_short"
            if n <= 180:
                return "q_medium"
            if n <= 350:
                return "q_long"
            return "q_very_long"

        def feature_bucket(text):
            q = str(text).lower().replace("ё", "е")
            return "|".join([
                length_bucket(q),
                script_bucket(q),
                "num" if re.search(r"\\d", q) else "nonnum",
                "expr" if re.search(r"\\d\\s*[+*×xх/:=-]\\s*\\d", q) else "noexpr",
                "open" if re.search(r"\\b(объясн|почему|напишите|сочин|эссе|опишите|перечислите|составьте|расскажите|докажите|explain|write|describe|list)\\b", q) else "closed",
            ])

        def final_line(text):
            s = str(text).strip()
            matches = re.findall(r"(?:Итоговый ответ|Ответ|Answer)\\s*[:：]\\s*(.+)", s, flags=re.IGNORECASE)
            if matches:
                return matches[-1].strip()
            lines = [line.strip() for line in s.splitlines() if line.strip()]
            return lines[-1] if lines else s

        def norm(text):
            s = final_line(text)
            s = s.lower().replace("ё", "е")
            s = re.sub(r"^(ответ|итоговый ответ|answer)\\s*[:：-]\\s*", "", s, flags=re.IGNORECASE)
            s = s.replace("−", "-").replace(",", ".")
            s = re.sub(r"\\s+", " ", s)
            s = re.sub(r"[^0-9a-zа-я+\\-*/=.,%√²³() ]+", "", s)
            return s.strip()

        def has_repetition_loop(text):
            parts = [p.strip().lower() for p in re.split(r"[,;\\n]+", str(text)) if p.strip()]
            if len(parts) >= 12 and len(set(parts[-12:])) <= 4:
                return True
            words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", str(text).lower())
            return len(words) >= 40 and len(set(words[-30:])) <= 6

        def update_metric(counter, answer, reference, base_answer=None):
            answer_s = str(answer or "")
            ref_s = str(reference or "")
            n_answer = norm(answer_s)
            n_ref = norm(ref_s)
            n_base = norm(base_answer or "")
            counter["rows"] += 1
            counter["exact"] += int(n_answer == n_ref and bool(n_ref))
            counter["final_line_exact"] += int(norm(final_line(answer_s)) == n_ref and bool(n_ref))
            counter["ref_in_output"] += int(bool(n_ref) and n_ref in n_answer)
            counter["output_in_ref"] += int(bool(n_answer) and n_answer in n_ref)
            counter["base_exact"] += int(n_base == n_ref and bool(n_ref))

        def stratified_sample(data):
            if "category" not in data.columns:
                return data.sample(n=min(SAMPLE_SIZE, len(data)), random_state=SEED)
            groups = []
            per_group = max(1, SAMPLE_SIZE // max(1, data["category"].nunique()))
            for _, group in data.groupby("category", dropna=False):
                groups.append(group.sample(n=min(per_group, len(group)), random_state=SEED))
            sample = pd.concat(groups, ignore_index=True)
            if len(sample) < min(SAMPLE_SIZE, len(data)):
                remaining = data.drop(index=sample.index, errors="ignore")
                if len(remaining):
                    extra = remaining.sample(n=min(SAMPLE_SIZE - len(sample), len(remaining)), random_state=SEED + 1)
                    sample = pd.concat([sample, extra], ignore_index=True)
            if len(sample) > SAMPLE_SIZE:
                sample = sample.sample(n=SAMPLE_SIZE, random_state=SEED + 2)
            return sample.reset_index(drop=True)

        def aggregate_table(table):
            out = {{}}
            for key, counts in table.items():
                rows = int(counts.get("rows", 0))
                item = {{k: int(v) for k, v in counts.items()}}
                if rows:
                    for name in ("exact", "final_line_exact", "ref_in_output", "output_in_ref", "base_exact"):
                        item[name + "_rate"] = item.get(name, 0) / rows
                out[str(key)] = item
            return out

        try:
            started = time.perf_counter()
            solution_path = Path("simple_solution/solution.py")
            weights_dir = Path("simple_solution/weights")
            if weights_dir.exists():
                shutil.rmtree(weights_dir)
            snapshot_download(repo_id=MODEL_ID, local_dir=weights_dir, local_dir_use_symlinks=False)

            spec = importlib.util.spec_from_file_location("task_c_solution_module", solution_path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            result["imports"]["solution"] = "ok"
            result["imports"]["pymorphy_available"] = bool(module.get_morph_analyzer())

            handlers = [
                ("expression_substitution", module.expression_substitution_answer),
                ("algebra_equation", module.algebra_equation_answer),
                ("exact_numeric", module.exact_numeric_answer),
                ("direct_arithmetic", module.direct_arithmetic_answer),
                ("chemistry_stoichiometry", module.chemistry_stoichiometry_answer),
                ("geometry_exact", module.geometry_exact_answer),
                ("formulaic_math_physics", module.formulaic_math_physics_answer),
                ("structured_school_task", module.structured_school_task_answer),
                ("calculator_written_arithmetic", module.calculator_written_arithmetic_answer),
                ("russian_morph_grammar", module.russian_morph_grammar_answer),
                ("quantity_conversion", module.quantity_conversion_answer),
                ("km_meters", module.km_meters_answer),
            ]

            data = pd.read_parquet(DATA_PATH).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            result["raw_task_data_read_remote_only"] = True
            sample = stratified_sample(data)
            result["sample_meta"] = {{
                "available_rows": int(len(data)),
                "sample_rows": int(len(sample)),
                "sample_size_requested": SAMPLE_SIZE,
                "seed": SEED,
                "category_counts": compact_counter(Counter(sample.get("category", pd.Series(["unknown"] * len(sample))).fillna("unknown"))),
            }}

            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams

            tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), use_fast=True)
            prompts = [module.build_prompt(tokenizer, str(row["question"])) for _, row in sample.iterrows()]
            llm = LLM(
                model=str(weights_dir),
                dtype="float16",
                quantization="awq_marlin",
                max_model_len=module.MAX_MODEL_LEN,
                gpu_memory_utilization=0.9,
                tokenizer_mode="auto",
                seed=0,
            )
            result["model_loaded"] = True
            sampling = SamplingParams(temperature=0.0, max_tokens=module.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
            gen_t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling)
            generation_s = time.perf_counter() - gen_t0

            quality = Counter()
            validity = Counter()
            by_category = defaultdict(Counter)
            by_bucket = defaultdict(Counter)
            by_target_label = defaultdict(Counter)
            by_handler = defaultdict(Counter)
            handler_counts = Counter()

            for (_, row), out in zip(sample.iterrows(), outputs):
                question = str(row["question"])
                reference = str(row["reference_answer"])
                completion = out.outputs[0]
                base_answer = str(completion.text or "").strip()
                token_ids = getattr(completion, "token_ids", None)
                output_tokens = len(token_ids) if token_ids is not None else len(tokenizer(base_answer).input_ids)
                first_handler = "fallback_model"
                final_answer = base_answer
                for name, func in handlers:
                    value = func(question)
                    if value is not None:
                        first_handler = name
                        final_answer = value
                        break
                if first_handler == "fallback_model":
                    final_answer = module.dedup_comma_loop(final_answer) or final_answer
                    final_answer = module.cleanup_english_cloze_answer(question, final_answer) or final_answer
                handler_counts[first_handler] += 1

                validity["rows"] += 1
                validity["base_hit_max_tokens"] += int(output_tokens >= module.MAX_NEW_TOKENS)
                validity["base_empty"] += int(not base_answer)
                validity["base_thinking_trace"] += int("<think" in base_answer or "</think>" in base_answer)
                validity["base_repetition_loop"] += int(has_repetition_loop(base_answer))
                validity["final_empty"] += int(not str(final_answer).strip())
                validity["deterministic_first_fire"] += int(first_handler != "fallback_model")
                validity["fallback_model"] += int(first_handler == "fallback_model")

                bucket = feature_bucket(question)
                label = answer_only_label(reference)
                category = str(row.get("category", "unknown"))
                update_metric(quality, final_answer, reference, base_answer)
                update_metric(by_category[category], final_answer, reference, base_answer)
                update_metric(by_bucket[bucket], final_answer, reference, base_answer)
                update_metric(by_target_label[label], final_answer, reference, base_answer)
                update_metric(by_handler[first_handler], final_answer, reference, base_answer)

            rows = int(quality.get("rows", 0))
            result["quality"] = aggregate_table({{"overall": quality}})["overall"]
            result["by_category"] = dict(sorted(aggregate_table(by_category).items()))
            result["by_bucket"] = dict(sorted(aggregate_table(by_bucket).items(), key=lambda kv: -kv[1].get("rows", 0))[:40])
            result["by_target_label"] = dict(sorted(aggregate_table(by_target_label).items()))
            result["by_first_handler"] = dict(sorted(aggregate_table(by_handler).items(), key=lambda kv: -kv[1].get("rows", 0)))
            result["handler_counts"] = compact_counter(handler_counts)
            result["validity"] = compact_counter(validity)
            result["runtime"] = {{
                "total_seconds": time.perf_counter() - started,
                "generation_seconds": generation_s,
                "projected_generation_4000_seconds": generation_s / max(1, rows) * 4000,
            }}
            result["status"] = "completed"
        except Exception as exc:
            result["error"] = f"{{type(exc).__name__}}: {{exc}}"
            result["traceback_tail"] = traceback.format_exc()[-2400:]

        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("status") == "completed" else 2)
        """
    ).strip()


def run_validation(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE", "runtime": {"total_seconds": 0.0}})
        return summary
    try:
        install_final_path_dependencies()
        env = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        proc = subprocess.run(
            [sys.executable, "-c", probe_source(args.sample_size, args.seed)],
            check=False,
            text=True,
            capture_output=True,
            timeout=3600,
            env=env,
        )
        probe_code = proc.returncode
        probe_log = ((proc.stdout or "") + (proc.stderr or "")).strip()
    except Exception as exc:
        probe_code = 999
        probe_log = f"{type(exc).__name__}: {exc}"
    paths["probe_log"].write_text(probe_log + "\n", encoding="utf-8")
    probe_json = None
    try:
        probe_json = c171.parse_probe_json(probe_log) or json.loads(probe_log)
        base.write_json(paths["probe"], probe_json)
    except Exception:
        probe_json = None
    ok = probe_code == 0 and probe_json is not None and probe_json.get("status") == "completed"
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "INVESTIGATE",
            "reason": "Aggregate current-stack validation completed." if ok else "Aggregate current-stack validation failed.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "imports": (probe_json or {}).get("imports"),
            "sample_meta": (probe_json or {}).get("sample_meta"),
            "quality": (probe_json or {}).get("quality"),
            "validity": (probe_json or {}).get("validity"),
            "handler_counts": (probe_json or {}).get("handler_counts"),
            "by_category": (probe_json or {}).get("by_category"),
            "by_bucket": (probe_json or {}).get("by_bucket"),
            "by_target_label": (probe_json or {}).get("by_target_label"),
            "by_first_handler": (probe_json or {}).get("by_first_handler"),
            "model_loaded": bool((probe_json or {}).get("model_loaded")),
            "raw_task_data_read_remote_only": bool((probe_json or {}).get("raw_task_data_read_remote_only")),
            "runtime": {"total_seconds": time.time() - start, "probe_runtime": (probe_json or {}).get("runtime")},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C193 Current-Stack Aggregate Validation Smoke",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Run the unchanged current final stack on a larger remote sample.",
        "- Return only aggregate metrics; no raw prompts, references, outputs, row ids, cached datasets, model weights, or adapter weights.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        f"- imports: `{summary.get('imports')}`",
        f"- model loaded: `{summary.get('model_loaded')}`",
        "",
        "## Sample",
        f"`{summary.get('sample_meta')}`",
        "",
        "## Overall Quality",
        f"`{summary.get('quality')}`",
        "",
        "## Validity",
        f"`{summary.get('validity')}`",
        "",
        "## Handler Counts",
        f"`{summary.get('handler_counts')}`",
        "",
        "## By Target Label",
        f"`{summary.get('by_target_label')}`",
        "",
        "## By First Handler",
        f"`{summary.get('by_first_handler')}`",
        "",
        "## Top Buckets",
    ]
    for key, item in list((summary.get("by_bucket") or {}).items())[:20]:
        lines.append(f"- `{key}`: `{item}`")
    lines.extend(
        [
            "",
            "## Category Metrics",
            f"`{summary.get('by_category')}`",
            "",
            "## Hygiene",
            f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
            f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
            f"- row ids returned: `{summary.get('row_ids_returned')}`",
            f"- outputs returned: `{summary.get('outputs_returned')}`",
            f"- model weights returned: `{summary.get('model_weights_returned')}`",
            f"- training started: `{summary.get('training_started')}`",
            f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
            "",
            "## Next",
            "Use the largest weak aggregate bucket/category to choose one broad follow-up; do not build a submission zip from this measurement alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    for key in ("reports_dir", "results_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    summary = run_validation(args, paths)
    base.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
