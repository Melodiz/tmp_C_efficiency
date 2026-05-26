from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg


EXPERIMENT_ID = "C198"
EXPERIMENT_SLUG = "C198_targeted_failure_retry"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C198_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
RETRY_PREFIX = "Реши кратко. Не объясняй подробно. В последней строке напиши только итоговый ответ."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C198 targeted retry for visible cap/loop fallback failures.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=198)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "zip": out_dir.with_suffix(".zip"),
    }


def load_solution() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("task_c_solution_module", Path("simple_solution/solution.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def risky_retry_slice(question: str, category: str) -> bool:
    bucket = agg.feature_bucket(question)
    if bucket in {
        "q_medium|cyrillic|num|noexpr|closed",
        "q_long|cyrillic|num|noexpr|closed",
        "q_short|cyrillic|num|expr|closed",
    }:
        return True
    return category in {"algebra/equations", "geometry", "history/geography/biology"}


def quality_update(counter: Counter[str], answer: str, reference: str) -> None:
    n_answer = agg.norm(answer)
    n_ref = agg.norm(reference)
    counter["rows"] += 1
    counter["exact"] += int(bool(n_ref) and n_answer == n_ref)
    counter["final_line_exact"] += int(bool(n_ref) and agg.norm(agg.final_line(answer)) == n_ref)
    counter["ref_in_output"] += int(bool(n_ref) and n_ref in n_answer)
    counter["output_in_ref"] += int(bool(n_answer) and n_answer in n_ref)


def valid_flags(answer: str, output_tokens: int, max_tokens: int) -> dict[str, bool]:
    return {
        "empty": not str(answer).strip(),
        "thinking": "<think" in str(answer) or "</think>" in str(answer),
        "hit_max_tokens": output_tokens >= max_tokens,
        "repetition_loop": probe.has_repetition_loop(str(answer)),
    }


def run_validation(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "retry_prefix": RETRY_PREFIX,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    agg.install_final_path_dependencies()
    solution = load_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    base_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, solution.USER_PREFIX) for row in rows
    ]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in base_prompts]

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    startup_t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=solution.MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=args.seed,
        trust_remote_code=False,
    )
    startup_s = time.perf_counter() - startup_t0
    sampling = SamplingParams(temperature=0.0, max_tokens=solution.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)

    base_t0 = time.perf_counter()
    base_outputs = llm.generate(base_prompts, sampling_params=sampling)
    base_generation_s = time.perf_counter() - base_t0

    base_records: list[dict[str, Any]] = []
    retry_indices: list[int] = []
    for idx, out in enumerate(base_outputs):
        completion = out.outputs[0]
        answer = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(answer).input_ids)
        row = rows[idx]
        handler, base_final = agg.first_handler(solution, str(row["question"]), answer)
        flags = valid_flags(answer, out_tokens, solution.MAX_NEW_TOKENS)
        should_retry = (
            handler == "fallback_model"
            and risky_retry_slice(str(row["question"]), str(row.get("category", "unknown")))
            and (flags["hit_max_tokens"] or flags["repetition_loop"])
        )
        if should_retry:
            retry_indices.append(idx)
        base_records.append(
            {
                "answer": answer,
                "answer_tokens": out_tokens,
                "final": base_final,
                "handler": handler,
                "flags": flags,
            }
        )

    retry_generation_s = 0.0
    retry_records: dict[int, dict[str, Any]] = {}
    if retry_indices:
        retry_prompts = [
            probe.apply_user_only_template(tokenizer, str(rows[idx]["question"]), True, RETRY_PREFIX)
            for idx in retry_indices
        ]
        retry_t0 = time.perf_counter()
        retry_outputs = llm.generate(retry_prompts, sampling_params=sampling)
        retry_generation_s = time.perf_counter() - retry_t0
        for idx, out in zip(retry_indices, retry_outputs):
            completion = out.outputs[0]
            answer = completion.text.strip()
            token_ids = getattr(completion, "token_ids", None)
            out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(answer).input_ids)
            handler, retry_final = agg.first_handler(solution, str(rows[idx]["question"]), answer)
            retry_records[idx] = {
                "answer_tokens": out_tokens,
                "final": retry_final,
                "handler": handler,
                "flags": valid_flags(answer, out_tokens, solution.MAX_NEW_TOKENS),
            }
    sampler.stop()

    base_quality: Counter[str] = Counter()
    retry_quality: Counter[str] = Counter()
    accepted_quality: Counter[str] = Counter()
    base_validity: Counter[str] = Counter()
    accepted_validity: Counter[str] = Counter()
    retry_attempt_quality: Counter[str] = Counter()
    retry_accept_reasons: Counter[str] = Counter()
    retry_by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    retry_by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for idx, row in enumerate(rows):
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(str(row["question"]))
        base_rec = base_records[idx]
        selected = base_rec
        retried = idx in retry_records
        accepted = False
        if retried:
            retry_rec = retry_records[idx]
            retry_flags = retry_rec["flags"]
            base_flags = base_rec["flags"]
            quality_update(retry_attempt_quality, retry_rec["final"], reference)
            retry_by_category[category]["attempts"] += 1
            retry_by_bucket[bucket]["attempts"] += 1
            accepted = (
                not retry_flags["empty"]
                and not retry_flags["thinking"]
                and not retry_flags["hit_max_tokens"]
                and not retry_flags["repetition_loop"]
                and int(retry_rec["answer_tokens"]) <= max(96, int(base_rec["answer_tokens"]) // 2)
            )
            if accepted:
                selected = retry_rec
                retry_accept_reasons["accepted_short_valid_retry"] += 1
                retry_by_category[category]["accepted"] += 1
                retry_by_bucket[bucket]["accepted"] += 1
            else:
                retry_accept_reasons["rejected_retry_invalid_or_too_long"] += 1
            retry_by_category[category]["base_cap"] += int(base_flags["hit_max_tokens"])
            retry_by_category[category]["base_rep"] += int(base_flags["repetition_loop"])
            retry_by_bucket[bucket]["base_cap"] += int(base_flags["hit_max_tokens"])
            retry_by_bucket[bucket]["base_rep"] += int(base_flags["repetition_loop"])

        quality_update(base_quality, base_rec["final"], reference)
        quality_update(accepted_quality, selected["final"], reference)
        quality_update(retry_quality, retry_records[idx]["final"] if retried else base_rec["final"], reference)
        for key, flags, counter in (
            ("base", base_rec["flags"], base_validity),
            ("accepted", selected["flags"], accepted_validity),
        ):
            counter["rows"] += 1
            counter[f"{key}_empty"] += int(flags["empty"])
            counter[f"{key}_thinking"] += int(flags["thinking"])
            counter[f"{key}_hit_max_tokens"] += int(flags["hit_max_tokens"])
            counter[f"{key}_repetition_loop"] += int(flags["repetition_loop"])

    total_generation_s = base_generation_s + retry_generation_s
    projected_generation_4000_s = (total_generation_s / max(1, len(rows))) * 4000
    projected_total_4000_s = startup_s + projected_generation_4000_s
    base_rates = agg.rates({"base": base_quality})["base"]
    retry_all_rates = agg.rates({"retry_all": retry_quality})["retry_all"]
    accepted_rates = agg.rates({"accepted": accepted_quality})["accepted"]
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE",
            "reason": "Targeted retry aggregate validation completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"solution": "ok", "pymorphy_available": bool(solution.get_morph_analyzer())},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "base_generation_s": base_generation_s,
                "retry_generation_s": retry_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {
                "avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens)),
                "retry_attempts": len(retry_indices),
            },
            "base_quality": base_rates,
            "retry_all_quality": retry_all_rates,
            "accepted_policy_quality": accepted_rates,
            "base_validity": {k: int(v) for k, v in base_validity.items()},
            "accepted_policy_validity": {k: int(v) for k, v in accepted_validity.items()},
            "retry_attempt_quality": agg.rates({"retry_attempt": retry_attempt_quality})["retry_attempt"]
            if retry_attempt_quality.get("rows")
            else {"rows": 0},
            "retry_policy": {
                "attempts": len(retry_indices),
                "accepted": int(retry_accept_reasons.get("accepted_short_valid_retry", 0)),
                "accept_reasons": {k: int(v) for k, v in retry_accept_reasons.items()},
                "by_category": {k: dict(v) for k, v in sorted(retry_by_category.items())},
                "by_bucket": {k: dict(v) for k, v in sorted(retry_by_bucket.items(), key=lambda kv: -kv[1].get("attempts", 0))},
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C198 Targeted Failure Retry",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- imports: `{summary.get('imports')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Quality",
        f"- base final stack: `{summary.get('base_quality')}`",
        f"- retry all: `{summary.get('retry_all_quality')}`",
        f"- accepted policy: `{summary.get('accepted_policy_quality')}`",
        f"- retry attempts only: `{summary.get('retry_attempt_quality')}`",
        "",
        "## Validity",
        f"- base: `{summary.get('base_validity')}`",
        f"- accepted policy: `{summary.get('accepted_policy_validity')}`",
        "",
        "## Retry Policy",
        f"`{summary.get('retry_policy')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- outputs returned: `{summary.get('outputs_returned')}`",
        f"- model weights returned: `{summary.get('model_weights_returned')}`",
        f"- training started: `{summary.get('training_started')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    summary = run_validation(args, paths)
    io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    agg.base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
