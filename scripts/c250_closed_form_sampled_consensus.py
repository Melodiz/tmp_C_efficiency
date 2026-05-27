from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c216_qwen3_14b_paired_bucket_aggregate as paired
import c246_failure_gated_same_model_512 as c246


EXPERIMENT_ID = "C250"
EXPERIMENT_SLUG = "C250_closed_form_sampled_consensus"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C250_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C250 same-model sampled consensus over C111 closed-form prompts.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=250)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--sample-max-tokens", type=int, default=160)
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


def is_closed_form_question(question: str) -> bool:
    q = question.lower()
    if any(word in q for word in ("перевед", "translate", "essay", "сочинен", "напишите текст")):
        return False
    if len(question) > 650:
        return False
    has_number = bool(re.search(r"\d|[=+\-*/^<>%]", question))
    has_unit = bool(
        re.search(
            r"\b(см|мм|м|км|дм|кг|г|мг|л|мл|байт|бит|кбайт|градус|радиан|час|мин|сек)\b",
            q,
        )
    )
    has_closed_cue = any(
        cue in q
        for cue in (
            "сколько",
            "найдите",
            "вычисл",
            "решите",
            "ответ",
            "чему рав",
            "calculate",
            "solve",
            "find",
            "what is",
            "how many",
            "which",
        )
    )
    return has_closed_cue and (has_number or has_unit)


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?i)^(ответ|answer)\s*[:：-]\s*", "", text)
    text = text.splitlines()[-1].strip() if "\n" in text else text
    text = text.strip(" .;,:")
    text = text.replace("−", "-").replace(",", ".")
    text = re.sub(r"\s+", " ", text.lower())
    text = re.sub(r"\s*([=+\-*/^])\s*", r"\1", text)
    return text


def consensus_eval(
    solution: Any,
    tokenizer: Any,
    row: dict[str, Any],
    out: Any,
    max_tokens: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    first_by_norm: dict[str, dict[str, Any]] = {}
    for completion in out.outputs:
        text = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(text).input_ids)
        final, handler = rollback.c111_stack(solution, str(row["question"]), text)
        flags = retry_base.valid_flags(text, out_tokens, max_tokens)
        norm = normalize_answer(final)
        item = {"final": final, "handler": handler, "out_tokens": out_tokens, "flags": flags, "norm": norm}
        candidates.append(item)
        if not c246.invalid(item) and norm and len(norm) <= 80:
            counts[norm] += 1
            first_by_norm.setdefault(norm, item)
    agreed = [norm for norm, count in counts.items() if count >= 2]
    if not agreed:
        return {"accepted": False, "item": None, "candidate_count": len(candidates), "agreement": None}
    agreed.sort(key=lambda norm: (-counts[norm], len(norm), norm))
    item = dict(first_by_norm[agreed[0]])
    item["handler"] = "sampled_consensus"
    item["out_tokens"] = min(int(candidate["out_tokens"]) for candidate in candidates)
    return {"accepted": True, "item": item, "candidate_count": len(candidates), "agreement": agreed[0]}


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
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
        "c111_commit": rollback.C111_COMMIT,
        "model_id": MODEL_ID,
        "route": "Route closed-form prompts to same-model sampled candidates; replace C111 only on strict normalized agreement.",
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in prompts]

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    startup_t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=c111.MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=args.seed,
        trust_remote_code=False,
    )
    startup_s = time.perf_counter() - startup_t0

    baseline_sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    baseline_t0 = time.perf_counter()
    baseline_outputs = llm.generate(prompts, sampling_params=baseline_sampling)
    baseline_generation_s = time.perf_counter() - baseline_t0
    baseline_eval = [
        c246.row_eval(c111, tokenizer, row, out, c111.MAX_NEW_TOKENS) for row, out in zip(rows, baseline_outputs)
    ]

    routed_indices = [idx for idx, row in enumerate(rows) if is_closed_form_question(str(row["question"]))]
    routed_prompts = [prompts[idx] for idx in routed_indices]
    routed_rows = [rows[idx] for idx in routed_indices]
    sample_generation_s = 0.0
    consensus_results: list[dict[str, Any]] = []
    if routed_prompts:
        sample_sampling = SamplingParams(
            temperature=0.7,
            max_tokens=args.sample_max_tokens,
            top_p=0.9,
            top_k=40,
            n=args.num_samples,
        )
        sample_t0 = time.perf_counter()
        sample_outputs = llm.generate(routed_prompts, sampling_params=sample_sampling)
        sample_generation_s = time.perf_counter() - sample_t0
        consensus_results = [
            consensus_eval(c111, tokenizer, row, out, args.sample_max_tokens)
            for row, out in zip(routed_rows, sample_outputs)
        ]
    sampler.stop()

    selected_eval = list(baseline_eval)
    consensus_eval_items: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    for idx, result in zip(routed_indices, consensus_results):
        route_counts["sampled_rows"] += 1
        route_counts["sampled_candidates"] += int(result["candidate_count"])
        if result["accepted"] and result["item"] is not None:
            selected_eval[idx] = result["item"]
            consensus_eval_items.append(result["item"])
            route_counts["accepted_consensus"] += 1
            route_counts["changed_from_c111"] += int(
                normalize_answer(result["item"]["final"]) != normalize_answer(baseline_eval[idx]["final"])
            )
        else:
            consensus_eval_items.append(baseline_eval[idx])
            route_counts["rejected_consensus"] += 1
    route_counts["rows"] = len(rows)
    route_counts["routed_rows"] = len(routed_rows)

    baseline_all = c246.quality_table(rows, baseline_eval)
    selected_all = c246.quality_table(rows, selected_eval)
    baseline_routed = c246.quality_table(routed_rows, [baseline_eval[idx] for idx in routed_indices])
    selected_routed = c246.quality_table(routed_rows, [selected_eval[idx] for idx in routed_indices])

    projected_total_4000_s = startup_s + ((baseline_generation_s + sample_generation_s) / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "C111 closed-form sampled consensus aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "baseline_generation_s": baseline_generation_s,
                "sample_generation_s": sample_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {"avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens))},
            "sampling": {
                "num_samples": args.num_samples,
                "sample_max_tokens": args.sample_max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            },
            "route_counts": {k: int(v) for k, v in route_counts.items()},
            "baseline_all": baseline_all,
            "selected_all": selected_all,
            "delta_selected_minus_baseline_all": paired.overall_delta(selected_all, baseline_all),
            "baseline_routed_only": baseline_routed,
            "selected_routed_only": selected_routed,
            "delta_selected_minus_baseline_routed_only": paired.overall_delta(selected_routed, baseline_routed),
            "delta_by_category_selected": paired.keyed_delta(selected_all, baseline_all, "by_category"),
            "delta_by_bucket_selected": paired.keyed_delta(selected_all, baseline_all, "top_buckets"),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C250 C111 Closed-Form Sampled Consensus Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- route: {summary.get('route')}",
        f"- sampling: `{summary.get('sampling')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Route Counts",
        f"`{summary.get('route_counts')}`",
        "",
        "## Selected Minus Baseline",
        f"`{summary.get('delta_selected_minus_baseline_all')}`",
        "",
        "## Routed Rows: Selected Minus Baseline",
        f"`{summary.get('delta_selected_minus_baseline_routed_only')}`",
        "",
        "## Baseline All",
        f"`{summary.get('baseline_all')}`",
        "",
        "## Selected All",
        f"`{summary.get('selected_all')}`",
        "",
        "## Routed Baseline",
        f"`{summary.get('baseline_routed_only')}`",
        "",
        "## Routed Selected",
        f"`{summary.get('selected_routed_only')}`",
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
    summary = run_validation(args)
    io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    agg.base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
