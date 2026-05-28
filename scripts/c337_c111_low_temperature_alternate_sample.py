from __future__ import annotations

import argparse
import gc
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202
import c235_c111_max_tokens_512 as c235


EXPERIMENT_ID = "C337"
EXPERIMENT_SLUG = "C337_c111_low_temperature_alternate_sample"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C337_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
ALT_TEMPERATURE = 0.2
ALT_TOP_P = 0.95


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C337 compare C111 greedy with one low-temperature alternate sample.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=337)
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


def row_quality(answer: str, reference: str) -> dict[str, bool]:
    n_answer = agg.norm(answer)
    n_ref = agg.norm(reference)
    return {
        "exact": bool(n_ref) and n_answer == n_ref,
        "final_line_exact": bool(n_ref) and agg.norm(agg.final_line(answer)) == n_ref,
        "ref_in_output": bool(n_ref) and n_ref in n_answer,
        "output_in_ref": bool(n_answer) and n_answer in n_ref,
    }


def paired_complementarity(solution: Any, tokenizer: Any, rows: list[dict[str, Any]], baseline_outputs: Any, alt_outputs: Any) -> dict[str, Any]:
    metrics = {
        "rows": len(rows),
        "changed_outputs": 0,
        "alt_exact_win": 0,
        "alt_exact_loss": 0,
        "alt_ref_in_output_win": 0,
        "alt_ref_in_output_loss": 0,
        "alt_output_in_ref_win": 0,
        "alt_output_in_ref_loss": 0,
        "alt_any_quality_win_without_any_loss": 0,
        "alt_validity_clean_when_baseline_invalid": 0,
        "alt_invalid_when_baseline_clean": 0,
    }
    by_category: dict[str, dict[str, int]] = {}

    for row, base_out, alt_out in zip(rows, baseline_outputs, alt_outputs):
        base_completion = base_out.outputs[0]
        alt_completion = alt_out.outputs[0]
        base_answer_raw = base_completion.text.strip()
        alt_answer_raw = alt_completion.text.strip()
        base_tokens = len(base_completion.token_ids) if getattr(base_completion, "token_ids", None) is not None else len(tokenizer(base_answer_raw).input_ids)
        alt_tokens = len(alt_completion.token_ids) if getattr(alt_completion, "token_ids", None) is not None else len(tokenizer(alt_answer_raw).input_ids)

        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        base_final, _ = rollback.c111_stack(solution, question, base_answer_raw)
        alt_final, _ = rollback.c111_stack(solution, question, alt_answer_raw)
        base_quality = row_quality(base_final, reference)
        alt_quality = row_quality(alt_final, reference)
        base_flags = retry_base.valid_flags(base_answer_raw, base_tokens, solution.MAX_NEW_TOKENS)
        alt_flags = retry_base.valid_flags(alt_answer_raw, alt_tokens, solution.MAX_NEW_TOKENS)
        base_invalid = any(base_flags[k] for k in ("empty", "thinking", "hit_max_tokens", "repetition_loop"))
        alt_invalid = any(alt_flags[k] for k in ("empty", "thinking", "hit_max_tokens", "repetition_loop"))

        cat = by_category.setdefault(
            category,
            {
                "rows": 0,
                "changed_outputs": 0,
                "alt_exact_win": 0,
                "alt_exact_loss": 0,
                "alt_ref_in_output_win": 0,
                "alt_ref_in_output_loss": 0,
                "alt_output_in_ref_win": 0,
                "alt_output_in_ref_loss": 0,
            },
        )
        cat["rows"] += 1

        if agg.norm(base_final) != agg.norm(alt_final):
            metrics["changed_outputs"] += 1
            cat["changed_outputs"] += 1
        quality_wins = 0
        quality_losses = 0
        for key, win_name, loss_name in (
            ("exact", "alt_exact_win", "alt_exact_loss"),
            ("ref_in_output", "alt_ref_in_output_win", "alt_ref_in_output_loss"),
            ("output_in_ref", "alt_output_in_ref_win", "alt_output_in_ref_loss"),
        ):
            if alt_quality[key] and not base_quality[key]:
                metrics[win_name] += 1
                cat[win_name] += 1
                quality_wins += 1
            if base_quality[key] and not alt_quality[key]:
                metrics[loss_name] += 1
                cat[loss_name] += 1
                quality_losses += 1
        if quality_wins and not quality_losses:
            metrics["alt_any_quality_win_without_any_loss"] += 1
        if base_invalid and not alt_invalid:
            metrics["alt_validity_clean_when_baseline_invalid"] += 1
        if not base_invalid and alt_invalid:
            metrics["alt_invalid_when_baseline_clean"] += 1

    by_category_sorted = dict(sorted(by_category.items(), key=lambda kv: (-kv[1]["rows"], kv[0])))
    return {"overall": metrics, "by_category": by_category_sorted}


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "prompts_returned": False,
        "references_returned": False,
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "c111_commit": rollback.C111_COMMIT,
        "model_id": MODEL_ID,
        "mechanism": "C111 prompt/handlers/max_tokens=320 with one low-temperature alternate sample; no model, prompt, handler, SFT, or packaging change.",
        "baseline_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": None},
        "alternate_sampling": {"temperature": ALT_TEMPERATURE, "top_p": ALT_TOP_P, "top_k": -1, "max_tokens": None},
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
    alternate_sampling = SamplingParams(temperature=ALT_TEMPERATURE, max_tokens=c111.MAX_NEW_TOKENS, top_p=ALT_TOP_P, top_k=-1)

    baseline_t0 = time.perf_counter()
    baseline_outputs = llm.generate(prompts, sampling_params=baseline_sampling)
    baseline_generation_s = time.perf_counter() - baseline_t0
    alternate_t0 = time.perf_counter()
    alternate_outputs = llm.generate(prompts, sampling_params=alternate_sampling)
    alternate_generation_s = time.perf_counter() - alternate_t0
    sampler.stop()

    baseline = c202.summarize_rows(c111, tokenizer, rows, baseline_outputs)
    alternate = c202.summarize_rows(c111, tokenizer, rows, alternate_outputs)
    delta = c235.delta(alternate, baseline)
    complementarity = paired_complementarity(c111, tokenizer, rows, baseline_outputs, alternate_outputs)
    total_generation_s = baseline_generation_s + alternate_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    gate = {
        "runtime_under_12_min": projected_total_4000_s < 720,
        "hit_max_tokens_not_worse": delta.get("hit_max_tokens", 0) <= 0,
        "repetition_not_worse": delta.get("repetition_loop", 0) <= 0,
        "both_containment_nonnegative": delta.get("ref_in_output", 0) >= 0 and delta.get("output_in_ref", 0) >= 0,
        "broad_complementarity": complementarity["overall"].get("alt_any_quality_win_without_any_loss", 0) >= 8,
    }
    gate["pass"] = bool(
        gate["runtime_under_12_min"]
        and gate["hit_max_tokens_not_worse"]
        and gate["repetition_not_worse"]
        and gate["both_containment_nonnegative"]
        and gate["broad_complementarity"]
    )
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if gate["pass"] else "KILL",
            "reason": "Low-temperature alternate sample cleared diagnostic gate." if gate["pass"] else "Low-temperature alternate sample failed diagnostic gate.",
            "raw_task_data_read_remote_only": True,
            "model_loaded": True,
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "baseline_generation_s": baseline_generation_s,
                "alternate_generation_s": alternate_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {"avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens))},
            "baseline_c111_greedy": baseline,
            "alternate_low_temperature": alternate,
            "delta_alternate_minus_c111": delta,
            "paired_complementarity": complementarity,
            "gate": gate,
        }
    )
    del llm
    gc.collect()
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C337 C111 Low-Temperature Alternate Sample",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- baseline sampling: `{summary.get('baseline_sampling')}`",
        f"- alternate sampling: `{summary.get('alternate_sampling')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Gate",
        f"`{summary.get('gate')}`",
        "",
        "## Delta Alternate Minus C111",
        f"`{summary.get('delta_alternate_minus_c111')}`",
        "",
        "## Paired Complementarity",
        f"`{summary.get('paired_complementarity')}`",
        "",
        "## Baseline C111 Greedy",
        f"`{summary.get('baseline_c111_greedy')}`",
        "",
        "## Alternate Low Temperature",
        f"`{summary.get('alternate_low_temperature')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- prompts returned: `{summary.get('prompts_returned')}`",
        f"- references returned: `{summary.get('references_returned')}`",
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
