from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202
import c239_c111_system_prefix_aggregate as c239


EXPERIMENT_ID = "C240"
EXPERIMENT_SLUG = "C240_failure_gated_system_prefix_fallback"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C240_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C240 route C111 visible failures to system-prefix fallback.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=240)
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


def invalid(flags: dict[str, bool]) -> bool:
    return bool(flags["empty"] or flags["thinking"] or flags["hit_max_tokens"] or flags["repetition_loop"])


def summarize_selected(solution: Any, tokenizer: Any, rows: list[dict[str, Any]], control_outputs: Any, system_outputs: Any) -> dict[str, Any]:
    quality: Counter[str] = Counter()
    validity: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    route_counts: Counter[str] = Counter()
    handlers: Counter[str] = Counter()
    output_tokens: list[int] = []

    for row, control_out, system_out in zip(rows, control_outputs, system_outputs):
        control_completion = control_out.outputs[0]
        system_completion = system_out.outputs[0]
        control_text = control_completion.text.strip()
        system_text = system_completion.text.strip()
        control_tokens = getattr(control_completion, "token_ids", None)
        system_tokens = getattr(system_completion, "token_ids", None)
        control_n = len(control_tokens) if control_tokens is not None else len(tokenizer(control_text).input_ids)
        system_n = len(system_tokens) if system_tokens is not None else len(tokenizer(system_text).input_ids)
        control_flags = retry_base.valid_flags(control_text, control_n, solution.MAX_NEW_TOKENS)
        system_flags = retry_base.valid_flags(system_text, system_n, solution.MAX_NEW_TOKENS)

        use_system = invalid(control_flags) and not invalid(system_flags)
        selected_text = system_text if use_system else control_text
        selected_tokens = system_n if use_system else control_n
        selected_flags = system_flags if use_system else control_flags
        route_counts["selected_system"] += int(use_system)
        route_counts["kept_control"] += int(not use_system)
        route_counts["control_invalid"] += int(invalid(control_flags))
        route_counts["system_invalid"] += int(invalid(system_flags))
        route_counts["eligible_system_clean"] += int(invalid(control_flags) and not invalid(system_flags))

        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(question)
        final, handler = rollback.c111_stack(solution, question, selected_text)

        handlers[handler] += 1
        output_tokens.append(selected_tokens)
        retry_base.quality_update(quality, final, reference)
        retry_base.quality_update(by_category[category], final, reference)
        retry_base.quality_update(by_bucket[bucket], final, reference)
        validity["rows"] += 1
        validity["empty"] += int(selected_flags["empty"])
        validity["thinking"] += int(selected_flags["thinking"])
        validity["hit_max_tokens"] += int(selected_flags["hit_max_tokens"])
        validity["repetition_loop"] += int(selected_flags["repetition_loop"])
        validity["deterministic_first_fire"] += int(handler != "fallback_model")

    return {
        "quality": agg.rates({"overall": quality})["overall"],
        "validity": {k: int(v) for k, v in validity.items()},
        "route_counts": {k: int(v) for k, v in route_counts.items()},
        "handler_counts": {k: int(v) for k, v in handlers.items()},
        "tokens": {
            "avg_output_tokens": sum(output_tokens) / max(1, len(output_tokens)),
            "max_output_tokens": max(output_tokens) if output_tokens else None,
        },
        "by_category": dict(sorted(agg.rates(by_category).items())),
        "top_buckets": dict(sorted(agg.rates(by_bucket).items(), key=lambda kv: -kv[1].get("rows", 0))[:20]),
    }


def delta(variant: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact": variant["quality"].get("exact", 0) - control["quality"].get("exact", 0),
        "final_line_exact": variant["quality"].get("final_line_exact", 0)
        - control["quality"].get("final_line_exact", 0),
        "ref_in_output": variant["quality"].get("ref_in_output", 0) - control["quality"].get("ref_in_output", 0),
        "output_in_ref": variant["quality"].get("output_in_ref", 0) - control["quality"].get("output_in_ref", 0),
        "hit_max_tokens": variant["validity"].get("hit_max_tokens", 0)
        - control["validity"].get("hit_max_tokens", 0),
        "repetition_loop": variant["validity"].get("repetition_loop", 0)
        - control["validity"].get("repetition_loop", 0),
        "avg_output_tokens": variant["tokens"].get("avg_output_tokens", 0)
        - control["tokens"].get("avg_output_tokens", 0),
    }


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
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    control_prompts = [
        probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows
    ]
    system_prompts = [
        c239.apply_system_prefix_template(tokenizer, str(row["question"]), c111.USER_PREFIX) for row in rows
    ]

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
    sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    control_t0 = time.perf_counter()
    control_outputs = llm.generate(control_prompts, sampling_params=sampling)
    control_generation_s = time.perf_counter() - control_t0
    system_t0 = time.perf_counter()
    system_outputs = llm.generate(system_prompts, sampling_params=sampling)
    system_generation_s = time.perf_counter() - system_t0
    sampler.stop()

    control = c202.summarize_rows(c111, tokenizer, rows, control_outputs)
    system = c202.summarize_rows(c111, tokenizer, rows, system_outputs)
    selected = summarize_selected(c111, tokenizer, rows, control_outputs, system_outputs)
    total_generation_s = control_generation_s + system_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "Failure-gated system-prefix fallback aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "control_generation_s": control_generation_s,
                "system_generation_s": system_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "control_user_prefix": control,
            "system_prefix_all": system,
            "selected_failure_gated": selected,
            "delta_selected_minus_control": delta(selected, control),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C240 Failure-Gated System-Prefix Fallback",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- model: `{summary.get('model_id')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Delta Selected Minus Control",
        f"`{summary.get('delta_selected_minus_control')}`",
        "",
        "## User-Prefix Control",
        f"`{summary.get('control_user_prefix')}`",
        "",
        "## System-Prefix All",
        f"`{summary.get('system_prefix_all')}`",
        "",
        "## Selected Failure-Gated",
        f"`{summary.get('selected_failure_gated')}`",
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
