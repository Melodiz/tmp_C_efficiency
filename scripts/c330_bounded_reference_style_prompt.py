from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c201_c111_vs_current_stack_aggregate as rollback
import c270_adaptive_length_paired_aggregate as c270


EXPERIMENT_ID = "C330"
EXPERIMENT_SLUG = "C330_bounded_reference_style_prompt"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C330_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
BOUNDED_REFERENCE_PREFIX = (
    "Ответь на языке задания. Если нужен развернутый ответ, дай полный, но сжатый "
    "учебный ответ: ключевое определение, краткое объяснение и итог. Используй 3-5 "
    "предложений или до 4 пунктов, не более 120 слов. Не повторяй условие."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C330 bounded reference-style routed prompt aggregate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=330)
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
        "mechanism": "C269 long-answer route uses bounded reference-style prompt; non-routed rows preserve C111 prompt; max_tokens remains 320",
        "bounded_reference_prefix": BOUNDED_REFERENCE_PREFIX,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    routed_indices = set(c270.route_indices(rows))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    baseline_prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]
    variant_prompts = [
        probe.apply_user_only_template(
            tokenizer,
            str(row["question"]),
            True,
            BOUNDED_REFERENCE_PREFIX if idx in routed_indices else c111.USER_PREFIX,
        )
        for idx, row in enumerate(rows)
    ]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in baseline_prompts]

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
    baseline_t0 = time.perf_counter()
    baseline_outputs = llm.generate(baseline_prompts, sampling_params=sampling)
    baseline_generation_s = time.perf_counter() - baseline_t0
    variant_t0 = time.perf_counter()
    variant_outputs = llm.generate(variant_prompts, sampling_params=sampling)
    variant_generation_s = time.perf_counter() - variant_t0
    sampler.stop()

    caps = [c111.MAX_NEW_TOKENS] * len(rows)
    baseline = c270.summarize_rows_with_caps(c111, tokenizer, rows, baseline_outputs, caps)
    variant = c270.summarize_rows_with_caps(c111, tokenizer, rows, variant_outputs, caps)
    routed_rows = [row for idx, row in enumerate(rows) if idx in routed_indices]
    routed_baseline_outputs = [out for idx, out in enumerate(baseline_outputs) if idx in routed_indices]
    routed_variant_outputs = [out for idx, out in enumerate(variant_outputs) if idx in routed_indices]
    routed_baseline = (
        c270.summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_baseline_outputs, [c111.MAX_NEW_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )
    routed_variant = (
        c270.summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_variant_outputs, [c111.MAX_NEW_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )
    overall_delta = c270.delta(variant, baseline)
    routed_delta = c270.delta(routed_variant, routed_baseline) if routed_rows else {}
    total_generation_s = baseline_generation_s + variant_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    gate = {
        "ref_in_output_nonnegative": overall_delta.get("ref_in_output", 0) >= 0,
        "output_in_ref_nonnegative": overall_delta.get("output_in_ref", 0) >= 0,
        "one_containment_positive": overall_delta.get("ref_in_output", 0) > 0 or overall_delta.get("output_in_ref", 0) > 0,
        "hit_max_tokens_not_worse": overall_delta.get("hit_max_tokens", 0) <= 0,
        "repetition_not_worse": overall_delta.get("repetition_loop", 0) <= 0,
        "runtime_under_12_min": projected_total_4000_s < 720,
    }
    gate["pass"] = bool(
        gate["ref_in_output_nonnegative"]
        and gate["output_in_ref_nonnegative"]
        and gate["one_containment_positive"]
        and gate["hit_max_tokens_not_worse"]
        and gate["repetition_not_worse"]
        and gate["runtime_under_12_min"]
    )
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if gate["pass"] else "KILL",
            "reason": "Bounded reference-style prompt cleared local gate." if gate["pass"] else "Bounded reference-style prompt failed local gate.",
            "raw_task_data_read_remote_only": True,
            "model_loaded": True,
            "sample_meta": {
                "rows": len(rows),
                "routed_rows": len(routed_indices),
                "routed_share": len(routed_indices) / max(1, len(rows)),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "startup_s": startup_s,
                "baseline_generation_s": baseline_generation_s,
                "variant_generation_s": variant_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {"avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens))},
            "baseline_c111": baseline,
            "variant_bounded_reference_style": variant,
            "delta_bounded_reference_style_minus_c111": overall_delta,
            "routed_baseline_c111": routed_baseline,
            "routed_variant_bounded_reference_style": routed_variant,
            "routed_delta_bounded_reference_style_minus_c111": routed_delta,
            "success_gate": gate,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C330 Bounded Reference-Style Prompt",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Success Gate",
        f"`{summary.get('success_gate')}`",
        "",
        "## Delta Bounded Reference Style Minus C111",
        f"`{summary.get('delta_bounded_reference_style_minus_c111')}`",
        "",
        "## Routed Delta Bounded Reference Style Minus C111",
        f"`{summary.get('routed_delta_bounded_reference_style_minus_c111')}`",
        "",
        "## Baseline C111",
        f"`{summary.get('baseline_c111')}`",
        "",
        "## Variant Bounded Reference Style",
        f"`{summary.get('variant_bounded_reference_style')}`",
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
