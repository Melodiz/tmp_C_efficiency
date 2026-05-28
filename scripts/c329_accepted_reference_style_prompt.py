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
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202
import c270_adaptive_length_paired_aggregate as c270
import c328_long_route_reference_style_prompt as c328


EXPERIMENT_ID = "C329"
EXPERIMENT_SLUG = "C329_accepted_reference_style_prompt"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C329_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
ACCEPT_TOKEN_LIMIT = 300


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C329 accepted C328 reference-style routed prompt aggregate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=329)
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


def completion_text_and_tokens(tokenizer: Any, output: Any) -> tuple[str, int]:
    completion = output.outputs[0]
    text = completion.text.strip()
    token_ids = getattr(completion, "token_ids", None)
    tokens = len(token_ids) if token_ids is not None else len(tokenizer(text).input_ids)
    return text, tokens


def accept_variant(tokenizer: Any, solution: Any, variant_output: Any) -> tuple[bool, dict[str, Any]]:
    text, tokens = completion_text_and_tokens(tokenizer, variant_output)
    flags = c202.retry_base.valid_flags(text, tokens, solution.MAX_NEW_TOKENS)
    accepted = bool(
        not flags["empty"]
        and not flags["thinking"]
        and not flags["hit_max_tokens"]
        and not flags["repetition_loop"]
        and tokens <= ACCEPT_TOKEN_LIMIT
    )
    return accepted, {
        "empty": bool(flags["empty"]),
        "thinking": bool(flags["thinking"]),
        "hit_max_tokens": bool(flags["hit_max_tokens"]),
        "repetition_loop": bool(flags["repetition_loop"]),
        "over_accept_token_limit": tokens > ACCEPT_TOKEN_LIMIT,
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
        "mechanism": "Accept C328 routed reference-style prompt output only when visible validity is clean; otherwise keep C111 output.",
        "accept_token_limit": ACCEPT_TOKEN_LIMIT,
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
            c328.REFERENCE_STYLE_PREFIX if idx in routed_indices else c111.USER_PREFIX,
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

    accepted_outputs = list(baseline_outputs)
    reject_reasons: dict[str, int] = {}
    accepted_count = 0
    for idx in routed_indices:
        accepted, reasons = accept_variant(tokenizer, c111, variant_outputs[idx])
        if accepted:
            accepted_outputs[idx] = variant_outputs[idx]
            accepted_count += 1
        else:
            for reason, fired in reasons.items():
                if fired:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    caps = [c111.MAX_NEW_TOKENS] * len(rows)
    baseline = c270.summarize_rows_with_caps(c111, tokenizer, rows, baseline_outputs, caps)
    raw_variant = c270.summarize_rows_with_caps(c111, tokenizer, rows, variant_outputs, caps)
    accepted = c270.summarize_rows_with_caps(c111, tokenizer, rows, accepted_outputs, caps)

    routed_rows = [row for idx, row in enumerate(rows) if idx in routed_indices]
    routed_baseline_outputs = [out for idx, out in enumerate(baseline_outputs) if idx in routed_indices]
    routed_raw_outputs = [out for idx, out in enumerate(variant_outputs) if idx in routed_indices]
    routed_accepted_outputs = [out for idx, out in enumerate(accepted_outputs) if idx in routed_indices]
    routed_baseline = (
        c270.summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_baseline_outputs, [c111.MAX_NEW_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )
    routed_raw = (
        c270.summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_raw_outputs, [c111.MAX_NEW_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )
    routed_accepted = (
        c270.summarize_rows_with_caps(c111, tokenizer, routed_rows, routed_accepted_outputs, [c111.MAX_NEW_TOKENS] * len(routed_rows))
        if routed_rows
        else {}
    )

    accepted_delta = c270.delta(accepted, baseline)
    raw_delta = c270.delta(raw_variant, baseline)
    routed_accepted_delta = c270.delta(routed_accepted, routed_baseline) if routed_rows else {}
    routed_raw_delta = c270.delta(routed_raw, routed_baseline) if routed_rows else {}
    total_generation_s = baseline_generation_s + variant_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    gate = {
        "ref_in_output_nonnegative": accepted_delta.get("ref_in_output", 0) >= 0,
        "output_in_ref_nonnegative": accepted_delta.get("output_in_ref", 0) >= 0,
        "one_containment_positive": accepted_delta.get("ref_in_output", 0) > 0 or accepted_delta.get("output_in_ref", 0) > 0,
        "hit_max_tokens_not_worse": accepted_delta.get("hit_max_tokens", 0) <= 0,
        "repetition_not_worse": accepted_delta.get("repetition_loop", 0) <= 0,
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
            "reason": "Accepted reference-style prompt cleared local gate." if gate["pass"] else "Accepted reference-style prompt failed local gate.",
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
            "acceptance": {
                "accepted_rows": accepted_count,
                "accepted_share_of_routed": accepted_count / max(1, len(routed_indices)),
                "rejected_rows": len(routed_indices) - accepted_count,
                "reject_reasons": dict(sorted(reject_reasons.items())),
            },
            "baseline_c111": baseline,
            "raw_reference_style": raw_variant,
            "accepted_reference_style": accepted,
            "delta_raw_reference_style_minus_c111": raw_delta,
            "delta_accepted_reference_style_minus_c111": accepted_delta,
            "routed_baseline_c111": routed_baseline,
            "routed_raw_reference_style": routed_raw,
            "routed_accepted_reference_style": routed_accepted,
            "routed_delta_raw_reference_style_minus_c111": routed_raw_delta,
            "routed_delta_accepted_reference_style_minus_c111": routed_accepted_delta,
            "success_gate": gate,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C329 Accepted Reference-Style Prompt",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- accept token limit: `{summary.get('accept_token_limit')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Acceptance",
        f"`{summary.get('acceptance')}`",
        "",
        "## Success Gate",
        f"`{summary.get('success_gate')}`",
        "",
        "## Delta Accepted Reference Style Minus C111",
        f"`{summary.get('delta_accepted_reference_style_minus_c111')}`",
        "",
        "## Delta Raw Reference Style Minus C111",
        f"`{summary.get('delta_raw_reference_style_minus_c111')}`",
        "",
        "## Routed Delta Accepted Reference Style Minus C111",
        f"`{summary.get('routed_delta_accepted_reference_style_minus_c111')}`",
        "",
        "## Routed Delta Raw Reference Style Minus C111",
        f"`{summary.get('routed_delta_raw_reference_style_minus_c111')}`",
        "",
        "## Baseline C111",
        f"`{summary.get('baseline_c111')}`",
        "",
        "## Accepted Reference Style",
        f"`{summary.get('accepted_reference_style')}`",
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
