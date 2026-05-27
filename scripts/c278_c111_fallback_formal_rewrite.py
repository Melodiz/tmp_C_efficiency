from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c220_paired_answer_judge_selector_aggregate as judge_base
import c222_c111_fallback_answer_extraction_aggregate as c222


EXPERIMENT_ID = "C278"
EXPERIMENT_SLUG = "C278_c111_fallback_formal_rewrite"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C278_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
REWRITE_PREFIX = (
    "Rewrite the previous solution in a concise formal textbook style. Preserve the facts, numbers, "
    "language, units, and final answer. Do not introduce new claims."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C278 formal rewrite of C111 fallback outputs.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=278)
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


def rewrite_prompt(question: str, previous_solution: str) -> str:
    return (
        f"Задание:\n{question}\n\n"
        f"Предыдущий ответ:\n{previous_solution}\n\n"
        "Переписанный ответ:"
    )


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
        "mechanism": "same-model formal rewrite on C111 fallback outputs only",
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

    control_t0 = time.perf_counter()
    control_outputs = llm.generate(
        control_prompts,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1),
    )
    control_generation_s = time.perf_counter() - control_t0
    controls = c222.control_records(c111, tokenizer, rows, control_outputs)
    fallback_indices = [i for i, record in enumerate(controls) if record["handler"] == "fallback_model"]
    rewrite_prompts = [
        probe.apply_user_only_template(
            tokenizer,
            rewrite_prompt(str(rows[idx]["question"]), str(controls[idx]["raw_answer"])),
            True,
            REWRITE_PREFIX,
        )
        for idx in fallback_indices
    ]
    rewrite_t0 = time.perf_counter()
    rewrite_outputs = llm.generate(
        rewrite_prompts,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=256, top_p=1.0, top_k=-1),
    )
    rewrite_generation_s = time.perf_counter() - rewrite_t0
    sampler.stop()

    selected, rewrite_stats = c222.extracted_records(c111, tokenizer, rows, controls, rewrite_outputs, fallback_indices)
    control_summary = c222.summarize_records(rows, controls)
    selected_summary = c222.summarize_records(rows, selected)
    delta = judge_base.delta(selected_summary, control_summary)
    total_generation_s = control_generation_s + rewrite_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    decision = (
        "MUTATE"
        if delta.get("ref_in_output", 0) > 0
        and delta.get("output_in_ref", 0) >= 0
        and delta.get("hit_max_tokens", 0) <= 0
        and delta.get("repetition_loop", 0) <= 0
        else "KILL"
    )

    summary.update(
        {
            "status": "completed",
            "decision_recommendation": decision,
            "reason": "Formal rewrite aggregate completed.",
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
                "rewrite_generation_s": rewrite_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "rewrite_stats": {
                **rewrite_stats,
                "fallback_rows": len(fallback_indices),
                "preserved_deterministic_rows": len(rows) - len(fallback_indices),
            },
            "control_c111": control_summary,
            "formal_rewrite_selected": selected_summary,
            "delta_rewrite_minus_control": delta,
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C278 C111 Fallback Formal Rewrite",
        "",
        "## Objective",
        "- No leaderboard submission or submission zip.",
        "- Only one mechanism: same-model formal rewrite of C111 fallback outputs; deterministic C111 rows stay unchanged.",
        "- Return aggregate metrics only; no raw prompts, references, outputs, row ids, datasets, weights, or adapters.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Rewrite Stats",
        f"`{summary.get('rewrite_stats')}`",
        "",
        "## Delta Rewrite Minus C111",
        f"`{summary.get('delta_rewrite_minus_control')}`",
        "",
        "## C111 Control",
        f"`{summary.get('control_c111')}`",
        "",
        "## Formal Rewrite Selected",
        f"`{summary.get('formal_rewrite_selected')}`",
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
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    summary = run_validation(args)
    io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    io.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
