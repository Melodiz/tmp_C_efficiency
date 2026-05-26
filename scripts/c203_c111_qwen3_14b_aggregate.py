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


EXPERIMENT_ID = "C203"
EXPERIMENT_SLUG = "C203_c111_qwen3_14b_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C203_artifacts"
MODEL_ID = "Qwen/Qwen3-14B-AWQ"
C202_CONTROL_BASELINE = {
    "artifact": "C202_artifacts_20260526T215002Z",
    "sample_source": "locked_val",
    "sample_size": 512,
    "seed": 202,
    "model": "Qwen/Qwen3-8B-AWQ",
    "quality": {"exact": 2, "final_line_exact": 2, "ref_in_output": 22, "output_in_ref": 63},
    "validity": {"hit_max_tokens": 15, "repetition_loop": 6, "empty": 0, "thinking": 0},
    "avg_output_tokens": 87.888671875,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C203 evaluate Qwen3-14B-AWQ with C111 stack.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=202)
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


def quality_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    quality = current.get("quality") or {}
    validity = current.get("validity") or {}
    base_quality = baseline["quality"]
    base_validity = baseline["validity"]
    return {
        "exact": quality.get("exact", 0) - base_quality["exact"],
        "final_line_exact": quality.get("final_line_exact", 0) - base_quality["final_line_exact"],
        "ref_in_output": quality.get("ref_in_output", 0) - base_quality["ref_in_output"],
        "output_in_ref": quality.get("output_in_ref", 0) - base_quality["output_in_ref"],
        "hit_max_tokens": validity.get("hit_max_tokens", 0) - base_validity["hit_max_tokens"],
        "repetition_loop": validity.get("repetition_loop", 0) - base_validity["repetition_loop"],
        "avg_output_tokens": (current.get("tokens") or {}).get("avg_output_tokens", 0)
        - baseline["avg_output_tokens"],
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
        "comparison_baseline": C202_CONTROL_BASELINE,
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
    sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    generation_t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling)
    generation_s = time.perf_counter() - generation_t0
    sampler.stop()

    result = c202.summarize_rows(c111, tokenizer, rows, outputs)
    projected_total_4000_s = startup_s + (generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE",
            "reason": "Qwen3-14B-AWQ C111-stack aggregate completed.",
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
                "generation_s": generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {"avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens))},
            "qwen3_14b": result,
            "delta_vs_c202_qwen3_8b_same_sample": quality_delta(result, C202_CONTROL_BASELINE),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C203 C111 Qwen3-14B Aggregate",
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
        "## Delta vs C202 Qwen3-8B Same-Sample Baseline",
        f"`{summary.get('delta_vs_c202_qwen3_8b_same_sample')}`",
        "",
        "## Qwen3-14B",
        f"`{summary.get('qwen3_14b')}`",
        "",
        "## Baseline",
        f"`{summary.get('comparison_baseline')}`",
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
