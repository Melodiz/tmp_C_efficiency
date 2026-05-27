from __future__ import annotations

import argparse
import gc
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
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as paired
import c216_qwen3_14b_paired_bucket_aggregate as delta_base


EXPERIMENT_ID = "C289"
EXPERIMENT_SLUG = "C289_routed_dense_algebra_translation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C289_artifacts"
BASELINE_MODEL_ID = "Qwen/Qwen3-8B-AWQ"
VARIANT_MODEL_ID = "Qwen/Qwen3-8B"

TRANSLATION_RE = re.compile(r"перевед|перевод|translate|translation", re.IGNORECASE)
ALGEBRA_RE = re.compile(r"уравнен|систем[аы]? урав|реши[^.?!\\n]{0,80}урав|solve[^.?!\\n]{0,80}equation", re.IGNORECASE)
SYMBOLIC_EQUATION_RE = re.compile(r"[=].*(^|\\W)[xy](\\W|$)|(^|\\W)[xy](\\W|$).*[=]", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C289 route algebra/translation rows to dense Qwen3-8B.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=289)
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


def route(question: str) -> str:
    text = str(question)
    if TRANSLATION_RE.search(text):
        return "dense_translation"
    if ALGEBRA_RE.search(text) or SYMBOLIC_EQUATION_RE.search(text):
        return "dense_algebra"
    return "awq_default"


def run_model(
    model_id: str,
    c111: Any,
    rows: list[dict[str, Any]],
    seed: int,
    quantization: str | None,
) -> tuple[Any, Any, dict[str, Any]]:
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in prompts]
    startup_t0 = time.perf_counter()
    llm = LLM(
        model=model_id,
        dtype="float16",
        quantization=quantization,
        max_model_len=c111.MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=seed,
        trust_remote_code=False,
    )
    startup_s = time.perf_counter() - startup_t0
    sampling = SamplingParams(temperature=0.0, max_tokens=c111.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    generation_t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling)
    generation_s = time.perf_counter() - generation_t0
    runtime = {
        "startup_s": startup_s,
        "generation_s": generation_s,
        "avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens)),
    }
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return tokenizer, outputs, runtime


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
        "baseline_model_id": BASELINE_MODEL_ID,
        "variant_model_id": VARIANT_MODEL_ID,
        "mechanism": "route algebra/equation and translation-like rows to dense Qwen3-8B",
        "package_risk": {"awq_selected_files_gb": 6.11, "dense_selected_files_gb": 16.40},
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    routes = [route(str(row["question"])) for row in rows]
    route_counts = Counter(routes)

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    baseline_tokenizer, baseline_outputs, baseline_runtime = run_model(
        BASELINE_MODEL_ID, c111, rows, args.seed, "awq_marlin"
    )
    dense_tokenizer, dense_outputs, dense_runtime = run_model(VARIANT_MODEL_ID, c111, rows, args.seed, None)
    sampler.stop()

    selected_outputs = [
        dense_out if route_name.startswith("dense_") else awq_out
        for route_name, awq_out, dense_out in zip(routes, baseline_outputs, dense_outputs)
    ]
    baseline = paired.summarize_rows(c111, baseline_tokenizer, rows, baseline_outputs)
    dense = paired.summarize_rows(c111, dense_tokenizer, rows, dense_outputs)
    selected = paired.summarize_rows(c111, baseline_tokenizer, rows, selected_outputs)

    total_generation_s = baseline_runtime["generation_s"] + dense_runtime["generation_s"]
    total_startup_s = baseline_runtime["startup_s"] + dense_runtime["startup_s"]
    projected_total_4000_s = total_startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "Routed dense algebra/translation aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
                "route_counts": {k: int(v) for k, v in route_counts.items()},
            },
            "runtime": {
                "baseline": baseline_runtime,
                "dense": dense_runtime,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "baseline_awq": baseline,
            "dense_all": dense,
            "selected_routed_dense": selected,
            "delta_dense_all_minus_awq": delta_base.overall_delta(dense, baseline),
            "delta_selected_minus_awq": delta_base.overall_delta(selected, baseline),
            "selected_delta_by_category": delta_base.keyed_delta(selected, baseline, "by_category"),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C289 Routed Dense Algebra/Translation Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- baseline model: `{summary.get('baseline_model_id')}`",
        f"- dense model: `{summary.get('variant_model_id')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        f"- package risk: `{summary.get('package_risk')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Delta Dense-All Minus AWQ",
        f"`{summary.get('delta_dense_all_minus_awq')}`",
        "",
        "## Delta Selected Minus AWQ",
        f"`{summary.get('delta_selected_minus_awq')}`",
        "",
        "## Selected Delta By Category",
        f"`{summary.get('selected_delta_by_category')}`",
        "",
        "## Baseline AWQ",
        f"`{summary.get('baseline_awq')}`",
        "",
        "## Dense All",
        f"`{summary.get('dense_all')}`",
        "",
        "## Selected Routed Dense",
        f"`{summary.get('selected_routed_dense')}`",
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
