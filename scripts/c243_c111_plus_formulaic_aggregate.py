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


EXPERIMENT_ID = "C243"
EXPERIMENT_SLUG = "C243_c111_plus_formulaic_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C243_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C243 compare C111 with isolated C119 formulaic solver.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=243)
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


def c111_plus_formulaic_stack(current: Any, c111: Any, question: str, answer: str) -> tuple[str, str]:
    c111_answer, c111_handler = rollback.c111_stack(c111, question, answer)
    if c111_handler != "fallback_model":
        return c111_answer, c111_handler
    formulaic = current.formulaic_math_physics_answer(question)
    if formulaic is not None:
        return formulaic, "formulaic_math_physics"
    return c111_answer, "fallback_model"


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
        "mechanism": "C111 plus only formulaic_math_physics_answer on C111 fallback rows",
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    agg.install_final_path_dependencies()
    current = retry_base.load_solution()
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

    base_validity: Counter[str] = Counter()
    c111_quality: Counter[str] = Counter()
    formulaic_quality: Counter[str] = Counter()
    changed_c111_quality: Counter[str] = Counter()
    changed_formulaic_quality: Counter[str] = Counter()
    c111_handlers: Counter[str] = Counter()
    formulaic_handlers: Counter[str] = Counter()
    change_pairs: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    output_tokens: list[int] = []

    for row, out in zip(rows, outputs):
        completion = out.outputs[0]
        base_answer = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        out_tokens = len(token_ids) if token_ids is not None else len(tokenizer(base_answer).input_ids)
        output_tokens.append(out_tokens)
        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(question)
        flags = retry_base.valid_flags(base_answer, out_tokens, c111.MAX_NEW_TOKENS)
        base_validity["rows"] += 1
        base_validity["empty"] += int(flags["empty"])
        base_validity["thinking"] += int(flags["thinking"])
        base_validity["hit_max_tokens"] += int(flags["hit_max_tokens"])
        base_validity["repetition_loop"] += int(flags["repetition_loop"])

        c111_answer, c111_handler = rollback.c111_stack(c111, question, base_answer)
        formulaic_answer, formulaic_handler = c111_plus_formulaic_stack(current, c111, question, base_answer)
        c111_handlers[c111_handler] += 1
        formulaic_handlers[formulaic_handler] += 1
        retry_base.quality_update(c111_quality, c111_answer, reference)
        retry_base.quality_update(formulaic_quality, formulaic_answer, reference)

        changed = agg.norm(c111_answer) != agg.norm(formulaic_answer)
        by_category[category]["rows"] += 1
        by_bucket[bucket]["rows"] += 1
        if changed:
            change_pairs[f"{c111_handler}->{formulaic_handler}"] += 1
            by_category[category]["changed"] += 1
            by_bucket[bucket]["changed"] += 1
            retry_base.quality_update(changed_c111_quality, c111_answer, reference)
            retry_base.quality_update(changed_formulaic_quality, formulaic_answer, reference)

    c111_rates = agg.rates({"c111": c111_quality})["c111"]
    formulaic_rates = agg.rates({"formulaic": formulaic_quality})["formulaic"]
    changed_c111_rates = (
        agg.rates({"c111_changed": changed_c111_quality})["c111_changed"] if changed_c111_quality else {"rows": 0}
    )
    changed_formulaic_rates = (
        agg.rates({"formulaic_changed": changed_formulaic_quality})["formulaic_changed"]
        if changed_formulaic_quality
        else {"rows": 0}
    )
    projected_total_4000_s = startup_s + (generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "C111 plus isolated formulaic solver aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {
                "current_solution": "ok",
                "c111_solution": "ok",
                "pymorphy_available": bool(current.get_morph_analyzer()),
            },
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
            "tokens": {
                "avg_input_tokens": sum(input_tokens) / max(1, len(input_tokens)),
                "avg_output_tokens": sum(output_tokens) / max(1, len(output_tokens)),
                "max_output_tokens": max(output_tokens) if output_tokens else None,
            },
            "base_validity": {k: int(v) for k, v in base_validity.items()},
            "c111_quality": c111_rates,
            "formulaic_quality": formulaic_rates,
            "delta_formulaic_minus_c111": {
                "exact": formulaic_rates.get("exact", 0) - c111_rates.get("exact", 0),
                "final_line_exact": formulaic_rates.get("final_line_exact", 0)
                - c111_rates.get("final_line_exact", 0),
                "ref_in_output": formulaic_rates.get("ref_in_output", 0) - c111_rates.get("ref_in_output", 0),
                "output_in_ref": formulaic_rates.get("output_in_ref", 0) - c111_rates.get("output_in_ref", 0),
            },
            "changed_rows": {
                "count": int(sum(change_pairs.values())),
                "c111_quality": changed_c111_rates,
                "formulaic_quality": changed_formulaic_rates,
                "delta": {
                    "exact": changed_formulaic_rates.get("exact", 0) - changed_c111_rates.get("exact", 0),
                    "final_line_exact": changed_formulaic_rates.get("final_line_exact", 0)
                    - changed_c111_rates.get("final_line_exact", 0),
                    "ref_in_output": changed_formulaic_rates.get("ref_in_output", 0)
                    - changed_c111_rates.get("ref_in_output", 0),
                    "output_in_ref": changed_formulaic_rates.get("output_in_ref", 0)
                    - changed_c111_rates.get("output_in_ref", 0),
                },
            },
            "handler_counts": {
                "c111": {k: int(v) for k, v in c111_handlers.items()},
                "formulaic": {k: int(v) for k, v in formulaic_handlers.items()},
                "change_pairs": {k: int(v) for k, v in change_pairs.most_common(40)},
            },
            "changed_by_category": {
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in sorted(by_category.items(), key=lambda kv: -kv[1].get("changed", 0))
            },
            "changed_by_bucket": {
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in sorted(by_bucket.items(), key=lambda kv: -kv[1].get("changed", 0))[:30]
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C243 C111 Plus Isolated Formulaic Solver Aggregate",
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
        "## Base Validity",
        f"`{summary.get('base_validity')}`",
        "",
        "## Quality",
        f"- C111 stack: `{summary.get('c111_quality')}`",
        f"- C111 plus formulaic: `{summary.get('formulaic_quality')}`",
        f"- delta formulaic minus C111: `{summary.get('delta_formulaic_minus_c111')}`",
        "",
        "## Changed Rows",
        f"`{summary.get('changed_rows')}`",
        "",
        "## Handler Counts",
        f"`{summary.get('handler_counts')}`",
        "",
        "## Changed Slices",
        f"- by category: `{summary.get('changed_by_category')}`",
        f"- by bucket: `{summary.get('changed_by_bucket')}`",
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
