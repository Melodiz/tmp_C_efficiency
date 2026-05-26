from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c198_targeted_failure_retry as retry_base


EXPERIMENT_ID = "C201"
EXPERIMENT_SLUG = "C201_c111_vs_current_stack_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C201_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
C111_COMMIT = "9426eb7"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C201 compare C111 and current final stacks on shared outputs.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=201)
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


def load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_c111_solution() -> Any:
    source = subprocess.check_output(
        ["git", "show", f"{C111_COMMIT}:simple_solution/solution.py"],
        text=True,
        cwd=Path.cwd(),
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="c201_c111_"))
    path = tmp_dir / "solution_c111.py"
    path.write_text(source, encoding="utf-8")
    return load_module_from_path("task_c_c111_solution", path)


def c111_stack(module: Any, question: str, answer: str) -> tuple[str, str]:
    for name, func in (
        ("expression_substitution", lambda: module.expression_substitution_answer(question)),
        ("comma_loop_dedup", lambda: module.dedup_comma_loop(answer)),
        ("english_cloze_cleanup", lambda: module.cleanup_english_cloze_answer(question, answer)),
        ("quantity_conversion", lambda: module.quantity_conversion_answer(question)),
        ("km_meters", lambda: module.km_meters_answer(question)),
    ):
        value = func()
        if value is not None:
            return value, name
    return answer, "fallback_model"


def current_stack(module: Any, question: str, answer: str) -> tuple[str, str]:
    return agg.first_handler(module, question, answer)


def update_quality(counter: Counter[str], answer: str, reference: str) -> None:
    retry_base.quality_update(counter, answer, reference)


def rates(table: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    return agg.rates(table)


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
        "c111_commit": C111_COMMIT,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    agg.install_final_path_dependencies()
    current = retry_base.load_solution()
    c111 = load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, current.USER_PREFIX) for row in rows]
    input_tokens = [len(tokenizer(prompt).input_ids) for prompt in prompts]

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    startup_t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=current.MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=args.seed,
        trust_remote_code=False,
    )
    startup_s = time.perf_counter() - startup_t0
    sampling = SamplingParams(temperature=0.0, max_tokens=current.MAX_NEW_TOKENS, top_p=1.0, top_k=-1)
    generation_t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling)
    generation_s = time.perf_counter() - generation_t0
    sampler.stop()

    base_validity: Counter[str] = Counter()
    c111_quality: Counter[str] = Counter()
    current_quality: Counter[str] = Counter()
    changed_quality_c111: Counter[str] = Counter()
    changed_quality_current: Counter[str] = Counter()
    c111_handlers: Counter[str] = Counter()
    current_handlers: Counter[str] = Counter()
    change_pairs: Counter[str] = Counter()
    by_category_delta: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket_delta: defaultdict[str, Counter[str]] = defaultdict(Counter)
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
        flags = retry_base.valid_flags(base_answer, out_tokens, current.MAX_NEW_TOKENS)
        base_validity["rows"] += 1
        base_validity["empty"] += int(flags["empty"])
        base_validity["thinking"] += int(flags["thinking"])
        base_validity["hit_max_tokens"] += int(flags["hit_max_tokens"])
        base_validity["repetition_loop"] += int(flags["repetition_loop"])

        c111_answer, c111_handler = c111_stack(c111, question, base_answer)
        current_answer, current_handler = current_stack(current, question, base_answer)
        c111_handlers[c111_handler] += 1
        current_handlers[current_handler] += 1
        update_quality(c111_quality, c111_answer, reference)
        update_quality(current_quality, current_answer, reference)

        changed = agg.norm(c111_answer) != agg.norm(current_answer)
        if changed:
            change_pairs[f"{c111_handler}->{current_handler}"] += 1
            update_quality(changed_quality_c111, c111_answer, reference)
            update_quality(changed_quality_current, current_answer, reference)
            by_category_delta[category]["changed"] += 1
            by_bucket_delta[bucket]["changed"] += 1
        by_category_delta[category]["rows"] += 1
        by_bucket_delta[bucket]["rows"] += 1

    c111_rates = rates({"c111": c111_quality})["c111"]
    current_rates = rates({"current": current_quality})["current"]
    changed_c111_rates = rates({"c111_changed": changed_quality_c111})["c111_changed"] if changed_quality_c111 else {"rows": 0}
    changed_current_rates = (
        rates({"current_changed": changed_quality_current})["current_changed"] if changed_quality_current else {"rows": 0}
    )
    projected_total_4000_s = startup_s + (generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE",
            "reason": "C111-vs-current stack aggregate comparison completed.",
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
            "current_quality": current_rates,
            "delta": {
                "exact": current_rates.get("exact", 0) - c111_rates.get("exact", 0),
                "final_line_exact": current_rates.get("final_line_exact", 0) - c111_rates.get("final_line_exact", 0),
                "ref_in_output": current_rates.get("ref_in_output", 0) - c111_rates.get("ref_in_output", 0),
                "output_in_ref": current_rates.get("output_in_ref", 0) - c111_rates.get("output_in_ref", 0),
            },
            "changed_rows": {
                "count": int(sum(change_pairs.values())),
                "c111_quality": changed_c111_rates,
                "current_quality": changed_current_rates,
                "delta": {
                    "exact": changed_current_rates.get("exact", 0) - changed_c111_rates.get("exact", 0),
                    "final_line_exact": changed_current_rates.get("final_line_exact", 0)
                    - changed_c111_rates.get("final_line_exact", 0),
                    "ref_in_output": changed_current_rates.get("ref_in_output", 0)
                    - changed_c111_rates.get("ref_in_output", 0),
                    "output_in_ref": changed_current_rates.get("output_in_ref", 0)
                    - changed_c111_rates.get("output_in_ref", 0),
                },
            },
            "handler_counts": {
                "c111": {k: int(v) for k, v in c111_handlers.items()},
                "current": {k: int(v) for k, v in current_handlers.items()},
                "change_pairs": {k: int(v) for k, v in change_pairs.most_common(40)},
            },
            "changed_by_category": {
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in sorted(by_category_delta.items(), key=lambda kv: -kv[1].get("changed", 0))
            },
            "changed_by_bucket": {
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in sorted(by_bucket_delta.items(), key=lambda kv: -kv[1].get("changed", 0))[:30]
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C201 C111 vs Current Final-Stack Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
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
        f"- current stack: `{summary.get('current_quality')}`",
        f"- delta current minus C111: `{summary.get('delta')}`",
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
