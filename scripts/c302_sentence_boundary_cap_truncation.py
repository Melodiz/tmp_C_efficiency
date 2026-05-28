from __future__ import annotations

import argparse
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import c071_probe as probe
import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202


EXPERIMENT_ID = "C302"
EXPERIMENT_SLUG = "C302_sentence_boundary_cap_truncation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C302_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C302 C111 sentence-boundary cap truncation aggregate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=302)
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


def truncate_at_sentence(text: str) -> tuple[str, bool]:
    stripped = str(text).strip()
    if not stripped:
        return stripped, False
    matches = list(re.finditer(r"[.!?。！？](?:[\"'»”)\]]+)?(?:\s|$)", stripped))
    if not matches:
        return stripped, False
    cut = matches[-1].end()
    candidate = stripped[:cut].strip()
    if len(candidate) < max(40, int(len(stripped) * 0.35)):
        return stripped, False
    return candidate, candidate != stripped


def summarize_rows(
    solution: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    outputs: Any,
    apply_truncation: bool,
) -> dict[str, Any]:
    quality: Counter[str] = Counter()
    validity: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handlers: Counter[str] = Counter()
    output_tokens: list[int] = []
    changed_count = 0
    eligible_count = 0

    for row, out in zip(rows, outputs):
        completion = out.outputs[0]
        raw_answer = completion.text.strip()
        token_ids = getattr(completion, "token_ids", None)
        raw_tokens = len(token_ids) if token_ids is not None else len(tokenizer(raw_answer).input_ids)
        raw_flags = c202.retry_base.valid_flags(raw_answer, raw_tokens, solution.MAX_NEW_TOKENS)
        answer = raw_answer
        changed = False
        if apply_truncation and raw_flags["hit_max_tokens"]:
            eligible_count += 1
            answer, changed = truncate_at_sentence(raw_answer)
        out_tokens = len(tokenizer(answer).input_ids)
        output_tokens.append(out_tokens)

        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(question)
        final, handler_name = rollback.c111_stack(solution, question, answer)
        flags = c202.retry_base.valid_flags(answer, out_tokens, solution.MAX_NEW_TOKENS)

        handlers[handler_name] += 1
        c202.retry_base.quality_update(quality, final, reference)
        c202.retry_base.quality_update(by_category[category], final, reference)
        c202.retry_base.quality_update(by_bucket[bucket], final, reference)
        validity["rows"] += 1
        validity["empty"] += int(flags["empty"])
        validity["thinking"] += int(flags["thinking"])
        validity["hit_max_tokens"] += int(flags["hit_max_tokens"])
        validity["repetition_loop"] += int(flags["repetition_loop"])
        validity["deterministic_first_fire"] += int(handler_name != "fallback_model")
        changed_count += int(changed)

    return {
        "quality": agg.rates({"overall": quality})["overall"],
        "validity": {k: int(v) for k, v in validity.items()},
        "handler_counts": {k: int(v) for k, v in handlers.items()},
        "tokens": {
            "avg_output_tokens": sum(output_tokens) / max(1, len(output_tokens)),
            "max_output_tokens": max(output_tokens) if output_tokens else None,
        },
        "changed_output_count": int(changed_count),
        "eligible_cap_count": int(eligible_count),
        "by_category": dict(sorted(agg.rates(by_category).items())),
        "by_bucket": dict(sorted(agg.rates(by_bucket).items())),
    }


def delta(variant: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact": variant["quality"].get("exact", 0) - baseline["quality"].get("exact", 0),
        "final_line_exact": variant["quality"].get("final_line_exact", 0)
        - baseline["quality"].get("final_line_exact", 0),
        "ref_in_output": variant["quality"].get("ref_in_output", 0)
        - baseline["quality"].get("ref_in_output", 0),
        "output_in_ref": variant["quality"].get("output_in_ref", 0)
        - baseline["quality"].get("output_in_ref", 0),
        "hit_max_tokens": variant["validity"].get("hit_max_tokens", 0)
        - baseline["validity"].get("hit_max_tokens", 0),
        "repetition_loop": variant["validity"].get("repetition_loop", 0)
        - baseline["validity"].get("repetition_loop", 0),
        "empty": variant["validity"].get("empty", 0) - baseline["validity"].get("empty", 0),
        "avg_output_tokens": variant["tokens"].get("avg_output_tokens", 0.0)
        - baseline["tokens"].get("avg_output_tokens", 0.0),
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
    prompts = [probe.apply_user_only_template(tokenizer, str(row["question"]), True, c111.USER_PREFIX) for row in rows]

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

    baseline = summarize_rows(c111, tokenizer, rows, outputs, apply_truncation=False)
    variant = summarize_rows(c111, tokenizer, rows, outputs, apply_truncation=True)
    overall_delta = delta(variant, baseline)
    projected_total_4000_s = startup_s + (generation_s / max(1, len(rows))) * 4000
    success_gate = {
        "changed_rows_positive": variant.get("changed_output_count", 0) > 0,
        "cap_hits_drop": overall_delta.get("hit_max_tokens", 0) < 0,
        "ref_in_output_nonnegative": overall_delta.get("ref_in_output", 0) >= 0,
        "output_in_ref_nonnegative": overall_delta.get("output_in_ref", 0) >= 0,
        "repetition_not_increased": overall_delta.get("repetition_loop", 0) <= 0,
        "runtime_under_12_min": projected_total_4000_s < 720,
    }
    success_gate["pass"] = bool(
        success_gate["changed_rows_positive"]
        and success_gate["cap_hits_drop"]
        and success_gate["ref_in_output_nonnegative"]
        and success_gate["output_in_ref_nonnegative"]
        and success_gate["repetition_not_increased"]
        and success_gate["runtime_under_12_min"]
    )
    return {
        **summary,
        "status": "completed",
        "decision_recommendation": "MUTATE" if success_gate["pass"] else "KILL",
        "reason": "Sentence-boundary truncation passed local gates." if success_gate["pass"] else "Sentence-boundary truncation did not clear local gates.",
        "sample_source": args.sample_source,
        "sample_size": len(rows),
        "baseline": baseline,
        "variant": variant,
        "overall_delta": overall_delta,
        "success_gate": success_gate,
        "runtime": {
            "startup_s": startup_s,
            "generation_s": generation_s,
            "projected_total_4000_s": projected_total_4000_s,
            "peak_gpu_mb": sampler.peak_mb,
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    baseline = summary.get("baseline") or {}
    variant = summary.get("variant") or {}
    delta_data = summary.get("overall_delta") or {}
    runtime = summary.get("runtime") or {}
    gate = summary.get("success_gate") or {}
    lines = [
        "# C302 Sentence-Boundary Cap Truncation",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- One mechanism: only when the raw C111 generation hits max tokens, truncate to the last complete sentence before C111 postprocessing.",
        "- Aggregate diagnostics only; no raw prompts, references, outputs, row ids, model weights, or adapter weights returned.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- sample source / size: `{summary.get('sample_source')}` / `{summary.get('sample_size')}`",
        "",
        "## Aggregate Metrics",
        f"- baseline quality: `{baseline.get('quality')}`",
        f"- variant quality: `{variant.get('quality')}`",
        f"- delta: `{delta_data}`",
        f"- baseline validity: `{baseline.get('validity')}`",
        f"- variant validity: `{variant.get('validity')}`",
        f"- eligible cap rows: `{variant.get('eligible_cap_count')}`",
        f"- changed output count: `{variant.get('changed_output_count')}`",
        f"- baseline tokens: `{baseline.get('tokens')}`",
        f"- variant tokens: `{variant.get('tokens')}`",
        "",
        "## Gate",
        f"- success gate: `{gate}`",
        "",
        "## Runtime",
        f"- startup seconds: `{runtime.get('startup_s')}`",
        f"- generation seconds: `{runtime.get('generation_s')}`",
        f"- projected 4000-query seconds: `{runtime.get('projected_total_4000_s')}`",
        f"- peak GPU MB: `{runtime.get('peak_gpu_mb')}`",
        "",
        "## Hygiene",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- prompts/references/outputs returned: `{summary.get('prompts_returned')}` / `{summary.get('references_returned')}` / `{summary.get('outputs_returned')}`",
        "",
        "## Next",
        "Scale only if cap hits drop with nonnegative containment and no repetition increase.",
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
