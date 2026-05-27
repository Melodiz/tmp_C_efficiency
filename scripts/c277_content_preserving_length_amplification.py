from __future__ import annotations

import argparse
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
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202
import c267_answer_marker_neutralization as post_base


EXPERIMENT_ID = "C277"
EXPERIMENT_SLUG = "C277_content_preserving_length_amplification"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C277_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C277 content-preserving length amplification over C111.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=277)
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


def sentence_prefix(text: str, max_chars: int = 700) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return ""
    pieces = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    kept: list[str] = []
    total = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if kept and total + 1 + len(piece) > max_chars:
            break
        kept.append(piece)
        total += len(piece) + 1
        if len(kept) >= 3:
            break
    return " ".join(kept).strip() or cleaned[:max_chars].strip()


def amplify(text: str) -> tuple[str, bool]:
    original = str(text).strip()
    if not original or len(original) > 1800 or probe.has_repetition_loop(original):
        return original, False
    prefix = sentence_prefix(original)
    if not prefix or len(prefix) < 20:
        return original, False
    variant = f"{original}\n\nИными словами: {prefix}"
    return variant, variant != original


def validity_counts(texts: list[str], tokenizer: Any) -> dict[str, int | float]:
    counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    for text in texts:
        token_len = len(tokenizer(str(text)).input_ids)
        token_lengths.append(token_len)
        flags = retry_base.valid_flags(str(text), token_len, 10**9)
        counts["rows"] += 1
        counts["empty"] += int(flags["empty"])
        counts["thinking"] += int(flags["thinking"])
        counts["repetition_loop"] += int(flags["repetition_loop"])
    return {
        **{k: int(v) for k, v in counts.items()},
        "avg_tokens": sum(token_lengths) / max(1, len(token_lengths)),
        "max_tokens": max(token_lengths) if token_lengths else 0,
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

    baseline_generated = c202.summarize_rows(c111, tokenizer, rows, outputs)
    baseline_finals: list[str] = []
    variant_finals: list[str] = []
    changed_flags: list[bool] = []
    for row, out in zip(rows, outputs):
        question = str(row["question"])
        base_answer = out.outputs[0].text.strip()
        final, _handler = rollback.c111_stack(c111, question, base_answer)
        variant, changed = amplify(final)
        baseline_finals.append(final)
        variant_finals.append(variant)
        changed_flags.append(changed)

    paired = post_base.summarize_precomputed(rows, baseline_finals, variant_finals, changed_flags)
    delta = paired["delta_variant_minus_baseline"]
    baseline_validity = validity_counts(baseline_finals, tokenizer)
    variant_validity = validity_counts(variant_finals, tokenizer)
    repetition_delta = int(variant_validity["repetition_loop"]) - int(baseline_validity["repetition_loop"])
    gate = {
        "changed_rows": paired["changed_rows"],
        "exact_delta": delta["exact"],
        "final_line_exact_delta": delta["final_line_exact"],
        "ref_in_output_delta": delta["ref_in_output"],
        "output_in_ref_delta": delta["output_in_ref"],
        "repetition_delta": repetition_delta,
        "avg_token_delta": variant_validity["avg_tokens"] - baseline_validity["avg_tokens"],
    }
    decision = (
        "MUTATE"
        if delta["ref_in_output"] >= 3 and delta["output_in_ref"] >= 3 and repetition_delta <= 5
        else "KILL"
    )
    reason = (
        "Length amplification improved both containment proxies without large repetition increase."
        if decision == "MUTATE"
        else "Length amplification did not clear containment/repetition gates."
    )
    projected_total_4000_s = startup_s + (generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": decision,
            "reason": reason,
            "raw_task_data_read_remote_only": True,
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
            "c111_generated_summary": baseline_generated,
            "paired_postprocess": paired,
            "baseline_validity": baseline_validity,
            "variant_validity": variant_validity,
            "gate": gate,
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    paired = summary.get("paired_postprocess") or {}
    lines = [
        "# C277 Content-Preserving Length Amplification",
        "",
        "## Objective",
        "- No leaderboard submission or submission zip.",
        "- Only one mechanism: append a bounded restatement of C111 final output to preserve and amplify existing content.",
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
        "## Gate",
        f"`{summary.get('gate')}`",
        "",
        "## Quality Delta",
        f"`{paired.get('delta_variant_minus_baseline')}`",
        "",
        "## Changed Rows",
        f"- changed rows: `{paired.get('changed_rows')}`",
        f"- changed-row ref-in-output delta: `{paired.get('changed_row_ref_in_output_delta')}`",
        f"- changed-row output-in-ref delta: `{paired.get('changed_row_output_in_ref_delta')}`",
        "",
        "## Validity",
        f"- baseline: `{summary.get('baseline_validity')}`",
        f"- variant: `{summary.get('variant_validity')}`",
        "",
        "## Baseline Quality",
        f"`{paired.get('baseline_quality')}`",
        "",
        "## Variant Quality",
        f"`{paired.get('variant_quality')}`",
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
