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
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202
import c266_c111_reference_style_gap as style_gap


EXPERIMENT_ID = "C267"
EXPERIMENT_SLUG = "C267_answer_marker_neutralization"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C267_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


MARKER_RE = re.compile(
    r"(?im)^\s*(?:ответ|итоговый ответ|итог|answer|final answer)\s*[:：\-]\s*"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C267 answer-marker neutralization prototype.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=267)
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


def neutralize_answer_markers(text: str) -> tuple[str, bool]:
    original = str(text)
    cleaned = MARKER_RE.sub("", original)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return original, False
    return cleaned, cleaned != original.strip()


def summarize_precomputed(
    rows: list[dict[str, Any]],
    baseline_finals: list[str],
    variant_finals: list[str],
    changed_flags: list[bool],
) -> dict[str, Any]:
    baseline_quality: Counter[str] = Counter()
    variant_quality: Counter[str] = Counter()
    by_category_delta: defaultdict[str, Counter[str]] = defaultdict(Counter)
    changed = 0
    changed_ref_delta = 0
    changed_output_ref_delta = 0
    baseline_pairs = []
    variant_pairs = []
    for row, base_final, variant_final, was_changed in zip(rows, baseline_finals, variant_finals, changed_flags):
        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        category = str(row.get("category", "unknown"))
        bucket = agg.feature_bucket(question)
        retry_base.quality_update(baseline_quality, base_final, reference)
        retry_base.quality_update(variant_quality, variant_final, reference)
        base_ref = int(retry_base.contains_normalized(base_final, reference))
        var_ref = int(retry_base.contains_normalized(variant_final, reference))
        base_out_ref = int(retry_base.contains_normalized(reference, base_final))
        var_out_ref = int(retry_base.contains_normalized(reference, variant_final))
        by_category_delta[category]["rows"] += 1
        by_category_delta[category]["changed"] += int(was_changed)
        by_category_delta[category]["ref_in_output_delta"] += var_ref - base_ref
        by_category_delta[category]["output_in_ref_delta"] += var_out_ref - base_out_ref
        changed += int(was_changed)
        if was_changed:
            changed_ref_delta += var_ref - base_ref
            changed_output_ref_delta += var_out_ref - base_out_ref
        baseline_pairs.append((bucket, reference, base_final))
        variant_pairs.append((bucket, reference, variant_final))
    baseline_rates = agg.rates({"overall": baseline_quality})["overall"]
    variant_rates = agg.rates({"overall": variant_quality})["overall"]
    deltas = {
        "exact": variant_rates.get("exact", 0) - baseline_rates.get("exact", 0),
        "final_line_exact": variant_rates.get("final_line_exact", 0) - baseline_rates.get("final_line_exact", 0),
        "ref_in_output": variant_rates.get("ref_in_output", 0) - baseline_rates.get("ref_in_output", 0),
        "output_in_ref": variant_rates.get("output_in_ref", 0) - baseline_rates.get("output_in_ref", 0),
    }
    return {
        "baseline_quality": baseline_rates,
        "variant_quality": variant_rates,
        "delta_variant_minus_baseline": deltas,
        "changed_rows": int(changed),
        "changed_row_ref_in_output_delta": int(changed_ref_delta),
        "changed_row_output_in_ref_delta": int(changed_output_ref_delta),
        "category_deltas": {
            k: {kk: int(vv) for kk, vv in v.items()} for k, v in sorted(by_category_delta.items())
        },
        "baseline_style": style_gap.summarize_style(baseline_pairs),
        "variant_style": style_gap.summarize_style(variant_pairs),
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
        variant, changed = neutralize_answer_markers(final)
        baseline_finals.append(final)
        variant_finals.append(variant)
        changed_flags.append(changed)

    paired = summarize_precomputed(rows, baseline_finals, variant_finals, changed_flags)
    delta = paired["delta_variant_minus_baseline"]
    baseline_style = paired["baseline_style"]
    variant_style = paired["variant_style"]
    marker_drop = baseline_style["output_markers"].get("answer_marker", 0) - variant_style["output_markers"].get(
        "answer_marker", 0
    )
    template_gain = variant_style.get("template_match", 0) - baseline_style.get("template_match", 0)
    quality_ok = (
        delta["exact"] >= 0
        and delta["final_line_exact"] >= 0
        and delta["ref_in_output"] >= 0
        and delta["output_in_ref"] >= 0
    )
    style_ok = marker_drop > 0 and template_gain >= 0
    decision = "MUTATE" if quality_ok and style_ok else "KILL"
    reason = (
        "Answer-marker neutralization preserved quality and improved marker/style aggregates."
        if decision == "MUTATE"
        else "Answer-marker neutralization failed quality or style gates."
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
            "gate": {
                "quality_nonnegative": quality_ok,
                "style_marker_drop": int(marker_drop),
                "template_match_delta": int(template_gain),
                "changed_rows": paired["changed_rows"],
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    paired = summary.get("paired_postprocess") or {}
    lines = [
        "# C267 Answer-Marker Neutralization Prototype",
        "",
        "## Objective",
        "- No leaderboard submission or submission zip.",
        "- Only one mechanism: remove leading answer/final markers from C111 final outputs while preserving content.",
        "- Compare unchanged C111 and marker-neutralized C111 on the same generated outputs.",
        "- Return aggregate metrics only; no raw prompts, references, outputs, row ids, datasets, weights, or adapters.",
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
        "## Baseline Quality",
        f"`{paired.get('baseline_quality')}`",
        "",
        "## Variant Quality",
        f"`{paired.get('variant_quality')}`",
        "",
        "## Style Baseline",
        f"`{paired.get('baseline_style')}`",
        "",
        "## Style Variant",
        f"`{paired.get('variant_style')}`",
        "",
        "## Category Deltas",
        f"`{paired.get('category_deltas')}`",
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
