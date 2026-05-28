from __future__ import annotations

import argparse
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
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c267_answer_marker_neutralization as post_base
import c293_strict_math_scratchpad_route as c293
import c294_strict_scratchpad_extraction as c294


EXPERIMENT_ID = "C305"
EXPERIMENT_SLUG = "C305_scratchpad_answer_augmented_c111"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C305_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C305 strict scratchpad answer hint plus preserved C111 output.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=305)
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


def normalize_answer_hint(text: str) -> str:
    hint = c294.extract_final_answer(text)
    hint = re.sub(r"\s+", " ", str(hint)).strip(" .,!?:;")
    if len(hint) > 80:
        return ""
    if len(hint) < 1:
        return ""
    return hint


def augment(base_final: str, scratchpad_text: str, route: str) -> tuple[str, bool]:
    base = str(base_final).strip()
    if route != "strict_math_scratchpad" or not base:
        return base, False
    hint = normalize_answer_hint(scratchpad_text)
    if not hint:
        return base, False
    if hint.lower() in base.lower():
        return base, False
    variant = f"Ответ: {hint}.\n\n{base}"
    return variant, variant != base


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
        "mechanism": "strict math scratchpad-derived answer hint prepended to otherwise preserved C111 final output",
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
    routes: list[str] = []
    scratchpad_prompts: list[str] = []
    route_counts: Counter[str] = Counter()
    for row in rows:
        route, prefix = c293.route_prefix(str(row["question"]), c111.USER_PREFIX)
        routes.append(route)
        route_counts[route] += 1
        scratchpad_prompts.append(probe.apply_user_only_template(tokenizer, str(row["question"]), True, prefix))

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
    control_t0 = time.perf_counter()
    control_outputs = llm.generate(control_prompts, sampling_params=sampling)
    control_generation_s = time.perf_counter() - control_t0
    scratch_t0 = time.perf_counter()
    scratchpad_outputs = llm.generate(scratchpad_prompts, sampling_params=sampling)
    scratchpad_generation_s = time.perf_counter() - scratch_t0
    sampler.stop()

    baseline_finals: list[str] = []
    variant_finals: list[str] = []
    changed_flags: list[bool] = []
    hint_counter: Counter[str] = Counter()
    for row, base_out, scratch_out, route in zip(rows, control_outputs, scratchpad_outputs, routes):
        question = str(row["question"])
        base_answer = base_out.outputs[0].text.strip()
        base_final, _handler = rollback.c111_stack(c111, question, base_answer)
        scratch_text = scratch_out.outputs[0].text.strip()
        variant, changed = augment(base_final, scratch_text, route)
        baseline_finals.append(base_final)
        variant_finals.append(variant)
        changed_flags.append(changed)
        hint_counter["changed"] += int(changed)
        hint_counter["route_strict_math"] += int(route == "strict_math_scratchpad")
        hint_counter["hint_extracted"] += int(bool(normalize_answer_hint(scratch_text)) and route == "strict_math_scratchpad")

    paired = post_base.summarize_precomputed(rows, baseline_finals, variant_finals, changed_flags)
    delta = paired["delta_variant_minus_baseline"]
    baseline_validity = validity_counts(baseline_finals, tokenizer)
    variant_validity = validity_counts(variant_finals, tokenizer)
    repetition_delta = int(variant_validity["repetition_loop"]) - int(baseline_validity["repetition_loop"])
    empty_delta = int(variant_validity["empty"]) - int(baseline_validity["empty"])
    thinking_delta = int(variant_validity["thinking"]) - int(baseline_validity["thinking"])
    projected_total_4000_s = startup_s + (
        (control_generation_s + scratchpad_generation_s) / max(1, len(rows))
    ) * 4000
    gate = {
        "changed_rows": paired["changed_rows"],
        "exact_delta": delta["exact"],
        "final_line_exact_delta": delta["final_line_exact"],
        "ref_in_output_delta": delta["ref_in_output"],
        "output_in_ref_delta": delta["output_in_ref"],
        "repetition_delta": repetition_delta,
        "empty_delta": empty_delta,
        "thinking_delta": thinking_delta,
        "avg_token_delta": variant_validity["avg_tokens"] - baseline_validity["avg_tokens"],
        "runtime_under_12_min": projected_total_4000_s < 720,
    }
    gate["pass"] = bool(
        delta["ref_in_output"] > 0
        and delta["output_in_ref"] >= 0
        and repetition_delta <= 0
        and empty_delta == 0
        and thinking_delta == 0
        and gate["runtime_under_12_min"]
    )
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if gate["pass"] else "KILL",
            "reason": "Scratchpad hint augmentation cleared local gates." if gate["pass"] else "Scratchpad hint augmentation did not clear local gates.",
            "raw_task_data_read_remote_only": True,
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
                "route_counts": {k: int(v) for k, v in route_counts.items()},
            },
            "runtime": {
                "startup_s": startup_s,
                "control_generation_s": control_generation_s,
                "scratchpad_generation_s": scratchpad_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "paired_postprocess": paired,
            "baseline_validity": baseline_validity,
            "variant_validity": variant_validity,
            "hint_counts": {k: int(v) for k, v in hint_counter.items()},
            "gate": gate,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    paired = summary.get("paired_postprocess") or {}
    lines = [
        "# C305 Scratchpad Answer-Augmented C111",
        "",
        "## Objective",
        "- No leaderboard submission or submission zip.",
        "- One mechanism: on strict math route, prepend only the extracted scratchpad final answer while preserving the original C111 final output.",
        "- This differs from C294 because it augments C111 instead of replacing C111 with the extracted short answer.",
        "- Return aggregate metrics only; no raw prompts, references, outputs, row ids, datasets, weights, or adapters.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Gate",
        f"`{summary.get('gate')}`",
        "",
        "## Hint Counts",
        f"`{summary.get('hint_counts')}`",
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
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
        "",
        "## Next",
        "Scale only if hint augmentation improves containment without validity regression.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    shutil.make_archive(str(paths["out_dir"]), "zip", paths["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
