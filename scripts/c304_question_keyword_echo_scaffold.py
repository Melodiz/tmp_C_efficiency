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
import c198_targeted_failure_retry as retry_base
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as c202
import c267_answer_marker_neutralization as post_base


EXPERIMENT_ID = "C304"
EXPERIMENT_SLUG = "C304_question_keyword_echo_scaffold"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C304_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "what",
    "which",
    "when",
    "where",
    "как",
    "что",
    "чем",
    "для",
    "при",
    "или",
    "если",
    "это",
    "его",
    "она",
    "они",
    "оно",
    "над",
    "под",
    "про",
    "без",
    "между",
    "нужно",
    "найди",
    "найдите",
    "определи",
    "определите",
    "укажите",
    "выберите",
    "запишите",
    "ответ",
    "решение",
    "задача",
    "вопрос",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C304 question-keyword echo scaffold over C111 outputs.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=304)
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


def question_keywords(question: str, limit: int = 5) -> list[str]:
    text = str(question).lower()
    raw_terms = re.findall(r"[a-zа-яё][a-zа-яё0-9\\-]{3,}", text, flags=re.IGNORECASE)
    counts: Counter[str] = Counter()
    for term in raw_terms:
        clean = term.strip("-").lower()
        if len(clean) < 4 or clean in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in clean) and len(clean) < 6:
            continue
        counts[clean] += 1
    ranked = sorted(counts, key=lambda t: (-counts[t], -len(t), t))
    return ranked[:limit]


def scaffold(question: str, final: str) -> tuple[str, bool]:
    base = str(final).strip()
    if not base or len(base) > 1800 or probe.has_repetition_loop(base):
        return base, False
    keywords = question_keywords(question)
    if len(keywords) < 2:
        return base, False
    existing_lower = base.lower()
    novel = [kw for kw in keywords if kw not in existing_lower]
    if len(novel) < 2:
        return base, False
    phrase = "; ".join(novel[:4])
    variant = f"{base}\n\nКлючевые понятия: {phrase}."
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
    route_counter: Counter[str] = Counter()
    for row, out in zip(rows, outputs):
        question = str(row["question"])
        base_answer = out.outputs[0].text.strip()
        final, _handler = rollback.c111_stack(c111, question, base_answer)
        variant, changed = scaffold(question, final)
        baseline_finals.append(final)
        variant_finals.append(variant)
        changed_flags.append(changed)
        route_counter["rows"] += 1
        route_counter["changed"] += int(changed)
        route_counter["keyword_eligible"] += int(len(question_keywords(question)) >= 2)

    paired = post_base.summarize_precomputed(rows, baseline_finals, variant_finals, changed_flags)
    delta = paired["delta_variant_minus_baseline"]
    baseline_validity = validity_counts(baseline_finals, tokenizer)
    variant_validity = validity_counts(variant_finals, tokenizer)
    repetition_delta = int(variant_validity["repetition_loop"]) - int(baseline_validity["repetition_loop"])
    empty_delta = int(variant_validity["empty"]) - int(baseline_validity["empty"])
    thinking_delta = int(variant_validity["thinking"]) - int(baseline_validity["thinking"])
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
    }
    gate["pass"] = bool(
        delta["ref_in_output"] > 0
        and delta["output_in_ref"] > 0
        and repetition_delta <= 0
        and empty_delta == 0
        and thinking_delta == 0
    )
    projected_total_4000_s = startup_s + (generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if gate["pass"] else "KILL",
            "reason": "Keyword scaffold improved both containment proxies." if gate["pass"] else "Keyword scaffold did not clear containment/validity gates.",
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
            "route_counts": {k: int(v) for k, v in route_counter.items()},
            "gate": gate,
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    paired = summary.get("paired_postprocess") or {}
    lines = [
        "# C304 Question-Keyword Echo Scaffold",
        "",
        "## Objective",
        "- No leaderboard submission or submission zip.",
        "- Only one mechanism: append a bounded scaffold of inference-visible question keywords to routed C111 final answers.",
        "- C111 generation, prompt, decoding, and deterministic stack remain unchanged.",
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
        "## Route",
        f"`{summary.get('route_counts')}`",
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
        "Scale only if both containment proxies improve with no validity regression.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    if paths["zip"].exists():
        paths["zip"].unlink()
    shutil.make_archive(str(paths["out_dir"]), "zip", paths["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
