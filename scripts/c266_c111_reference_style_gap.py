from __future__ import annotations

import argparse
import re
import shutil
import statistics
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


EXPERIMENT_ID = "C266"
EXPERIMENT_SLUG = "C266_c111_reference_style_gap"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C266_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C266 C111 output-vs-reference style gap diagnostic.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=266)
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


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def safe_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values = sorted(int(v) for v in values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0

    def quantile(p: float) -> float:
        if len(values) == 1:
            return float(values[0])
        pos = (len(values) - 1) * p
        lo = int(pos)
        hi = min(len(values) - 1, lo + (0 if pos == int(pos) else 1))
        if lo == hi:
            return float(values[lo])
        return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))

    return {
        "count": len(values),
        "mean": float(mean),
        "std": float(std),
        "std_over_mean": float(std / mean) if mean else None,
        "min": int(values[0]),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "max": int(values[-1]),
    }


def length_bucket(tokens: int) -> str:
    if tokens <= 3:
        return "tiny"
    if tokens <= 12:
        return "short"
    if tokens <= 40:
        return "medium"
    if tokens <= 120:
        return "long"
    return "very_long"


def style_profile(text: str) -> dict[str, Any]:
    text = str(text).strip()
    low = text.lower().replace("ё", "е")
    lines = [line for line in text.splitlines() if line.strip()]
    tokens = word_count(text)
    answer_any = bool(
        re.search(r"(?:^|\n)\s*(?:ответ|итоговый ответ|итог|answer|final answer)\s*[:：\-]", low, flags=re.I)
    )
    bullet = bool(re.search(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", text))
    equation = bool(re.search(r"[=<>]|\d\s*[+*/:×\-]\s*\d", text))
    ending_punct = bool(re.search(r"[.!?。]$", text))
    if answer_any:
        structure = "answer_marker"
    elif bullet:
        structure = "list"
    elif len(lines) > 1:
        structure = "multiline"
    else:
        structure = "plain"
    line_label = "single_line" if len(lines) <= 1 else "few_lines" if len(lines) <= 4 else "many_lines"
    return {
        "tokens": tokens,
        "lines": len(lines),
        "answer_marker": answer_any,
        "list_like": bullet,
        "multiline": len(lines) > 1,
        "equation_like": equation,
        "ending_punct": ending_punct,
        "template": "|".join(
            [
                length_bucket(tokens),
                structure,
                line_label,
                "sentential" if ending_punct else "bare",
                "equation" if equation else "no_equation",
            ]
        ),
    }


def summarize_style(pairs: list[tuple[str, str, str]]) -> dict[str, Any]:
    ref_tokens: list[int] = []
    out_tokens: list[int] = []
    ref_lines: list[int] = []
    out_lines: list[int] = []
    ref_markers: Counter[str] = Counter()
    out_markers: Counter[str] = Counter()
    ref_templates: Counter[str] = Counter()
    out_templates: Counter[str] = Counter()
    bucket_rows: defaultdict[str, int] = defaultdict(int)
    bucket_ref_tokens: defaultdict[str, list[int]] = defaultdict(list)
    bucket_out_tokens: defaultdict[str, list[int]] = defaultdict(list)
    bucket_ref_markers: defaultdict[str, Counter[str]] = defaultdict(Counter)
    bucket_out_markers: defaultdict[str, Counter[str]] = defaultdict(Counter)
    template_match = 0

    for bucket, reference, output in pairs:
        ref = style_profile(reference)
        out = style_profile(output)
        ref_tokens.append(ref["tokens"])
        out_tokens.append(out["tokens"])
        ref_lines.append(ref["lines"])
        out_lines.append(out["lines"])
        ref_templates[ref["template"]] += 1
        out_templates[out["template"]] += 1
        template_match += int(ref["template"] == out["template"])
        bucket_rows[bucket] += 1
        bucket_ref_tokens[bucket].append(ref["tokens"])
        bucket_out_tokens[bucket].append(out["tokens"])
        for key in ["answer_marker", "list_like", "multiline", "equation_like", "ending_punct"]:
            ref_markers[key] += int(ref[key])
            out_markers[key] += int(out[key])
            bucket_ref_markers[bucket][key] += int(ref[key])
            bucket_out_markers[bucket][key] += int(out[key])

    rows = len(pairs)
    by_bucket = []
    for bucket, count in sorted(bucket_rows.items(), key=lambda item: -item[1])[:25]:
        ref_mean = statistics.fmean(bucket_ref_tokens[bucket]) if bucket_ref_tokens[bucket] else 0.0
        out_mean = statistics.fmean(bucket_out_tokens[bucket]) if bucket_out_tokens[bucket] else 0.0
        by_bucket.append(
            {
                "bucket": bucket,
                "rows": int(count),
                "ref_mean_tokens": float(ref_mean),
                "output_mean_tokens": float(out_mean),
                "output_ref_token_ratio": float(out_mean / ref_mean) if ref_mean else None,
                "ref_multiline_share": bucket_ref_markers[bucket]["multiline"] / count,
                "output_multiline_share": bucket_out_markers[bucket]["multiline"] / count,
                "ref_list_share": bucket_ref_markers[bucket]["list_like"] / count,
                "output_list_share": bucket_out_markers[bucket]["list_like"] / count,
            }
        )

    return {
        "rows": rows,
        "reference_tokens": safe_stats(ref_tokens),
        "output_tokens": safe_stats(out_tokens),
        "reference_lines": safe_stats(ref_lines),
        "output_lines": safe_stats(out_lines),
        "output_ref_mean_token_ratio": (statistics.fmean(out_tokens) / statistics.fmean(ref_tokens))
        if ref_tokens and statistics.fmean(ref_tokens)
        else None,
        "template_match": int(template_match),
        "template_match_share": float(template_match / rows) if rows else 0.0,
        "reference_markers": {k: int(v) for k, v in ref_markers.items()},
        "output_markers": {k: int(v) for k, v in out_markers.items()},
        "top_reference_templates": [
            {"template": name, "count": int(count), "share": count / rows if rows else 0.0}
            for name, count in ref_templates.most_common(15)
        ],
        "top_output_templates": [
            {"template": name, "count": int(count), "share": count / rows if rows else 0.0}
            for name, count in out_templates.most_common(15)
        ],
        "by_bucket": by_bucket,
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

    quality = c202.summarize_rows(c111, tokenizer, rows, outputs)
    style_pairs: list[tuple[str, str, str]] = []
    for row, out in zip(rows, outputs):
        question = str(row["question"])
        reference = str(row.get("reference_answer", ""))
        base_answer = out.outputs[0].text.strip()
        final, _handler = rollback.c111_stack(c111, question, base_answer)
        style_pairs.append((agg.feature_bucket(question), reference, final))
    style = summarize_style(style_pairs)

    ratio = style.get("output_ref_mean_token_ratio") or 0.0
    ref_multiline = style["reference_markers"].get("multiline", 0)
    out_multiline = style["output_markers"].get("multiline", 0)
    ref_list = style["reference_markers"].get("list_like", 0)
    out_list = style["output_markers"].get("list_like", 0)
    rows_n = max(1, style.get("rows", 0))
    list_gap = (ref_list - out_list) / rows_n
    multiline_gap = (ref_multiline - out_multiline) / rows_n
    structural_gap = list_gap >= 0.20 or multiline_gap >= 0.20
    length_gap = ratio < 0.80
    gate_pass = bool(structural_gap or length_gap)
    decision = "MUTATE" if gate_pass else "KILL"
    reason = (
        "C111 outputs have a broad style/length gap versus references; queue S1 prototype."
        if gate_pass
        else "C111 outputs are already close enough in reference style that S1 postprocessing has low ceiling."
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
            "c111_quality": quality,
            "style_gap": style,
            "gate": {
                "s1_style_gap_pass": gate_pass,
                "output_ref_mean_token_ratio": ratio,
                "list_gap_share_ref_minus_output": list_gap,
                "multiline_gap_share_ref_minus_output": multiline_gap,
                "length_gap": length_gap,
                "structural_gap": structural_gap,
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C266 C111 Reference-Style Gap Diagnostic",
        "",
        "## Objective",
        "- No leaderboard submission or submission zip.",
        "- Compare C111 output style against reference style on a locked validation aggregate.",
        "- Return only aggregate metrics; no raw prompts, references, outputs, row ids, datasets, weights, or adapters.",
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
        "## C111 Quality",
        f"`{summary.get('c111_quality')}`",
        "",
        "## Style Gap",
        f"`{summary.get('style_gap')}`",
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
