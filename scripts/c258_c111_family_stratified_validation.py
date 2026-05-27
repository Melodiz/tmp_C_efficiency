from __future__ import annotations

import argparse
import os
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
import c201_c111_vs_current_stack_aggregate as c201


EXPERIMENT_ID = "C258"
EXPERIMENT_SLUG = "C258_c111_family_stratified_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C258_artifacts"
DEFAULT_SAMPLE_SIZE = 1200
DEFAULT_SEED = 258
MODEL_ID = "Qwen/Qwen3-8B-AWQ"


FAMILY_PATTERNS = {
    "date_time_calendar": [
        r"\b(?:date|day|month|year|calendar|weekday|hour|minute|second|time)\b",
        r"(?:дата|календар|день\s+недел|понедельник|вторник|среда|четверг|пятница|суббота|воскресенье|январ|феврал|март|апрел|июн|июл|август|сентябр|октябр|ноябр|декабр|час|минут|секунд|сутк)",
    ],
    "chemistry_formula": [
        r"\b(?:chemistry|chemical|mole|molar|reaction|acid|oxide|formula)\b",
        r"(?:хими|моляр|моль|реакц|кислот|оксид|формул|веществ|элемент|атом|ион|валентн)",
        r"\b(?:H2O|CO2|NaCl|HCl|H2SO4|O2|N2|CH4)\b",
    ],
    "sequence_progression": [
        r"\b(?:sequence|series|progression|next term|arithmetic progression|geometric progression)\b",
        r"(?:последовательн|прогресси|следующ(?:ее|ий|ая)?\s+числ|член\s+последовательности|ряд\s+чисел)",
    ],
    "base_number_system": [
        r"\b(?:binary|hexadecimal|octal|base\s*\d+|number system)\b",
        r"(?:двоичн|шестнадцатеричн|восьмеричн|систем[аеы]?\s+счислен|основани[ея]\s+\d+)",
    ],
    "geometry_coordinate": [
        r"\b(?:coordinate|radius|diameter|circle|triangle|rectangle|perimeter|area|angle|arc|slope)\b",
        r"(?:координат|радиус|диаметр|окружн|круг|треугольн|прямоугольн|периметр|площад|угол|дуг[аи]|наклон|центр)",
    ],
    "structured_language_list": [
        r"\b(?:anagram|letters|word form|part of speech|spelling|grammar|synonym|antonym)\b",
        r"(?:анаграм|букв|слова?|част[ьи]\s+речи|орфограф|граммат|синоним|антоним|ударен|падеж|морфолог)",
    ],
    "logic_table": [
        r"\b(?:truth table|logic|logical|boolean|statement)\b",
        r"(?:таблиц[аы]?\s+истин|логик|булев|высказыван|истинн|ложн)",
    ],
    "finance_percent": [
        r"\b(?:percent|discount|interest|price|cost|currency|dollar|euro)\b",
        r"(?:процент|скидк|стоимост|цена|рубл|доллар|евро|валют|прибыл|процентн)",
    ],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C258 C111 family-stratified aggregate validation.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def normalize_question(text: str) -> str:
    return str(text).lower().replace("ё", "е")


def families_for(question: str) -> list[str]:
    q = normalize_question(question)
    hits = []
    for family, patterns in FAMILY_PATTERNS.items():
        if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in patterns):
            hits.append(family)
    return hits or ["no_family"]


def compact(counter: Counter[str]) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def update_quality(counter: Counter[str], answer: str, reference: str) -> None:
    retry_base.quality_update(counter, answer, reference)


def add_validity(counter: Counter[str], base_answer: str, out_tokens: int, max_tokens: int, deterministic: bool) -> None:
    flags = retry_base.valid_flags(base_answer, out_tokens, max_tokens)
    counter["rows"] += 1
    counter["empty"] += int(flags["empty"])
    counter["thinking"] += int(flags["thinking"])
    counter["hit_max_tokens"] += int(flags["hit_max_tokens"])
    counter["repetition_loop"] += int(flags["repetition_loop"])
    counter["deterministic_first_fire"] += int(deterministic)
    counter["fallback_model"] += int(not deterministic)


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
        "c111_commit": c201.C111_COMMIT,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    from vllm import LLM, SamplingParams

    start = time.time()
    agg.install_final_path_dependencies()
    c111 = c201.load_c111_solution()
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

    overall_quality: Counter[str] = Counter()
    overall_validity: Counter[str] = Counter()
    by_family_quality: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_family_validity: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_family_handler: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_family_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    category_by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handler_counts: Counter[str] = Counter()
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
        final_answer, handler = c201.c111_stack(c111, question, base_answer)
        deterministic = handler != "fallback_model"
        handler_counts[handler] += 1
        update_quality(overall_quality, final_answer, reference)
        add_validity(overall_validity, base_answer, out_tokens, c111.MAX_NEW_TOKENS, deterministic)
        for family in families_for(question):
            family_counts[family] += 1
            update_quality(by_family_quality[family], final_answer, reference)
            add_validity(by_family_validity[family], base_answer, out_tokens, c111.MAX_NEW_TOKENS, deterministic)
            by_family_handler[family][handler] += 1
            by_family_bucket[family][bucket] += 1
            category_by_family[family][category] += 1

    projected_total_4000_s = startup_s + (generation_s / max(1, len(rows))) * 4000
    family_quality = rates(by_family_quality)
    family_validity = rates(by_family_validity)
    weak_families = []
    for family, counts in family_counts.most_common():
        q = family_quality.get(family, {})
        v = family_validity.get(family, {})
        rows_n = int(counts)
        visible_failures = int(v.get("hit_max_tokens", 0)) + int(v.get("repetition_loop", 0)) + int(v.get("empty", 0))
        weak_families.append(
            {
                "family": family,
                "rows": rows_n,
                "exact": int(q.get("exact", 0)),
                "ref_in_output": int(q.get("ref_in_output", 0)),
                "output_in_ref": int(q.get("output_in_ref", 0)),
                "final_line_exact": int(q.get("final_line_exact", 0)),
                "visible_failures": visible_failures,
                "deterministic_first_fire": int(v.get("deterministic_first_fire", 0)),
                "fallback_model": int(v.get("fallback_model", 0)),
            }
        )

    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE",
            "reason": "C111 family-stratified aggregate validation completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok", "pymorphy_available": bool(c111.get_morph_analyzer())},
            "sample_meta": {
                "source": args.sample_source,
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "total_seconds": time.time() - start,
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
            "overall_quality": rates({"overall": overall_quality})["overall"],
            "overall_validity": {k: int(v) for k, v in overall_validity.items()},
            "family_counts": compact(family_counts),
            "family_quality": family_quality,
            "family_validity": family_validity,
            "family_handlers": {family: compact(counts) for family, counts in by_family_handler.items()},
            "family_buckets": {
                family: [{"bucket": bucket, "count": int(count)} for bucket, count in counts.most_common(12)]
                for family, counts in by_family_bucket.items()
            },
            "family_categories": {
                family: [{"category": category, "count": int(count)} for category, count in counts.most_common(12)]
                for family, counts in category_by_family.items()
            },
            "weak_family_summary": weak_families,
            "handler_counts": compact(handler_counts),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C258 C111 Family-Stratified Aggregate Validation",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Measure proven C111 quality and visible failures by C257 family before any solver/model-route code port.",
        "- Return only aggregate metrics; no raw prompts, references, outputs, row ids, datasets, weights, or adapter files.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- imports: `{summary.get('imports')}`",
        "",
        "## Sample",
        f"`{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Overall",
        f"- quality: `{summary.get('overall_quality')}`",
        f"- validity: `{summary.get('overall_validity')}`",
        f"- handlers: `{summary.get('handler_counts')}`",
        "",
        "## Family Counts",
        f"`{summary.get('family_counts')}`",
        "",
        "## Weak Family Summary",
        f"`{summary.get('weak_family_summary')}`",
        "",
        "## Family Quality",
        f"`{summary.get('family_quality')}`",
        "",
        "## Family Validity",
        f"`{summary.get('family_validity')}`",
        "",
        "## Family Handlers",
        f"`{summary.get('family_handlers')}`",
        "",
        "## Family Buckets",
    ]
    for family, rows in (summary.get("family_buckets") or {}).items():
        lines.append(f"- {family}: `{rows}`")
    lines.extend(
        [
            "",
            "## Family Categories",
            f"`{summary.get('family_categories')}`",
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
    )
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
