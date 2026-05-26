from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import c071_probe as probe
import c169_lora_training_stack_import_smoke as base


EXPERIMENT_ID = "C195"
EXPERIMENT_SLUG = "C195_direct_probe_aggregate_validation"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C195_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
MORPH_PACKAGES = ("pymorphy3==2.0.6", "pymorphy3-dicts-ru", "razdel==0.5.0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C195 aggregate validation via proven direct probe path.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "probe_summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_probe_summary.json",
        "zip": out_dir.with_suffix(".zip"),
    }


def install_final_path_dependencies() -> None:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *MORPH_PACKAGES])


def final_line(text: str) -> str:
    matches = re.findall(r"(?:Итоговый ответ|Ответ|Answer)\s*[:：]\s*(.+)", str(text), flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else str(text).strip()


def norm(text: str) -> str:
    value = final_line(text).lower().replace("ё", "е")
    value = re.sub(r"^(ответ|итоговый ответ|answer)\s*[:：-]\s*", "", value, flags=re.IGNORECASE)
    value = value.replace("−", "-").replace(",", ".")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^0-9a-zа-я+\-*/=.,%√²³() ]+", "", value)
    return value.strip()


def answer_label(value: str) -> str:
    lines = [part.strip() for part in str(value).strip().splitlines() if part.strip()]
    if not lines:
        return "empty"
    if len(lines) > 1:
        return "multiline"
    if len(lines[0]) > 80:
        return "long"
    if len(lines[0].split()) > 14:
        return "essay_like"
    return "ok"


def feature_bucket(text: str) -> str:
    q = str(text).lower().replace("ё", "е")
    cyr = sum("а" <= ch <= "я" for ch in q)
    lat = sum("a" <= ch <= "z" for ch in q)
    script = "cyrillic" if cyr > lat else "latin" if lat > cyr else "mixed_or_symbolic"
    length = "q_short" if len(q) <= 80 else "q_medium" if len(q) <= 180 else "q_long" if len(q) <= 350 else "q_very_long"
    numeric = "num" if re.search(r"\d", q) else "nonnum"
    expr = "expr" if re.search(r"\d\s*[+*×xх/:=-]\s*\d", q) else "noexpr"
    openness = (
        "open"
        if re.search(r"\b(объясн|почему|напишите|сочин|эссе|опишите|перечислите|составьте|расскажите|докажите|explain|write|describe|list)\b", q)
        else "closed"
    )
    return "|".join([length, script, numeric, expr, openness])


def update(counter: Counter[str], answer: str, reference: str, base_answer: str) -> None:
    n_answer = norm(answer)
    n_ref = norm(reference)
    n_base = norm(base_answer)
    counter["rows"] += 1
    counter["exact"] += int(bool(n_ref) and n_answer == n_ref)
    counter["final_line_exact"] += int(bool(n_ref) and norm(final_line(answer)) == n_ref)
    counter["ref_in_output"] += int(bool(n_ref) and n_ref in n_answer)
    counter["output_in_ref"] += int(bool(n_answer) and n_answer in n_ref)
    counter["base_exact"] += int(bool(n_ref) and n_base == n_ref)


def rates(table: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, counts in table.items():
        rows = int(counts.get("rows", 0))
        item: dict[str, Any] = {name: int(value) for name, value in counts.items()}
        if rows:
            for name in ("exact", "final_line_exact", "ref_in_output", "output_in_ref", "base_exact"):
                item[name + "_rate"] = item.get(name, 0) / rows
        out[str(key)] = item
    return out


def first_handler(module: Any, question: str, base_answer: str) -> tuple[str, str]:
    handlers = [
        ("expression_substitution", module.expression_substitution_answer),
        ("algebra_equation", module.algebra_equation_answer),
        ("exact_numeric", module.exact_numeric_answer),
        ("direct_arithmetic", module.direct_arithmetic_answer),
        ("chemistry_stoichiometry", module.chemistry_stoichiometry_answer),
        ("geometry_exact", module.geometry_exact_answer),
        ("formulaic_math_physics", module.formulaic_math_physics_answer),
        ("structured_school_task", module.structured_school_task_answer),
        ("calculator_written_arithmetic", module.calculator_written_arithmetic_answer),
        ("russian_morph_grammar", module.russian_morph_grammar_answer),
        ("quantity_conversion", module.quantity_conversion_answer),
        ("km_meters", module.km_meters_answer),
    ]
    for name, func in handlers:
        value = func(question)
        if value is not None:
            return name, value
    cleaned = module.dedup_comma_loop(base_answer) or base_answer
    cleaned = module.cleanup_english_cloze_answer(question, cleaned) or cleaned
    return "fallback_model", cleaned


def run_validation(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
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
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    install_final_path_dependencies()
    spec = importlib.util.spec_from_file_location("task_c_solution_module", Path("simple_solution/solution.py"))
    solution = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(solution)
    run_args = SimpleNamespace(
        candidate="qwen3-8b-awq",
        model_id=MODEL_ID,
        baseline_local_path=str(probe.BASELINE_LOCAL_PATH),
        sample_source=args.sample_source,
        sample_size=args.sample_size,
        output_dir=str(paths["results_dir"]),
        max_model_len=solution.MAX_MODEL_LEN,
        max_tokens=solution.MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        dtype="float16",
        quantization="awq_marlin",
        gpu_memory_utilization=0.9,
        gpu_sample_interval=0.5,
        seed=args.seed,
        trust_remote_code=False,
        no_enable_thinking_false=False,
        user_prefix=solution.USER_PREFIX,
        skip_hf_metadata=True,
        save_prompts=False,
        dry_run=False,
        no_fail=True,
    )
    probe_summary = probe.run_probe(run_args)
    base.write_json(paths["probe_summary"], probe_summary)
    summary["raw_task_data_read_remote_only"] = True

    output_path = Path((probe_summary.get("paths") or {}).get("outputs", ""))
    samples_path = Path((probe_summary.get("paths") or {}).get("samples", ""))
    overall: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_label: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_handler: defaultdict[str, Counter[str]] = defaultdict(Counter)
    handler_counts: Counter[str] = Counter()
    validity: Counter[str] = Counter()

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question = str(row.get("question", ""))
            reference = str(row.get("reference_answer", ""))
            base_answer = str(row.get("answer", ""))
            handler, final_answer = first_handler(solution, question, base_answer)
            category = str(row.get("category", "unknown"))
            bucket = feature_bucket(question)
            label = answer_label(reference)
            handler_counts[handler] += 1
            validity["rows"] += 1
            validity["base_hit_max_tokens"] += int(bool(row.get("hit_max_tokens")))
            validity["base_empty"] += int(not base_answer)
            validity["base_thinking_trace"] += int(bool(row.get("has_thinking_trace")))
            validity["base_repetition_loop"] += int(bool(row.get("repetition_loop_suspected")))
            validity["deterministic_first_fire"] += int(handler != "fallback_model")
            validity["fallback_model"] += int(handler == "fallback_model")
            update(overall, final_answer, reference, base_answer)
            update(by_category[category], final_answer, reference, base_answer)
            update(by_bucket[bucket], final_answer, reference, base_answer)
            update(by_label[label], final_answer, reference, base_answer)
            update(by_handler[handler], final_answer, reference, base_answer)

    for raw_path in (output_path, samples_path):
        if raw_path.exists():
            raw_path.unlink()

    summary.update(
        {
            "status": "completed" if probe_summary.get("status") == "completed" else "failed",
            "decision_recommendation": "MUTATE" if probe_summary.get("status") == "completed" else "INVESTIGATE",
            "reason": "Direct-probe aggregate validation completed." if probe_summary.get("status") == "completed" else "Direct-probe aggregate validation failed.",
            "imports": {"solution": "ok", "pymorphy_available": bool(solution.get_morph_analyzer())},
            "sample_meta": probe_summary.get("sample"),
            "runtime": {"total_seconds": time.time() - start, "probe_runtime": probe_summary.get("runtime")},
            "tokens": probe_summary.get("tokens"),
            "validity": {k: int(v) for k, v in validity.items()},
            "quality": rates({"overall": overall})["overall"],
            "handler_counts": {k: int(v) for k, v in handler_counts.items()},
            "by_category": dict(sorted(rates(by_category).items())),
            "by_bucket": dict(sorted(rates(by_bucket).items(), key=lambda kv: -kv[1].get("rows", 0))[:40]),
            "by_target_label": dict(sorted(rates(by_label).items())),
            "by_first_handler": dict(sorted(rates(by_handler).items(), key=lambda kv: -kv[1].get("rows", 0))),
            "model_loaded": probe_summary.get("status") == "completed",
            "raw_temp_files_deleted": {
                "outputs": not output_path.exists(),
                "samples": not samples_path.exists(),
            },
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C195 Direct-Probe Aggregate Validation",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Use the previously proven direct vLLM probe path, then delete raw temporary outputs before packaging.",
        "- Return only aggregate metrics; no raw prompts, references, outputs, row ids, cached datasets, model weights, or adapter weights.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- imports: `{summary.get('imports')}`",
        f"- raw temp files deleted: `{summary.get('raw_temp_files_deleted')}`",
        "",
        "## Sample",
        f"`{summary.get('sample_meta')}`",
        "",
        "## Quality",
        f"`{summary.get('quality')}`",
        "",
        "## Validity",
        f"`{summary.get('validity')}`",
        "",
        "## Handler Counts",
        f"`{summary.get('handler_counts')}`",
        "",
        "## By Target Label",
        f"`{summary.get('by_target_label')}`",
        "",
        "## By First Handler",
        f"`{summary.get('by_first_handler')}`",
        "",
        "## Top Buckets",
    ]
    for key, item in list((summary.get("by_bucket") or {}).items())[:20]:
        lines.append(f"- `{key}`: `{item}`")
    lines.extend(
        [
            "",
            "## Category Metrics",
            f"`{summary.get('by_category')}`",
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
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    for key in ("reports_dir", "results_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    summary = run_validation(args, paths)
    base.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
