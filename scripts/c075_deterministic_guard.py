from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence

import c072_output_control as base
import c073_short_prefix_output_control as c073


EXPERIMENT_ID = "C075"
EXPERIMENT_SLUG = "C075_c073_deterministic_arithmetic_unit_guard"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C075_artifacts"
SOURCE_EXPERIMENT_ID = "C073"
KNOWN_C073_MISSES = {
    2987: "4829",
    7012: "498",
    9168: "-95,26",
    6234: "4,5 м²",
}


NUMBER_RE = r"[+-]?\d+(?:[,.]\d+)?"


def normalize_text(text: str) -> str:
    normalized = text.lower().replace("\u202f", " ").replace("\xa0", " ").replace("−", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def parse_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ".").replace("−", "-"))
    except (InvalidOperation, ValueError):
        return None


def format_decimal(value: Decimal, comma: bool = True, places: int | None = None) -> str:
    if places is not None:
        quant = Decimal("1").scaleb(-places)
        value = value.quantize(quant, rounding=ROUND_HALF_UP)
    if value == value.to_integral_value():
        text = str(int(value))
    else:
        text = format(value.normalize(), "f").rstrip("0").rstrip(".")
    return text.replace(".", ",") if comma else text


def final_answer(answer: str) -> str:
    return f"{answer}\n\nИтоговый ответ: {answer}"


def arithmetic_guard(question: str) -> dict[str, Any] | None:
    q = normalize_text(question)
    simple = re.fullmatch(rf"({NUMBER_RE})\s*([+\-*×xх])\s*({NUMBER_RE})\s*=*\s*", q)
    if simple:
        left = parse_decimal(simple.group(1))
        right = parse_decimal(simple.group(3))
        if left is None or right is None:
            return None
        op = simple.group(2)
        if op == "+":
            value = left + right
        elif op == "-":
            value = left - right
        elif op in {"*", "×", "x", "х"}:
            value = left * right
        else:
            return None
        answer = format_decimal(value)
        return {"kind": "simple_arithmetic", "answer": final_answer(answer), "value": answer}

    multiply_words = re.fullmatch(rf"({NUMBER_RE})\s+умнож(?:ить|ь)?\s+на\s+({NUMBER_RE})\s*", q)
    if multiply_words:
        left = parse_decimal(multiply_words.group(1))
        right = parse_decimal(multiply_words.group(2))
        if left is None or right is None:
            return None
        answer = format_decimal(left * right)
        return {"kind": "simple_arithmetic_words", "answer": final_answer(answer), "value": answer}

    divide_words = re.fullmatch(rf"({NUMBER_RE})\s+раздели(?:ть)?\s+на\s+({NUMBER_RE})(?:\s+в\s+столбик)?\s*", q)
    if divide_words:
        left = parse_decimal(divide_words.group(1))
        right = parse_decimal(divide_words.group(2))
        if left is None or right is None or right == 0:
            return None
        quotient = left / right
        if quotient == quotient.to_integral_value():
            answer = format_decimal(quotient)
        else:
            whole = int(left // right)
            remainder = left - (Decimal(whole) * right)
            if remainder == remainder.to_integral_value() and right == right.to_integral_value():
                answer = f"{whole} ост. {format_decimal(remainder)}"
            else:
                answer = format_decimal(quotient, places=6)
        return {"kind": "simple_division_words", "answer": final_answer(answer), "value": answer}

    percent_delta = re.fullmatch(rf"({NUMBER_RE})\s*([+\-])\s*({NUMBER_RE})\s*%\s*", q)
    if percent_delta:
        base_value = parse_decimal(percent_delta.group(1))
        percent = parse_decimal(percent_delta.group(3))
        if base_value is None or percent is None:
            return None
        delta = base_value * percent / Decimal(100)
        value = base_value + delta if percent_delta.group(2) == "+" else base_value - delta
        answer = format_decimal(value)
        return {"kind": "percent_delta", "answer": final_answer(answer), "value": answer}

    percent_of = re.fullmatch(rf"({NUMBER_RE})\s+процент(?:ов|а)?\s+от\s+({NUMBER_RE})\s*", q)
    if percent_of:
        percent = parse_decimal(percent_of.group(1))
        base_value = parse_decimal(percent_of.group(2))
        if percent is None or base_value is None:
            return None
        answer = format_decimal(base_value * percent / Decimal(100))
        return {"kind": "percent_of", "answer": final_answer(answer), "value": answer}

    return None


def unit_guard(question: str) -> dict[str, Any] | None:
    q = normalize_text(question)

    area_dm_to_m = re.fullmatch(
        rf"({NUMBER_RE})\s+(?:дм2|дм\^2|дм²|дециметр(?:ов)?\s+квадратн(?:ых|ые)?)\s+.*(?:в|переведи)\s+квадратн(?:ые|ых)?\s+метр(?:ы|ах)?",
        q,
    )
    if area_dm_to_m:
        value = parse_decimal(area_dm_to_m.group(1))
        if value is None:
            return None
        answer = f"{format_decimal(value / Decimal(100))} м²"
        return {"kind": "area_dm2_to_m2", "answer": final_answer(answer), "value": answer}

    area_m_to_ha = re.fullmatch(rf"({NUMBER_RE})\s+(?:м2|м\^2|м²)\s+.*(?:га|гектар)", q)
    if area_m_to_ha:
        value = parse_decimal(area_m_to_ha.group(1))
        if value is None:
            return None
        answer = f"{format_decimal(value / Decimal(10000))} га"
        return {"kind": "area_m2_to_ha", "answer": final_answer(answer), "value": answer}

    km_m_to_m = re.fullmatch(rf"({NUMBER_RE})\s+километр(?:ов|а)?\s+({NUMBER_RE})\s+метр(?:ов|а)?\s+.*сколько\s+метр", q)
    if km_m_to_m:
        km = parse_decimal(km_m_to_m.group(1))
        meters = parse_decimal(km_m_to_m.group(2))
        if km is None or meters is None:
            return None
        answer = f"{format_decimal(km * Decimal(1000) + meters)} метров"
        return {"kind": "km_m_to_m", "answer": final_answer(answer), "value": answer}

    m_dm_to_dm = re.fullmatch(rf"({NUMBER_RE})\s+м\s+({NUMBER_RE})\s+дм\s+.*сколько\s+дм", q)
    if m_dm_to_dm:
        meters = parse_decimal(m_dm_to_dm.group(1))
        dm = parse_decimal(m_dm_to_dm.group(2))
        if meters is None or dm is None:
            return None
        answer = f"{format_decimal(meters * Decimal(10) + dm)} дм"
        return {"kind": "m_dm_to_dm", "answer": final_answer(answer), "value": answer}

    tons_to_grams = re.fullmatch(rf"сколько\s+грамм(?:ов)?\s+в\s+({NUMBER_RE})\s+тонн(?:ах|е|ы)?(?:,.*)?", q)
    if tons_to_grams:
        tons = parse_decimal(tons_to_grams.group(1))
        if tons is None:
            return None
        answer = f"{format_decimal(tons * Decimal(1_000_000))} граммов"
        return {"kind": "tons_to_grams", "answer": final_answer(answer), "value": answer}

    speed_kmh_to_ms = re.fullmatch(rf"({NUMBER_RE})\s+км/ч\s+в\s+м/с", q)
    if speed_kmh_to_ms:
        kmh = parse_decimal(speed_kmh_to_ms.group(1))
        if kmh is None:
            return None
        answer = f"{format_decimal(kmh * Decimal(1000) / Decimal(3600), places=2)} м/с"
        return {"kind": "kmh_to_ms", "answer": final_answer(answer), "value": answer}

    return None


def deterministic_guard(question: str) -> dict[str, Any] | None:
    return arithmetic_guard(question) or unit_guard(question)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "zip": out_dir.with_suffix(".zip"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def estimate_output_tokens(answer: str) -> int:
    return max(1, len(re.findall(r"\S+", answer)))


def has_repetition_loop(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        most_common = max(lines.count(line) for line in set(lines))
        if most_common >= 4:
            return True
    words = text.split()
    if len(words) >= 80:
        tail = words[-40:]
        return len(set(tail)) / max(1, len(tail)) < 0.25
    return False


def apply_guard(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    guarded_rows: list[dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    applied_row_ids: list[int] = []
    present_known: list[int] = []
    corrected_known: dict[str, str] = {}
    for row in rows:
        new_row = dict(row)
        row_id = int(row["row_id"])
        if row_id in KNOWN_C073_MISSES:
            present_known.append(row_id)
        guard = deterministic_guard(str(row.get("question", "")))
        if guard is not None:
            by_kind[str(guard["kind"])] += 1
            applied_row_ids.append(row_id)
            if row_id in KNOWN_C073_MISSES:
                corrected_known[str(row_id)] = str(guard["value"])
            new_row["c073_answer"] = row.get("answer")
            new_row["answer"] = guard["answer"]
            new_row["output_tokens"] = estimate_output_tokens(guard["answer"])
            new_row["finish_reason"] = "deterministic_guard"
            new_row["stop_reason"] = str(guard["kind"])
            new_row["hit_max_tokens"] = False
            new_row["repetition_loop_suspected"] = False
            new_row["deterministic_guard"] = {
                "applied": True,
                "kind": guard["kind"],
                "value": guard["value"],
            }
        else:
            new_row["deterministic_guard"] = {"applied": False}
            new_row["repetition_loop_suspected"] = has_repetition_loop(str(new_row.get("answer", "")))
        guarded_rows.append(new_row)

    stats = {
        "applied_rows": len(applied_row_ids),
        "applied_row_ids": applied_row_ids,
        "by_kind": dict(sorted(by_kind.items())),
        "known_c073_misses": KNOWN_C073_MISSES,
        "present_known_misses": present_known,
        "corrected_known_misses": corrected_known,
    }
    return guarded_rows, stats


def build_summary(
    source_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    guard_stats: dict[str, Any],
    paths: dict[str, Path],
    source_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    output_token_counts = [int(row.get("output_tokens") or 0) for row in rows]
    summary = dict(source_summary)
    source_experiment = summary.get("experiment_id")
    if source_experiment and source_experiment != EXPERIMENT_ID:
        summary["source_experiment_id"] = source_experiment
    summary["experiment_id"] = EXPERIMENT_ID
    summary["experiment_slug"] = EXPERIMENT_SLUG
    summary["status"] = "completed"
    summary["source_c073_manifest"] = source_manifest
    config = summary.setdefault("config", {})
    config.update(
        {
            "c075_mechanism": "deterministic_arithmetic_unit_guard",
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "model_prompt_sampling_changed_from_c073": False,
            "guard_abstains_unless_high_confidence": True,
            "router_retrieval_cache_sft_lora": False,
        }
    )
    runtime = summary.setdefault("runtime", {})
    runtime["sample_rows"] = len(rows)
    summary["tokens"] = {
        **(summary.get("tokens") or {}),
        "avg_output_tokens_after_guard": sum(output_token_counts) / max(1, len(output_token_counts)),
        "max_output_tokens_after_guard": max(output_token_counts) if output_token_counts else None,
        "total_output_tokens_after_guard": sum(output_token_counts),
    }
    summary["validity"] = {
        "jsonl_rows": len(rows),
        "one_answer_per_input": len(rows) == int((summary.get("sample") or {}).get("rows") or len(rows)),
        "thinking_trace_rows": sum(1 for row in rows if "<think" in str(row.get("answer", "")) or "</think>" in str(row.get("answer", ""))),
        "max_token_hit_rows": sum(1 for row in rows if row.get("hit_max_tokens")),
        "empty_answer_rows": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
        "repetition_loop_suspected_rows": sum(1 for row in rows if row.get("repetition_loop_suspected")),
    }
    summary["deterministic_guard"] = guard_stats
    summary["paths"] = {
        "summary": str(paths["summary"]),
        "metrics": str(paths["metrics"]),
        "outputs": str(paths["outputs"]),
        "log": str(paths["log"]),
    }
    return summary


def build_metrics(summary: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    validity = summary.get("validity") or {}
    runtime = summary.get("runtime") or {}
    sample_rows = int(runtime.get("sample_rows") or validity.get("jsonl_rows") or 0)
    max_token_hit_rows = int(validity.get("max_token_hit_rows") or 0)
    empty_answer_rows = int(validity.get("empty_answer_rows") or 0)
    thinking_trace_rows = int(validity.get("thinking_trace_rows") or 0)
    repetition_rows = int(validity.get("repetition_loop_suspected_rows") or 0)
    projected = runtime.get("projected_total_4000_s")
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "status": summary.get("status"),
        "candidate": summary.get("candidate"),
        "model_ref": summary.get("model_ref"),
        "summary_path": str(paths["summary"]),
        "outputs_path": str(paths["outputs"]),
        "output_control": {
            "source": "C073_short_prefix_320",
            "model_prompt_sampling_changed_from_c073": False,
        },
        "deterministic_guard": summary.get("deterministic_guard"),
        "sample_rows": sample_rows,
        "runtime": runtime,
        "tokens": summary.get("tokens"),
        "validity": validity,
        "environment": summary.get("environment"),
        "hf_metadata": summary.get("hf_metadata"),
        "rates": {
            "max_token_hit_rate": max_token_hit_rows / sample_rows if sample_rows else None,
            "empty_answer_rate": empty_answer_rows / sample_rows if sample_rows else None,
            "thinking_trace_rate": thinking_trace_rows / sample_rows if sample_rows else None,
            "repetition_loop_suspected_rate": repetition_rows / sample_rows if sample_rows else None,
            "guard_fire_rate": (summary.get("deterministic_guard") or {}).get("applied_rows", 0) / sample_rows
            if sample_rows
            else None,
            "projected_total_4000_min": projected / 60 if projected is not None else None,
        },
        "basic_validity_pass": bool(
            summary.get("status") == "completed"
            and validity.get("one_answer_per_input") is True
            and thinking_trace_rows == 0
            and empty_answer_rows == 0
        ),
    }


def decision_recommendation(metrics: dict[str, Any], dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "INVESTIGATE", "Dry run only; no model or guard evidence was produced."
    guard_stats = metrics.get("deterministic_guard") or {}
    corrected = guard_stats.get("corrected_known_misses") or {}
    projected = (metrics.get("rates") or {}).get("projected_total_4000_min")
    validity = metrics.get("validity") or {}
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C075 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after deterministic guard application."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    present_known = guard_stats.get("present_known_misses") or []
    if present_known and not all(str(row_id) in corrected for row_id in present_known):
        return "MUTATE", "The guard is promising but did not cover every known C073 arithmetic/unit miss."
    return "MUTATE", "Known arithmetic/unit misses are corrected; English/grammar regressions and packaging remain unresolved."


def write_report(report_path: Path, metrics: dict[str, Any], args: argparse.Namespace, dry_run: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation, reason = decision_recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    guard_stats = metrics.get("deterministic_guard") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    applied_rows = guard_stats.get("applied_row_ids") or []
    corrected = guard_stats.get("corrected_known_misses") or {}
    lines = [
        "# C075 C073 Deterministic Arithmetic/Unit Guard Report",
        "",
        "## Objective",
        "- ID: C075",
        "- Mechanism: high-confidence deterministic arithmetic/fraction/unit guard on top of unchanged C073 fallback.",
        "- Leaderboard submission: NO.",
        "",
        "## Commands/config",
        f"- wrapper command: `python scripts/c075_deterministic_guard.py --out {args.out}`",
        f"- sample source: `{args.sample_source}`",
        f"- sample size: `{args.sample_size}`",
        f"- dry run: `{dry_run}`",
        "- C073 model/prefix/sampling changed: `False`.",
        "- forbidden methods: no new prompt, system prompt, retrieval, cache, SFT, LoRA, or broad solver.",
        "",
        "## Results",
        "| status | rows | guard fires | max-token hits | thinking traces | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {fires} | {cap_hits} | {thinking} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            fires=guard_stats.get("applied_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Guard Coverage",
        f"- applied rows: `{applied_rows}`",
        f"- by kind: `{guard_stats.get('by_kind', {})}`",
        f"- known C073 misses present in sample: `{guard_stats.get('present_known_misses', [])}`",
        f"- corrected known C073 misses: `{corrected}`",
        "",
        "## Remaining Known Risk",
        "- C075 does not address the C073 English comparative regression on row 2506.",
        "- C075 does not address Russian grammar, chemistry, essay, or open-ended reasoning errors.",
        "- Offline packaging was not performed by this runner.",
        "",
        "## Artifact Layout",
        f"- report: `reports/{EXPERIMENT_SLUG}_report.md`",
        f"- results: `results/{EXPERIMENT_ID}/*.summary.json`, `*.metrics.json`, `*.outputs.jsonl`",
        f"- logs: `logs/{EXPERIMENT_ID}/*.log`",
        "",
        "## Decision recommendation",
        "",
        recommendation,
        "",
        "## Strongest reason against recommendation",
        f"- {reason}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_dry_run(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"{base.utc_stamp()}_c075_dry_run"
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "run_id": run_id,
        "status": "dry_run",
        "candidate": "qwen3-4b",
        "model_ref": "Qwen/Qwen3-4B-Instruct-2507",
        "config": {
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "sample_source": args.sample_source,
            "sample_size_requested": args.sample_size,
            "c075_mechanism": "deterministic_arithmetic_unit_guard",
            "model_prompt_sampling_changed_from_c073": False,
        },
        "runtime": {"sample_rows": 0},
        "validity": {
            "jsonl_rows": 0,
            "one_answer_per_input": None,
            "thinking_trace_rows": 0,
            "max_token_hit_rows": 0,
            "empty_answer_rows": 0,
            "repetition_loop_suspected_rows": 0,
        },
        "deterministic_guard": {
            "applied_rows": 0,
            "applied_row_ids": [],
            "by_kind": {},
            "known_c073_misses": KNOWN_C073_MISSES,
            "present_known_misses": [],
            "corrected_known_misses": {},
        },
        "paths": {
            "summary": str(run_paths["summary"]),
            "metrics": str(run_paths["metrics"]),
            "outputs": str(run_paths["outputs"]),
            "log": str(run_paths["log"]),
        },
    }
    base.append_jsonl(run_paths["outputs"], [])
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].parent.mkdir(parents=True, exist_ok=True)
    run_paths["log"].write_text("dry_run=true\nNo GPU experiment was executed.\n", encoding="utf-8")
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def run_source_c073(out_dir: Path, args: argparse.Namespace) -> Path:
    source_out = out_dir.parent / f"{out_dir.name}_source_c073_{base.utc_stamp()}"
    if source_out.exists():
        shutil.rmtree(source_out)
    forwarded = [
        "--out",
        str(source_out),
        "--sample-source",
        args.sample_source,
        "--sample-size",
        str(args.sample_size),
        "--variants",
        "short_prefix_320",
        "--max-model-len",
        str(args.max_model_len),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
    ]
    if args.skip_hf_metadata:
        forwarded.append("--skip-hf-metadata")
    if args.trust_remote_code:
        forwarded.append("--trust-remote-code")
    code = c073.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C073 source runner failed with exit {code}")
    return source_out


def create_gpu_artifacts(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_out = run_source_c073(paths["out_dir"], args)
    source_manifest_path = source_out / "artifact_manifest.json"
    source_manifest = base.read_json(source_manifest_path)
    source_run = source_manifest["runs"][0]
    source_summary = base.read_json(Path(source_run["summary_path"]))
    source_outputs = read_jsonl(Path(source_run["outputs_path"]))
    guarded_rows, guard_stats = apply_guard(source_outputs)

    run_id = str(source_summary.get("run_id") or f"{base.utc_stamp()}_c075")
    run_paths = {
        **paths,
        "summary": paths["results_dir"] / f"{run_id}.summary.json",
        "metrics": paths["results_dir"] / f"{run_id}.metrics.json",
        "outputs": paths["results_dir"] / f"{run_id}.outputs.jsonl",
        "log": paths["logs_dir"] / f"{run_id}.log",
    }
    summary = build_summary(source_summary, guarded_rows, guard_stats, run_paths, source_manifest)
    summary.setdefault("runtime", {})["c075_postprocess_s"] = time.perf_counter() - started
    base.append_jsonl(run_paths["outputs"], guarded_rows)
    base.write_json(run_paths["summary"], summary)
    metrics = build_metrics(summary, run_paths)
    base.write_json(run_paths["metrics"], metrics)
    run_paths["log"].parent.mkdir(parents=True, exist_ok=True)
    run_paths["log"].write_text(
        "\n".join(
            [
                f"experiment_id={EXPERIMENT_ID}",
                f"source_out={source_out}",
                f"source_summary={source_run.get('summary_path')}",
                f"guard_applied_rows={guard_stats['applied_rows']}",
                f"guard_by_kind={guard_stats['by_kind']}",
                f"corrected_known_misses={guard_stats['corrected_known_misses']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "paths": run_paths, "summary": summary, "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C075 deterministic guard over the unchanged C073 fallback.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Artifact directory. A sibling .zip is also written.")
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="hard_audit")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create the artifact layout without loading a model.")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    archived_previous = base.prepare_out_dir(out_dir)
    paths = artifact_paths(out_dir)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        run_record = create_dry_run(paths, args)
    else:
        run_record = create_gpu_artifacts(paths, args)

    write_report(paths["report"], run_record["metrics"], args, dry_run=args.dry_run)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "created_utc": base.utc_stamp(),
        "dry_run": args.dry_run,
        "out_dir": str(paths["out_dir"]),
        "zip_path": str(paths["zip"]),
        "archived_previous_out_dir": str(archived_previous) if archived_previous else None,
        "runs": [
            {
                "run_id": run_record["run_id"],
                "summary_path": str(run_record["paths"]["summary"]),
                "metrics_path": str(run_record["paths"]["metrics"]),
                "outputs_path": str(run_record["paths"]["outputs"]),
                "log_path": str(run_record["paths"]["log"]),
                "metrics": run_record["metrics"],
            }
        ],
    }
    base.write_json(out_dir / "artifact_manifest.json", manifest)
    zip_path = base.make_zip(out_dir)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "packaged",
                "dry_run": args.dry_run,
                "out_dir": str(out_dir),
                "zip_path": str(zip_path),
                "report": str(paths["report"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
