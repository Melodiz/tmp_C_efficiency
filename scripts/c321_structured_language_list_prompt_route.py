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
import c201_c111_vs_current_stack_aggregate as rollback
import c211_c111_task_conditional_prompt_aggregate as task_prompt


EXPERIMENT_ID = "C321"
EXPERIMENT_SLUG = "C321_structured_language_list_prompt_route"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C321_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
STRUCTURED_LANGUAGE_PREFIX = (
    "Ответь на языке задания. Для заданий по грамматике, морфологии, пунктуации, "
    "словам, буквам или переводу дай именно требуемую форму, категорию, слово, "
    "фразу или список. Если нужен список, сохрани порядок и разделяй пункты "
    "переносами строк. Не добавляй ход рассуждений."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C321 structured-language/list prompt route.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=321)
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


def structured_route(question: str) -> str:
    q = str(question).lower().replace("ё", "е")
    if re.search(r"(?:морфолог|разбор\s+слов|форма\s+слов|част[ьи]\s+речи|глагол|существительн|прилагательн|причасти|деепричасти)", q):
        return "morphology_word_form"
    if re.search(r"(?:пунктуац|запят|тире|двоеточи|синтакс|предложени|однородн|придаточн|обособл)", q):
        return "punctuation_syntax"
    if re.search(r"(?:падеж|склонени|род\b|числ[оае]\b|именительн|родительн|дательн|винительн|творительн|предложн)", q):
        return "grammar_case_declension"
    if re.search(r"(?:анаграм|букв|слог|слова?\s+из\s+букв|перестав|letter|anagram|scrabble)", q):
        return "letters_anagram_wordplay"
    if re.search(r"(?:перевед|translation|translate|английск|русск|испанск|французск|немецк|китайск)", q):
        return "translation_language"
    return "default"


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
        "c111_commit": rollback.C111_COMMIT,
        "mechanism": "route C319 structured-language/list rows to a list-preserving language prompt; keep C111 prompt elsewhere",
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
    route_counts: Counter[str] = Counter()
    routed_indices: list[int] = []
    variant_prompts = []
    for idx, row in enumerate(rows):
        route = structured_route(str(row["question"]))
        route_counts[route] += 1
        prefix = STRUCTURED_LANGUAGE_PREFIX if route != "default" else c111.USER_PREFIX
        if route != "default":
            routed_indices.append(idx)
        variant_prompts.append(probe.apply_user_only_template(tokenizer, str(row["question"]), True, prefix))
    input_tokens_control = [len(tokenizer(prompt).input_ids) for prompt in control_prompts]
    input_tokens_variant = [len(tokenizer(prompt).input_ids) for prompt in variant_prompts]

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
    variant_t0 = time.perf_counter()
    variant_outputs = llm.generate(variant_prompts, sampling_params=sampling)
    variant_generation_s = time.perf_counter() - variant_t0
    sampler.stop()

    control = task_prompt.summarize_rows(c111, tokenizer, rows, control_outputs)
    variant = task_prompt.summarize_rows(c111, tokenizer, rows, variant_outputs)
    routed_rows = [rows[i] for i in routed_indices]
    routed_control_outputs = [control_outputs[i] for i in routed_indices]
    routed_variant_outputs = [variant_outputs[i] for i in routed_indices]
    routed_control = task_prompt.summarize_rows(c111, tokenizer, routed_rows, routed_control_outputs)
    routed_variant = task_prompt.summarize_rows(c111, tokenizer, routed_rows, routed_variant_outputs)

    def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return {
            "exact": b["quality"].get("exact", 0) - a["quality"].get("exact", 0),
            "final_line_exact": b["quality"].get("final_line_exact", 0) - a["quality"].get("final_line_exact", 0),
            "ref_in_output": b["quality"].get("ref_in_output", 0) - a["quality"].get("ref_in_output", 0),
            "output_in_ref": b["quality"].get("output_in_ref", 0) - a["quality"].get("output_in_ref", 0),
            "hit_max_tokens": b["validity"].get("hit_max_tokens", 0) - a["validity"].get("hit_max_tokens", 0),
            "repetition_loop": b["validity"].get("repetition_loop", 0) - a["validity"].get("repetition_loop", 0),
            "avg_output_tokens": b["tokens"].get("avg_output_tokens", 0) - a["tokens"].get("avg_output_tokens", 0),
        }

    overall_delta = delta(control, variant)
    routed_delta = delta(routed_control, routed_variant)
    gate_pass = (
        routed_delta["exact"] >= 0
        and routed_delta["ref_in_output"] >= 0
        and routed_delta["output_in_ref"] >= 0
        and routed_delta["hit_max_tokens"] <= 0
        and routed_delta["repetition_loop"] <= 0
        and (routed_delta["ref_in_output"] > 0 or routed_delta["output_in_ref"] > 0 or routed_delta["exact"] > 0)
    )
    total_generation_s = control_generation_s + variant_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE" if gate_pass else "KILL",
            "reason": "Structured-language/list routed prompt aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
            "sample_meta": {
                "rows": len(rows),
                "routed_rows": len(routed_indices),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
                "route_counts": {k: int(v) for k, v in route_counts.items()},
            },
            "runtime": {
                "startup_s": startup_s,
                "control_generation_s": control_generation_s,
                "variant_generation_s": variant_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {
                "avg_input_tokens_control": sum(input_tokens_control) / max(1, len(input_tokens_control)),
                "avg_input_tokens_variant": sum(input_tokens_variant) / max(1, len(input_tokens_variant)),
            },
            "gate_pass": gate_pass,
            "control_c111": control,
            "variant_routed_structured_prompt": variant,
            "delta_variant_minus_control": overall_delta,
            "routed_control_c111": routed_control,
            "routed_variant_structured_prompt": routed_variant,
            "routed_delta_variant_minus_control": routed_delta,
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C321 Structured-Language/List Prompt Route",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        f"- gate pass: `{summary.get('gate_pass')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Overall Delta",
        f"`{summary.get('delta_variant_minus_control')}`",
        "",
        "## Routed Delta",
        f"`{summary.get('routed_delta_variant_minus_control')}`",
        "",
        "## Routed C111 Control",
        f"`{summary.get('routed_control_c111')}`",
        "",
        "## Routed Structured Prompt Variant",
        f"`{summary.get('routed_variant_structured_prompt')}`",
        "",
        "## C111 Control",
        f"`{summary.get('control_c111')}`",
        "",
        "## Structured Prompt Variant",
        f"`{summary.get('variant_routed_structured_prompt')}`",
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
