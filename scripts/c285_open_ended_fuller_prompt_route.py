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
import c195_direct_probe_aggregate_validation as agg
import c201_c111_vs_current_stack_aggregate as rollback
import c202_c111_no_detailed_reasoning_prompt_aggregate as paired


EXPERIMENT_ID = "C285"
EXPERIMENT_SLUG = "C285_open_ended_fuller_prompt_route"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C285_artifacts"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
FULLER_OPEN_PREFIX = (
    "Ответь на языке задания полно, но без повторения условия. "
    "Для объяснения, перевода, текста или перечисления дай содержательный ответ с нужными деталями. "
    "Для задач с одним итогом дай только итог."
)

OPEN_PATTERNS = re.compile(
    r"объясн|почему|зачем|опиши|описать|расскаж|рассказать|напиши|написать|"
    r"сочинен|эссе|текст|перевед|перевод|translate|translation|"
    r"кратко изложи|сформулируй|характеристик|приведи пример|что такое|кто такой|"
    r"литератур|произведен|автор|герой|истори|географ|биолог",
    re.IGNORECASE,
)

CLOSED_GUARD = re.compile(
    r"реши|вычисл|найд[и]?|чему равн|сколько|уравнен|падеж|склонен|спряж|"
    r"част[ьи] речи|морфолог|граммат|баланс|коэффициент|периметр|площад|"
    r"\d|[=+\-*/^√]",
    re.IGNORECASE,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C285 route open-ended rows to a fuller-answer prefix.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=285)
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


def route_prefix(question: str, c111_prefix: str) -> tuple[str, str]:
    text = str(question)
    if OPEN_PATTERNS.search(text) and not CLOSED_GUARD.search(text):
        return "open_fuller", FULLER_OPEN_PREFIX
    return "c111_default", c111_prefix


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
        "mechanism": "question-text route open-ended rows to fuller-answer prefix",
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
    variant_prompts = []
    for row in rows:
        route, prefix = route_prefix(str(row["question"]), c111.USER_PREFIX)
        route_counts[route] += 1
        variant_prompts.append(probe.apply_user_only_template(tokenizer, str(row["question"]), True, prefix))
    control_input_tokens = [len(tokenizer(prompt).input_ids) for prompt in control_prompts]
    variant_input_tokens = [len(tokenizer(prompt).input_ids) for prompt in variant_prompts]

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

    control = paired.summarize_rows(c111, tokenizer, rows, control_outputs)
    variant = paired.summarize_rows(c111, tokenizer, rows, variant_outputs)
    total_generation_s = control_generation_s + variant_generation_s
    projected_total_4000_s = startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "MUTATE",
            "reason": "C111 stack paired open-ended fuller-answer route aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
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
                "variant_generation_s": variant_generation_s,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "tokens": {
                "avg_input_tokens_control": sum(control_input_tokens) / max(1, len(control_input_tokens)),
                "avg_input_tokens_variant": sum(variant_input_tokens) / max(1, len(variant_input_tokens)),
            },
            "control_c111_prefix": control,
            "variant_open_ended_fuller_route": variant,
            "delta_variant_minus_c111": {
                "exact": variant["quality"].get("exact", 0) - control["quality"].get("exact", 0),
                "final_line_exact": variant["quality"].get("final_line_exact", 0)
                - control["quality"].get("final_line_exact", 0),
                "ref_in_output": variant["quality"].get("ref_in_output", 0)
                - control["quality"].get("ref_in_output", 0),
                "output_in_ref": variant["quality"].get("output_in_ref", 0)
                - control["quality"].get("output_in_ref", 0),
                "hit_max_tokens": variant["validity"].get("hit_max_tokens", 0)
                - control["validity"].get("hit_max_tokens", 0),
                "repetition_loop": variant["validity"].get("repetition_loop", 0)
                - control["validity"].get("repetition_loop", 0),
                "avg_output_tokens": variant["tokens"].get("avg_output_tokens", 0)
                - control["tokens"].get("avg_output_tokens", 0),
            },
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C285 Open-Ended Fuller-Answer Prompt Route Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Input Tokens",
        f"`{summary.get('tokens')}`",
        "",
        "## Delta Open-Ended Route Minus C111",
        f"`{summary.get('delta_variant_minus_c111')}`",
        "",
        "## C111 Prefix Control",
        f"`{summary.get('control_c111_prefix')}`",
        "",
        "## Open-Ended Fuller Route Variant",
        f"`{summary.get('variant_open_ended_fuller_route')}`",
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
