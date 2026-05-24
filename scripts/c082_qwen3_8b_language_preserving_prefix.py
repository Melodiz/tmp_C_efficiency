from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c078_quantized_8b_awq_feasibility as c078


LANGUAGE_PRESERVING_PREFIX = (
    "Ответь кратко и точно на языке задания. Не повторяй условие. "
    "В конце дай только итоговый ответ."
)


def ensure_arg(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv.extend([flag, value])


def configure_experiment() -> None:
    c078.EXPERIMENT_ID = "C082"
    c078.EXPERIMENT_SLUG = "C082_qwen3_8b_language_preserving_prefix"
    c078.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C082_artifacts"
    c078.QUANTIZATION = "awq_marlin"
    c078.SOURCE_EXPERIMENT_ID = "C080"
    c078.RUNNER_SCRIPT = "scripts/c082_qwen3_8b_language_preserving_prefix.py"
    c078.SAMPLING_CHANGED_FROM_C073 = True
    c078.MECHANISM_DESCRIPTION = (
        "keep the C080 quantized 8B backend and greedy decoding, but replace "
        "the Russian-only C073 short prefix with a language-preserving short prefix."
    )
    c078.MODEL_CHANGE_SOURCE = "C080_awq_marlin_greedy"
    c078.FORBIDDEN_METHODS_DESCRIPTION = (
        "no deterministic guard, retrieval/RAG, cache, SFT, LoRA, system prompt, "
        "sampling change, or model/backend change."
    )
    c078.c073.SHORT_USER_PREFIX = LANGUAGE_PRESERVING_PREFIX


def run(argv: Sequence[str] | None = None) -> int:
    configure_experiment()
    forwarded = list(argv or [])
    ensure_arg(forwarded, "--dtype", "float16")
    ensure_arg(forwarded, "--quantization", c078.QUANTIZATION)
    ensure_arg(forwarded, "--temperature", "0.0")
    ensure_arg(forwarded, "--top-p", "1.0")
    ensure_arg(forwarded, "--top-k", "-1")
    return c078.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
