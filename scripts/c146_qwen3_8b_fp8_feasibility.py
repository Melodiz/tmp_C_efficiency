from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c078_quantized_8b_awq_feasibility as c078
import c082_qwen3_8b_language_preserving_prefix as c082


MODEL_ID = "Qwen/Qwen3-8B-FP8"


def ensure_arg(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv.extend([flag, value])


def configure_experiment() -> None:
    c078.EXPERIMENT_ID = "C146"
    c078.EXPERIMENT_SLUG = "C146_qwen3_8b_fp8_feasibility"
    c078.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C146_artifacts"
    c078.MODEL_ID = MODEL_ID
    c078.QUANTIZATION = "fp8"
    c078.SOURCE_EXPERIMENT_ID = "C082"
    c078.RUNNER_SCRIPT = "scripts/c146_qwen3_8b_fp8_feasibility.py"
    c078.SAMPLING_CHANGED_FROM_C073 = True
    c078.MECHANISM_DESCRIPTION = (
        "replace only the Qwen3-8B-AWQ model/runtime with Qwen3-8B-FP8 while keeping "
        "the C082 language-preserving prefix, greedy decoding, output cap, and no deterministic handlers."
    )
    c078.MODEL_CHANGE_SOURCE = "C082_qwen3_8b_awq_language_preserving_prefix"
    c078.FORBIDDEN_METHODS_DESCRIPTION = (
        "no deterministic handlers, retrieval/RAG, cache, SFT, LoRA, system prompt, "
        "prompt rewrite, sampling change, or output-cap change."
    )
    c078.c073.SHORT_USER_PREFIX = c082.LANGUAGE_PRESERVING_PREFIX


def run(argv: Sequence[str] | None = None) -> int:
    configure_experiment()
    forwarded = list(argv or [])
    ensure_arg(forwarded, "--dtype", "float16")
    ensure_arg(forwarded, "--quantization", c078.QUANTIZATION)
    ensure_arg(forwarded, "--temperature", "0.0")
    ensure_arg(forwarded, "--top-p", "1.0")
    ensure_arg(forwarded, "--top-k", "-1")
    ensure_arg(forwarded, "--sample-source", "locked_val")
    return c078.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
