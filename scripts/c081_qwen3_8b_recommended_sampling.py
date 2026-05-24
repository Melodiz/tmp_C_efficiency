from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c078_quantized_8b_awq_feasibility as c078


c078.EXPERIMENT_ID = "C081"
c078.EXPERIMENT_SLUG = "C081_qwen3_8b_recommended_sampling"
c078.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C081_artifacts"
c078.QUANTIZATION = "awq_marlin"
c078.SOURCE_EXPERIMENT_ID = "C080"
c078.RUNNER_SCRIPT = "scripts/c081_qwen3_8b_recommended_sampling.py"
c078.SAMPLING_CHANGED_FROM_C073 = True
c078.MECHANISM_DESCRIPTION = "use the C080 model/backend with the Qwen recommended non-thinking sampling profile."


def ensure_arg(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv.extend([flag, value])


def run(argv: Sequence[str] | None = None) -> int:
    forwarded = list(argv or [])
    ensure_arg(forwarded, "--dtype", "float16")
    ensure_arg(forwarded, "--quantization", c078.QUANTIZATION)
    ensure_arg(forwarded, "--temperature", "0.7")
    ensure_arg(forwarded, "--top-p", "0.8")
    ensure_arg(forwarded, "--top-k", "20")
    return c078.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
