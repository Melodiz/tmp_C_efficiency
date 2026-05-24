from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c078_quantized_8b_awq_feasibility as c078


c078.EXPERIMENT_ID = "C079"
c078.EXPERIMENT_SLUG = "C079_qwen3_8b_awq_float16_unblock"
c078.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C079_artifacts"
c078.SOURCE_EXPERIMENT_ID = "C078"
c078.RUNNER_SCRIPT = "scripts/c079_qwen3_8b_awq_float16_unblock.py"


def run(argv: Sequence[str] | None = None) -> int:
    forwarded = list(argv or [])
    if "--dtype" not in forwarded:
        forwarded.extend(["--dtype", "float16"])
    return c078.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
