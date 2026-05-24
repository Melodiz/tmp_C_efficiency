from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c083_qwen3_8b_expression_substitution_guard as c083


def ensure_arg(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv.extend([flag, value])


def configure_experiment() -> None:
    c083.EXPERIMENT_ID = "C085"
    c083.EXPERIMENT_SLUG = "C085_c084_max_tokens_384"
    c083.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C085_artifacts"
    c083.RUNNER_SCRIPT = "scripts/c085_c084_max_tokens_384.py"


def run(argv: Sequence[str] | None = None) -> int:
    configure_experiment()
    forwarded = list(argv or [])
    ensure_arg(forwarded, "--sample-source", "hard_audit")
    ensure_arg(forwarded, "--max-tokens", "384")
    return c083.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
