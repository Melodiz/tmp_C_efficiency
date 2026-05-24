from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c083_qwen3_8b_expression_substitution_guard as c083


def ensure_arg(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv.extend([flag, value])


def configure_experiment() -> None:
    c083.EXPERIMENT_ID = "C084"
    c083.EXPERIMENT_SLUG = "C084_c083_hard_audit_validation"
    c083.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C084_artifacts"
    c083.RUNNER_SCRIPT = "scripts/c084_c083_hard_audit_validation.py"


def run(argv: Sequence[str] | None = None) -> int:
    configure_experiment()
    forwarded = list(argv or [])
    ensure_arg(forwarded, "--sample-source", "hard_audit")
    return c083.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
