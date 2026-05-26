from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c169_lora_training_stack_import_smoke as base
import c193_current_stack_aggregate_validation as c193


EXPERIMENT_ID = "C194"
EXPERIMENT_SLUG = "C194_aggregate_validation_unblock"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C194_artifacts"


def run(argv: Sequence[str] | None = None) -> int:
    c193.EXPERIMENT_ID = EXPERIMENT_ID
    c193.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c193.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    args = c193.parse_args(argv)
    if "--sample-size" not in list(argv or []):
        args.sample_size = 64
    paths = c193.artifact_paths(Path(args.out))
    for key in ("reports_dir", "results_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    summary = c193.run_validation(args, paths)
    base.write_json(paths["summary"], summary)
    c193.write_report(paths["report"], summary)
    base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
