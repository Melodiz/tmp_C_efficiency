from __future__ import annotations

import argparse
from typing import Sequence

import c072_output_control


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch named Task C experiments from the command line.")
    parser.add_argument("--id", required=True, choices=["C072"], help="Experiment ID to run.")
    parser.add_argument("--out", required=True, help="Artifact directory. The experiment writes a sibling .zip.")
    parser.add_argument("--dry-run", action="store_true", help="Create the artifact layout without a GPU/model run.")
    parser.add_argument(
        "experiment_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded after `--` to the experiment runner.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    forwarded = ["--out", args.out]
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.experiment_args:
        forwarded.extend(arg for arg in args.experiment_args if arg != "--")
    if args.id == "C072":
        return c072_output_control.run(forwarded)
    raise ValueError(f"Unsupported experiment id: {args.id}")


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
