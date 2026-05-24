from __future__ import annotations

import argparse
from typing import Sequence

import c072_output_control
import c073_short_prefix_output_control
import c075_deterministic_guard
import c076_c075_guard_heldout_validation
import c077_slash_fraction_guard_abstention
import c078_quantized_8b_awq_feasibility


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch named Task C experiments from the command line.")
    parser.add_argument(
        "--id",
        required=True,
        choices=["C072", "C073", "C075", "C076", "C077", "C078"],
        help="Experiment ID to run.",
    )
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
    if args.id == "C073":
        return c073_short_prefix_output_control.run(forwarded)
    if args.id == "C075":
        return c075_deterministic_guard.run(forwarded)
    if args.id == "C076":
        return c076_c075_guard_heldout_validation.run(forwarded)
    if args.id == "C077":
        return c077_slash_fraction_guard_abstention.run(forwarded)
    if args.id == "C078":
        return c078_quantized_8b_awq_feasibility.run(forwarded)
    raise ValueError(f"Unsupported experiment id: {args.id}")


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
