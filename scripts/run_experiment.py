from __future__ import annotations

import argparse
from typing import Sequence

import c072_output_control
import c073_short_prefix_output_control
import c075_deterministic_guard
import c076_c075_guard_heldout_validation
import c077_slash_fraction_guard_abstention
import c078_quantized_8b_awq_feasibility
import c079_qwen3_8b_awq_float16_unblock
import c080_qwen3_8b_awq_marlin
import c081_qwen3_8b_recommended_sampling
import c082_qwen3_8b_language_preserving_prefix
import c083_qwen3_8b_expression_substitution_guard
import c084_c083_hard_audit_validation
import c085_c084_max_tokens_384
import c086_c084_repetition_list_dedup
import c087_c086_locked_val_validation
import c088_simple_solution_candidate_smoke


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch named Task C experiments from the command line.")
    parser.add_argument(
        "--id",
        required=True,
        choices=[
            "C072",
            "C073",
            "C075",
            "C076",
            "C077",
            "C078",
            "C079",
            "C080",
            "C081",
            "C082",
            "C083",
            "C084",
            "C085",
            "C086",
            "C087",
            "C088",
        ],
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
    if args.id == "C079":
        return c079_qwen3_8b_awq_float16_unblock.run(forwarded)
    if args.id == "C080":
        return c080_qwen3_8b_awq_marlin.run(forwarded)
    if args.id == "C081":
        return c081_qwen3_8b_recommended_sampling.run(forwarded)
    if args.id == "C082":
        return c082_qwen3_8b_language_preserving_prefix.run(forwarded)
    if args.id == "C083":
        return c083_qwen3_8b_expression_substitution_guard.run(forwarded)
    if args.id == "C084":
        return c084_c083_hard_audit_validation.run(forwarded)
    if args.id == "C085":
        return c085_c084_max_tokens_384.run(forwarded)
    if args.id == "C086":
        return c086_c084_repetition_list_dedup.run(forwarded)
    if args.id == "C087":
        return c087_c086_locked_val_validation.run(forwarded)
    if args.id == "C088":
        return c088_simple_solution_candidate_smoke.run(forwarded)
    raise ValueError(f"Unsupported experiment id: {args.id}")


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
