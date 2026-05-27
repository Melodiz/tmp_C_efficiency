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
import c089_english_final_answer_cleanup
import c090_strict_english_cloze_cleanup
import c091_c090_hard_audit_validation
import c092_true_c090_hard_audit_validation
import c093_c092_simple_solution_smoke
import c094_km_meters_guard
import c096_qwen3_8b_thinking_final_only_prompt
import c097_qwen3_8b_answer_only_prompt
import c098_c097_with_c093_handlers_hard_audit
import c099_qwen3_8b_task_conditional_prompt
import c100_qwen3_8b_no_detailed_reasoning_prompt
import c101_c100_with_c093_handlers_hard_audit
import c102_qwen3_8b_c093_minimal_no_reasoning_prompt
import c103_c094_locked_val_validation
import c104_c094_simple_solution_smoke
import c106_qwen3_14b_awq_feasibility
import c107_qwen3_14b_awq_c104_handlers_hard_audit
import c108_qwen25_14b_awq_feasibility
import c111_quantity_conversion_final_smoke
import c113_numeric_exact_final_smoke
import c116_chemistry_stoichiometry_final_smoke
import c119_formulaic_math_physics_final_smoke
import c120_qwen3_14b_with_c119_exact_stack_audit
import c123_structured_school_task_final_smoke
import c125_direct_arithmetic_final_smoke
import c131_russian_morph_grammar_final_smoke
import c135_calculator_written_arithmetic_final_smoke
import c140_algebra_equation_final_smoke
import c146_qwen3_8b_fp8_feasibility
import c152_selective_retry
import c156_geometry_exact_final_smoke
import c160_lora_inference_compat_smoke
import c161_tiny_lora_training_smoke
import c169_lora_training_stack_import_smoke
import c170_lora_training_stack_target_import_smoke
import c171_lora_training_stack_torchao_import_smoke
import c172_synthetic_tiny_lora_training_step_smoke
import c173_qwen3_8b_synthetic_qlora_smoke
import c175_remote_tiny_task_sft_smoke
import c177_base_vs_lora_aggregate_validation_smoke
import c178_sft_aggregate_metric_cap_diagnostic
import c181_answer_only_tiny_sft_smoke
import c182_answer_only_sft_confirmation_smoke
import c184_answer_only_scaled_sft_smoke
import c186_answer_only_route_harm_diagnostic
import c188_answer_only_input_route_audit
import c190_final_stack_coverage_audit
import c191_dependency_parity_coverage_audit
import c193_current_stack_aggregate_validation
import c194_aggregate_validation_unblock
import c195_direct_probe_aggregate_validation
import c196_current_stack_scaled_aggregate_validation
import c197_failure_slice_aggregate_validation
import c198_targeted_failure_retry
import c199_answer_first_prompt_aggregate
import c201_c111_vs_current_stack_aggregate
import c202_c111_no_detailed_reasoning_prompt_aggregate
import c203_c111_qwen3_14b_aggregate
import c204_qwen3_14b_relaxed_prefix_aggregate
import c207_routed_answer_only_adapter_diagnostic
import c209_c111_thinking_mode_aggregate
import c211_c111_task_conditional_prompt_aggregate
import c216_qwen3_14b_paired_bucket_aggregate
import c218_qwen3_4b_2507_fp8_paired_aggregate
import c220_paired_answer_judge_selector_aggregate
import c222_c111_fallback_answer_extraction_aggregate
import c227_phi4_mini_paired_aggregate
import c228_gemma3_4b_paired_aggregate
import c229_qwen25_7b_awq_paired_aggregate
import c231_c111_large_failure_map
import c232_failure_gated_qwen25_fallback
import c234_semantic_proxy_calibration
import c235_c111_max_tokens_512
import c236_c111_max_tokens_512_scaled
import c237_c111_max_tokens_512_scaled_setup_retry
import c238_qwen3_4b_thinking_2507_fp8_paired_aggregate
import c239_c111_system_prefix_aggregate
import c240_failure_gated_system_prefix_fallback
import c243_c111_plus_formulaic_aggregate
import c244_c111_plus_numeric_aggregate
import c246_failure_gated_same_model_512
import c248_failure_gated_concise_reanswer
import c250_closed_form_sampled_consensus
import c251_c111_plus_algebra_equation_aggregate
import c253_answer_shape_router_audit
import c254_final_answer_target_audit
import c255_broad_final_answer_sft_smoke


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
            "C089",
            "C090",
            "C091",
            "C092",
            "C093",
            "C094",
            "C096",
            "C097",
            "C098",
            "C099",
            "C100",
            "C101",
            "C102",
            "C103",
            "C104",
            "C106",
            "C107",
            "C108",
            "C111",
            "C113",
            "C116",
            "C119",
            "C120",
            "C123",
            "C125",
            "C131",
            "C135",
            "C140",
            "C146",
            "C152",
            "C156",
            "C160",
            "C161",
            "C169",
            "C170",
            "C171",
            "C172",
            "C173",
            "C175",
            "C177",
            "C178",
            "C181",
            "C182",
            "C184",
            "C186",
            "C188",
            "C190",
            "C191",
            "C193",
            "C194",
            "C195",
            "C196",
            "C197",
            "C198",
            "C199",
            "C201",
            "C202",
            "C203",
            "C204",
            "C207",
            "C209",
            "C211",
            "C216",
            "C218",
            "C220",
            "C222",
            "C227",
            "C228",
            "C229",
            "C231",
            "C232",
            "C234",
            "C235",
            "C236",
            "C237",
            "C238",
            "C239",
            "C240",
            "C243",
            "C244",
            "C246",
            "C248",
            "C250",
            "C251",
            "C253",
            "C254",
            "C255",
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
    if args.id == "C089":
        return c089_english_final_answer_cleanup.run(forwarded)
    if args.id == "C090":
        return c090_strict_english_cloze_cleanup.run(forwarded)
    if args.id == "C091":
        return c091_c090_hard_audit_validation.run(forwarded)
    if args.id == "C092":
        return c092_true_c090_hard_audit_validation.run(forwarded)
    if args.id == "C093":
        return c093_c092_simple_solution_smoke.run(forwarded)
    if args.id == "C094":
        return c094_km_meters_guard.run(forwarded)
    if args.id == "C096":
        return c096_qwen3_8b_thinking_final_only_prompt.run(forwarded)
    if args.id == "C097":
        return c097_qwen3_8b_answer_only_prompt.run(forwarded)
    if args.id == "C098":
        return c098_c097_with_c093_handlers_hard_audit.run(forwarded)
    if args.id == "C099":
        return c099_qwen3_8b_task_conditional_prompt.run(forwarded)
    if args.id == "C100":
        return c100_qwen3_8b_no_detailed_reasoning_prompt.run(forwarded)
    if args.id == "C101":
        return c101_c100_with_c093_handlers_hard_audit.run(forwarded)
    if args.id == "C102":
        return c102_qwen3_8b_c093_minimal_no_reasoning_prompt.run(forwarded)
    if args.id == "C103":
        return c103_c094_locked_val_validation.run(forwarded)
    if args.id == "C104":
        return c104_c094_simple_solution_smoke.run(forwarded)
    if args.id == "C106":
        return c106_qwen3_14b_awq_feasibility.run(forwarded)
    if args.id == "C107":
        return c107_qwen3_14b_awq_c104_handlers_hard_audit.run(forwarded)
    if args.id == "C108":
        return c108_qwen25_14b_awq_feasibility.run(forwarded)
    if args.id == "C111":
        return c111_quantity_conversion_final_smoke.run(forwarded)
    if args.id == "C113":
        return c113_numeric_exact_final_smoke.run(forwarded)
    if args.id == "C116":
        return c116_chemistry_stoichiometry_final_smoke.run(forwarded)
    if args.id == "C119":
        return c119_formulaic_math_physics_final_smoke.run(forwarded)
    if args.id == "C120":
        return c120_qwen3_14b_with_c119_exact_stack_audit.run(forwarded)
    if args.id == "C123":
        return c123_structured_school_task_final_smoke.run(forwarded)
    if args.id == "C125":
        return c125_direct_arithmetic_final_smoke.run(forwarded)
    if args.id == "C131":
        return c131_russian_morph_grammar_final_smoke.run(forwarded)
    if args.id == "C135":
        return c135_calculator_written_arithmetic_final_smoke.run(forwarded)
    if args.id == "C140":
        return c140_algebra_equation_final_smoke.run(forwarded)
    if args.id == "C146":
        return c146_qwen3_8b_fp8_feasibility.run(forwarded)
    if args.id == "C152":
        return c152_selective_retry.run(forwarded)
    if args.id == "C156":
        return c156_geometry_exact_final_smoke.run(forwarded)
    if args.id == "C160":
        return c160_lora_inference_compat_smoke.run(forwarded)
    if args.id == "C161":
        return c161_tiny_lora_training_smoke.run(forwarded)
    if args.id == "C169":
        return c169_lora_training_stack_import_smoke.run(forwarded)
    if args.id == "C170":
        return c170_lora_training_stack_target_import_smoke.run(forwarded)
    if args.id == "C171":
        return c171_lora_training_stack_torchao_import_smoke.run(forwarded)
    if args.id == "C172":
        return c172_synthetic_tiny_lora_training_step_smoke.run(forwarded)
    if args.id == "C173":
        return c173_qwen3_8b_synthetic_qlora_smoke.run(forwarded)
    if args.id == "C175":
        return c175_remote_tiny_task_sft_smoke.run(forwarded)
    if args.id == "C177":
        return c177_base_vs_lora_aggregate_validation_smoke.run(forwarded)
    if args.id == "C178":
        return c178_sft_aggregate_metric_cap_diagnostic.run(forwarded)
    if args.id == "C181":
        return c181_answer_only_tiny_sft_smoke.run(forwarded)
    if args.id == "C182":
        return c182_answer_only_sft_confirmation_smoke.run(forwarded)
    if args.id == "C184":
        return c184_answer_only_scaled_sft_smoke.run(forwarded)
    if args.id == "C186":
        return c186_answer_only_route_harm_diagnostic.run(forwarded)
    if args.id == "C188":
        return c188_answer_only_input_route_audit.run(forwarded)
    if args.id == "C190":
        return c190_final_stack_coverage_audit.run(forwarded)
    if args.id == "C191":
        return c191_dependency_parity_coverage_audit.run(forwarded)
    if args.id == "C193":
        return c193_current_stack_aggregate_validation.run(forwarded)
    if args.id == "C194":
        return c194_aggregate_validation_unblock.run(forwarded)
    if args.id == "C195":
        return c195_direct_probe_aggregate_validation.run(forwarded)
    if args.id == "C196":
        return c196_current_stack_scaled_aggregate_validation.run(forwarded)
    if args.id == "C197":
        return c197_failure_slice_aggregate_validation.run(forwarded)
    if args.id == "C198":
        return c198_targeted_failure_retry.run(forwarded)
    if args.id == "C199":
        return c199_answer_first_prompt_aggregate.run(forwarded)
    if args.id == "C201":
        return c201_c111_vs_current_stack_aggregate.run(forwarded)
    if args.id == "C202":
        return c202_c111_no_detailed_reasoning_prompt_aggregate.run(forwarded)
    if args.id == "C203":
        return c203_c111_qwen3_14b_aggregate.run(forwarded)
    if args.id == "C204":
        return c204_qwen3_14b_relaxed_prefix_aggregate.run(forwarded)
    if args.id == "C207":
        return c207_routed_answer_only_adapter_diagnostic.run(forwarded)
    if args.id == "C209":
        return c209_c111_thinking_mode_aggregate.run(forwarded)
    if args.id == "C211":
        return c211_c111_task_conditional_prompt_aggregate.run(forwarded)
    if args.id == "C216":
        return c216_qwen3_14b_paired_bucket_aggregate.run(forwarded)
    if args.id == "C218":
        return c218_qwen3_4b_2507_fp8_paired_aggregate.run(forwarded)
    if args.id == "C220":
        return c220_paired_answer_judge_selector_aggregate.run(forwarded)
    if args.id == "C222":
        return c222_c111_fallback_answer_extraction_aggregate.run(forwarded)
    if args.id == "C227":
        return c227_phi4_mini_paired_aggregate.run(forwarded)
    if args.id == "C228":
        return c228_gemma3_4b_paired_aggregate.run(forwarded)
    if args.id == "C229":
        return c229_qwen25_7b_awq_paired_aggregate.run(forwarded)
    if args.id == "C231":
        return c231_c111_large_failure_map.run(forwarded)
    if args.id == "C232":
        return c232_failure_gated_qwen25_fallback.run(forwarded)
    if args.id == "C234":
        return c234_semantic_proxy_calibration.run(forwarded)
    if args.id == "C235":
        return c235_c111_max_tokens_512.run(forwarded)
    if args.id == "C236":
        return c236_c111_max_tokens_512_scaled.run(forwarded)
    if args.id == "C237":
        return c237_c111_max_tokens_512_scaled_setup_retry.run(forwarded)
    if args.id == "C238":
        return c238_qwen3_4b_thinking_2507_fp8_paired_aggregate.run(forwarded)
    if args.id == "C239":
        return c239_c111_system_prefix_aggregate.run(forwarded)
    if args.id == "C240":
        return c240_failure_gated_system_prefix_fallback.run(forwarded)
    if args.id == "C243":
        return c243_c111_plus_formulaic_aggregate.run(forwarded)
    if args.id == "C244":
        return c244_c111_plus_numeric_aggregate.run(forwarded)
    if args.id == "C246":
        return c246_failure_gated_same_model_512.run(forwarded)
    if args.id == "C248":
        return c248_failure_gated_concise_reanswer.run(forwarded)
    if args.id == "C250":
        return c250_closed_form_sampled_consensus.run(forwarded)
    if args.id == "C251":
        return c251_c111_plus_algebra_equation_aggregate.run(forwarded)
    if args.id == "C253":
        return c253_answer_shape_router_audit.run(forwarded)
    if args.id == "C254":
        return c254_final_answer_target_audit.run(forwarded)
    if args.id == "C255":
        return c255_broad_final_answer_sft_smoke.run(forwarded)
    raise ValueError(f"Unsupported experiment id: {args.id}")


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
