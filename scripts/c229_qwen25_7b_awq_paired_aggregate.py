from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as io
import c195_direct_probe_aggregate_validation as agg
import c201_c111_vs_current_stack_aggregate as rollback
import c216_qwen3_14b_paired_bucket_aggregate as paired


EXPERIMENT_ID = "C229"
EXPERIMENT_SLUG = "C229_qwen25_7b_awq_paired_aggregate"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C229_artifacts"
BASELINE_MODEL_ID = "Qwen/Qwen3-8B-AWQ"
VARIANT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_PACKAGE_METADATA = {
    "baseline_selected_files_gb": 6.10,
    "variant_selected_files_gb": 5.57,
    "variant_license": "apache-2.0",
    "metadata_source": "Hugging Face API with blobs=true, checked 2026-05-27",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C229 paired Qwen3-8B-AWQ vs Qwen2.5-7B-AWQ aggregate.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-source", choices=["hard_audit", "locked_val", "dataset"], default="locked_val")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=229)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "zip": out_dir.with_suffix(".zip"),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_task_data_read_remote_only": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "outputs_returned": False,
        "model_weights_returned": False,
        "training_started": False,
        "adapter_weights_returned": False,
        "c111_commit": rollback.C111_COMMIT,
        "baseline_model_id": BASELINE_MODEL_ID,
        "variant_model_id": VARIANT_MODEL_ID,
        "model_package_metadata": MODEL_PACKAGE_METADATA,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE"})
        return summary

    import c071_probe as probe

    c111 = rollback.load_c111_solution()
    sample = probe.load_sample(args.sample_source, args.sample_size, args.seed)
    rows = sample.to_dict(orient="records")

    sampler = probe.GpuMemorySampler(interval_s=0.5)
    sampler.start()
    baseline, baseline_runtime = paired.run_model(BASELINE_MODEL_ID, c111, rows, args.seed, "awq_marlin")
    variant, variant_runtime = paired.run_model(VARIANT_MODEL_ID, c111, rows, args.seed, "awq_marlin")
    sampler.stop()

    total_generation_s = baseline_runtime["generation_s"] + variant_runtime["generation_s"]
    total_startup_s = baseline_runtime["startup_s"] + variant_runtime["startup_s"]
    projected_total_4000_s = total_startup_s + (total_generation_s / max(1, len(rows))) * 4000
    summary.update(
        {
            "status": "completed",
            "decision_recommendation": "REVIEW",
            "reason": "Paired Qwen3-8B-AWQ vs Qwen2.5-7B-AWQ aggregate completed.",
            "raw_task_data_read_remote_only": True,
            "imports": {"c111_solution": "ok"},
            "sample_meta": {
                "rows": len(rows),
                "category_counts": sample["category"].value_counts().sort_index().to_dict()
                if "category" in sample
                else {},
            },
            "runtime": {
                "baseline": baseline_runtime,
                "variant": variant_runtime,
                "projected_total_4000_s": projected_total_4000_s,
                "peak_vram_used_mb_nvidia_smi": sampler.peak_used_mb,
            },
            "baseline_8b": baseline,
            "variant_qwen25_7b_awq": variant,
            "delta_qwen25_7b_minus_8b": paired.overall_delta(variant, baseline),
            "delta_by_category": paired.keyed_delta(variant, baseline, "by_category"),
            "delta_by_bucket": paired.keyed_delta(variant, baseline, "top_buckets"),
            "model_loaded": True,
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C229 Qwen2.5-7B-AWQ Paired Aggregate",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- C111 commit: `{summary.get('c111_commit')}`",
        f"- baseline model: `{summary.get('baseline_model_id')}`",
        f"- variant model: `{summary.get('variant_model_id')}`",
        f"- sample: `{summary.get('sample_meta')}`",
        "",
        "## Package Metadata",
        f"`{summary.get('model_package_metadata')}`",
        "",
        "## Runtime",
        f"`{summary.get('runtime')}`",
        "",
        "## Delta Qwen2.5 7B Minus 8B",
        f"`{summary.get('delta_qwen25_7b_minus_8b')}`",
        "",
        "## Delta By Category",
        f"`{summary.get('delta_by_category')}`",
        "",
        "## Delta By Bucket",
        f"`{summary.get('delta_by_bucket')}`",
        "",
        "## Baseline 8B",
        f"`{summary.get('baseline_8b')}`",
        "",
        "## Variant Qwen2.5 7B",
        f"`{summary.get('variant_qwen25_7b_awq')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{summary.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
        f"- row ids returned: `{summary.get('row_ids_returned')}`",
        f"- outputs returned: `{summary.get('outputs_returned')}`",
        f"- model weights returned: `{summary.get('model_weights_returned')}`",
        f"- training started: `{summary.get('training_started')}`",
        f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    summary = run_validation(args)
    io.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    agg.base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
