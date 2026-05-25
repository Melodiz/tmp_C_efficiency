from __future__ import annotations

import c098_c097_with_c093_handlers_hard_audit as c098
import c100_qwen3_8b_no_detailed_reasoning_prompt as c100


EXPERIMENT_ID = "C101"
EXPERIMENT_SLUG = "C101_c100_with_c093_handlers_hard_audit"
ORIGINAL_BUILD_SUMMARY = c098.build_summary


def run_source_c100(out_dir, args):
    source_out = out_dir.parent / f"{out_dir.name}_source_c100_{c098.base.utc_stamp()}"
    if source_out.exists():
        c098.shutil.rmtree(source_out)
    forwarded = [
        "--out",
        str(source_out),
        "--sample-source",
        args.sample_source,
        "--sample-size",
        str(args.sample_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
    ]
    if args.skip_hf_metadata:
        forwarded.append("--skip-hf-metadata")
    code = c100.run(forwarded)
    if code != 0:
        raise RuntimeError(f"C100 source runner failed with exit {code}")
    return source_out


def recommendation(metrics, dry_run):
    if dry_run:
        return "INVESTIGATE", "Dry run only; no hard-audit evidence was produced."
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    projected = rates.get("projected_total_4000_min")
    if metrics.get("status") != "completed":
        return "INVESTIGATE", "The C101 runner did not complete."
    if validity.get("one_answer_per_input") is not True or validity.get("thinking_trace_rows") or validity.get("empty_answer_rows"):
        return "KILL", "Basic validity failed after applying existing handlers."
    if isinstance(projected, (int, float)) and projected >= 12:
        return "KILL", "Projected runtime misses the 12 minute gate."
    if int(validity.get("max_token_hit_rows") or 0) > 2:
        return "KILL", "Hard-audit truncation is not better than C092/C093."
    if int(validity.get("repetition_loop_suspected_rows") or 0) > 1:
        return "KILL", "Repetition risk remains too high."
    return "MUTATE", "C100 prompt plus existing handlers passed hard-audit validity gates; row-level review and final smoke are needed."


def build_summary(source_summary, rows, handler_stats, paths):
    summary = ORIGINAL_BUILD_SUMMARY(source_summary, rows, handler_stats, paths)
    summary["experiment_id"] = EXPERIMENT_ID
    summary["experiment_slug"] = EXPERIMENT_SLUG
    config = summary.setdefault("config", {})
    config.pop("c098_mechanism", None)
    config.update(
        {
            "c101_mechanism": "c100_prompt_with_existing_c093_handlers",
            "source_experiment_id": "C100",
            "new_handlers_added": False,
            "handlers_match_c093": True,
        }
    )
    return summary


def write_report(report_path, metrics, args, dry_run):
    rec, reason = recommendation(metrics, dry_run)
    validity = metrics.get("validity") or {}
    rates = metrics.get("rates") or {}
    handlers = metrics.get("handler_stats") or {}
    projected = rates.get("projected_total_4000_min")
    projected_text = f"{projected:.2f}" if isinstance(projected, (int, float)) else "n/a"
    lines = [
        "# C101 C100 With C093 Handlers Hard-Audit Report",
        "",
        "## Objective",
        "- ID: C101",
        "- Mechanism: validate C100 no-detailed-reasoning prompt with the existing C093 handler stack.",
        "- Leaderboard submission: NO.",
        "",
        "## Results",
        "| status | rows | max-token hits | thinking traces | empty answers | repetition suspects | projected 4000q min |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {status} | {rows} | {cap_hits} | {thinking} | {empty} | {repetition} | {projected} |".format(
            status=metrics.get("status"),
            rows=metrics.get("sample_rows", 0),
            cap_hits=validity.get("max_token_hit_rows", "n/a"),
            thinking=validity.get("thinking_trace_rows", "n/a"),
            empty=validity.get("empty_answer_rows", "n/a"),
            repetition=validity.get("repetition_loop_suspected_rows", "n/a"),
            projected=projected_text,
        ),
        "",
        "## Handler Coverage",
        f"- expression guard: `{(handlers.get('expression_guard') or {}).get('applied_row_ids', [])}`",
        f"- comma dedup: `{(handlers.get('comma_repetition_dedup') or {}).get('applied_row_ids', [])}`",
        f"- strict English cleanup: `{(handlers.get('strict_english_cleanup') or {}).get('applied_row_ids', [])}`",
        "",
        "## Decision recommendation",
        "",
        rec,
        "",
        "## Strongest reason against recommendation",
        f"- {reason}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv=None):
    c098.EXPERIMENT_ID = EXPERIMENT_ID
    c098.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c098.DEFAULT_OUT_DIR = c098.Path("artifacts") / "tmp" / "C101_artifacts"
    c098.run_source_c097 = run_source_c100
    c098.build_summary = build_summary
    c098.recommendation = recommendation
    c098.write_report = write_report
    return c098.run(argv)


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
