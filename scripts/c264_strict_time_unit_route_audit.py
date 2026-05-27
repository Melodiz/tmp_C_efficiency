from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence

import c169_lora_training_stack_import_smoke as base


EXPERIMENT_ID = "C264"
EXPERIMENT_SLUG = "C264_strict_time_unit_route_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C264_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C264 strict time-unit conversion route audit.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "out_dir": out_dir,
        "reports_dir": out_dir / "reports",
        "results_dir": out_dir / "results" / EXPERIMENT_ID,
        "logs_dir": out_dir / "logs" / EXPERIMENT_ID,
        "report": out_dir / "reports" / f"{EXPERIMENT_SLUG}_report.md",
        "summary": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_summary.json",
        "probe": out_dir / "results" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_probe.json",
        "probe_log": out_dir / "logs" / EXPERIMENT_ID / f"{EXPERIMENT_SLUG}_probe.log",
        "zip": out_dir.with_suffix(".zip"),
    }


def probe_source() -> str:
    return textwrap.dedent(
        f"""
        import json
        import re
        import traceback
        from collections import Counter, defaultdict
        from pathlib import Path

        result = {{
            "status": "failed",
            "leaderboard_submission": False,
            "raw_task_data_read_remote_only": False,
            "raw_examples_returned": False,
            "row_ids_returned": False,
            "outputs_returned": False,
            "targets_returned": False,
            "model_loaded": False,
            "training_started": False,
            "data_meta": {{}},
            "route_counts": {{}},
            "route_target_shape": {{}},
            "route_script_counts": {{}},
            "route_bucket_counts": {{}},
            "route_overlap_counts": [],
        }}

        TIME_UNIT = r"(?:час(?:а|ов)?|ч\\.?|минут(?:а|ы)?|мин\\.?|секунд(?:а|ы)?|сек\\.?|сут(?:ки|ок)?|дн(?:я|ей)?|hour|hours|minute|minutes|second|seconds|day|days)"
        CONVERT_WORD = r"(?:перевед|выраз|сколько|convert|express|in\\s+(?:seconds|minutes|hours|days))"
        ARITH_WORD = r"(?:через|назад|спустя|после|до|прибав|вычт|duration|elapsed|after|before)"
        NUM_UNIT = rf"\\d+(?:[,.]\\d+)?\\s*{{TIME_UNIT}}"

        ROUTES = {{
            "strict_convert_two_units": [CONVERT_WORD, NUM_UNIT, TIME_UNIT],
            "strict_numeric_two_units": [NUM_UNIT, rf"{{TIME_UNIT}}.*{{TIME_UNIT}}"],
            "strict_compound_time_to_unit": [rf"\\d+\\s*(?:час|hour)", rf"\\d+\\s*(?:мин|minute)", CONVERT_WORD],
            "strict_seconds_minutes_hours": [r"(?:секунд|минут|час|second|minute|hour)", CONVERT_WORD, r"\\d"],
            "loose_c262_time_conversion": [r"(?:час|минут|секунд|сутк|time|hour|minute|second)", r"(?:перевед|сколько|convert|in\\s+seconds|in\\s+minutes|in\\s+hours)"],
            "time_arithmetic_not_conversion": [TIME_UNIT, ARITH_WORD],
        }}

        def normalize(text):
            return str(text).lower().replace("ё", "е")

        def clean_target(text):
            text = str(text).strip()
            text = re.sub(r"^(ответ|итоговый ответ|итог|answer|final answer)\\s*[:：-]\\s*", "", text, flags=re.I).strip()
            return text.strip(" .;:")

        def target_shape(ref):
            lines = [clean_target(line) for line in str(ref).splitlines() if line.strip()]
            short_lines = [line for line in lines if line and len(line) <= 100 and len(line.split()) <= 18]
            if short_lines:
                target = short_lines[-1]
                if len(target) <= 12:
                    return "tiny_short"
                if len(target) <= 40:
                    return "short"
                return "medium_short"
            if not lines:
                return "empty"
            if len(str(ref)) > 250 or len(str(ref).split()) > 45:
                return "long_essay"
            return "nonshort"

        def script_bucket(text):
            cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "е" for ch in str(text))
            lat = sum("a" <= ch.lower() <= "z" for ch in str(text))
            if cyr > lat:
                return "cyrillic"
            if lat > cyr:
                return "latin"
            return "mixed_or_symbolic"

        def feature_bucket(question):
            q = normalize(question)
            n = len(q)
            length = "q_short" if n <= 80 else "q_medium" if n <= 180 else "q_long" if n <= 350 else "q_very_long"
            numeric = "num" if re.search(r"\\d", q) else "nonnum"
            unit_count = len(re.findall(TIME_UNIT, q, flags=re.I))
            units = "two_plus_units" if unit_count >= 2 else "one_unit" if unit_count == 1 else "no_unit"
            convert = "convert_word" if re.search(CONVERT_WORD, q, flags=re.I) else "no_convert"
            return "|".join([length, script_bucket(q), numeric, units, convert])

        def routes_for(question):
            q = normalize(question)
            hits = []
            for name, patterns in ROUTES.items():
                if all(re.search(pattern, q, flags=re.I) for pattern in patterns):
                    hits.append(name)
            return hits

        def compact(counter):
            return {{str(k): int(v) for k, v in counter.items()}}

        try:
            import pandas as pd
            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            result["raw_task_data_read_remote_only"] = True
            counts = Counter()
            by_target = defaultdict(Counter)
            by_script = defaultdict(Counter)
            by_bucket = defaultdict(Counter)
            overlaps = Counter()
            matched_rows = 0
            for _, row in data.iterrows():
                hits = routes_for(row["question"])
                if not hits:
                    continue
                matched_rows += 1
                shape = target_shape(row["reference_answer"])
                script = script_bucket(row["question"])
                bucket = feature_bucket(row["question"])
                for hit in hits:
                    counts[hit] += 1
                    by_target[hit][shape] += 1
                    by_script[hit][script] += 1
                    by_bucket[hit][bucket] += 1
                for i, a in enumerate(hits):
                    for b in hits[i + 1:]:
                        overlaps[" & ".join(sorted([a, b]))] += 1
            result["data_meta"] = {{"rows": int(len(data)), "matched_rows": int(matched_rows), "routes": sorted(ROUTES)}}
            result["route_counts"] = compact(counts)
            result["route_target_shape"] = {{route: compact(c) for route, c in by_target.items()}}
            result["route_script_counts"] = {{route: compact(c) for route, c in by_script.items()}}
            result["route_bucket_counts"] = {{
                route: [{{"bucket": bucket, "count": int(count)}} for bucket, count in c.most_common(12)]
                for route, c in by_bucket.items()
            }}
            result["route_overlap_counts"] = [
                {{"routes": key, "count": int(count)}} for key, count in overlaps.most_common(20)
            ]
            result["status"] = "completed"
        except Exception as exc:
            result["error"] = f"{{type(exc).__name__}}: {{exc}}"
            result["traceback_tail"] = traceback.format_exc()[-2400:]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "completed" else 2)
        """
    ).strip()


def run_audit(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    start = time.time()
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_slug": EXPERIMENT_SLUG,
        "leaderboard_submission": False,
        "raw_examples_returned": False,
        "row_ids_returned": False,
        "outputs_returned": False,
        "targets_returned": False,
        "model_loaded": False,
        "training_started": False,
    }
    if args.dry_run:
        summary.update({"status": "dry_run", "decision_recommendation": "INVESTIGATE", "runtime": {"total_seconds": 0.0}})
        return summary
    try:
        result = subprocess.run([sys.executable, "-c", probe_source()], check=False, text=True, capture_output=True, timeout=900)
        probe_code = result.returncode
        probe_log = ((result.stdout or "") + (result.stderr or "")).strip()
    except Exception as exc:
        probe_code = 999
        probe_log = f"{type(exc).__name__}: {exc}"
    paths["probe_log"].write_text(probe_log + "\n", encoding="utf-8")
    try:
        probe_json = json.loads(probe_log)
        base.write_json(paths["probe"], probe_json)
    except Exception:
        probe_json = None
    ok = probe_code == 0 and probe_json is not None and probe_json.get("status") == "completed"
    counts = (probe_json or {}).get("route_counts") or {}
    strict_counts = {k: int(v) for k, v in counts.items() if k not in {"loose_c262_time_conversion", "time_arithmetic_not_conversion"}}
    best_name, best_count = (None, 0)
    if strict_counts:
        best_name, best_count = max(strict_counts.items(), key=lambda item: int(item[1]))
    decision = "MUTATE" if ok and int(best_count) >= 50 else "KILL"
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": decision if ok else "KILL",
            "reason": "A strict time-unit conversion route cleared count gates." if decision == "MUTATE" else "No strict time-unit conversion route cleared count gates.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "best_strict_route": best_name,
            "best_strict_route_count": int(best_count),
            "data_meta": (probe_json or {}).get("data_meta"),
            "route_counts": counts,
            "route_target_shape": (probe_json or {}).get("route_target_shape"),
            "route_script_counts": (probe_json or {}).get("route_script_counts"),
            "route_bucket_counts": (probe_json or {}).get("route_bucket_counts"),
            "route_overlap_counts": (probe_json or {}).get("route_overlap_counts"),
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    probe = summary.get("probe") or {}
    lines = [
        "# C264 Strict Time-Unit Conversion Route Audit",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Remote-only aggregate route audit after C263 showed broad but noisy date/time surfaces.",
        "- No raw prompts, references, row ids, outputs, targets, datasets, weights, or adapter files returned.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- best strict route/count: `{summary.get('best_strict_route')}` / `{summary.get('best_strict_route_count')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        f"- runtime: `{summary.get('runtime')}`",
        "",
        "## Data",
        f"`{summary.get('data_meta')}`",
        "",
        "## Route Counts",
        f"`{summary.get('route_counts')}`",
        "",
        "## Target Shape",
        f"`{summary.get('route_target_shape')}`",
        "",
        "## Script Counts",
        f"`{summary.get('route_script_counts')}`",
        "",
        "## Buckets",
    ]
    for name, rows in (summary.get("route_bucket_counts") or {}).items():
        lines.append(f"- {name}: `{rows}`")
    lines.extend(
        [
            "",
            "## Overlaps",
            f"`{summary.get('route_overlap_counts')}`",
            "",
            "## Hygiene",
            f"- raw task data read remote only: `{probe.get('raw_task_data_read_remote_only')}`",
            f"- raw examples returned: `{probe.get('raw_examples_returned')}`",
            f"- row ids returned: `{probe.get('row_ids_returned')}`",
            f"- outputs returned: `{probe.get('outputs_returned')}`",
            f"- targets returned: `{probe.get('targets_returned')}`",
            f"- model loaded: `{probe.get('model_loaded')}`",
            f"- training started: `{probe.get('training_started')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    summary = run_audit(args, paths)
    base.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
