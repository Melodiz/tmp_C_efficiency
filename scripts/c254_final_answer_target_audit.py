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


EXPERIMENT_ID = "C254"
EXPERIMENT_SLUG = "C254_final_answer_target_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C254_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C254 final-answer target extraction audit.")
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
            "model_loaded": False,
            "training_started": False,
            "adapter_weights_returned": False,
            "data_meta": {{}},
            "method_counts": {{}},
            "target_by_bucket": {{}},
            "target_by_script": {{}},
        }}

        def clean(text):
            text = str(text).strip()
            text = re.sub(r"^(ответ|итоговый ответ|итог|answer|final answer)\\s*[:：-]\\s*", "", text, flags=re.I).strip()
            return text.strip(" .;:")

        def label(text):
            text = clean(text)
            if not text:
                return "empty"
            if len(text) > 100:
                return "long"
            if len(text.split()) > 18:
                return "essay_like"
            if "\\n" in text:
                return "multiline"
            return "short"

        def extract_methods(ref):
            text = str(ref).strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            methods = {{}}
            if lines:
                methods["first_line"] = clean(lines[0])
                methods["last_line"] = clean(lines[-1])
            match = re.search(r"(?:итоговый ответ|итог|ответ|final answer|answer)\\s*[:：-]\\s*(.+)$", text, flags=re.I | re.S)
            if match:
                tail = match.group(1).strip().splitlines()[0].strip()
                methods["explicit_final_marker"] = clean(tail)
            if lines:
                short_lines = [clean(line) for line in lines if label(line) == "short"]
                if short_lines:
                    methods["last_short_line"] = short_lines[-1]
            return methods

        def script_bucket(text):
            cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in str(text))
            lat = sum("a" <= ch.lower() <= "z" for ch in str(text))
            if cyr > lat:
                return "cyrillic"
            if lat > cyr:
                return "latin"
            return "mixed_or_symbolic"

        def feature_bucket(question):
            q = str(question).lower().replace("ё", "е")
            n = len(q)
            length = "q_short" if n <= 80 else "q_medium" if n <= 180 else "q_long" if n <= 350 else "q_very_long"
            numeric = "num" if re.search(r"\\d", q) else "nonnum"
            expr = "expr" if re.search(r"\\d\\s*[+*×xх/:=\\-]\\s*\\d", q) else "noexpr"
            openish = "open" if re.search(r"\\b(объясн|почему|напишите|сочин|эссе|опишите|перечислите|составьте|расскажите|докажите|explain|write|describe|list)\\b", q) else "closed"
            return "|".join([length, script_bucket(q), numeric, expr, openish])

        def compact(counter):
            return {{str(k): int(v) for k, v in counter.items()}}

        try:
            import pandas as pd
            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            result["raw_task_data_read_remote_only"] = True
            method_counts = defaultdict(Counter)
            by_bucket = defaultdict(lambda: defaultdict(Counter))
            by_script = defaultdict(lambda: defaultdict(Counter))
            best_short = Counter()
            for _, row in data.iterrows():
                bucket = feature_bucket(row["question"])
                script = script_bucket(row["question"])
                methods = extract_methods(row["reference_answer"])
                has_short = False
                for name, target in methods.items():
                    lab = label(target)
                    method_counts[name][lab] += 1
                    by_bucket[name][bucket][lab] += 1
                    by_script[name][script][lab] += 1
                    has_short = has_short or lab == "short"
                best_short["short"] += int(has_short)
                best_short["not_short"] += int(not has_short)
            result["data_meta"] = {{"rows": int(len(data)), "any_short_extractable": compact(best_short)}}
            result["method_counts"] = {{name: compact(counts) for name, counts in method_counts.items()}}
            result["target_by_bucket"] = {{
                name: [
                    {{"bucket": bucket, "total": int(sum(counts.values())), "labels": compact(counts)}}
                    for bucket, counts in sorted(groups.items(), key=lambda item: -sum(item[1].values()))[:30]
                ]
                for name, groups in by_bucket.items()
            }}
            result["target_by_script"] = {{
                name: {{script: compact(counts) for script, counts in groups.items()}}
                for name, groups in by_script.items()
            }}
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
        "model_loaded": False,
        "training_started": False,
        "adapter_weights_returned": False,
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
    probe_json = None
    try:
        probe_json = json.loads(probe_log)
        base.write_json(paths["probe"], probe_json)
    except Exception:
        probe_json = None
    ok = probe_code == 0 and probe_json is not None and probe_json.get("status") == "completed"
    any_short = ((probe_json or {}).get("data_meta") or {}).get("any_short_extractable") or {}
    short_count = int(any_short.get("short", 0))
    decision = "MUTATE" if short_count >= 1500 else "KILL"
    reason = "Final-answer target extraction found broad short targets." if decision == "MUTATE" else "Final-answer target extraction is not broad enough."
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": decision if ok else "KILL",
            "reason": reason if ok else "Final-answer target audit failed.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "method_counts": (probe_json or {}).get("method_counts"),
            "target_by_bucket": (probe_json or {}).get("target_by_bucket"),
            "target_by_script": (probe_json or {}).get("target_by_script"),
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C254 Final-Answer Target Extraction Audit",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Remote-only aggregate audit of extractable short final-answer targets from references.",
        "- No model load, adapter training, package build, raw examples, row ids, or extracted targets returned.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        f"- runtime: `{summary.get('runtime')}`",
        "",
        "## Data",
        f"`{summary.get('data_meta')}`",
        "",
        "## Method Counts",
        f"`{summary.get('method_counts')}`",
        "",
        "## Target By Script",
        f"`{summary.get('target_by_script')}`",
        "",
        "## Target By Bucket",
    ]
    for method, rows in (summary.get("target_by_bucket") or {}).items():
        lines.append(f"- {method}: `{rows}`")
    probe = summary.get("probe") or {}
    lines.extend(
        [
            "",
            "## Hygiene",
            f"- raw task data read remote only: `{probe.get('raw_task_data_read_remote_only')}`",
            f"- raw examples returned: `{probe.get('raw_examples_returned')}`",
            f"- row ids returned: `{probe.get('row_ids_returned')}`",
            f"- outputs returned: `{probe.get('outputs_returned')}`",
            f"- model loaded: `{probe.get('model_loaded')}`",
            f"- training started: `{probe.get('training_started')}`",
            f"- adapter weights returned: `{probe.get('adapter_weights_returned')}`",
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
