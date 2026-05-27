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


EXPERIMENT_ID = "C253"
EXPERIMENT_SLUG = "C253_answer_shape_router_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C253_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C253 learned input-side answer-shape router audit.")
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
        import math
        import re
        import traceback
        from collections import Counter
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
            "feature_names": [],
            "baseline_c188_reference": {{"best_ok": 92, "best_total": 248, "best_ok_rate": 0.371}},
            "route_candidates": [],
            "selected_routes": [],
            "validation_summary": {{}},
        }}

        def answer_only_target(value):
            text = str(value).strip()
            text = re.sub(r"^(ответ|итог|answer)\\s*[:：-]\\s*", "", text, flags=re.IGNORECASE).strip()
            lines = [part.strip() for part in text.splitlines() if part.strip()]
            if not lines:
                return "empty"
            target = lines[0]
            if len(lines) > 1:
                return "multiline"
            if len(target) > 80:
                return "long"
            if len(target.split()) > 14:
                return "essay_like"
            return "ok"

        def script_bucket(text):
            text = str(text)
            cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
            lat = sum("a" <= ch.lower() <= "z" for ch in text)
            if cyr > lat:
                return "cyrillic"
            if lat > cyr:
                return "latin"
            return "mixed_or_symbolic"

        def length_bucket(text):
            n = len(str(text))
            if n <= 60:
                return "q_tiny"
            if n <= 120:
                return "q_short"
            if n <= 220:
                return "q_medium"
            if n <= 420:
                return "q_long"
            return "q_very_long"

        def feature_row(question):
            q = str(question)
            low = q.lower().replace("ё", "е")
            token_count = len(re.findall(r"\\w+", low))
            number_count = len(re.findall(r"\\d+", low))
            return {{
                "script": script_bucket(q),
                "length": length_bucket(q),
                "token_bucket": "tok_le_8" if token_count <= 8 else "tok_le_16" if token_count <= 16 else "tok_le_32" if token_count <= 32 else "tok_gt_32",
                "num_bucket": "num_0" if number_count == 0 else "num_1" if number_count == 1 else "num_2_3" if number_count <= 3 else "num_gt_3",
                "has_expr": "yes" if re.search(r"\\d\\s*[+*×xх/:=\\-]\\s*\\d", low) else "no",
                "has_blank": "yes" if re.search(r"(_{{2,}}|\\.\\.\\.|\\(\\s*\\)|\\[\\s*\\])", low) else "no",
                "short_cue": "yes" if re.search(r"\\b(найдите|найди|вычисл|посчитай|сколько|чему равн|укажите|определите|запишите ответ|answer|calculate|find|solve)\\b", low) else "no",
                "conversion_cue": "yes" if re.search(r"\\b(перевед|выраз|convert|метр|грамм|литр|байт|секунд|минут|час|см|дм|км|кг)\\b", low) else "no",
                "grammar_cue": "yes" if re.search(r"\\b(падеж|склонен|спряжен|часть речи|морфолог|синтакс|предложен|граммат|cloze|tense|verb|noun|adjective)\\b", low) else "no",
                "open_cue": "yes" if re.search(r"\\b(объясн|почему|напишите|сочин|эссе|опишите|перечислите|составьте|расскажите|докажите|explain|write|describe|list)\\b", low) else "no",
                "option_cue": "yes" if re.search(r"\\b(вариант|выберите|choice|option|а\\)|б\\)|a\\)|b\\))", low) else "no",
                "punct_bucket": "many_punct" if len(re.findall(r"[?;:,]", low)) >= 3 else "some_punct" if re.search(r"[?;:,]", low) else "no_punct",
            }}

        def is_train(index):
            return (index * 1103515245 + 12345) % 100 < 70

        def metrics(rows):
            total = len(rows)
            ok = sum(1 for row in rows if row["label"] == "ok")
            return {{
                "total": int(total),
                "ok": int(ok),
                "ok_rate": round(ok / total, 4) if total else 0.0,
                "labels": dict(Counter(row["label"] for row in rows)),
            }}

        def route_key(row, features):
            return tuple((feature, row["features"][feature]) for feature in features)

        try:
            import pandas as pd

            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            rows = []
            for index, row in data.iterrows():
                rows.append({{
                    "split": "train" if is_train(index) else "valid",
                    "label": answer_only_target(row["reference_answer"]),
                    "features": feature_row(row["question"]),
                }})
            feature_names = sorted(rows[0]["features"].keys()) if rows else []
            train_rows = [row for row in rows if row["split"] == "train"]
            valid_rows = [row for row in rows if row["split"] == "valid"]
            result["raw_task_data_read_remote_only"] = True
            result["feature_names"] = feature_names
            result["data_meta"] = {{
                "rows": int(len(rows)),
                "train_rows": int(len(train_rows)),
                "valid_rows": int(len(valid_rows)),
                "all": metrics(rows),
                "train": metrics(train_rows),
                "valid": metrics(valid_rows),
            }}

            feature_sets = [
                ("basic_len_script", ["length", "script"]),
                ("cue_shape", ["short_cue", "open_cue", "grammar_cue", "option_cue"]),
                ("numeric_shape", ["num_bucket", "has_expr", "conversion_cue", "short_cue", "open_cue"]),
                ("blank_grammar", ["has_blank", "grammar_cue", "open_cue", "length"]),
                ("full_compact", ["length", "script", "token_bucket", "num_bucket", "has_expr", "has_blank", "short_cue", "conversion_cue", "grammar_cue", "open_cue", "option_cue"]),
            ]
            candidates = []
            for name, features in feature_sets:
                grouped = {{}}
                for row in train_rows:
                    grouped.setdefault(route_key(row, features), []).append(row)
                for key, group in grouped.items():
                    train_m = metrics(group)
                    if train_m["total"] < 25:
                        continue
                    if train_m["ok_rate"] < 0.55:
                        continue
                    key_set = set(key)
                    valid_group = [row for row in valid_rows if set(route_key(row, features)) == key_set]
                    valid_m = metrics(valid_group)
                    if valid_m["total"] < 10:
                        continue
                    candidates.append({{
                        "route_family": name,
                        "features": dict(key),
                        "train": train_m,
                        "valid": valid_m,
                    }})
            candidates.sort(key=lambda item: (-item["valid"]["ok_rate"], -item["valid"]["ok"], -item["valid"]["total"], -item["train"]["ok_rate"]))
            result["route_candidates"] = candidates[:50]

            selected = []
            selected_keys = []
            selected_valid = []
            for candidate in candidates:
                if candidate["valid"]["ok_rate"] < 0.6:
                    continue
                features = tuple(sorted(candidate["features"].items()))
                if features in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.append(features)
                cand_features = candidate["features"]
                for row in valid_rows:
                    if all(row["features"].get(k) == v for k, v in cand_features.items()):
                        selected_valid.append(row)
                if len(selected) >= 8:
                    break
            result["selected_routes"] = selected
            result["validation_summary"] = metrics(selected_valid)
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
    validation = (probe_json or {}).get("validation_summary") or {}
    best = ((probe_json or {}).get("route_candidates") or [None])[0]
    decision = "REVIEW"
    reason = "Answer-shape router audit completed."
    if ok and (validation.get("ok_rate", 0.0) < 0.6 or validation.get("ok", 0) < 50):
        decision = "KILL"
        reason = "No validation route with enough purity and coverage over C188."
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": decision if ok else "KILL",
            "reason": reason if ok else "Answer-shape router audit failed.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "feature_names": (probe_json or {}).get("feature_names"),
            "best_route_candidate": best,
            "route_candidates": (probe_json or {}).get("route_candidates") or [],
            "selected_routes": (probe_json or {}).get("selected_routes") or [],
            "validation_summary": validation,
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# C253 Answer-Shape Router Feasibility Audit",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Remote-only aggregate audit for input-side answer-only adapter routing.",
        "- No model load, adapter training, package build, or raw artifact return.",
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
        "## Best Route Candidate",
        f"`{summary.get('best_route_candidate')}`",
        "",
        "## Selected Routes",
        f"`{summary.get('selected_routes')}`",
        "",
        "## Validation Summary",
        f"`{summary.get('validation_summary')}`",
        "",
        "## Route Candidates",
    ]
    for candidate in (summary.get("route_candidates") or [])[:20]:
        lines.append(f"- `{candidate}`")
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
