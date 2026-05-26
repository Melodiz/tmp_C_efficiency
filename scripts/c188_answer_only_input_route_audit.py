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


EXPERIMENT_ID = "C188"
EXPERIMENT_SLUG = "C188_answer_only_input_route_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C188_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C188 answer-only input-feature route audit.")
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
        from collections import Counter
        from pathlib import Path

        result = {{
            "status": "failed",
            "leaderboard_submission": False,
            "raw_task_data_read_remote_only": False,
            "raw_examples_returned": False,
            "row_ids_returned": False,
            "model_loaded": False,
            "training_started": False,
            "adapter_weights_returned": False,
            "data_meta": {{}},
            "feature_counts": {{}},
            "route_candidates": [],
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
            if n <= 80:
                return "q_short"
            if n <= 180:
                return "q_medium"
            if n <= 350:
                return "q_long"
            return "q_very_long"

        def cue_flags(text):
            q = str(text).lower().replace("ё", "е")
            flags = {{}}
            flags["has_number"] = bool(re.search(r"\\d", q))
            flags["has_expression"] = bool(re.search(r"\\d\\s*[+*×xх/:=-]\\s*\\d", q))
            flags["has_blank"] = bool(re.search(r"(_{{2,}}|\\.\\.\\.|\\(\\s*\\)|\\[\\s*\\])", q))
            flags["short_answer_cue"] = bool(re.search(r"\\b(найдите|найди|вычисл|посчитай|сколько|чему равн|укажите|определите|запишите ответ|answer|calculate|find)\\b", q))
            flags["conversion_cue"] = bool(re.search(r"\\b(перевед|выраз|сколько.*(метр|грамм|литр|байт|секунд|минут|час|см|дм|км|кг)|convert)\\b", q))
            flags["grammar_cue"] = bool(re.search(r"\\b(падеж|склонен|спряжен|часть речи|морфолог|синтакс|предложен|граммат|cloze|tense|verb|noun|adjective)\\b", q))
            flags["open_ended_cue"] = bool(re.search(r"\\b(объясн|почему|напишите|сочин|эссе|опишите|перечислите|составьте|расскажите|докажите|explain|write|describe|list)\\b", q))
            flags["option_cue"] = bool(re.search(r"\\b(вариант|выберите|choice|option|а\\)|б\\)|a\\)|b\\))", q))
            return {{k: ("yes" if v else "no") for k, v in flags.items()}}

        def ok_rate(counter):
            total = sum(counter.values())
            return round(counter.get("ok", 0) / total, 4) if total else 0.0

        try:
            import pandas as pd

            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            data["target_label"] = data["reference_answer"].map(answer_only_target)
            data["script_bucket"] = data["question"].map(script_bucket)
            data["length_bucket"] = data["question"].map(length_bucket)
            for name in ["has_number", "has_expression", "has_blank", "short_answer_cue", "conversion_cue", "grammar_cue", "open_ended_cue", "option_cue"]:
                data[name] = data["question"].map(lambda value, key=name: cue_flags(value)[key])

            result["raw_task_data_read_remote_only"] = True
            result["data_meta"] = {{
                "data_rows": int(len(data)),
                "target_label_counts": dict(Counter(data["target_label"].astype(str))),
            }}

            feature_names = [
                "script_bucket",
                "length_bucket",
                "has_number",
                "has_expression",
                "has_blank",
                "short_answer_cue",
                "conversion_cue",
                "grammar_cue",
                "open_ended_cue",
                "option_cue",
            ]
            feature_counts = {{}}
            for feature in feature_names:
                rows = []
                for value, group in data.groupby(feature):
                    counts = Counter(group["target_label"].astype(str))
                    total = int(sum(counts.values()))
                    rows.append({{"value": str(value), "total": total, "ok": int(counts.get("ok", 0)), "ok_rate": ok_rate(counts), "labels": dict(counts)}})
                feature_counts[feature] = sorted(rows, key=lambda item: (-item["ok_rate"], -item["total"], item["value"]))
            result["feature_counts"] = feature_counts

            combos = [
                ["length_bucket", "script_bucket"],
                ["short_answer_cue", "open_ended_cue"],
                ["has_number", "has_expression", "conversion_cue"],
                ["has_number", "short_answer_cue", "open_ended_cue"],
                ["grammar_cue", "has_blank", "open_ended_cue"],
                ["length_bucket", "has_number", "short_answer_cue", "open_ended_cue"],
                ["length_bucket", "script_bucket", "has_number", "has_expression", "short_answer_cue", "open_ended_cue"],
            ]
            candidates = []
            for combo in combos:
                for values, group in data.groupby(combo):
                    if not isinstance(values, tuple):
                        values = (values,)
                    counts = Counter(group["target_label"].astype(str))
                    total = int(sum(counts.values()))
                    if total < 20:
                        continue
                    ok = int(counts.get("ok", 0))
                    candidates.append({{
                        "features": dict(zip(combo, map(str, values))),
                        "total": total,
                        "ok": ok,
                        "ok_rate": ok_rate(counts),
                        "labels": dict(counts),
                    }})
            result["route_candidates"] = sorted(candidates, key=lambda item: (-item["ok_rate"], -item["ok"], -item["total"]))[:40]
            result["status"] = "completed"
        except Exception as exc:
            result["error"] = f"{{type(exc).__name__}}: {{exc}}"
            result["traceback_tail"] = traceback.format_exc()[-2000:]

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
    candidates = (probe_json or {}).get("route_candidates") or []
    best = candidates[0] if candidates else None
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": "Input-feature route audit completed." if ok else "Input-feature route audit failed.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "feature_counts": (probe_json or {}).get("feature_counts"),
            "route_candidates": candidates,
            "best_route_candidate": best,
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    data_meta = summary.get("data_meta") or {}
    lines = [
        "# C188 Answer-Only Input Route Audit",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Remote-only CPU aggregate audit; no model load or training.",
        "- Return only input-feature counts and target-shape label distributions; no raw text, answers, row ids, outputs, model weights, or adapter weights.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        "",
        "## Data",
        f"- data rows: `{data_meta.get('data_rows')}`",
        f"- target label counts: `{data_meta.get('target_label_counts')}`",
        "",
        "## Best Route Candidate",
        f"- candidate: `{summary.get('best_route_candidate')}`",
        "",
        "## Feature Counts",
    ]
    for feature, rows in (summary.get("feature_counts") or {}).items():
        lines.append(f"- {feature}: `{rows}`")
    lines.extend(["", "## Route Candidates"])
    for candidate in (summary.get("route_candidates") or [])[:20]:
        lines.append(f"- `{candidate}`")
    probe = summary.get("probe") or {}
    lines.extend(
        [
            "",
            "## Hygiene",
            f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
            f"- row ids returned: `{summary.get('row_ids_returned')}`",
            f"- model loaded: `{summary.get('model_loaded')}`",
            f"- training started: `{summary.get('training_started')}`",
            f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
            f"- error: `{probe.get('error')}`" if probe.get("error") else "- error: none",
            "",
            "## Next",
            "Use route purity and coverage to decide whether answer-only LoRA deployment remains viable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.out))
    if paths["out_dir"].exists():
        shutil.rmtree(paths["out_dir"])
    for key in ("reports_dir", "results_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    summary = run_audit(args, paths)
    base.write_json(paths["summary"], summary)
    write_report(paths["report"], summary)
    base.zip_artifacts(paths)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
