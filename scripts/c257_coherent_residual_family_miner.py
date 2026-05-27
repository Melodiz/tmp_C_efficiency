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


EXPERIMENT_ID = "C257"
EXPERIMENT_SLUG = "C257_coherent_residual_family_miner"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C257_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C257 coherent residual family miner.")
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
            "family_counts": {{}},
            "family_by_script": {{}},
            "family_by_bucket": {{}},
            "family_target_shape": {{}},
            "family_intersections": [],
        }}

        FAMILY_PATTERNS = {{
            "date_time_calendar": [
                r"\\b(?:date|day|month|year|calendar|weekday|hour|minute|second|time)\\b",
                r"(?:дата|календар|день\\s+недел|понедельник|вторник|среда|четверг|пятница|суббота|воскресенье|январ|феврал|март|апрел|июн|июл|август|сентябр|октябр|ноябр|декабр|час|минут|секунд|сутк)",
            ],
            "chemistry_formula": [
                r"\\b(?:chemistry|chemical|mole|molar|reaction|acid|oxide|formula)\\b",
                r"(?:хими|моляр|моль|реакц|кислот|оксид|формул|веществ|элемент|атом|ион|валентн)",
                r"\\b(?:H2O|CO2|NaCl|HCl|H2SO4|O2|N2|CH4)\\b",
            ],
            "sequence_progression": [
                r"\\b(?:sequence|series|progression|next term|arithmetic progression|geometric progression)\\b",
                r"(?:последовательн|прогресси|следующ(?:ее|ий|ая)?\\s+числ|член\\s+последовательности|ряд\\s+чисел)",
            ],
            "base_number_system": [
                r"\\b(?:binary|hexadecimal|octal|base\\s*\\d+|number system)\\b",
                r"(?:двоичн|шестнадцатеричн|восьмеричн|систем[аеы]?\\s+счислен|основани[ея]\\s+\\d+)",
            ],
            "geometry_coordinate": [
                r"\\b(?:coordinate|radius|diameter|circle|triangle|rectangle|perimeter|area|angle|arc|slope)\\b",
                r"(?:координат|радиус|диаметр|окружн|круг|треугольн|прямоугольн|периметр|площад|угол|дуг[аи]|наклон|центр)",
            ],
            "structured_language_list": [
                r"\\b(?:anagram|letters|word form|part of speech|spelling|grammar|synonym|antonym)\\b",
                r"(?:анаграм|букв|слова?|част[ьи]\\s+речи|орфограф|граммат|синоним|антоним|ударен|падеж|морфолог)",
            ],
            "logic_table": [
                r"\\b(?:truth table|logic|logical|boolean|statement)\\b",
                r"(?:таблиц[аы]?\\s+истин|логик|булев|высказыван|истинн|ложн)",
            ],
            "finance_percent": [
                r"\\b(?:percent|discount|interest|price|cost|currency|dollar|euro)\\b",
                r"(?:процент|скидк|стоимост|цена|рубл|доллар|евро|валют|прибыл|процентн)",
            ],
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
            expr = "expr" if re.search(r"\\d\\s*[+*×xх/:=\\-]\\s*\\d", q) else "noexpr"
            return "|".join([length, script_bucket(q), numeric, expr])

        def families_for(question):
            q = normalize(question)
            hits = []
            for family, patterns in FAMILY_PATTERNS.items():
                if any(re.search(pattern, q, flags=re.I) for pattern in patterns):
                    hits.append(family)
            return hits

        def compact(counter):
            return {{str(k): int(v) for k, v in counter.items()}}

        try:
            import pandas as pd
            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            result["raw_task_data_read_remote_only"] = True

            family_counts = Counter()
            by_script = defaultdict(Counter)
            by_bucket = defaultdict(Counter)
            by_target = defaultdict(Counter)
            intersections = Counter()
            no_family = 0
            multi_family = 0
            for _, row in data.iterrows():
                fams = families_for(row["question"])
                if not fams:
                    no_family += 1
                    continue
                multi_family += int(len(fams) > 1)
                bucket = feature_bucket(row["question"])
                script = script_bucket(row["question"])
                shape = target_shape(row["reference_answer"])
                for family in fams:
                    family_counts[family] += 1
                    by_script[family][script] += 1
                    by_bucket[family][bucket] += 1
                    by_target[family][shape] += 1
                for i, a in enumerate(fams):
                    for b in fams[i + 1:]:
                        intersections[" & ".join(sorted([a, b]))] += 1

            result["data_meta"] = {{
                "rows": int(len(data)),
                "no_family_rows": int(no_family),
                "multi_family_rows": int(multi_family),
                "families": sorted(FAMILY_PATTERNS),
            }}
            result["family_counts"] = compact(family_counts)
            result["family_by_script"] = {{family: compact(counts) for family, counts in by_script.items()}}
            result["family_by_bucket"] = {{
                family: [
                    {{"bucket": bucket, "count": int(count)}}
                    for bucket, count in counts.most_common(20)
                ]
                for family, counts in by_bucket.items()
            }}
            result["family_target_shape"] = {{family: compact(counts) for family, counts in by_target.items()}}
            result["family_intersections"] = [
                {{"families": key, "count": int(count)}} for key, count in intersections.most_common(30)
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
    probe_json = None
    try:
        probe_json = json.loads(probe_log)
        base.write_json(paths["probe"], probe_json)
    except Exception:
        probe_json = None
    ok = probe_code == 0 and probe_json is not None and probe_json.get("status") == "completed"
    counts = (probe_json or {}).get("family_counts") or {}
    best_family, best_count = (None, 0)
    if counts:
        best_family, best_count = max(counts.items(), key=lambda item: int(item[1]))
    decision = "MUTATE" if ok and int(best_count) >= 300 else "KILL"
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": decision if ok else "KILL",
            "reason": "Family miner found at least one broad candidate surface." if decision == "MUTATE" else "No broad coherent family surface cleared the count gate.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "best_family": best_family,
            "best_family_count": int(best_count),
            "data_meta": (probe_json or {}).get("data_meta"),
            "family_counts": counts,
            "family_by_script": (probe_json or {}).get("family_by_script"),
            "family_by_bucket": (probe_json or {}).get("family_by_bucket"),
            "family_target_shape": (probe_json or {}).get("family_target_shape"),
            "family_intersections": (probe_json or {}).get("family_intersections"),
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    probe = summary.get("probe") or {}
    lines = [
        "# C257 Coherent Residual Family Miner",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Remote-only aggregate family mining before any code port.",
        "- No raw prompts, references, row ids, outputs, extracted targets, datasets, model weights, or adapter weights returned.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- best family/count: `{summary.get('best_family')}` / `{summary.get('best_family_count')}`",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        f"- runtime: `{summary.get('runtime')}`",
        "",
        "## Data",
        f"`{summary.get('data_meta')}`",
        "",
        "## Family Counts",
        f"`{summary.get('family_counts')}`",
        "",
        "## Family Target Shape",
        f"`{summary.get('family_target_shape')}`",
        "",
        "## Family By Script",
        f"`{summary.get('family_by_script')}`",
        "",
        "## Family By Bucket",
    ]
    for family, rows in (summary.get("family_by_bucket") or {}).items():
        lines.append(f"- {family}: `{rows}`")
    lines.extend(
        [
            "",
            "## Intersections",
            f"`{summary.get('family_intersections')}`",
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
