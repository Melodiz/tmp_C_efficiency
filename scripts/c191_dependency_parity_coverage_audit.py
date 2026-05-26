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
import c171_lora_training_stack_torchao_import_smoke as c171


EXPERIMENT_ID = "C191"
EXPERIMENT_SLUG = "C191_dependency_parity_coverage_audit"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C191_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")
MORPH_PACKAGES = ("pymorphy3==2.0.6", "pymorphy3-dicts-ru", "razdel==0.5.0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C191 dependency-parity final-stack coverage audit.")
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


def install_final_path_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *MORPH_PACKAGES])


def probe_source() -> str:
    return textwrap.dedent(
        f"""
        import importlib.util
        import json
        import re
        import sys
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
            "imports": {{}},
            "data_meta": {{}},
            "handler_counts": {{}},
            "all_handler_counts": {{}},
            "handler_by_target_label": {{}},
            "handler_by_feature_bucket": {{}},
            "fallback_buckets": [],
        }}

        def answer_only_label(value):
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
            cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in str(text))
            lat = sum("a" <= ch.lower() <= "z" for ch in str(text))
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

        def feature_bucket(text):
            q = str(text).lower().replace("ё", "е")
            parts = [
                length_bucket(q),
                script_bucket(q),
                "num" if re.search(r"\\d", q) else "nonnum",
                "expr" if re.search(r"\\d\\s*[+*×xх/:=-]\\s*\\d", q) else "noexpr",
                "open" if re.search(r"\\b(объясн|почему|напишите|сочин|эссе|опишите|перечислите|составьте|расскажите|докажите|explain|write|describe|list)\\b", q) else "closed",
            ]
            return "|".join(parts)

        def compact_counter(counter):
            return {{str(k): int(v) for k, v in counter.items()}}

        try:
            import pandas as pd

            solution_path = Path("simple_solution/solution.py")
            spec = importlib.util.spec_from_file_location("task_c_solution_module", solution_path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            result["imports"]["solution"] = "ok"
            result["imports"]["pymorphy_available"] = bool(module.get_morph_analyzer())

            handlers = [
                ("expression_substitution", module.expression_substitution_answer),
                ("algebra_equation", module.algebra_equation_answer),
                ("exact_numeric", module.exact_numeric_answer),
                ("direct_arithmetic", module.direct_arithmetic_answer),
                ("chemistry_stoichiometry", module.chemistry_stoichiometry_answer),
                ("geometry_exact", module.geometry_exact_answer),
                ("formulaic_math_physics", module.formulaic_math_physics_answer),
                ("structured_school_task", module.structured_school_task_answer),
                ("calculator_written_arithmetic", module.calculator_written_arithmetic_answer),
                ("russian_morph_grammar", module.russian_morph_grammar_answer),
                ("quantity_conversion", module.quantity_conversion_answer),
                ("km_meters", module.km_meters_answer),
            ]

            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            result["raw_task_data_read_remote_only"] = True

            first_counts = Counter()
            all_counts = Counter()
            label_counts = Counter()
            by_label = defaultdict(Counter)
            by_bucket = defaultdict(Counter)
            fallback_bucket_counts = defaultdict(Counter)
            multi_fire = Counter()
            errors = Counter()

            for _, row in data.iterrows():
                question = str(row["question"])
                label = answer_only_label(row["reference_answer"])
                bucket = feature_bucket(question)
                label_counts[label] += 1
                fired = []
                for name, func in handlers:
                    try:
                        value = func(question)
                    except Exception as exc:
                        errors[name + ":" + type(exc).__name__] += 1
                        value = None
                    if value is not None:
                        fired.append(name)
                        all_counts[name] += 1
                if fired:
                    first = fired[0]
                    first_counts[first] += 1
                    by_label[first][label] += 1
                    by_bucket[first][bucket] += 1
                    if len(fired) > 1:
                        multi_fire["+".join(fired[:4])] += 1
                else:
                    first_counts["fallback_model"] += 1
                    by_label["fallback_model"][label] += 1
                    fallback_bucket_counts[bucket][label] += 1

            total = int(len(data))
            result["data_meta"] = {{
                "data_rows": total,
                "target_label_counts": compact_counter(label_counts),
                "deterministic_first_fire_rows": int(total - first_counts.get("fallback_model", 0)),
                "fallback_rows": int(first_counts.get("fallback_model", 0)),
            }}
            result["handler_counts"] = compact_counter(first_counts)
            result["all_handler_counts"] = compact_counter(all_counts)
            result["handler_by_target_label"] = {{name: compact_counter(counts) for name, counts in by_label.items()}}
            result["handler_by_feature_bucket"] = {{
                name: dict(sorted(((bucket, int(count)) for bucket, count in counts.items()), key=lambda item: -item[1])[:30])
                for name, counts in by_bucket.items()
            }}
            result["fallback_buckets"] = [
                {{"bucket": bucket, "total": int(sum(counts.values())), "labels": compact_counter(counts)}}
                for bucket, counts in sorted(fallback_bucket_counts.items(), key=lambda item: -sum(item[1].values()))[:40]
            ]
            result["multi_fire_patterns"] = compact_counter(multi_fire)
            result["handler_errors"] = compact_counter(errors)
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
        install_final_path_dependencies()
        result = subprocess.run([sys.executable, "-c", probe_source()], check=False, text=True, capture_output=True, timeout=1200)
        probe_code = result.returncode
        probe_log = ((result.stdout or "") + (result.stderr or "")).strip()
    except Exception as exc:
        probe_code = 999
        probe_log = f"{type(exc).__name__}: {exc}"
    paths["probe_log"].write_text(probe_log + "\n", encoding="utf-8")
    probe_json = None
    try:
        probe_json = c171.parse_probe_json(probe_log) or json.loads(probe_log)
        base.write_json(paths["probe"], probe_json)
    except Exception:
        probe_json = None
    ok = probe_code == 0 and probe_json is not None and probe_json.get("status") == "completed"
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": "MUTATE" if ok else "KILL",
            "reason": "Dependency-parity final-stack coverage audit completed." if ok else "Dependency-parity final-stack coverage audit failed.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "imports": (probe_json or {}).get("imports"),
            "handler_counts": (probe_json or {}).get("handler_counts"),
            "all_handler_counts": (probe_json or {}).get("all_handler_counts"),
            "handler_by_target_label": (probe_json or {}).get("handler_by_target_label"),
            "handler_by_feature_bucket": (probe_json or {}).get("handler_by_feature_bucket"),
            "fallback_buckets": (probe_json or {}).get("fallback_buckets"),
            "multi_fire_patterns": (probe_json or {}).get("multi_fire_patterns"),
            "handler_errors": (probe_json or {}).get("handler_errors"),
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    data_meta = summary.get("data_meta") or {}
    probe = summary.get("probe") or {}
    lines = [
        "# C191 Dependency-Parity Final-Stack Coverage Audit",
        "",
        "## Objective",
        "- No leaderboard submission.",
        "- Remote CPU aggregate audit; no model load or training.",
        "- Install the final morphology dependency set before probing, matching the final-entrypoint C131+ path.",
        "- Return only handler-family and bucket counts; no raw text, answers, outputs, row ids, model weights, or adapter weights.",
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
        f"- deterministic first-fire rows: `{data_meta.get('deterministic_first_fire_rows')}`",
        f"- fallback rows: `{data_meta.get('fallback_rows')}`",
        f"- imports: `{summary.get('imports')}`",
        "",
        "## Handler Counts",
        f"- first-fire counts: `{summary.get('handler_counts')}`",
        f"- all-fire counts: `{summary.get('all_handler_counts')}`",
        f"- by target label: `{summary.get('handler_by_target_label')}`",
        f"- multi-fire patterns: `{summary.get('multi_fire_patterns')}`",
        f"- handler errors: `{summary.get('handler_errors')}`",
        "",
        "## Handler Buckets",
        f"`{summary.get('handler_by_feature_bucket')}`",
        "",
        "## Fallback Buckets",
    ]
    for item in (summary.get("fallback_buckets") or [])[:30]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Hygiene",
            f"- raw examples returned: `{summary.get('raw_examples_returned')}`",
            f"- row ids returned: `{summary.get('row_ids_returned')}`",
            f"- outputs returned: `{summary.get('outputs_returned')}`",
            f"- model loaded: `{summary.get('model_loaded')}`",
            f"- training started: `{summary.get('training_started')}`",
            f"- adapter weights returned: `{summary.get('adapter_weights_returned')}`",
            f"- error: `{probe.get('error')}`" if probe.get("error") else "- error: none",
            "",
            "## Next",
            "Compare with C190 to decide whether dependency-enabled morphology creates a broad safe follow-up.",
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
