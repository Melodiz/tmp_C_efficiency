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


EXPERIMENT_ID = "C265"
EXPERIMENT_SLUG = "C265_reference_style_aggregate_analysis"
DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C265_artifacts"
DATA_PATH = Path("data/dataset_ml_challenge.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C265 S1 reference-style aggregate analysis.")
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
        import statistics
        import traceback
        from collections import Counter, defaultdict
        from pathlib import Path

        result = {{
            "status": "failed",
            "leaderboard_submission": False,
            "raw_task_data_read_remote_only": False,
            "raw_examples_returned": False,
            "row_ids_returned": False,
            "prompts_returned": False,
            "references_returned": False,
            "outputs_returned": False,
            "targets_returned": False,
            "model_loaded": False,
            "training_started": False,
            "adapter_weights_returned": False,
            "data_meta": {{}},
            "length_stats": {{}},
            "bucket_length_stats": [],
            "marker_counts": {{}},
            "register_counts": {{}},
            "template_counts": [],
            "template_by_bucket": [],
            "gate": {{}},
        }}

        ANSWER_MARKER = r"^(?:ответ|итоговый ответ|итог|answer|final answer)\\s*[:：\\-]"
        ANY_ANSWER_MARKER = r"(?:^|\\n)\\s*(?:ответ|итоговый ответ|итог|answer|final answer)\\s*[:：\\-]"
        FORMAL_MARKERS = {{
            "solution": r"\\b(?:решение|solution)\\b",
            "given": r"\\b(?:дано|given)\\b",
            "therefore": r"\\b(?:следовательно|таким образом|значит|therefore|thus)\\b",
            "obtain": r"\\b(?:получаем|получится|получим|we get|we obtain)\\b",
            "because": r"\\b(?:потому что|так как|because|since)\\b",
            "formula": r"(?:формул|formula)",
        }}

        def normalize(text):
            return str(text).strip()

        def script_bucket(text):
            text = str(text)
            cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
            lat = sum("a" <= ch.lower() <= "z" for ch in text)
            if cyr > lat:
                return "cyrillic"
            if lat > cyr:
                return "latin"
            return "mixed_or_symbolic"

        def question_bucket(question):
            q = str(question).lower().replace("ё", "е")
            n = len(q)
            length = "q_short" if n <= 80 else "q_medium" if n <= 180 else "q_long" if n <= 350 else "q_very_long"
            numeric = "num" if re.search(r"\\d", q) else "nonnum"
            openish = "open" if re.search(r"(объясн|почему|напиши|сочин|эссе|опиши|перечисл|расскаж|докаж|обосну|explain|write|describe|list|why)", q) else "closed"
            stemish = "stem" if re.search(r"(уравнен|вычисл|найд|сколько|формул|хими|физик|геометр|процент|задач|equation|calculate|find|formula|chem|physics|geometry)", q) else "nonstem"
            return "|".join([length, script_bucket(q), numeric, openish, stemish])

        def word_count(text):
            return len(re.findall(r"\\S+", str(text)))

        def quantiles(values):
            if not values:
                return {{}}
            values = sorted(values)
            def q(p):
                if len(values) == 1:
                    return float(values[0])
                pos = (len(values) - 1) * p
                lo = math.floor(pos)
                hi = math.ceil(pos)
                if lo == hi:
                    return float(values[lo])
                return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))
            return {{"p10": q(0.10), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}}

        def stats(values):
            values = [int(v) for v in values]
            if not values:
                return {{"count": 0}}
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            out = {{
                "count": int(len(values)),
                "mean": float(mean),
                "std": float(std),
                "std_over_mean": float(std / mean) if mean else None,
                "min": int(min(values)),
                "max": int(max(values)),
            }}
            out.update(quantiles(values))
            return out

        def length_bucket(tokens):
            if tokens <= 3:
                return "tiny"
            if tokens <= 12:
                return "short"
            if tokens <= 40:
                return "medium"
            if tokens <= 120:
                return "long"
            return "very_long"

        def marker_profile(ref):
            text = normalize(ref)
            low = text.lower().replace("ё", "е")
            lines = [line for line in text.splitlines() if line.strip()]
            tokens = word_count(text)
            answer_start = bool(re.search(ANSWER_MARKER, low, flags=re.I))
            answer_any = bool(re.search(ANY_ANSWER_MARKER, low, flags=re.I))
            multiline = len(lines) > 1
            bullet = bool(re.search(r"(^|\\n)\\s*(?:[-*•]|\\d+[.)])\\s+", text))
            equation = bool(re.search(r"[=<>]|\\d\\s*[+*/:×\\-]\\s*\\d", text))
            ending_punct = bool(re.search(r"[.!?。]$", text))
            if answer_start:
                structure = "answer_start"
            elif answer_any:
                structure = "answer_later"
            elif bullet:
                structure = "list"
            elif multiline:
                structure = "multiline"
            else:
                structure = "plain"
            line_label = "single_line" if len(lines) <= 1 else "few_lines" if len(lines) <= 4 else "many_lines"
            punct_label = "sentential" if ending_punct else "bare"
            eq_label = "equation" if equation else "no_equation"
            return {{
                "tokens": tokens,
                "chars": len(text),
                "lines": len(lines),
                "answer_start": answer_start,
                "answer_any": answer_any,
                "multiline": multiline,
                "bullet_or_numbered": bullet,
                "equation_like": equation,
                "ending_punct": ending_punct,
                "template": "|".join([length_bucket(tokens), structure, line_label, punct_label, eq_label]),
            }}

        def compact_counter(counter):
            return {{str(k): int(v) for k, v in counter.items()}}

        try:
            import pandas as pd
            data = pd.read_parquet(Path("{DATA_PATH}")).reset_index(drop=True)
            data = data.rename(columns={{"query": "question", "answer": "reference_answer"}})
            data = data.dropna(subset=["question", "reference_answer"]).copy()
            result["raw_task_data_read_remote_only"] = True

            token_values = []
            char_values = []
            line_values = []
            marker_counts = Counter()
            register_counts = Counter()
            templates = Counter()
            bucket_tokens = defaultdict(list)
            bucket_templates = defaultdict(Counter)
            bucket_markers = defaultdict(Counter)

            for _, row in data.iterrows():
                ref = row["reference_answer"]
                bucket = question_bucket(row["question"])
                prof = marker_profile(ref)
                token_values.append(prof["tokens"])
                char_values.append(prof["chars"])
                line_values.append(prof["lines"])
                bucket_tokens[bucket].append(prof["tokens"])
                templates[prof["template"]] += 1
                bucket_templates[bucket][prof["template"]] += 1

                for name in ["answer_start", "answer_any", "multiline", "bullet_or_numbered", "equation_like", "ending_punct"]:
                    marker_counts[name] += int(prof[name])
                    bucket_markers[bucket][name] += int(prof[name])
                low = normalize(ref).lower().replace("ё", "е")
                for name, pattern in FORMAL_MARKERS.items():
                    hit = bool(re.search(pattern, low, flags=re.I))
                    register_counts[name] += int(hit)
                    bucket_markers[bucket][f"reg_{name}"] += int(hit)

            rows = int(len(data))
            top_template, top_template_count = ("", 0)
            if templates:
                top_template, top_template_count = templates.most_common(1)[0]
            top_template_share = float(top_template_count / rows) if rows else 0.0
            actionable = []
            for bucket, counts in bucket_templates.items():
                support = sum(counts.values())
                if support < 100:
                    continue
                name, count = counts.most_common(1)[0]
                share = count / support
                markers = bucket_markers[bucket]
                answer_share = markers.get("answer_any", 0) / support
                if share >= 0.30 or answer_share >= 0.30:
                    actionable.append({{
                        "bucket": bucket,
                        "support": int(support),
                        "top_template": name,
                        "top_template_share": float(share),
                        "answer_marker_share": float(answer_share),
                    }})
            actionable.sort(key=lambda item: (-item["support"], -item["top_template_share"]))

            token_stats = stats(token_values)
            heterogeneous = bool((token_stats.get("std_over_mean") or 0.0) > 2.0)
            gate_pass = bool(top_template_share >= 0.30 or len(actionable) >= 3)
            if heterogeneous and top_template_share < 0.30 and len(actionable) < 3:
                gate_pass = False

            result["data_meta"] = {{"rows": rows}}
            result["length_stats"] = {{
                "tokens": token_stats,
                "chars": stats(char_values),
                "lines": stats(line_values),
            }}
            result["bucket_length_stats"] = [
                {{"bucket": bucket, "token_stats": stats(values)}}
                for bucket, values in sorted(bucket_tokens.items(), key=lambda item: -len(item[1]))[:40]
            ]
            result["marker_counts"] = compact_counter(marker_counts)
            result["register_counts"] = compact_counter(register_counts)
            result["template_counts"] = [
                {{"template": name, "count": int(count), "share": float(count / rows) if rows else 0.0}}
                for name, count in templates.most_common(30)
            ]
            result["template_by_bucket"] = actionable[:30]
            result["gate"] = {{
                "s1_gate1_pass": gate_pass,
                "top_template": top_template,
                "top_template_count": int(top_template_count),
                "top_template_share": top_template_share,
                "token_std_over_mean": token_stats.get("std_over_mean"),
                "heterogeneous_length": heterogeneous,
                "actionable_bucket_count": int(len(actionable)),
                "kill_reason": None if gate_pass else "Reference styles are too heterogeneous for safe global or bucketed S1 postprocessing.",
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
        "prompts_returned": False,
        "references_returned": False,
        "outputs_returned": False,
        "targets_returned": False,
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
    try:
        probe_json = json.loads(probe_log)
        base.write_json(paths["probe"], probe_json)
    except Exception:
        probe_json = None
    ok = probe_code == 0 and probe_json is not None and probe_json.get("status") == "completed"
    gate = (probe_json or {}).get("gate") or {}
    decision = "MUTATE" if ok and gate.get("s1_gate1_pass") else "KILL"
    reason = (
        "S1 Gate 1 found a dominant or bucket-actionable reference style pattern."
        if decision == "MUTATE"
        else "S1 Gate 1 did not find a safe dominant reference-style pattern."
    )
    summary.update(
        {
            "status": "completed" if ok else "failed",
            "decision_recommendation": decision if ok else "KILL",
            "reason": reason if ok else "Reference-style aggregate analysis failed.",
            "probe_returncode": probe_code,
            "probe": probe_json,
            "data_meta": (probe_json or {}).get("data_meta"),
            "length_stats": (probe_json or {}).get("length_stats"),
            "bucket_length_stats": (probe_json or {}).get("bucket_length_stats"),
            "marker_counts": (probe_json or {}).get("marker_counts"),
            "register_counts": (probe_json or {}).get("register_counts"),
            "template_counts": (probe_json or {}).get("template_counts"),
            "template_by_bucket": (probe_json or {}).get("template_by_bucket"),
            "gate": gate,
            "runtime": {"total_seconds": time.time() - start},
        }
    )
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    probe = summary.get("probe") or {}
    gate = summary.get("gate") or {}
    lines = [
        "# C265 S1 Reference-Style Aggregate Analysis",
        "",
        "## Objective",
        "- No leaderboard submission or submission zip.",
        "- Execute S1 Gate 1 from supervisor strategy.",
        "- Analyze training reference style/length/register/structure remotely and return aggregate metrics only.",
        "- No raw prompts, references, row ids, outputs, targets, datasets, model weights, or adapter files returned.",
        "",
        "## Result",
        f"- status: `{summary.get('status')}`",
        f"- decision recommendation: `{summary.get('decision_recommendation')}`",
        f"- reason: {summary.get('reason')}",
        f"- probe return code: `{summary.get('probe_returncode')}`",
        f"- runtime: `{summary.get('runtime')}`",
        "",
        "## Gate",
        f"`{gate}`",
        "",
        "## Data",
        f"`{summary.get('data_meta')}`",
        "",
        "## Length Stats",
        f"`{summary.get('length_stats')}`",
        "",
        "## Markers",
        f"- structural: `{summary.get('marker_counts')}`",
        f"- register: `{summary.get('register_counts')}`",
        "",
        "## Top Templates",
        f"`{summary.get('template_counts')}`",
        "",
        "## Actionable Buckets",
        f"`{summary.get('template_by_bucket')}`",
        "",
        "## Bucket Length Stats",
        f"`{summary.get('bucket_length_stats')}`",
        "",
        "## Hygiene",
        f"- raw task data read remote only: `{probe.get('raw_task_data_read_remote_only')}`",
        f"- raw examples returned: `{probe.get('raw_examples_returned')}`",
        f"- row ids returned: `{probe.get('row_ids_returned')}`",
        f"- prompts returned: `{probe.get('prompts_returned')}`",
        f"- references returned: `{probe.get('references_returned')}`",
        f"- outputs returned: `{probe.get('outputs_returned')}`",
        f"- targets returned: `{probe.get('targets_returned')}`",
        f"- model loaded: `{probe.get('model_loaded')}`",
        f"- training started: `{probe.get('training_started')}`",
        f"- adapter weights returned: `{probe.get('adapter_weights_returned')}`",
    ]
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
