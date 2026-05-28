from __future__ import annotations

from pathlib import Path
from typing import Sequence

import c177_base_vs_lora_aggregate_validation_smoke as c177
import c178_sft_aggregate_metric_cap_diagnostic as c178
import c273_min_full_reference_sft_unblock as c273


EXPERIMENT_ID = "C325"
EXPERIMENT_SLUG = "C325_compressed_reference_sft_smoke"


def task_probe_source(
    model_id: str,
    train_rows: int,
    val_rows: int,
    steps: int,
    max_seq_len: int,
    max_new_tokens: int,
    seed: int,
) -> str:
    source = c273.task_probe_source(model_id, train_rows, val_rows, steps, max_seq_len, max_new_tokens, seed)
    source = source.replace(
        '    data["question_len"] = data["question"].astype(str).str.len()\n'
        '    data["answer_len"] = data["reference_answer"].astype(str).str.len()\n'
        '    pool = data[(data["question_len"] <= 500) & (data["answer_len"] <= 3600)].copy()\n',
        '    def compressed_reference_answer(text, target_tokens=160):\n'
        '        text = str(text).strip()\n'
        '        tokens = re.findall(r"\\\\S+", text)\n'
        '        if len(tokens) <= target_tokens:\n'
        '            return text\n'
        '        sentences = re.split(r"(?<=[.!?。！？])\\\\s+|\\\\n+", text)\n'
        '        kept = []\n'
        '        kept_len = 0\n'
        '        for sentence in sentences:\n'
        '            sentence = sentence.strip()\n'
        '            if not sentence:\n'
        '                continue\n'
        '            sentence_len = len(re.findall(r"\\\\S+", sentence))\n'
        '            if kept and kept_len + sentence_len > target_tokens:\n'
        '                break\n'
        '            if not kept and sentence_len > target_tokens:\n'
        '                return " ".join(tokens[:target_tokens])\n'
        '            kept.append(sentence)\n'
        '            kept_len += sentence_len\n'
        '        return "\\\\n".join(kept) if kept else " ".join(tokens[:target_tokens])\n'
        '    data["question_len"] = data["question"].astype(str).str.len()\n'
        '    data["answer_len"] = data["reference_answer"].astype(str).str.len()\n'
        '    data["training_answer"] = data["reference_answer"].map(compressed_reference_answer)\n'
        '    data["training_answer_len"] = data["training_answer"].astype(str).str.len()\n'
        '    data["compressed_changed"] = (data["training_answer"].astype(str) != data["reference_answer"].astype(str))\n'
        '    pool = data[(data["question_len"] <= 500) & (data["training_answer_len"] <= 1200)].copy()\n',
    )
    source = source.replace(
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        '        "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),\n',
        '        "training_target": "compressed_reference_answer_160_tokens",\n'
        '        "train_compressed_changed": int(sum(bool(r.get("compressed_changed")) for r in train)),\n'
        '        "val_compressed_changed": int(sum(bool(r.get("compressed_changed")) for r in val)),\n'
        '        "train_shape_counts": dict(Counter(shape_bucket(r) for r in train)),\n'
        '        "val_shape_counts": dict(Counter(shape_bucket(r) for r in val)),\n',
    )
    source = source.replace(
        '        full = tokenizer.apply_chat_template(build_messages(row["question"], row["reference_answer"]), tokenize=False)\n',
        '        full = tokenizer.apply_chat_template(build_messages(row["question"], row.get("training_answer", row["reference_answer"])), tokenize=False)\n',
    )
    if "compressed_reference_answer" not in source or 'row.get("training_answer"' not in source:
        raise RuntimeError("C325 source patch failed")
    return source


def write_report(path: Path, summary: dict) -> None:
    c178.write_report(path, summary)
    text = path.read_text(encoding="utf-8")
    text = text.replace("# C178 SFT Aggregate Metric/Cap Diagnostic", "# C325 Compressed-Reference SFT Smoke", 1)
    text = text.replace(
        "If containment/final-line metrics remain zero or cap-dominated, kill or park tiny SFT rather than scaling it.",
        "Kill unless both containment proxies improve without cap/invalid/repetition worsening versus the base on the same validation rows.",
    )
    path.write_text(text, encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    c177.EXPERIMENT_ID = EXPERIMENT_ID
    c177.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    c177.DEFAULT_OUT_DIR = Path("artifacts") / "tmp" / "C325_artifacts"
    c177.DEFAULT_TARGET_DIR = Path("/content/c325_train_site")
    c177.REMOTE_ADAPTER_DIR = Path("/content/c325_adapter_scratch")
    c177.task_probe_source = task_probe_source
    c177.write_report = write_report
    forwarded = list(argv or [])
    defaults = (
        ("--train-rows", "96"),
        ("--val-rows", "24"),
        ("--steps", "24"),
        ("--max-seq-len", "512"),
        ("--max-new-tokens", "224"),
        ("--seed", "325"),
    )
    for flag, value in defaults:
        if flag not in forwarded:
            forwarded.extend([flag, value])
    return c177.run(forwarded)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
