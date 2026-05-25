"""C092 candidate: vLLM + Qwen3-8B-AWQ with narrow postprocessing.

Input:  /workspace/input.pickle
Output: /workspace/output.json
Weights: ./weights should contain Qwen/Qwen3-8B-AWQ files.
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


MODEL_DIR = "./weights"
USER_PREFIX = "Ответь кратко и точно на языке задания. Не повторяй условие. В конце дай только итоговый ответ."
MAX_NEW_TOKENS = 320
MAX_MODEL_LEN = 4096
NUMBER_RE = r"[+-]?\d+(?:[,.]\d+)?"


class UnsafeExpression(ValueError):
    pass


def parse_fraction(raw: str) -> Fraction:
    return Fraction(raw.replace(",", ".").replace("−", "-"))


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def normalize_expression(raw: str) -> str | None:
    expr = raw.strip().strip("$").strip()
    expr = expr.replace("\\cdot", "*").replace("×", "*").replace("−", "-")
    expr = expr.replace("{", "(").replace("}", ")")
    expr = re.sub(r"\s+", "", expr)
    if not expr or len(expr) > 160:
        return None
    if re.search(r"[^0-9A-Za-z+\-*/^().,]", expr):
        return None
    expr = expr.replace(",", ".").replace("^", "**")
    expr = re.sub(r"(\d)([A-Za-z])", r"\1*\2", expr)
    expr = re.sub(r"([A-Za-z]|\d|\))\(", r"\1*(", expr)
    expr = re.sub(r"\)([A-Za-z]|\d)", r")*\1", expr)
    return expr


def eval_ast(node: ast.AST, variables: dict[str, Fraction]) -> Fraction:
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise UnsafeExpression(f"unbound variable {node.id}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = eval_ast(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
    if isinstance(node, ast.BinOp):
        left = eval_ast(node.left, variables)
        right = eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsafeExpression("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 6:
                raise UnsafeExpression("unsafe exponent")
            return left**right.numerator
    raise UnsafeExpression(f"unsupported expression node {type(node).__name__}")


def expression_substitution_answer(question: str) -> str | None:
    text = " ".join(question.replace("\u202f", " ").replace("\xa0", " ").split())
    if not re.search(r"найд[иите]\s+значение\s+выражения", text, flags=re.IGNORECASE):
        return None
    match = re.search(r"значение\s+выражения\s+(.+?)\s+при\s+(.+)", text, flags=re.IGNORECASE)
    if not match:
        return None

    expr_raw = match.group(1)
    assign_text = match.group(2)
    dollar = re.search(r"\$(.+?)\$", expr_raw)
    if dollar:
        expr_raw = dollar.group(1)

    assignments = {
        name: parse_fraction(value)
        for name, value in re.findall(r"\b([A-Za-z])\s*=\s*(" + NUMBER_RE + r")", assign_text)
    }
    if not assignments:
        return None

    expr = normalize_expression(expr_raw)
    if expr is None:
        return None
    used_names = set(re.findall(r"[A-Za-z]", expr))
    if not used_names or not used_names.issubset(assignments):
        return None

    try:
        value = eval_ast(ast.parse(expr, mode="eval"), assignments)
    except (SyntaxError, UnsafeExpression, ValueError, ZeroDivisionError):
        return None

    answer = format_fraction(value)
    return f"{answer}\n\nИтоговый ответ: {answer}"


def has_repetition_loop(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        most_common = max(lines.count(line) for line in set(lines))
        if most_common >= 4:
            return True
    words = text.split()
    if len(words) >= 80:
        tail = words[-40:]
        return len(set(tail)) / max(1, len(tail)) < 0.25
    return False


def normalize_item(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().strip(" .;:!?")).replace("ё", "е")


def dedup_comma_loop(answer: str) -> str | None:
    if "," not in answer or not has_repetition_loop(answer):
        return None
    compact = " ".join(answer.replace("\n", " ").split())
    prefix = ""
    body = compact
    if ":" in compact[:80]:
        prefix, body = compact.split(":", 1)
        prefix = prefix.strip() + ": "

    items = [item.strip(" .;:!?") for item in body.split(",") if item.strip(" .;:!?")]
    if len(items) < 12:
        return None

    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = normalize_item(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    if len(unique) < 2 or len(unique) / len(items) > 0.45:
        return None

    answer_list = ", ".join(unique[:30])
    first_line = f"{prefix}{answer_list}".strip()
    return f"{first_line}\n\nИтоговый ответ: {answer_list}"


def is_english_prompt(question: str) -> bool:
    latin = len(re.findall(r"[A-Za-z]", question))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", question))
    return latin >= 10 and latin > cyrillic * 2


def is_cloze_prompt(question: str) -> bool:
    text = " ".join(question.split())
    if "____" in text or "＿" in text:
        return True
    if re.search(r"\b[A-Z]{2,}\b\s*(?:[.!?])?$", text):
        return True
    if re.search(r"\bchoose\b", text, flags=re.IGNORECASE) and len(text.split()) <= 18:
        return True
    return False


def cleanup_english_cloze_answer(question: str, answer: str) -> str | None:
    if not is_english_prompt(question) or not is_cloze_prompt(question):
        return None
    if not re.search(r"[А-Яа-яЁё]", answer) or not re.search(r"ответ\s*:", answer, flags=re.IGNORECASE):
        return None

    before_marker = re.split(r"\*{0,2}\s*Ответ\s*:\s*\*{0,2}", answer, maxsplit=1, flags=re.IGNORECASE)[0]
    first_lines = [line.strip(" *") for line in before_marker.splitlines() if line.strip(" *")]
    if len(first_lines) != 1:
        return None
    first = first_lines[0].strip()
    if not re.search(r"[A-Za-z]", first) or re.search(r"[А-Яа-яЁё]", first):
        return None
    if len(first.split()) > 5:
        return None
    return first


def parse_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ".").replace("−", "-"))
    except (InvalidOperation, ValueError):
        return None


def format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f").rstrip("0").rstrip(".").replace(".", ",")


def km_meters_answer(question: str) -> str | None:
    text = question.lower().replace("\u202f", " ").replace("\xa0", " ").replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip()
    match = re.fullmatch(
        rf"({NUMBER_RE})\s+километр(?:ов|а)?\s+({NUMBER_RE})\s+метр(?:ов|а)?\s+.*сколько\s+метр(?:ов)?",
        text,
    )
    if not match:
        return None
    km = parse_decimal(match.group(1))
    meters = parse_decimal(match.group(2))
    if km is None or meters is None:
        return None
    value = format_decimal(km * Decimal(1000) + meters)
    return f"{value} метров\n\nИтоговый ответ: {value} метров"


def build_prompt(tokenizer: Any, question: str) -> str:
    content = f"{USER_PREFIX}\n\n{question}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    llm = LLM(
        model=MODEL_DIR,
        dtype="float16",
        quantization="awq_marlin",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=0,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,
        top_k=-1,
    )

    prompts = [build_prompt(tokenizer, row["question"]) for row in rows]
    outputs = llm.generate(prompts, sampling_params=sampling)

    result = []
    for row, out in zip(rows, outputs):
        answer = out.outputs[0].text.strip()
        answer = expression_substitution_answer(row["question"]) or answer
        answer = dedup_comma_loop(answer) or answer
        answer = cleanup_english_cloze_answer(row["question"], answer) or answer
        answer = km_meters_answer(row["question"]) or answer
        result.append({"rid": row["rid"], "answer": answer})

    with open("output.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
