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


UNIT_FACTORS: dict[str, tuple[str, Decimal, int]] = {
    "мм": ("мм", Decimal("0.001"), 1),
    "миллиметр": ("мм", Decimal("0.001"), 1),
    "см": ("см", Decimal("0.01"), 1),
    "сантиметр": ("см", Decimal("0.01"), 1),
    "дм": ("дм", Decimal("0.1"), 1),
    "дециметр": ("дм", Decimal("0.1"), 1),
    "м": ("м", Decimal("1"), 1),
    "метр": ("м", Decimal("1"), 1),
    "км": ("км", Decimal("1000"), 1),
    "километр": ("км", Decimal("1000"), 1),
    "ар": ("ар", Decimal("100"), 2),
    "га": ("га", Decimal("10000"), 2),
    "гектар": ("га", Decimal("10000"), 2),
}


UNIT_LABELS = {
    (1, "мм"): "мм",
    (1, "см"): "см",
    (1, "дм"): "дециметров",
    (1, "м"): "метров",
    (1, "км"): "км",
    (2, "мм"): "мм²",
    (2, "см"): "см²",
    (2, "дм"): "дм²",
    (2, "м"): "м²",
    (2, "км"): "км²",
    (2, "ар"): "ар",
    (2, "га"): "га",
}


NAMED_NUMBER_POWERS = {
    "миллион": 6,
    "миллионов": 6,
    "миллиард": 9,
    "миллиардов": 9,
    "триллион": 12,
    "триллионов": 12,
    "квадриллион": 15,
    "квадриллионов": 15,
    "квинтиллион": 18,
    "квинтиллионов": 18,
    "секстиллион": 21,
    "секстиллионов": 21,
    "септиллион": 24,
    "септиллионов": 24,
    "октиллион": 27,
    "октиллионов": 27,
    "нониллион": 30,
    "нониллионов": 30,
    "дециллион": 33,
    "дециллионов": 33,
    "ундециллион": 36,
    "ундециллионов": 36,
}


def normalize_conversion_text(question: str) -> str:
    text = question.lower().replace("\u202f", " ").replace("\xa0", " ").replace("−", "-")
    text = text.replace("ё", "е").replace("²", "^2").replace("³", "^3").replace(",", ".")
    text = re.sub(r"\bкв\.\s*", "квадратных ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_conversion_number(raw: str) -> Decimal | None:
    try:
        return Decimal(re.sub(r"\s+", "", raw))
    except (InvalidOperation, ValueError):
        return None


def parse_metric_unit(raw: str) -> tuple[str, Decimal, int] | None:
    text = raw.strip()
    if "/" in text:
        return None

    square = False
    if re.match(r"^(?:квадратн\w*|кв)\s+", text):
        square = True
        text = re.sub(r"^(?:квадратн\w*|кв)\s+", "", text)

    match = re.match(r"^(миллиметр\w*|сантиметр\w*|дециметр\w*|километр\w*|гектар\w*|метр\w*|мм|см|дм|км|м|га|ар)(?:\^2)?", text)
    if not match:
        return None

    token = match.group(1)
    key = None
    for candidate in sorted(UNIT_FACTORS, key=len, reverse=True):
        if token == candidate or token.startswith(candidate):
            key = candidate
            break
    if key is None:
        return None

    canon, factor, forced_dim = UNIT_FACTORS[key]
    tail = text[match.end() :]
    if re.search(r"\s+квадратн\w*", tail) or "^2" in match.group(0):
        square = True
    dim = forced_dim if forced_dim == 2 else (2 if square else 1)
    if dim == 2 and forced_dim != 2:
        factor *= factor
    return canon, factor, dim


def conversion_label(unit: tuple[str, Decimal, int]) -> str:
    canon, _, dim = unit
    return UNIT_LABELS.get((dim, canon), "м²" if dim == 2 else "метров")


def conversion_answer(value: Decimal, unit: tuple[str, Decimal, int]) -> str:
    rendered = format_decimal(value)
    label = conversion_label(unit)
    return f"{rendered} {label}\n\nИтоговый ответ: {rendered} {label}"


def quantity_conversion_answer(question: str) -> str | None:
    text = normalize_conversion_text(question)

    mixed = re.fullmatch(
        rf"({NUMBER_RE})\s+километр\w*\s+({NUMBER_RE})\s+метр\w*\s+.*сколько\s+метр\w*",
        text,
    )
    if mixed:
        km = parse_conversion_number(mixed.group(1))
        meters = parse_conversion_number(mixed.group(2))
        if km is not None and meters is not None:
            value = km * Decimal(1000) + meters
            return f"{format_decimal(value)} метров\n\nИтоговый ответ: {format_decimal(value)} метров"

    how_many = re.fullmatch(r"сколько\s+(.+?)\s+в\s+(?:одном\s+)?([\d.\s]+)?\s*(.+)", text)
    if how_many:
        target = parse_metric_unit(how_many.group(1))
        source = parse_metric_unit(how_many.group(3))
        amount = parse_conversion_number(how_many.group(2) or "1")
        if source and target and amount is not None and source[2] == target[2]:
            return conversion_answer(amount * source[1] / target[1], target)

    default_area = re.fullmatch(rf"({NUMBER_RE})\s+(ар|га)\s+это\s+сколько", text)
    if default_area:
        source = parse_metric_unit(default_area.group(2))
        amount = parse_conversion_number(default_area.group(1))
        if source and amount is not None:
            target = ("м", Decimal(1), 2)
            return conversion_answer(amount * source[1], target)

    patterns = [
        rf"^([\d.\s]+)\s+(.+?)\s+(?:переведи|переведите|перевести)\s+в\s+(.+?)(?:\.|$)",
        rf"^(?:переведи|переведите|перевести)\s+([\d.\s]+(?:\.\d+)?)\s+(.+?)\s+в\s+(.+?)(?:\.|$)",
        rf"^([\d.\s]+)\s+(.+?)\s+в\s+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        source_text = match.group(2)
        target_text = match.group(3)
        if "/" in source_text or "/" in target_text:
            return None
        amount = parse_conversion_number(match.group(1))
        source = parse_metric_unit(source_text)
        target = parse_metric_unit(target_text)
        if source and target and source[2] == 2 and target[2] == 1 and re.match(r"метр\w*", target_text.strip()):
            target = ("м", Decimal(1), 2)
        if source and target and amount is not None and source[2] == target[2]:
            return conversion_answer(amount * source[1] / target[1], target)

    named = re.search(
        r"(ундециллион|дециллион|нониллион|октиллион|септиллион|секстиллион|квинтиллион|квадриллион|триллион|миллиард|миллион)\s*(?:—|-|это|\s)*\s*сколько\s+(триллионов|миллиардов|миллионов)",
        text,
    )
    if named:
        source_power = NAMED_NUMBER_POWERS.get(named.group(1))
        target_power = NAMED_NUMBER_POWERS.get(named.group(2))
        if source_power is not None and target_power is not None and source_power >= target_power:
            diff = source_power - target_power
            value = "1" if diff == 0 else f"10^{diff}"
            target_name = named.group(2)
            return f"{value} {target_name}\n\nИтоговый ответ: {value} {target_name}"

    return None


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
        answer = quantity_conversion_answer(row["question"]) or answer
        answer = km_meters_answer(row["question"]) or answer
        result.append({"rid": row["rid"], "answer": answer})

    with open("output.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
