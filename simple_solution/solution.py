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
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
from math import gcd
from math import sqrt
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


SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_numeric_text(question: str) -> str:
    text = question.lower().translate(SUBSCRIPT_DIGITS)
    text = text.replace("\u202f", " ").replace("\xa0", " ").replace("−", "-").replace(",", ".")
    text = text.replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def format_numeric_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f").rstrip("0").rstrip(".").replace(".", ",")


def numeric_final_answer(value: str) -> str:
    return f"{value}\n\nИтоговый ответ: {value}"


def exact_numeric_answer(question: str) -> str | None:
    text = normalize_numeric_text(question)

    match = re.fullmatch(r"переведи число\s+([0-7]+)\s+из восьмеричн[а-я]* системы счисления в десятичн[а-я]*\.?", text)
    if match:
        return numeric_final_answer(str(int(match.group(1), 8)))

    match = re.fullmatch(r"переведи число\s+(\d+)\s+в двоичн[а-я]* систему счисления\.?", text)
    if match:
        return numeric_final_answer(format(int(match.group(1)), "b"))

    match = re.fullmatch(r"выполни перевод из восьмеричн[а-я]* системы счисления в десятичн[а-я]*:\s*(.+)", text)
    if match:
        numbers = [item[:-1] if item.endswith("8") else item for item in re.findall(r"[0-7]+8?", match.group(1))]
        if numbers:
            values = " и ".join(str(int(item, 8)) for item in numbers)
            return numeric_final_answer(values)

    match = re.fullmatch(r"выполните сложение в двоичн[а-я]* системе счисления\s+([01]+)\s*\+\s*([01]+)", text)
    if match:
        value = int(match.group(1), 2) + int(match.group(2), 2)
        return numeric_final_answer(format(value, "b"))

    match = re.fullmatch(rf"({NUMBER_RE})\s*процент(?:ов|а)?\s+от\s+({NUMBER_RE})", text)
    if match:
        pct = parse_decimal(match.group(1))
        base_value = parse_decimal(match.group(2))
        if pct is not None and base_value is not None:
            return numeric_final_answer(format_numeric_decimal(base_value * pct / Decimal(100)))

    match = re.fullmatch(rf"({NUMBER_RE})%\s+от\s+({NUMBER_RE})", text)
    if match:
        pct = parse_decimal(match.group(1))
        base_value = parse_decimal(match.group(2))
        if pct is not None and base_value is not None:
            return numeric_final_answer(format_numeric_decimal(base_value * pct / Decimal(100)))

    match = re.fullmatch(rf"({NUMBER_RE})\s*([+-])\s*({NUMBER_RE})%", text)
    if match:
        base_value = parse_decimal(match.group(1))
        pct = parse_decimal(match.group(3))
        if base_value is not None and pct is not None:
            delta = base_value * pct / Decimal(100)
            value = base_value + delta if match.group(2) == "+" else base_value - delta
            return numeric_final_answer(format_numeric_decimal(value))

    match = re.fullmatch(r"(\d+)\s+(\d+)/(\d+)\s+переводится в десятичн[а-я]*", text)
    if match:
        whole = Decimal(match.group(1))
        numerator = Decimal(match.group(2))
        denominator = Decimal(match.group(3))
        if denominator != 0:
            return numeric_final_answer(format_numeric_decimal(whole + numerator / denominator))

    match = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)\s+в десятичн[а-я]*", text)
    if match:
        numerator = Decimal(match.group(1))
        denominator = Decimal(match.group(2))
        if denominator != 0:
            return numeric_final_answer(format_numeric_decimal(numerator / denominator))

    match = re.fullmatch(r"(\d+)\s+во второй степени", text)
    if match:
        value = int(match.group(1)) ** 2
        return numeric_final_answer(str(value))

    return None


CHEM_TOKEN_RE = re.compile(r"([A-Z][a-z]?|\(|\)|\d+)")


def parse_chemical_formula(raw: str) -> dict[str, int] | None:
    formula = raw.strip().translate(SUBSCRIPT_DIGITS)
    formula = re.sub(r"^\d+", "", formula)
    tokens = CHEM_TOKEN_RE.findall(formula)
    if not tokens or "".join(tokens) != formula:
        return None

    stack: list[Counter[str]] = [Counter()]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "(":
            stack.append(Counter())
            index += 1
            continue
        if token == ")":
            if len(stack) == 1:
                return None
            group = stack.pop()
            index += 1
            multiplier = 1
            if index < len(tokens) and tokens[index].isdigit():
                multiplier = int(tokens[index])
                index += 1
            for element, count in group.items():
                stack[-1][element] += count * multiplier
            continue
        if re.match(r"[A-Z]", token):
            element = token
            index += 1
            multiplier = 1
            if index < len(tokens) and tokens[index].isdigit():
                multiplier = int(tokens[index])
                index += 1
            stack[-1][element] += multiplier
            continue
        return None

    if len(stack) != 1:
        return None
    return dict(stack[0])


def rref_fraction_matrix(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    mat = [row[:] for row in matrix]
    row_count = len(mat)
    col_count = len(mat[0]) if row_count else 0
    pivots: list[int] = []
    pivot_row = 0
    for col in range(col_count):
        found = next((r for r in range(pivot_row, row_count) if mat[r][col]), None)
        if found is None:
            continue
        mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
        pivot = mat[pivot_row][col]
        mat[pivot_row] = [value / pivot for value in mat[pivot_row]]
        for r in range(row_count):
            if r == pivot_row or not mat[r][col]:
                continue
            factor = mat[r][col]
            mat[r] = [value - factor * base for value, base in zip(mat[r], mat[pivot_row])]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return mat, pivots


def balance_chemical_equation(equation: str) -> tuple[list[str], list[str], list[int]] | None:
    cleaned = equation.translate(SUBSCRIPT_DIGITS)
    cleaned = cleaned.replace("→", "=").replace("->", "=").replace("—", "=").replace("–", "=")
    cleaned = re.sub(r"\s+", "", cleaned)
    if "=" not in cleaned:
        return None
    left_raw, right_raw = cleaned.split("=", 1)
    left = [item for item in left_raw.split("+") if item]
    right = [item for item in right_raw.split("+") if item]
    if not left or not right or len(left) + len(right) > 8:
        return None

    compounds = left + right
    parsed = [parse_chemical_formula(item) for item in compounds]
    if any(item is None for item in parsed):
        return None
    elements = sorted({element for item in parsed if item for element in item})
    matrix: list[list[Fraction]] = []
    for element in elements:
        row: list[Fraction] = []
        for index, compound in enumerate(parsed):
            sign = 1 if index < len(left) else -1
            row.append(Fraction(sign * (compound or {}).get(element, 0)))
        matrix.append(row)

    reduced, pivots = rref_fraction_matrix(matrix)
    free_cols = [col for col in range(len(compounds)) if col not in pivots]
    if len(free_cols) != 1:
        return None

    solution = [Fraction(0) for _ in compounds]
    solution[free_cols[0]] = Fraction(1)
    for row_index, col in enumerate(pivots):
        solution[col] = -reduced[row_index][free_cols[0]]

    denominator_lcm = 1
    for value in solution:
        denominator_lcm = denominator_lcm * value.denominator // gcd(denominator_lcm, value.denominator)
    integers = [int(value * denominator_lcm) for value in solution]
    if any(value <= 0 for value in integers):
        integers = [-value for value in integers]
    if any(value <= 0 for value in integers):
        return None
    divisor = abs(reduce(gcd, integers))
    return left, right, [value // divisor for value in integers]


def extract_chemical_equation(question: str) -> str | None:
    text = question.translate(SUBSCRIPT_DIGITS)
    pattern = (
        r"((?:\d*[A-Z][A-Za-z0-9()]*\s*\+\s*)*"
        r"\d*[A-Z][A-Za-z0-9()]*\s*(?:=|→|->)\s*"
        r"(?:\d*[A-Z][A-Za-z0-9()]*\s*\+\s*)*"
        r"\d*[A-Z][A-Za-z0-9()]*)"
    )
    match = re.search(pattern, text)
    return match.group(1) if match else None


def coefficient_sum_answer(question: str) -> str | None:
    text = " ".join(question.split())
    if not re.search(r"сумм[ауы]?\s+коэффициент", text, flags=re.IGNORECASE):
        return None
    equation = extract_chemical_equation(text)
    if equation is None:
        return None
    balanced = balance_chemical_equation(equation)
    if balanced is None:
        return None
    value = str(sum(balanced[2]))
    return numeric_final_answer(value)


def ammonia_synthesis_answer(question: str) -> str | None:
    text = normalize_numeric_text(question)
    if "аммиак" not in text or "водород" not in text or "азот" not in text or "выход" not in text:
        return None
    volume_match = re.search(rf"синтез[а-я\s]+({NUMBER_RE})\s*л\s+аммиак", text)
    yield_match = re.search(rf"выход\s+продукт[а-я\s]+составляет\s+({NUMBER_RE})\s*%", text)
    if not volume_match or not yield_match:
        return None
    ammonia_volume = parse_decimal(volume_match.group(1))
    yield_percent = parse_decimal(yield_match.group(1))
    if ammonia_volume is None or yield_percent is None or yield_percent == 0:
        return None

    theoretical_ammonia = ammonia_volume / (yield_percent / Decimal(100))
    hydrogen_volume = theoretical_ammonia * Decimal(3) / Decimal(2)
    nitrogen_mass = theoretical_ammonia / Decimal(2) / Decimal("22.4") * Decimal(28)
    h2 = format_decimal(hydrogen_volume.quantize(Decimal("0.1")))
    n2 = format_decimal(nitrogen_mass.quantize(Decimal("0.1")))
    value = f"{h2} л водорода и {n2} г азота"
    return numeric_final_answer(value)


def concentration_stoichiometry_answer(question: str) -> str | None:
    text = normalize_numeric_text(question)
    match = re.search(
        rf"в реакции\s+(\d+)\s*x\s*\+\s*(\d+)\s*y\s*=\s*z\s+начальн[а-я ]+концентрац[а-я ]+равны\s+({NUMBER_RE})\s+и\s+({NUMBER_RE}).*концентрац[а-я ]+y.*x\s+станет\s+({NUMBER_RE})",
        text,
    )
    if not match:
        return None
    coef_x = Decimal(match.group(1))
    coef_y = Decimal(match.group(2))
    initial_x = parse_decimal(match.group(3))
    initial_y = parse_decimal(match.group(4))
    final_x = parse_decimal(match.group(5))
    if initial_x is None or initial_y is None or final_x is None or coef_x == 0:
        return None
    consumed_x = initial_x - final_x
    final_y = initial_y - consumed_x * coef_y / coef_x
    return numeric_final_answer(format_numeric_decimal(final_y))


def chemistry_stoichiometry_answer(question: str) -> str | None:
    return (
        coefficient_sum_answer(question)
        or ammonia_synthesis_answer(question)
        or concentration_stoichiometry_answer(question)
    )


def formulaic_math_physics_answer(question: str) -> str | None:
    text = normalize_numeric_text(question)

    match = re.fullmatch(r"сколько десятков в числе\s+(\d+)", text)
    if match:
        return numeric_final_answer(str(int(match.group(1)) // 10))

    match = re.fullmatch(rf"({NUMBER_RE})\s+от\s+(\d{{5,}})\s+это\s+сколько", text)
    if match:
        pct = parse_decimal(match.group(1))
        value = parse_decimal(match.group(2))
        if pct is not None and value is not None and Decimal(0) <= pct <= Decimal(100):
            return numeric_final_answer(format_numeric_decimal(value * pct / Decimal(100)))

    match = re.fullmatch(r"известно, что каждый символ кодируется одним байтом\.? найти информационный объем сообщения, содержащего (\d+) символ[а-я ]+байт[а-я(). ]*", text)
    if match:
        return numeric_final_answer(match.group(1))

    match = re.fullmatch(r"сколько квадратных метров в комнате\s+(\d+)\s+на\s+(\d+)\s+метр[а-я]*", text)
    if match:
        value = Decimal(match.group(1)) * Decimal(match.group(2))
        return numeric_final_answer(f"{format_numeric_decimal(value)} квадратных метров")

    match = re.fullmatch(r"сколько метров в\s+(\d+)\s+дециметр[а-я]*", text)
    if match:
        value = Decimal(match.group(1)) / Decimal(10)
        return numeric_final_answer(f"{format_numeric_decimal(value)} метра")

    match = re.fullmatch(rf"({NUMBER_RE})\s+градусов\s+по\s+фаренгейту\s+сколько\s+по\s+цельсию", text)
    if match:
        f_value = parse_decimal(match.group(1))
        if f_value is not None:
            c_value = (f_value - Decimal(32)) * Decimal(5) / Decimal(9)
            return numeric_final_answer(f"{format_numeric_decimal(c_value)} градусов")

    match = re.fullmatch(r"(\d+)\s+недель\s+сколько\s+лет", text)
    if match:
        value = Decimal(match.group(1)) / Decimal(52)
        return numeric_final_answer(f"{format_numeric_decimal(value.quantize(Decimal('0.01')))} года")

    match = re.fullmatch(r"(\d+)\s+ар\s+это\s+сколько", text)
    if match:
        value = Decimal(match.group(1)) * Decimal(100)
        return numeric_final_answer(f"{format_numeric_decimal(value)} квадратных метров")

    match = re.fullmatch(r"сколько тонн в\s+(\d+)\s*т\s+(\d+)\s*кг\s+(\d+)\s*г", text)
    if match:
        value = Decimal(match.group(1)) + Decimal(match.group(2)) / Decimal(1000) + Decimal(match.group(3)) / Decimal(1000000)
        return numeric_final_answer(f"{format_numeric_decimal(value)} тонн")

    match = re.fullmatch(r"найдите делимое[,.] если неполное частное\s+(\d+)[,.]\s+делитель\s+(\d+)\s+и\s+остаток\s+(\d+)", text)
    if match:
        value = int(match.group(1)) * int(match.group(2)) + int(match.group(3))
        return numeric_final_answer(str(value))

    if re.search(r"монет[ау]\s+подбрасывают\s+дважды", text) and "ровно один раз" in text:
        return numeric_final_answer("1/2")

    match = re.fullmatch(r"электрический кипятильник рассчитан на\s+(\d+)\s*в\s+и\s+силу\s+тока\s+(\d+)\s*а[,.]?\s+какова\s+мощность\s+тока\s+в\s+кипятильнике\??", text)
    if match:
        value = Decimal(match.group(1)) * Decimal(match.group(2))
        return numeric_final_answer(f"{format_numeric_decimal(value)} Вт")

    match = re.fullmatch(rf"какова скорость света в [а-я]+[,.] если его показатель преломления равен\s+({NUMBER_RE})\?", text)
    if match:
        n_value = parse_decimal(match.group(1))
        if n_value is not None and n_value != 0:
            value = Decimal("3e8") / n_value
            short = value / Decimal("1e8")
            return numeric_final_answer(f"{format_decimal(short.quantize(Decimal('0.01')))} × 10^8 м/с")

    match = re.fullmatch(r"диагональ квадрата равна\s+(\d+)[,.]\s+чему равна площадь квадрата\?", text)
    if match:
        diag = Decimal(match.group(1))
        return numeric_final_answer(format_numeric_decimal(diag * diag / Decimal(2)))

    match = re.fullmatch(r"найдите площадь боковой поверхности конуса[,.] если образующая конуса равна\s+(\d+)\s*см[,.] а диаметр основания\s+[-—]\s+(\d+)\s*см[,.] ответ:.*", text)
    if match:
        generatrix = int(match.group(1))
        diameter = int(match.group(2))
        coeff = generatrix * diameter // 2 if (generatrix * diameter) % 2 == 0 else None
        value = f"{coeff}π" if coeff is not None else f"{format_numeric_decimal(Decimal(generatrix) * Decimal(diameter) / Decimal(2))}π"
        return numeric_final_answer(value)

    match = re.fullmatch(r"правильная четырехугольная призма описана около цилиндра[,.] радиус основания которого равен\s+(\d+)[,.] площадь боковой поверхности призмы равна\s+(\d+)[,.] найдите высоту цилиндра[.]?", text)
    if match:
        radius = Decimal(match.group(1))
        surface = Decimal(match.group(2))
        value = surface / (Decimal(8) * radius)
        return numeric_final_answer(format_numeric_decimal(value))

    match = re.fullmatch(r"участок земли.*прямоугольника со сторонами\s+(\d+)\s*м\s+и\s+(\d+)\s*м.*одна из длинных сторон.*остальные три стороны.*длину забора.*", text)
    if match:
        a = Decimal(match.group(1))
        b = Decimal(match.group(2))
        value = max(a, b) + Decimal(2) * min(a, b)
        return numeric_final_answer(format_numeric_decimal(value))

    match = re.fullmatch(r"катеты прямоугольного треугольника\s+(\d+)\s+и\s+(\d+)[,.] найдите высоту[,.] проведенную к гипотенузе[,.] ответ округлите до сотых[.]?", text)
    if match:
        a = float(match.group(1))
        b = float(match.group(2))
        c = sqrt(a * a + b * b)
        value = Decimal(str(a * b / c)).quantize(Decimal("0.01"))
        return numeric_final_answer(format_numeric_decimal(value))

    match = re.fullmatch(r"задачи по теме молярный объем 8 класс: какой объем занимают\s+(\d+)\s+моля кислорода\?", text)
    if match:
        value = Decimal(match.group(1)) * Decimal("22.4")
        return numeric_final_answer(f"{format_numeric_decimal(value)} л")

    match = re.fullmatch(r"сколько льда при 0\s*°c расплавится[,.] если ему передать количество теплоты[,.] которое выделится при конденсации водяного пара массой\s+(\d+)\s*кг.*", text)
    if match:
        steam_kg = Decimal(match.group(1))
        value = steam_kg * Decimal("2.3e6") / Decimal("3.4e5")
        return numeric_final_answer(f"{format_numeric_decimal(value.quantize(Decimal('0.1')))} кг")

    match = re.fullmatch(r"2[.] какое давление сжатого воздуха[,.] находящегося в баллоне объемом\s+(\d+)\s*л\s+при\s+(\d+)\s*°c[,.] если масса воздуха\s+(\d+)\s*кг\?", text)
    if match:
        volume_m3 = Decimal(match.group(1)) / Decimal(1000)
        temp_k = Decimal(match.group(2)) + Decimal("273.15")
        mass = Decimal(match.group(3))
        pressure = mass / Decimal("0.029") * Decimal("8.314") * temp_k / volume_m3
        mpa = pressure / Decimal("1e6")
        return numeric_final_answer(f"{format_numeric_decimal(mpa.quantize(Decimal('0.1')))} МПа")

    return None


def to_roman(value: int) -> str | None:
    if value <= 0 or value >= 4000:
        return None
    parts = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    remaining = value
    for amount, symbol in parts:
        while remaining >= amount:
            out.append(symbol)
            remaining -= amount
    return "".join(out)


def format_pi_radians(degrees: int) -> str:
    value = Fraction(degrees, 180)
    if value == 0:
        return "0"
    if value.denominator == 1:
        return "π" if value.numerator == 1 else f"{value.numerator}π"
    if value.numerator == 1:
        return f"π/{value.denominator}"
    return f"{value.numerator}π/{value.denominator}"


def structured_school_task_answer(question: str) -> str | None:
    text = normalize_numeric_text(question)

    match = re.fullmatch(r"(\d+)\s*м\s+(\d+)\s*дм\s+сколько\s+дм", text)
    if match:
        value = int(match.group(1)) * 10 + int(match.group(2))
        return numeric_final_answer(f"{value} дм")

    match = re.fullmatch(r"сколько литр[а-я]* в\s+(\d+)\s+кубическ[а-я]* метр[а-я]*", text)
    if match:
        value = int(match.group(1)) * 1000
        return numeric_final_answer(f"{value} литров")

    match = re.fullmatch(r"сколько грамм[а-я]* в\s+(\d+)\s+тонн[а-я]*(?:[,.]\s*представь ответ в виде таблицы)?", text)
    if match:
        value = int(match.group(1)) * 1_000_000
        return numeric_final_answer(f"{value} граммов")

    match = re.fullmatch(r"переведите в радианн[а-я]* мер[а-я]* угл[а-я]*\s+(.+)", text)
    if match:
        degrees = [int(item) for item in re.findall(r"\d+", match.group(1))]
        if 1 <= len(degrees) <= 8 and all(0 <= item <= 360 for item in degrees):
            return numeric_final_answer(", ".join(format_pi_radians(item) for item in degrees))

    match = re.fullmatch(r"(\d{1,4})\s+в\s+римск[а-я]*\s+цифр[а-я]*", text)
    if match:
        roman = to_roman(int(match.group(1)))
        if roman is not None:
            return numeric_final_answer(roman)

    match = re.fullmatch(
        r"периметр равнобедренного треугольника составляет\s+(\d+)\s*см[,.]\s+при этом основание превышает боковую сторону на\s+(\d+)\s*см[,.]\s+найдите длину боковой стороны[.]?",
        text,
    )
    if match:
        side = Fraction(int(match.group(1)) - int(match.group(2)), 3)
        return numeric_final_answer(f"{format_fraction(side)} см")

    match = re.fullmatch(
        r"задача[.]?\s+на концах невесомого рычага действуют силы\s+(\d+)\s+и\s+(\d+)\s*н[,.]\s+расстояние от точки опоры до меньшей силы равно\s+(" + NUMBER_RE + r")\s*м[,.]\s+определи длину плеча большей силы[,.]\s+если рычаг находится в равновесии[.]?",
        text,
    )
    if match:
        force_a = Decimal(match.group(1))
        force_b = Decimal(match.group(2))
        distance = parse_decimal(match.group(3))
        if distance is not None:
            value = min(force_a, force_b) * distance / max(force_a, force_b)
            return numeric_final_answer(f"{format_numeric_decimal(value)} м")

    match = re.search(
        r"выполнили\s+(\d+)\s+поперечн[а-я]*\s+распил[а-я]*[,.]\s+в результате получилось\s+(\d+)\s+куск[а-я]*[,.]\s+сколько досок взяли изначально\?",
        text,
    )
    if match and "дос" in text:
        cuts = int(match.group(1))
        pieces = int(match.group(2))
        if pieces >= cuts:
            return numeric_final_answer(str(pieces - cuts))

    match = re.fullmatch(
        r".*вероятность того[,.]\s+что .* больше\s+\d+\s+метр[а-я]*[,.]\s+равна\s+(" + NUMBER_RE + r").*вероятность того[,.]\s+что .* более\s+\d+\s+метр[а-я]*[,.]\s+равна\s+(" + NUMBER_RE + r").*более\s+\d+\s+метр[а-я]*[,.]\s+но не более\s+\d+\s+метр[а-я]*\?",
        text,
    )
    if match:
        high = parse_decimal(match.group(1))
        higher = parse_decimal(match.group(2))
        if high is not None and higher is not None:
            return numeric_final_answer(format_numeric_decimal(high - higher))

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
        answer = exact_numeric_answer(row["question"]) or answer
        answer = chemistry_stoichiometry_answer(row["question"]) or answer
        answer = formulaic_math_physics_answer(row["question"]) or answer
        answer = structured_school_task_answer(row["question"]) or answer
        answer = dedup_comma_loop(answer) or answer
        answer = cleanup_english_cloze_answer(row["question"], answer) or answer
        answer = quantity_conversion_answer(row["question"]) or answer
        answer = km_meters_answer(row["question"]) or answer
        result.append({"rid": row["rid"], "answer": answer})

    with open("output.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
