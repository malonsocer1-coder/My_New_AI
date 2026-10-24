"""Generadores de datos sintéticos con respuestas verificadas por programa.

Todo ejemplo generado aquí se comprueba con un verificador independiente
(Python evalúa la operación, la herramienta se ejecuta de verdad, etc.)
antes de emitirse. Los conjuntos de evaluación se generan con semillas
distintas y se excluyen explícitamente de los de entrenamiento.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

NAMES = ["María", "José", "Lupita", "Carlos", "Ana", "Diego", "Sofía", "Juan", "Valeria", "Miguel",
         "Fernanda", "Luis", "Ximena", "Jorge", "Camila", "Ricardo"]
OBJECTS = [("manzanas", "manzana"), ("tacos", "taco"), ("libros", "libro"), ("canicas", "canica"),
           ("pesos", "peso"), ("lápices", "lápiz"), ("naranjas", "naranja"), ("boletos", "boleto")]


def _column_addition_steps(a: int, b: int) -> str:
    """Explicación columna a columna (algoritmo verificable que el modelo puede imitar)."""
    da, db = str(a)[::-1], str(b)[::-1]
    carry, digits, steps = 0, [], []
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        s = x + y + carry
        steps.append(f"{x}+{y}" + (f"+{carry}" if carry else "") + f"={s}")
        digits.append(str(s % 10))
        carry = s // 10
    if carry:
        digits.append(str(carry))
    result = "".join(reversed(digits))
    assert int(result) == a + b
    return "; ".join(steps) + f" → {result}"


def make_arith_problem(rng: random.Random, max_digits: int = 3, lang: str = "es") -> dict:
    op = rng.choice(["+", "-", "*", "+", "-"])
    nd = rng.randint(1, max_digits)
    a = rng.randint(0, 10 ** nd - 1)
    b = rng.randint(0, 10 ** rng.randint(1, max_digits) - 1)
    if op == "*":
        b = rng.randint(0, 99 if max_digits > 2 else 9)
    if op == "-" and b > a:
        a, b = b, a
    answer = {"+": a + b, "-": a - b, "*": a * b}[op]
    assert eval(f"{a}{op}{b}") == answer  # verificador independiente
    style = rng.random()
    if lang == "es":
        if style < 0.4 or op == "*" or min(a, b) < 2:
            q = f"¿Cuánto es {a} {op} {b}?"
        elif op == "+":
            n = rng.choice(NAMES)
            pl, _ = rng.choice(OBJECTS)
            q = f"{n} tiene {a} {pl} y le dan {b} más. ¿Cuántos {pl} tiene en total?"
        else:
            n = rng.choice(NAMES)
            pl, _ = rng.choice(OBJECTS)
            q = f"{n} tenía {a} {pl} y regaló {b}. ¿Cuántos {pl} le quedan?"
    else:
        q = f"What is {a} {op} {b}?"
    return {"a": a, "b": b, "op": op, "question": q, "answer": str(answer)}


def _key(p: dict) -> tuple:
    return (p["a"], p["op"], p["b"])


def load_exclusions(path: str | None) -> set[tuple]:
    if not path or not Path(path).exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "a" in row:
                out.add((row["a"], row["op"], row["b"]))
    return out


def arithmetic_documents(n_docs: int = 2000, problems_per_doc: int = 12, seed: int = 0,
                         max_digits: int = 3, exclude_file: str | None = None,
                         with_steps_prob: float = 0.5) -> Iterator[str]:
    """Documentos de práctica aritmética para el preentrenamiento (dominio 'math')."""
    rng = random.Random(seed)
    excl = load_exclusions(exclude_file)
    for _ in range(n_docs):
        lines = []
        while len(lines) < problems_per_doc:
            p = make_arith_problem(rng, max_digits, lang="es" if rng.random() < 0.8 else "en")
            if _key(p) in excl:
                continue
            if p["op"] == "+" and rng.random() < with_steps_prob:
                lines.append(f"Pregunta: {p['question']}\nRazonamiento: {_column_addition_steps(p['a'], p['b'])}"
                             f"\nRespuesta: {p['answer']}")
            else:
                lines.append(f"Pregunta: {p['question']}\nRespuesta: {p['answer']}")
        yield "\n\n".join(lines)


def arithmetic_eval_items(n: int = 300, seed: int = 999_001, max_digits: int = 3) -> list[dict]:
    rng = random.Random(seed)
    seen, items = set(), []
    while len(items) < n:
        p = make_arith_problem(rng, max_digits, "es")
        if _key(p) in seen:
            continue
        seen.add(_key(p))
        items.append({"id": f"arith-{len(items)}", **p, "prompt": f"Pregunta: {p['question']}\nRespuesta:"})
    return items
