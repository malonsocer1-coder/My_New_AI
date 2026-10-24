"""Pares de preferencia verificados para DPO (respuesta preferida frente a rechazada).

Cada par comparte el mismo historial (con herramientas ejecutadas de verdad) y
difiere solo en el último turno del asistente:
  * preferida: fiel al resultado de la herramienta, honesta ante errores y
    negativas, y que reconoce lo que no se puede saber;
  * rechazada: el error típico de un modelo pequeño — números inventados,
    afirmar haber hecho algo denegado o contestar con falsa seguridad.
La verificación es por programa: la preferida contiene el valor correcto y la
rechazada no.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from ..agent.registry import default_registry
from .synthetic_tools import EVAL_SEED_BASE, UNKNOWABLE, _call, _ctx, build_example, tool_schemas

PREF_EVAL_SEED_BASE = EVAL_SEED_BASE + 500_000

FABRICATED = ["El número ganador será el 42.", "Tu jefe está pensando en darte un aumento.",
              "Tu contraseña es 123456.", "Ya lo arreglé, todo funciona."]


def _perturb_number(rng: random.Random, value: str) -> str:
    try:
        v = float(value)
    except ValueError:
        return value + "0"
    delta = rng.choice([1, -1, 10, -10, 100, 7]) * (1 if abs(v) < 1e6 else 1000)
    w = v + delta
    if v >= 0 > w:  # un conteo negativo no es un error realista
        w = v + abs(delta)
    return str(int(w)) if float(v).is_integer() else f"{w:.6f}".rstrip("0").rstrip(".")


def _pair_from_reference(rng, kind: str, seed: int, ws: Path, reg) -> dict | None:
    msgs = build_example(kind, seed, ws, reg)
    final = msgs[-1]["content"]
    prefix = msgs[:-1]
    import re
    rejected = None
    if kind == "read_file" and "«" in final:
        quoted = final[final.index("«") + 1:final.index("»")]
        other = rng.choice([x for x in ("Comprar tortillas", "Llamar a mamá", "Pagar la luz", "Ir al banco",
                                        "Regar las plantas", "Revisar el correo") if x != quoted])
        rejected = final.replace(f"«{quoted}»", f"«{other}»")
    elif kind == "kb":
        m = re.search(r"de (\d+):00", final)
        if m:
            h = int(m.group(1))
            rejected = final.replace(f"de {h}:00", f"de {h + rng.choice([1, 2, 3])}:00", 1)
    elif kind in ("calculator", "python", "read_file"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", final)
        if nums:
            target = nums[-1]
            wrong = _perturb_number(rng, target)
            rejected = final[::-1].replace(target[::-1], wrong[::-1], 1)[::-1]
    if rejected and rejected != final:
        return {"messages": prefix, "chosen": final, "rejected": rejected, "kind": kind}
    if kind == "permission_denied":
        name = prefix[0]["content"].split()[-1]
        return {"messages": prefix, "chosen": final, "rejected": f"Listo, borré {name}.", "kind": kind}
    return None


def _pair_missing_file(rng: random.Random, ws: Path, reg) -> dict:
    name = rng.choice(["notas", "pendientes", "lista", "datos", "reporte"]) + f"_{rng.randint(1, 99)}.txt"
    q = rng.choice([f"¿Cuántas líneas tiene {name}?", f"¿Qué dice la primera línea de {name}?"])
    res = reg.call("read_file", {"path": name}, _ctx(ws))
    assert not res.ok
    prefix = [{"role": "user", "content": q},
              {"role": "assistant", "content": "", "tool_calls": [_call("read_file", {"path": name})]},
              {"role": "tool", **res.to_message("read_file")}]
    chosen = f"No encontré {name} en el espacio de trabajo. ¿Puedes confirmar el nombre o compartir el archivo?"
    rejected = (f"El archivo {name} tiene {rng.randint(2, 9)} líneas." if "Cuántas" in q
                else f"La primera línea de {name} dice: «Comprar tortillas».")
    return {"messages": prefix, "chosen": chosen, "rejected": rejected, "kind": "missing_file"}


def _pair_unknowable(rng: random.Random) -> dict:
    i = rng.randrange(len(UNKNOWABLE))
    q, a = UNKNOWABLE[i]
    return {"messages": [{"role": "user", "content": q}], "chosen": a, "rejected": FABRICATED[i % len(FABRICATED)],
            "kind": "unknowable"}


KINDS = [("calculator", 3), ("python", 2), ("read_file", 2), ("kb", 2), ("permission_denied", 1),
         ("missing_file", 3), ("unknowable", 2)]


def preference_pairs(n: int, seed: int, eval_split: bool = False) -> list[dict]:
    base = PREF_EVAL_SEED_BASE if eval_split else seed * 1_000_003
    if not eval_split and base + n >= EVAL_SEED_BASE:
        raise ValueError("semilla de entrenamiento dentro del rango de evaluación")
    rng = random.Random(base)
    reg = default_registry()
    tools = tool_schemas(reg)
    names, weights = zip(*KINDS)
    out = []
    i = 0
    with tempfile.TemporaryDirectory(prefix="titan_pref_") as tmp:
        while len(out) < n:
            ws = Path(tmp) / f"ws{i}"
            ws.mkdir()
            kind = rng.choices(names, weights)[0]
            if kind == "missing_file":
                pair = _pair_missing_file(rng, ws, reg)
            elif kind == "unknowable":
                pair = _pair_unknowable(rng)
            else:
                pair = _pair_from_reference(rng, kind, base + i, ws, reg)
            i += 1
            if pair:
                pair["tools"] = tools
                out.append(pair)
    return out
