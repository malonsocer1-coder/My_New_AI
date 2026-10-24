"""Conversaciones sintéticas de uso de herramientas, con ejecución REAL de las herramientas.

Cada ejemplo se construye ejecutando las herramientas de verdad en un espacio
de trabajo temporal; la respuesta final se deriva del resultado obtenido y se
verifica por programa. Se cubren: selección de herramienta, llamadas
estructuradas, interpretación de resultados, recuperación de errores, no usar
herramientas cuando no hacen falta y reconocer incertidumbre.

Las semillas de entrenamiento y de evaluación son disjuntas.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from ..agent.permissions import DEFAULT_POLICY, PermissionPolicy
from ..agent.protocol import ToolContext
from ..agent.registry import default_registry
from ..agent.tools.builtin import safe_eval

TOOLSETS = {
    "basic": ["calculator", "read_file", "list_dir", "write_file", "python_exec", "kb_search"],
}

TOPICS = ["la biblioteca", "el gimnasio", "la cafetería", "la clínica", "el museo", "la oficina de correos"]
DAYS = ["lunes a viernes", "martes a sábado", "lunes a sábado"]
GREETINGS = [("Hola, ¿cómo estás?", "¡Hola! Estoy bien, gracias. ¿En qué te puedo ayudar hoy?"),
             ("Buenos días", "¡Buenos días! ¿Qué necesitas?"),
             ("Gracias por tu ayuda", "¡Con gusto! Si necesitas algo más, aquí estoy."),
             ("¿Quién eres?", "Soy TITAN, un asistente entrenado desde cero por el proyecto TITAN AI. "
                              "Puedo conversar, hacer cálculos y usar herramientas cuando me das permiso.")]
UNKNOWABLE = [("¿Qué número saldrá en la lotería de mañana?",
               "No lo sé y nadie puede saberlo: el resultado de un sorteo es aleatorio."),
              ("¿Qué está pensando mi jefe ahora mismo?",
               "No tengo forma de saberlo. Si quieres, te ayudo a preparar cómo preguntárselo."),
              ("¿Cuál es mi contraseña del banco?",
               "No conozco tu contraseña y no debo intentar averiguarla. Si la olvidaste, usa la opción "
               "oficial de recuperación de tu banco."),
              ("Arréglalo.", "¿Podrías darme más detalles? No sé a qué te refieres: dime qué quieres que arregle "
                             "y comparte el texto, archivo o error.")]


def _ctx(ws: Path) -> ToolContext:
    pol = PermissionPolicy(rules={k: "allow" for k in DEFAULT_POLICY}, approver=lambda r: "allow_once")
    return ToolContext(workspace=str(ws), permissions=pol, settings={"sandbox_backend": "auto"})


def _call(name: str, args: dict) -> dict:
    return {"name": name, "arguments": args}


def _arith_expr(rng: random.Random) -> str:
    a, b, c = rng.randint(100, 9999), rng.randint(10, 999), rng.randint(2, 99)
    return rng.choice([f"{a} * {b}", f"{a} * {b} + {c}", f"({a} + {b}) * {c}", f"{a} / {c}", f"{a} - {b} * {c}",
                       f"sqrt({c * c})", f"{a} ** 2"])


def _ex_calculator(rng, reg, ctx, ws):
    expr = _arith_expr(rng)
    q = rng.choice([f"¿Cuánto es {expr}?", f"Calcula {expr}, por favor.", f"Necesito el resultado exacto de {expr}."])
    res = reg.call("calculator", {"expression": expr}, ctx)
    assert res.ok and res.content["result"] == safe_eval(expr)
    val = res.content["result"]
    if isinstance(val, float):
        val = round(val, 6)
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": "Lo calculo con la calculadora.", "tool_calls": [_call("calculator", {"expression": expr})]},
            {"role": "tool", **res.to_message("calculator")},
            {"role": "assistant", "content": f"{expr} = {val}"}]


def _ex_read_file(rng, reg, ctx, ws):
    name = rng.choice(["notas", "pendientes", "lista", "datos", "recordatorio"]) + f"_{rng.randint(1, 99)}.txt"
    lines = [rng.choice(["Comprar tortillas", "Llamar a mamá", "Pagar la luz", "Revisar el correo",
                         "Estudiar matemáticas", "Ir al banco", "Regar las plantas"]) for _ in range(rng.randint(2, 5))]
    (ws / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    ask = rng.choice(["first", "count"])
    q = f"¿Qué dice la primera línea de {name}?" if ask == "first" else f"¿Cuántas líneas tiene {name}?"
    res = reg.call("read_file", {"path": name}, ctx)
    assert res.ok
    content_lines = [ln for ln in res.content.split("\n") if ln.strip()]
    ans = (f"La primera línea de {name} dice: «{content_lines[0]}»." if ask == "first"
           else f"El archivo {name} tiene {len(content_lines)} líneas.")
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": "", "tool_calls": [_call("read_file", {"path": name})]},
            {"role": "tool", **res.to_message("read_file")},
            {"role": "assistant", "content": ans}]


def _ex_error_recovery(rng, reg, ctx, ws):
    real = f"informe_{rng.randint(100, 999)}.txt"
    wrong = real.replace("informe", "reporte")
    (ws / real).write_text(f"Ventas del mes: {rng.randint(1000, 9000)} pesos\n", encoding="utf-8")
    q = f"Lee el archivo {wrong} y dime las ventas."
    r1 = reg.call("read_file", {"path": wrong}, ctx)
    assert not r1.ok
    r2 = reg.call("list_dir", {}, ctx)
    assert r2.ok and any(i["name"] == real for i in r2.content)
    r3 = reg.call("read_file", {"path": real}, ctx)
    assert r3.ok
    ventas = r3.content.split(":")[1].strip()
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": "", "tool_calls": [_call("read_file", {"path": wrong})]},
            {"role": "tool", **r1.to_message("read_file")},
            {"role": "assistant", "content": "Ese archivo no existe. Reviso qué archivos hay.",
             "tool_calls": [_call("list_dir", {})]},
            {"role": "tool", **r2.to_message("list_dir")},
            {"role": "assistant", "content": f"Encontré {real}, que parece ser el correcto.",
             "tool_calls": [_call("read_file", {"path": real})]},
            {"role": "tool", **r3.to_message("read_file")},
            {"role": "assistant", "content": f"No existía {wrong}, pero en {real} las ventas del mes son {ventas}."}]


def _ex_python(rng, reg, ctx, ws):
    n = rng.randint(10, 200)
    kind = rng.choice(["squares", "primes", "fib"])
    if kind == "squares":
        q, code = f"Calcula la suma de los cuadrados del 1 al {n} ejecutando código.", f"print(sum(i * i for i in range(1, {n + 1})))"
    elif kind == "primes":
        q = f"¿Cuántos números primos hay menores que {n}? Usa Python."
        code = (f"n = {n}\nprimos = [k for k in range(2, n) if all(k % d for d in range(2, int(k ** 0.5) + 1))]\n"
                "print(len(primos))")
    else:
        q = f"Dame el término {n % 40 + 5} de la sucesión de Fibonacci (F(0)=0) calculándolo con código."
        code = f"a, b = 0, 1\nfor _ in range({n % 40 + 5}):\n    a, b = b, a + b\nprint(a)"
    res = reg.call("python_exec", {"code": code}, ctx)
    assert res.ok, res.error
    out = res.content["stdout"].strip()
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": "Escribo y ejecuto un programa corto.", "tool_calls": [_call("python_exec", {"code": code})]},
            {"role": "tool", **res.to_message("python_exec")},
            {"role": "assistant", "content": f"El resultado es {out}."}]


def _ex_kb(rng, reg, ctx, ws):
    kb = ws / "knowledge"
    kb.mkdir(exist_ok=True)
    topic = rng.choice(TOPICS)
    h1, h2 = rng.randint(6, 10), rng.randint(15, 21)
    days = rng.choice(DAYS)
    fname = topic.split()[-1] + ".md"
    (kb / fname).write_text(f"# {topic.capitalize()}\n\n{topic.capitalize()} abre de {h1}:00 a {h2}:00 horas, de {days}.\n\n"
                            f"Para más información llame a recepción.", encoding="utf-8")
    q = rng.choice([f"¿A qué hora abre {topic}?", f"¿Cuál es el horario de {topic}?"])
    res = reg.call("kb_search", {"query": q}, ctx)
    assert res.ok and res.content and str(h1) in res.content[0]["text"]
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": "Lo busco en la base de conocimiento.", "tool_calls": [_call("kb_search", {"query": q})]},
            {"role": "tool", **res.to_message("kb_search")},
            {"role": "assistant", "content": f"Según {res.content[0]['source']}, {topic} abre de {h1}:00 a {h2}:00 horas, de {days}."}]


def _ex_write(rng, reg, ctx, ws):
    name = rng.choice(["saludo", "mensaje", "nota", "aviso"]) + f"_{rng.randint(1, 999)}.txt"
    text = rng.choice(["Hola a todos", "Reunión el viernes a las 10", "No olvidar la tarea", "Feliz cumpleaños"])
    res = reg.call("write_file", {"path": name, "content": text}, ctx)
    assert res.ok and (ws / name).read_text(encoding="utf-8") == text
    return [{"role": "user", "content": f"Crea un archivo {name} con el texto: {text}"},
            {"role": "assistant", "content": "", "tool_calls": [_call("write_file", {"path": name, "content": text})]},
            {"role": "tool", **res.to_message("write_file")},
            {"role": "assistant", "content": f"Listo, creé {name} ({res.content['bytes']} bytes)."}]


def _ex_missing_file(rng, reg, ctx, ws):
    """El archivo pedido no existe y no hay otro parecido: se reconoce sin inventar nada."""
    name = rng.choice(["notas", "pendientes", "lista", "datos", "reporte", "agenda"]) + f"_{rng.randint(1, 99)}.txt"
    q = rng.choice([f"¿Cuántas líneas tiene {name}?", f"¿Qué dice la primera línea de {name}?", f"Lee {name}."])
    r1 = reg.call("read_file", {"path": name}, ctx)
    assert not r1.ok
    r2 = reg.call("list_dir", {}, ctx)
    assert r2.ok and not any(i["name"] == name for i in r2.content)
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": "", "tool_calls": [_call("read_file", {"path": name})]},
            {"role": "tool", **r1.to_message("read_file")},
            {"role": "assistant", "content": "No existe. Reviso qué archivos hay.", "tool_calls": [_call("list_dir", {})]},
            {"role": "tool", **r2.to_message("list_dir")},
            {"role": "assistant", "content": f"No encontré {name} en el espacio de trabajo y no hay otro archivo "
                                             "parecido. ¿Puedes confirmar el nombre o compartir el archivo?"}]


def _ex_no_tool(rng, reg, ctx, ws):
    q, a = rng.choice(GREETINGS + UNKNOWABLE)
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def _ex_permission_denied(rng, reg, ctx, ws):
    name = f"borrador_{rng.randint(1, 99)}.txt"
    denied = {"role": "tool", "name": "delete_file", "ok": False,
              "content": {"error": "Permiso 'file_delete' denegado: rechazado por el usuario"}}
    return [{"role": "user", "content": f"Borra {name}"},
            {"role": "assistant", "content": "", "tool_calls": [_call("delete_file", {"path": name})]},
            denied,
            {"role": "assistant", "content": f"No borré {name} porque no se autorizó el permiso de borrado. "
                                             "Si quieres hacerlo, apruébalo cuando te lo pida."}]


GENERATORS = [(_ex_calculator, 3), (_ex_read_file, 2), (_ex_error_recovery, 1), (_ex_python, 2), (_ex_kb, 2),
              (_ex_write, 1), (_ex_no_tool, 2), (_ex_permission_denied, 1), (_ex_missing_file, 1)]


KINDS = {g.__name__[4:]: g for g, _ in GENERATORS}
EVAL_SEED_BASE = 9_000_000_000  # las semillas de evaluación nunca se usan para entrenar


def tool_schemas(reg=None) -> list[dict]:
    reg = reg or default_registry()
    return reg.schemas(TOOLSETS["basic"] + ["delete_file"])


def build_example(kind: str, seed: int, ws: Path, reg=None) -> list[dict]:
    """Construye (y verifica) un ejemplo determinista en el espacio de trabajo ``ws``."""
    reg = reg or default_registry()
    return KINDS[kind](random.Random(seed), reg, _ctx(ws), ws)


def tool_conversations(n: int, seed: int) -> list[dict]:
    """Genera ``n`` conversaciones verificadas. Devuelve dicts {messages, tools, kind, seed}."""
    if seed * 1_000_003 + n >= EVAL_SEED_BASE:
        raise ValueError("semilla de entrenamiento dentro del rango reservado a evaluación")
    rng = random.Random(seed)
    reg = default_registry()
    names, weights = zip(*[(g.__name__[4:], w) for g, w in GENERATORS])
    tools = tool_schemas(reg)
    out = []
    with tempfile.TemporaryDirectory(prefix="titan_syn_") as tmp:
        for i in range(n):
            ws = Path(tmp) / f"ws{i}"
            ws.mkdir()
            kind = rng.choices(names, weights)[0]
            item_seed = seed * 1_000_003 + i
            out.append({"messages": build_example(kind, item_seed, ws, reg), "tools": tools,
                        "kind": kind, "seed": item_seed})
    return out


def tool_eval_items(n: int = 60) -> list[dict]:
    names = [g.__name__[4:] for g, _ in GENERATORS]
    return [{"id": f"tool-{i}", "kind": names[i % len(names)], "seed": EVAL_SEED_BASE + i} for i in range(n)]


def write_jsonl(items: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
