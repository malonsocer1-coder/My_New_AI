"""Banco de evaluación reproducible de TITAN.

Suites (``--suite all`` ejecuta todas; se pueden combinar con comas):

  bpb           pérdida y bits por byte por dominio en los shards de validación
  mc            opción múltiple por verosimilitud (conocimiento general en español),
                con exactitud, calibración (ECE), consistencia ante el orden de
                las opciones y robustez ante entradas sin acentos ni mayúsculas
  arith         aritmética (generación voraz, coincidencia exacta)
  code          funciones Python con tests ejecutados en el sandbox (pass@1)
  instructions  seguimiento de instrucciones con restricciones verificables
  uncertainty   reconocimiento de incertidumbre, premisas falsas y ambigüedad
  tool_use      agente con herramientas reales: selección, llamadas válidas y respuesta
  retrieval     recuperar un dato insertado en un contexto largo ("aguja en un pajar")

Cada resultado se acompaña del azar o línea base correspondiente. No se usa
ningún modelo externo como juez; todo se verifica por programa.
"""

from __future__ import annotations

import json
import math
import random
import re
import tempfile
import time
import unicodedata
from pathlib import Path

import torch

from ..agent.permissions import DEFAULT_POLICY, PermissionPolicy
from ..agent.protocol import ToolContext
from ..agent.registry import default_registry
from ..agent.sandbox import run_python
from ..infer.engine import Engine, load_engine
from ..utils.io import read_jsonl, write_json
from .checkers import check_constraint, check_uncertainty

DATA = Path(__file__).parent / "data"
SUITES = ["bpb", "mc", "arith", "code", "instructions", "uncertainty", "tool_use", "retrieval"]


def _strip_accents(t: str) -> str:
    t = unicodedata.normalize("NFD", t)
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


# ----------------------------------------------------------------------
def eval_bpb(eng: Engine, manifest: str | None, windows: int = 64) -> dict:
    if not manifest or not Path(manifest).exists():
        return {"skipped": "sin manifest de datos (pase --data o use una ejecución con datos locales)"}
    from ..data.loader import EvalLoader
    ev = EvalLoader(manifest, "val", min(eng.model.cfg.max_seq_len, 1024), eng.device)
    out = {}
    with torch.no_grad():
        for d in ev.domains():
            tot = n = 0.0
            for x, y in ev.batches(d, windows, 8):
                o = eng.model(x, targets=y)
                k = (y != -100).sum().item()
                tot += o.ce_loss.item() * k
                n += k
            loss = tot / n
            r = {"loss": round(loss, 4), "ppl": round(math.exp(loss), 2)}
            if ev.bytes_per_token.get(d):
                r["bpb"] = round(loss / math.log(2) / ev.bytes_per_token[d], 4)
            out[d] = r
    return out


def _mc_scores(eng: Engine, question: str, choices: list[str]) -> tuple[list[float], list[float]]:
    ctx = f"Pregunta: {question}\nRespuesta:"
    raw, norm = [], []
    for c in choices:
        ll, _ = eng.loglikelihood(ctx, " " + c)
        raw.append(ll)
        norm.append(ll / max(1, len(c.encode("utf-8"))))
    return raw, norm


def eval_mc(eng: Engine, limit: int | None = None) -> dict:
    items = read_jsonl(DATA / "mc_general_es.jsonl")[:limit]
    correct = correct_norm = consistent = robust = 0
    bins = [[0, 0, 0.0] for _ in range(10)]  # [n, aciertos, confianza]
    for it in items:
        ch, ans = it["choices"], it["answer"]
        raw, norm = _mc_scores(eng, it["question"], ch)
        pred = max(range(len(ch)), key=lambda i: raw[i])
        pred_n = max(range(len(ch)), key=lambda i: norm[i])
        correct += pred == ans
        correct_norm += pred_n == ans
        probs = torch.softmax(torch.tensor(norm), 0)
        conf = float(probs.max())
        b = min(9, int(conf * 10))
        bins[b][0] += 1
        bins[b][1] += pred_n == ans
        bins[b][2] += conf
        # consistencia: mismas opciones en otro orden
        rot = ch[1:] + ch[:1]
        _, norm_r = _mc_scores(eng, it["question"], rot)
        consistent += rot[max(range(len(rot)), key=lambda i: norm_r[i])] == ch[pred_n]
        # robustez: sin acentos y en minúsculas
        _, norm_s = _mc_scores(eng, _strip_accents(it["question"]).lower(), ch)
        robust += max(range(len(ch)), key=lambda i: norm_s[i]) == ans
    n = len(items)
    ece = sum(abs(b[1] / b[0] - b[2] / b[0]) * b[0] / n for b in bins if b[0])
    return {"n": n, "acc": round(correct / n, 4), "acc_norm": round(correct_norm / n, 4), "chance": 0.25,
            "ece": round(ece, 4), "consistency_under_reorder": round(consistent / n, 4),
            "acc_norm_no_accents_lowercase": round(robust / n, 4)}


def eval_arith(eng: Engine, limit: int | None = None) -> dict:
    items = read_jsonl(DATA / "arith_es.jsonl")[:limit]
    ok = 0
    examples = []
    for it in items:
        out = eng.complete(it["prompt"], max_new_tokens=12, temperature=0.0)
        m = re.search(r"-?\d+", out)
        got = m.group(0) if m else None
        ok += got == it["answer"]
        if len(examples) < 5:
            examples.append({"prompt": it["question"], "expected": it["answer"], "got": out.strip()[:40]})
    return {"n": len(items), "exact_match": round(ok / len(items), 4), "examples": examples}


def eval_code(eng: Engine, limit: int | None = None) -> dict:
    items = read_jsonl(DATA / "code_python_es.jsonl")[:limit]
    passed = 0
    for it in items:
        comp = eng.complete(it["prompt"], max_new_tokens=160, temperature=0.0)
        comp = re.split(r"\n(?=def |class |if __name__|print\()", comp)[0]
        r = run_python(it["prompt"] + comp + "\n\n" + it["tests"], timeout_s=10)
        passed += r.returncode == 0 and not r.timed_out
    # control: la solución de referencia debe pasar (valida el propio benchmark)
    ref_ok = sum(run_python(it["prompt"] + it["reference"] + "\n" + it["tests"]).returncode == 0 for it in items)
    return {"n": len(items), "pass@1": round(passed / len(items), 4), "reference_pass_rate": ref_ok / len(items)}


def _ctx_len(eng) -> int:
    return getattr(eng, "max_seq_len", None) or eng.model.cfg.max_seq_len


def _overflows(eng, messages, tools=None, reserve: int = 48) -> bool:
    """True si el prompt de chat no deja al menos ``reserve`` tokens para responder."""
    return len(eng.chat_format.prompt_ids(messages, tools)) > _ctx_len(eng) - reserve


def eval_instructions(eng: Engine, limit: int | None = None) -> dict:
    items = read_jsonl(DATA / "instructions_es.jsonl")[:limit]
    all_ok = parts_ok = parts = overflow = 0
    for it in items:
        if _overflows(eng, [{"role": "user", "content": it["prompt"]}]):
            overflow += 1
            continue
        out = eng.chat([{"role": "user", "content": it["prompt"]}], max_new_tokens=200, temperature=0.0)
        res = [check_constraint(out, c) for c in it["checks"]]
        all_ok += all(res)
        parts_ok += sum(res)
        parts += len(res)
    if overflow == len(items):
        return {"n": len(items), "context_overflow": overflow, "skipped": "contexto del modelo insuficiente"}
    return {"n": len(items), "prompt_level_acc": round(all_ok / len(items), 4),
            "constraint_level_acc": round(parts_ok / max(1, parts), 4), "context_overflow": overflow}


def eval_uncertainty(eng: Engine, limit: int | None = None) -> dict:
    items = read_jsonl(DATA / "uncertainty_es.jsonl")[:limit]
    ok = overflow = 0
    by = {}
    for it in items:
        if _overflows(eng, [{"role": "user", "content": it["prompt"]}]):
            overflow += 1
            continue
        out = eng.chat([{"role": "user", "content": it["prompt"]}], max_new_tokens=80, temperature=0.0)
        hit = check_uncertainty(out, it["expected_behavior"])
        ok += hit
        b = by.setdefault(it["expected_behavior"], [0, 0])
        b[0] += 1
        b[1] += hit
    if overflow == len(items):
        return {"n": len(items), "context_overflow": overflow, "skipped": "contexto del modelo insuficiente"}
    return {"n": len(items), "acc": round(ok / len(items), 4), "context_overflow": overflow,
            "by_behavior": {k: round(v[1] / v[0], 4) for k, v in by.items()},
            "note": "verificación por marcadores léxicos: indicativa, no definitiva"}


def eval_tool_use(eng: Engine, limit: int | None = None) -> dict:
    from ..agent.loop import Agent
    from ..sft.synthetic_tools import build_example, tool_schemas
    items = read_jsonl(DATA / "tool_use_es.jsonl")[:limit]
    reg = default_registry()
    if _overflows(eng, [{"role": "user", "content": "¿Cuánto es 2 + 2?"}], tool_schemas(reg), reserve=96):
        return {"n": len(items), "context_overflow": len(items), "skipped": "contexto del modelo insuficiente"}
    sel = valid = answered = 0
    by = {}
    for it in items:
        with tempfile.TemporaryDirectory(prefix="titan_eval_") as tmp:
            ws = Path(tmp)
            ref = build_example(it["kind"], it["seed"], ws, reg)
            # se rehace el espacio de trabajo para que el agente no vea efectos de la referencia
            for f in list(ws.iterdir()):
                if f.is_file() and f.name.startswith(("saludo", "mensaje", "nota_", "aviso")):
                    f.unlink()
            expected_tool = next((m["tool_calls"][0]["name"] for m in ref if m["role"] == "assistant"
                                  and m.get("tool_calls")), None)
            pol = PermissionPolicy(rules={**{k: "allow" for k in DEFAULT_POLICY}},
                                   approver=lambda r: "deny" if r.capability == "file_delete" else "allow_once")
            ctx = ToolContext(workspace=str(ws), permissions=pol)
            agent = Agent(eng, reg, ctx, max_steps=5, sampling={"temperature": 0.0, "max_new_tokens": 200})
            res = agent.run([ref[0]])
            first = next((m["tool_calls"][0]["name"] for m in res.messages if m["role"] == "assistant"
                          and m.get("tool_calls")), None)
            n_calls = sum(1 for t in res.trace if "call" in t)
            n_invalid = sum(1 for t in res.trace if "invalid_call" in t)
            s = first == expected_tool
            v = n_invalid == 0 and (n_calls > 0 or expected_tool is None)
            keys = re.findall(r"-?\d+(?:\.\d+)?|«[^»]+»", ref[-1]["content"])
            a = bool(keys) and all(k.strip("«»") in res.final for k in keys) if expected_tool else s
            sel += s
            valid += v
            answered += a
            b = by.setdefault(it["kind"], [0, 0])
            b[0] += 1
            b[1] += s
    n = len(items)
    return {"n": n, "tool_selection_acc": round(sel / n, 4), "valid_call_rate": round(valid / n, 4),
            "answer_acc": round(answered / n, 4), "selection_by_kind": {k: round(v[1] / v[0], 3) for k, v in by.items()}}


def eval_retrieval(eng: Engine, n: int = 20, seed: int = 31337) -> dict:
    rng = random.Random(seed)
    filler = ("El clima de la región es templado durante gran parte del año. Las lluvias se concentran en verano. "
              "Los mercados abren temprano y venden frutas, verduras y flores. ")
    max_len = eng.model.cfg.max_seq_len
    res = {}
    for frac in (0.25, 0.5, 0.9):
        ok = 0
        for i in range(n):
            name = rng.choice(["Ana", "Luis", "Marta", "Pedro", "Sofía"]) + f" {rng.randint(1, 99)}"
            code = str(rng.randint(1000, 9999))
            fact = f"El código secreto de {name} es {code}. "
            budget = int(max_len * frac) - 40
            reps = max(1, budget // max(1, len(eng.tok.encode(filler))))
            pos = rng.randint(0, reps)
            ctx = filler * pos + fact + filler * (reps - pos)
            prompt = f"{ctx}\nPregunta: ¿Cuál es el código secreto de {name}?\nRespuesta: El código secreto de {name} es"
            out = eng.complete(prompt, max_new_tokens=6, temperature=0.0)
            ok += code in out
        res[f"context_{int(frac * 100)}pct"] = round(ok / n, 4)
    res["n_per_length"] = n
    res["max_seq_len"] = max_len
    return res


def _flat(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if k == "seconds":
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
        elif isinstance(v, dict):
            out.update({f"{k}.{k2}": v2 for k2, v2 in v.items() if isinstance(v2, (int, float))})
    return out


def run_suite(checkpoint: str, suite: str = "all", out_path: str | None = None, device: str = "auto",
              limit: int | None = None, data: str | None = None) -> dict:
    eng = load_engine(checkpoint, device=device)
    names = SUITES if suite == "all" else suite.split(",")
    if data is None:
        info = Path(checkpoint) / "run_info.json"
        if info.exists():
            data = json.loads(info.read_text())["train_config"].get("data")
    fns = {"bpb": lambda: eval_bpb(eng, data), "mc": lambda: eval_mc(eng, limit),
           "arith": lambda: eval_arith(eng, limit), "code": lambda: eval_code(eng, limit),
           "instructions": lambda: eval_instructions(eng, limit), "uncertainty": lambda: eval_uncertainty(eng, limit),
           "tool_use": lambda: eval_tool_use(eng, limit), "retrieval": lambda: eval_retrieval(eng)}
    results = {}
    t0 = time.time()
    for n in names:
        t = time.time()
        results[n] = fns[n]()
        results[n]["seconds"] = round(time.time() - t, 1)
    report = {"checkpoint": str(checkpoint), "suites": names, "results": results,
              "params": eng.model.num_params(), "elapsed_s": round(time.time() - t0, 1),
              "summary": {n: _flat(r) for n, r in results.items()}}
    if out_path:
        write_json(out_path, report)
    return report
