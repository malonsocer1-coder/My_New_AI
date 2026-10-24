"""Importadores de conversaciones abiertas (con licencia permisiva) para el SFT.

Convierte conjuntos públicos al formato propio ``{"messages": [...], "kind": ...}``
y les aplica las mismas garantías que a los datos de preentrenamiento:
normalización Unicode, eliminación de datos personales (correos, teléfonos…),
descontaminación con n-gramas frente a los benchmarks propios, deduplicación
exacta y límites de longitud.

Formatos admitidos:

* ``pairs``: una fila = instrucción y respuesta (p. ej. Aya, Apache-2.0:
  campos ``inputs``/``targets``, idioma en ``language_code``).
* ``oasst``: árboles de mensajes de OpenAssistant (OASST1/OASST2, Apache-2.0):
  se reconstruye, desde cada raíz, el hilo de mejor rango (``rank`` más bajo).

Orígenes: ``loader: hf`` (streaming de Hugging Face, requiere ``datasets`` e
Internet) o ``loader: jsonl`` (archivo local, p. ej. una descarga previa).
Con ``optional: true`` un fallo (sin red, conjunto renombrado, columnas
distintas) se registra y el resto del SFT continúa; sin él, el error detiene
la fase para que no pase inadvertido.
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterator

from ..data.clean import normalize_text, scrub_pii
from ..data.decontam import ContaminationIndex


def _rows(spec: dict) -> Iterator[dict]:
    loader = spec.get("loader", "hf")
    p = spec.get("params", {})
    if loader == "jsonl":
        for path in sorted(glob.glob(p["path"])):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        return
    if loader != "hf":
        raise ValueError(f"loader desconocido para importar chats: {loader}")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("importar desde Hugging Face requiere `pip install datasets`") from e
    kw = {k: p[k] for k in ("name", "data_dir", "revision") if p.get(k)}
    yield from load_dataset(p["path"], split=p.get("split", "train"), streaming=True, **kw)


def _passes(row: dict, filters: dict) -> bool:
    for fld, allowed in (filters or {}).items():
        if fld not in row:
            raise KeyError(f"la columna de filtro '{fld}' no existe; columnas: {sorted(row)}")
        val = row[fld]
        vals = val if isinstance(val, list) else [val]
        if not any(v in allowed for v in vals):
            return False
    return True


def _pairs(spec: dict) -> Iterator[list[dict]]:
    pf, rf = spec.get("prompt_field", "inputs"), spec.get("response_field", "targets")
    cf = spec.get("context_field")
    for row in _rows(spec):
        missing = [f for f in (pf, rf) if f not in row]
        if missing:
            raise KeyError(f"faltan las columnas {missing}; columnas disponibles: {sorted(row)}")
        if not _passes(row, spec.get("filter")):
            continue
        prompt = str(row[pf] or "")
        if cf and row.get(cf):
            prompt = f"{prompt}\n\n{row[cf]}"
        yield [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(row[rf] or "")}]


def _oasst(spec: dict) -> Iterator[list[dict]]:
    langs = set(spec.get("langs", ["es"]))
    by_id: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for row in _rows(spec):
        for f in ("message_id", "parent_id", "text", "role", "lang"):
            if f not in row:
                raise KeyError(f"falta la columna '{f}' (formato oasst); columnas: {sorted(row)}")
        if row["lang"] not in langs or row.get("deleted") or row.get("review_result") is False:
            continue
        by_id[row["message_id"]] = row
        if row["parent_id"] is None:
            roots.append(row["message_id"])
        else:
            children.setdefault(row["parent_id"], []).append(row["message_id"])

    def best_child(mid: str, role: str) -> dict | None:
        kids = [by_id[c] for c in children.get(mid, []) if c in by_id and by_id[c]["role"] == role]
        if not kids:
            return None
        return min(kids, key=lambda r: (r.get("rank") if r.get("rank") is not None else 1e9,
                                        str(r.get("created_date", "")), r["message_id"]))

    max_turns = int(spec.get("max_turns", 6))
    for rid in roots:
        root = by_id[rid]
        if root["role"] != "prompter":
            continue
        msgs, cur = [{"role": "user", "content": root["text"]}], root
        while len(msgs) < 2 * max_turns:
            ans = best_child(cur["message_id"], "assistant")
            if ans is None:
                break
            msgs.append({"role": "assistant", "content": ans["text"]})
            nxt = best_child(ans["message_id"], "prompter")
            if nxt is None:
                break
            msgs.append({"role": "user", "content": nxt["text"]})
            cur = nxt
        if msgs[-1]["role"] == "user":
            msgs = msgs[:-1]
        if len(msgs) >= 2:
            yield msgs


FORMATS = {"pairs": _pairs, "oasst": _oasst}


def import_chats(spec: dict, decontam: ContaminationIndex | None = None, log=print) -> tuple[list[dict], dict]:
    """Devuelve (conversaciones, estadísticas) para una fuente ``spec`` (ver docstring del módulo)."""
    name = spec["name"]
    fmt = FORMATS[spec.get("format", "pairs")]
    max_conv = int(spec.get("max_conversations", 10**9))
    min_chars, max_chars = int(spec.get("min_chars", 2)), int(spec.get("max_chars", 6000))
    stats = {"name": name, "license": spec.get("license", "unknown"), "kept": 0, "too_short": 0,
             "too_long": 0, "contaminated": 0, "duplicates": 0}
    seen: set[str] = set()
    out: list[dict] = []
    for msgs in fmt(spec):
        clean = []
        for m in msgs:
            text = normalize_text(m["content"]).strip()
            if spec.get("scrub_pii", True):
                text = scrub_pii(text)
            clean.append({"role": m["role"], "content": text})
        lens = [len(m["content"]) for m in clean]
        if min(lens) < min_chars:
            stats["too_short"] += 1
            continue
        if sum(lens) > max_chars:
            stats["too_long"] += 1
            continue
        joined = "\n".join(m["content"] for m in clean)
        if decontam is not None and decontam.is_contaminated(joined):
            stats["contaminated"] += 1
            continue
        h = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        if h in seen:
            stats["duplicates"] += 1
            continue
        seen.add(h)
        out.append({"messages": clean, "kind": name})
        if len(out) >= max_conv:
            break
    stats["kept"] = len(out)
    log(f"[sft] {name}: {len(out)} conversaciones importadas ({stats})")
    return out, stats


def eval_index(patterns: list[str], n: int = 13) -> ContaminationIndex:
    """Índice de n-gramas de los benchmarks propios (mismo criterio que el preentrenamiento)."""
    from ..data.pipeline import _load_eval_texts
    idx = ContaminationIndex(n=n)
    idx.add_eval_texts(_load_eval_texts(patterns))
    return idx
