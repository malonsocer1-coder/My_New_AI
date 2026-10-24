"""Verificadores automáticos para seguimiento de instrucciones e incertidumbre."""

from __future__ import annotations

import json
import re
import unicodedata

from ..data.clean import detect_language

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(t: str) -> list[str]:
    return _WORD.findall(t)


def check_constraint(text: str, c: dict) -> bool:
    t, v = c["type"], c.get("value")
    s = text.strip()
    if t == "max_words":
        return len(_words(s)) <= v
    if t == "min_words":
        return len(_words(s)) >= v
    if t == "contains":
        return v.lower() in s.lower()
    if t == "starts_with":
        return s.startswith(v)
    if t == "ends_with":
        return s.endswith(v)
    if t == "lowercase":
        return s == s.lower() and any(ch.isalpha() for ch in s)
    if t == "uppercase":
        return s == s.upper() and any(ch.isalpha() for ch in s)
    if t == "no_comma":
        return "," not in s
    if t == "no_letter":
        return v.lower() not in s.lower()
    if t == "one_of":
        return s.strip(" .!¡") in v
    if t == "regex":
        return re.search(v, s) is not None
    if t == "language":
        return detect_language(s)[0] == v
    if t == "num_bullets":
        return sum(1 for ln in s.split("\n") if re.match(r"^\s*[-*•]\s+", ln)) == v
    if t == "num_numbered":
        return sum(1 for ln in s.split("\n") if re.match(r"^\s*\d+[.)]\s+", ln)) == v
    if t == "num_lines":
        return len([ln for ln in s.split("\n") if ln.strip()]) == v
    if t == "num_paragraphs":
        return len([p for p in re.split(r"\n\s*\n", s) if p.strip()]) == v
    if t == "count_word":
        word, n = v
        return sum(1 for w in _words(s.lower()) if w == word.lower()) == n
    if t == "json_keys":
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return False
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return False
        return isinstance(obj, dict) and all(k in obj for k in v)
    raise ValueError(f"Restricción desconocida: {t}")


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


UNCERTAINTY_MARKERS = ["no lo se", "no se ", "no tengo forma", "no puedo saber", "nadie puede saber", "no es posible saber",
                       "no tengo informacion", "no tengo acceso", "no conozco", "desconozco", "no hay manera",
                       "no puedo predecir", "es aleatorio", "imposible saber", "no dispongo"]
FALSE_PREMISE_MARKERS = ["no existe", "no ha ganado", "no es cierto", "en realidad", "no esta hecha", "premisa",
                         "no hay", "nunca", "falso", "no tiene"] + UNCERTAINTY_MARKERS
CLARIFY_MARKERS = ["podrias", "puedes darme", "a que te refieres", "mas detalles", "que quieres", "cual ", "?",
                   "necesito saber", "especifica", "aclarar"]


def check_uncertainty(text: str, expected: str) -> bool:
    n = _norm(text)
    markers = {"abstain": UNCERTAINTY_MARKERS, "false_premise": FALSE_PREMISE_MARKERS,
               "clarify": CLARIFY_MARKERS}[expected]
    return any(m in n for m in markers)
