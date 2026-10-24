"""Base de conocimiento local (recuperación de información, RAG) con BM25.

El conocimiento vive en documentos del disco del usuario (``knowledge/``):
texto, Markdown, PDF, Word, Excel, PowerPoint... Se trocean en pasajes y se
indexan con BM25 (sin modelos externos). El índice se reconstruye solo cuando
cambian los archivos. Así un modelo pequeño puede responder con información
que no cabe en sus parámetros, citando el archivo de origen.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .documents import SUPPORTED, read_document

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = set("de la que el en y a los se del las un por con no una su para es al lo como más o the of and to in is".split())


def _terms(text: str) -> list[str]:
    return [w for w in (t.lower() for t in _WORD.findall(text)) if w not in _STOP and len(w) > 1]


@dataclass
class Passage:
    source: str
    index: int
    text: str


class KnowledgeBase:
    def __init__(self, root: str | Path, passage_chars: int = 800, k1: float = 1.5, b: float = 0.75):
        self.root = Path(root)
        self.passage_chars, self.k1, self.b = passage_chars, k1, b
        self._sig: tuple | None = None
        self.passages: list[Passage] = []

    def _signature(self) -> tuple:
        if not self.root.exists():
            return ()
        return tuple(sorted((str(p), p.stat().st_mtime, p.stat().st_size)
                            for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED))

    def _split(self, text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        out, cur = [], ""
        for p in paras:
            if len(cur) + len(p) > self.passage_chars and cur:
                out.append(cur)
                cur = ""
            cur = (cur + "\n\n" + p).strip()
        if cur:
            out.append(cur)
        return out

    def refresh(self) -> None:
        sig = self._signature()
        if sig == self._sig:
            return
        self.passages = []
        for path, _, _ in sig:
            try:
                text = read_document(path)
            except Exception:
                continue
            rel = str(Path(path).relative_to(self.root))
            self.passages += [Passage(rel, i, t) for i, t in enumerate(self._split(text))]
        self.tf = [Counter(_terms(p.text)) for p in self.passages]
        self.lens = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.passages)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._sig = sig

    def search(self, query: str, k: int = 3) -> list[dict]:
        self.refresh()
        q = _terms(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [{"source": self.passages[i].source, "passage": self.passages[i].index,
                 "score": round(s, 3), "text": self.passages[i].text} for s, i in scores[:k]]
