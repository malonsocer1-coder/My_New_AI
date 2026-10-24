"""Detección de contaminación: impide que textos de los benchmarks entren en el entrenamiento.

Se indexan los n-gramas (por defecto 13 palabras, como en GPT-3) de todos los
conjuntos de evaluación; cualquier documento de entrenamiento que comparta
al menos ``min_hits`` n-gramas con la evaluación se descarta y se registra.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _ngrams(text: str, n: int) -> Iterable[bytes]:
    toks = _TOKEN.findall(text.lower())
    for i in range(len(toks) - n + 1):
        yield hashlib.blake2b(" ".join(toks[i:i + n]).encode("utf-8"), digest_size=8).digest()


class ContaminationIndex:
    def __init__(self, n: int = 13, min_hits: int = 1):
        self.n = n
        self.min_hits = min_hits
        self.grams: set[bytes] = set()
        self.short_texts: set[str] = set()

    def add_eval_text(self, text: str) -> None:
        toks = _TOKEN.findall(text.lower())
        if len(toks) < self.n:
            # Textos cortos (p. ej. respuestas): coincidencia literal normalizada
            if len(toks) >= 4:
                self.short_texts.add(" ".join(toks))
            return
        self.grams.update(_ngrams(text, self.n))

    def add_eval_texts(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add_eval_text(t)

    def __len__(self) -> int:
        return len(self.grams) + len(self.short_texts)

    def hits(self, text: str) -> int:
        if not self.grams and not self.short_texts:
            return 0
        count = sum(1 for g in _ngrams(text, self.n) if g in self.grams)
        if self.short_texts:
            # Límites de palabra: "1 56" no debe coincidir con "1 567"
            norm = " " + " ".join(_TOKEN.findall(text.lower())) + " "
            count += sum(1 for s in self.short_texts if f" {s} " in norm)
        return count

    def is_contaminated(self, text: str) -> bool:
        return self.hits(text) >= self.min_hits
