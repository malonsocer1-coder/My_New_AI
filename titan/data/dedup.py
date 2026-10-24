"""Deduplicación exacta (SHA-256) y aproximada (MinHash + LSH).

Para corpus de cientos de GB se recomienda ejecutar la misma lógica de forma
distribuida (p. ej. con ``datatrove`` en nodos CPU baratos); esta
implementación sirve para corpus medianos y como referencia verificable.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict

import numpy as np

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX32 = np.uint64((1 << 32) - 1)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def exact_key(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class ExactDeduper:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        k = exact_key(text)
        if k in self.seen:
            return True
        self.seen.add(k)
        return False


class MinHashLSH:
    """Detecta documentos con similitud de Jaccard alta sobre n-gramas de palabras.

    Con ``bands`` x ``rows`` = ``num_perm``, la probabilidad de colisión para
    similitud s es 1 - (1 - s^rows)^bands (umbral ~ (1/bands)^(1/rows)).
    """

    def __init__(self, num_perm: int = 128, bands: int = 16, ngram: int = 5, seed: int = 1234):
        if num_perm % bands:
            raise ValueError("num_perm debe ser múltiplo de bands")
        self.num_perm, self.bands, self.rows, self.ngram = num_perm, bands, num_perm // bands, ngram
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, np.iinfo(np.int64).max, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.b = rng.randint(0, np.iinfo(np.int64).max, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.buckets: list[dict[bytes, int]] = [defaultdict(int) for _ in range(bands)]
        self.count = 0

    @property
    def threshold(self) -> float:
        return (1 / self.bands) ** (1 / self.rows)

    def signature(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        if len(toks) < self.ngram:
            shingles = {" ".join(toks)}
        else:
            shingles = {" ".join(toks[i:i + self.ngram]) for i in range(len(toks) - self.ngram + 1)}
        hv = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in shingles), dtype=np.uint64, count=len(shingles))
        with np.errstate(over="ignore"):
            phv = ((hv[:, None] * self.a[None, :] + self.b[None, :]) % _MERSENNE) & _MAX32
        return phv.min(axis=0)

    def insert_if_new(self, text: str) -> bool:
        """Devuelve True si el documento es nuevo (y lo inserta); False si es casi duplicado."""
        sig = self.signature(text)
        keys = [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]
        if any(k in self.buckets[i] for i, k in enumerate(keys)):
            return False
        for i, k in enumerate(keys):
            self.buckets[i][k] = self.count
        self.count += 1
        return True

    @staticmethod
    def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float((sig_a == sig_b).mean())
