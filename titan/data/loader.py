"""Cargador de tokens determinista y reanudable para preentrenamiento.

Propiedades:
  * Los shards se leen con ``np.memmap``: la RAM no limita el tamaño del corpus.
  * Mezcla de dominios por pesos, con una secuencia de elecciones derivada
    solo de (semilla, índice global de muestra).
  * Dentro de cada dominio, los fragmentos se recorren sin repetición por
    época mediante una permutación pseudoaleatoria (Feistel) de memoria O(1).
  * El estado completo es un único entero (``samples_seen``): reanudar tras
    una interrupción reproduce exactamente los mismos lotes.
  * Cada rango de un entrenamiento distribuido recibe muestras disjuntas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

_BLOCK = 1 << 16


def _mix64(x: int) -> int:
    """splitmix64: mezcla de bits de alta calidad para la función de ronda."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class FeistelPermutation:
    """Biyección pseudoaleatoria de [0, n) sin almacenar la permutación."""

    def __init__(self, n: int, key: int, rounds: int = 4):
        if n <= 0:
            raise ValueError("n debe ser > 0")
        self.n = n
        bits = max(2, (n - 1).bit_length())
        bits += bits % 2
        self.half = bits // 2
        self.mask = (1 << self.half) - 1
        self.keys = [_mix64(key * 1000003 + r) for r in range(rounds)]

    def __call__(self, i: int) -> int:
        if not 0 <= i < self.n:
            raise IndexError(i)
        x = i
        while True:  # cycle-walking hasta caer en [0, n)
            left, right = x >> self.half, x & self.mask
            for k in self.keys:
                left, right = right, left ^ (_mix64(right ^ k) & self.mask)
            x = (left << self.half) | right
            if x < self.n:
                return x


class TokenStream:
    """Concatenación lógica de los shards de un dominio."""

    def __init__(self, paths: list[Path], dtype: str):
        self.arrays = [np.memmap(p, dtype=dtype, mode="r") for p in paths]
        sizes = [a.size for a in self.arrays]
        self.offsets = np.cumsum([0] + sizes)
        self.size = int(self.offsets[-1])

    def read(self, start: int, length: int) -> np.ndarray:
        out = np.empty(length, dtype=np.int64)
        pos, filled = start, 0
        while filled < length:
            k = int(np.searchsorted(self.offsets, pos, side="right") - 1)
            local = pos - int(self.offsets[k])
            take = min(length - filled, self.arrays[k].size - local)
            out[filled:filled + take] = self.arrays[k][local:local + take]
            filled += take
            pos += take
        return out


def _seed_int(*parts) -> int:
    h = hashlib.blake2b(json.dumps(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


class PretrainLoader:
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1234,
        domain_weights: dict[str, float] | None = None,
        device: str | torch.device = "cpu",
    ):
        manifest_path = Path(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        self.seq_len, self.batch_size = seq_len, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.device = torch.device(device)
        by_domain: dict[str, list[Path]] = {}
        for s in self.manifest["shards"]:
            if s["split"] == split:
                by_domain.setdefault(s["domain"], []).append(root / s["path"])
        if domain_weights:
            missing = sorted(set(by_domain) - set(domain_weights))
            if missing:
                print(f"[datos] aviso: dominios sin peso explícito, se excluyen: {missing}", flush=True)
            weights = {d: domain_weights.get(d, 0.0) for d in by_domain}
        else:
            weights = self.manifest.get("domain_weights") or {}
        self.domains = [d for d in sorted(by_domain) if weights.get(d, 1.0) > 0]
        if not self.domains:
            raise ValueError(f"No hay shards para split={split}")
        self.streams = {d: TokenStream(sorted(by_domain[d]), self.manifest["dtype"]) for d in self.domains}
        self.n_chunks = {d: (self.streams[d].size - 1) // seq_len for d in self.domains}
        for d, n in self.n_chunks.items():
            if n <= 0:
                raise ValueError(f"El dominio {d} tiene menos de seq_len+1 tokens")
        w = np.array([weights.get(d, 1.0) for d in self.domains], dtype=np.float64)
        self.probs = w / w.sum()
        self.samples_seen = 0
        self._block_id = -1
        self._block_base = np.zeros(len(self.domains), dtype=np.int64)
        self._perms: dict[tuple[int, int], FeistelPermutation] = {}

    # -- estado ---------------------------------------------------------
    def state_dict(self) -> dict:
        return {"samples_seen": self.samples_seen, "seed": self.seed, "seq_len": self.seq_len,
                "domains": self.domains}

    def load_state_dict(self, state: dict) -> None:
        if state.get("seq_len", self.seq_len) != self.seq_len:
            raise ValueError("seq_len del checkpoint no coincide con el cargador")
        self.samples_seen = int(state["samples_seen"])
        self._block_id = -1

    # -- muestreo -------------------------------------------------------
    def _load_block(self, block: int) -> None:
        if block < self._block_id or self._block_id < 0:
            base = np.zeros(len(self.domains), dtype=np.int64)
            start = 0
        else:
            base = self._block_base + np.bincount(self._choices, minlength=len(self.domains))
            start = self._block_id + 1
        for b in range(start, block):
            base += np.bincount(self._block_choices(b), minlength=len(self.domains))
        self._choices = self._block_choices(block)
        self._rank_in_dom = np.empty(_BLOCK, dtype=np.int64)
        for d in range(len(self.domains)):
            idx = np.nonzero(self._choices == d)[0]
            self._rank_in_dom[idx] = np.arange(idx.size)
        self._block_base = base
        self._block_id = block

    def _block_choices(self, block: int) -> np.ndarray:
        rng = np.random.default_rng(_seed_int("mix", self.seed, block))
        return rng.choice(len(self.domains), size=_BLOCK, p=self.probs)

    def sample_location(self, s: int) -> tuple[str, int]:
        block, off = divmod(s, _BLOCK)
        if block != self._block_id:
            self._load_block(block)
        d = int(self._choices[off])
        c = int(self._block_base[d] + self._rank_in_dom[off])
        dom = self.domains[d]
        n = self.n_chunks[dom]
        epoch, j = divmod(c, n)
        key = (d, epoch)
        if key not in self._perms:
            if len(self._perms) > 64:
                self._perms.clear()
            self._perms[key] = FeistelPermutation(n, _seed_int("perm", self.seed, dom, epoch))
        return dom, self._perms[key](j)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.samples_seen + self.rank * self.batch_size
        rows = []
        for b in range(self.batch_size):
            dom, chunk = self.sample_location(base + b)
            rows.append(self.streams[dom].read(chunk * self.seq_len, self.seq_len + 1))
        self.samples_seen += self.batch_size * self.world_size
        arr = torch.from_numpy(np.stack(rows))
        if self.device.type == "cuda":
            arr = arr.pin_memory().to(self.device, non_blocking=True)
        return arr[:, :-1], arr[:, 1:]

    def tokens_per_step(self) -> int:
        return self.batch_size * self.world_size * self.seq_len


class EvalLoader:
    """Ventanas fijas y deterministas por dominio para medir la pérdida de validación."""

    def __init__(self, manifest_path: str | Path, split: str, seq_len: int, device="cpu"):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        by_domain: dict[str, list[Path]] = {}
        for s in manifest["shards"]:
            if s["split"] == split:
                by_domain.setdefault(s["domain"], []).append(root / s["path"])
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.streams = {d: TokenStream(sorted(p), manifest["dtype"]) for d, p in sorted(by_domain.items())}
        self.bytes_per_token = {
            d: manifest["totals"].get(f"{d}.{split}", {}).get("bytes_per_token") for d in self.streams
        }

    def domains(self) -> list[str]:
        return [d for d, s in self.streams.items() if s.size > self.seq_len + 1]

    def batches(self, domain: str, max_windows: int, batch_size: int):
        st = self.streams[domain]
        n = (st.size - 1) // self.seq_len
        k = min(n, max_windows)
        idx = np.linspace(0, n - 1, k).astype(np.int64) if k < n else np.arange(n)
        for i in range(0, len(idx), batch_size):
            rows = [st.read(int(c) * self.seq_len, self.seq_len + 1) for c in idx[i:i + batch_size]]
            arr = torch.from_numpy(np.stack(rows)).to(self.device)
            yield arr[:, :-1], arr[:, 1:]
