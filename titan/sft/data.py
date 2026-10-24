"""Preparación de datos de SFT y cargadores con máscara de pérdida.

Entrada: archivos JSONL con conversaciones ``{"messages": [...], "tools": [...]}``
(propias, sintéticas verificadas o conjuntos abiertos con licencia permisiva,
p. ej. OASST2 o Aya, convertidos a este formato).

Salida (``out_dir``): filas de longitud fija ``seq_len + 1`` empaquetadas sin
partir conversaciones (si una conversación no cabe, se descarta y se contabiliza), con
``tokens.bin`` (IDs), ``mask.bin`` (1 = se entrena) y ``meta.json``.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..tokenizer import TitanTokenizer
from ..utils.io import read_yaml, sha256_file, write_json
from .chat import ChatFormat


def _iter_conversations(patterns: list[str]):
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def pack(encoded: list[tuple[list[int], list[int]]], row_len: int, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    rows_t, rows_m = [], []
    cur_t, cur_m = [], []

    def flush():
        if cur_t:
            n = row_len - len(cur_t)
            rows_t.append(cur_t + [pad_id] * n)
            rows_m.append(cur_m + [0] * n)

    for ids, labels in encoded:
        if len(ids) > row_len:
            continue  # no cabe: se descarta (prepare_sft lo contabiliza)
        mask = [0 if l == -100 else 1 for l in labels]
        if len(cur_t) + len(ids) > row_len:
            flush()
            cur_t, cur_m = [], []
        cur_t += ids
        cur_m += mask
    flush()
    return np.asarray(rows_t, dtype=np.int64), np.asarray(rows_m, dtype=np.uint8)


def prepare_sft(config: str | dict, log=print) -> dict:
    cfg = read_yaml(config) if isinstance(config, (str, Path)) else dict(config)
    tok = TitanTokenizer.load(cfg["tokenizer"])
    fmt = ChatFormat(tok)
    seq_len = int(cfg["seq_len"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    convs = list(_iter_conversations(cfg["files"]))
    rng = random.Random(cfg.get("seed", 0))
    rng.shuffle(convs)
    n_val = max(1, int(len(convs) * cfg.get("val_fraction", 0.05)))
    splits = {"val": convs[:n_val], "train": convs[n_val:]}
    meta = {"seq_len": seq_len, "tokenizer_sha256": tok.sha256, "splits": {}}
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    for split, items in splits.items():
        enc = []
        trunc = 0
        for c in items:
            e = fmt.encode_messages(c["messages"], c.get("tools"))
            ids = e.ids + [tok.eos_id]
            labels = e.labels + [-100]
            if len(ids) > seq_len + 1:
                trunc += 1  # demasiado larga para el contexto: se descarta
            enc.append((ids, labels))
        toks, mask = pack(enc, seq_len + 1, tok.pad_id)
        toks.astype(dtype).tofile(out / f"{split}_tokens.bin")
        mask.tofile(out / f"{split}_mask.bin")
        meta["splits"][split] = {"conversations": len(items), "rows": int(toks.shape[0]),
                                 "trained_tokens": int(mask.sum()), "dropped_too_long": trunc,
                                 "sha256_tokens": sha256_file(out / f"{split}_tokens.bin")}
        log(f"[sft] {split}: {len(items)} conversaciones -> {toks.shape[0]} filas, "
            f"{int(mask.sum()):,} tokens con pérdida, {trunc} descartadas por longitud")
        if items and trunc == len(items):
            raise ValueError(f"Ninguna conversación de '{split}' cabe en seq_len={seq_len}")
    meta["dtype"] = np.dtype(dtype).name
    write_json(out / "meta.json", meta)
    return meta


class _SFTRows:
    def __init__(self, data_dir: str | Path, split: str, seq_len: int):
        d = Path(data_dir)
        meta = json.loads((d / "meta.json").read_text())
        if meta["seq_len"] != seq_len:
            raise ValueError(f"Los datos SFT se prepararon con seq_len={meta['seq_len']}")
        self.toks = np.memmap(d / f"{split}_tokens.bin", dtype=meta["dtype"], mode="r").reshape(-1, seq_len + 1)
        self.mask = np.memmap(d / f"{split}_mask.bin", dtype=np.uint8, mode="r").reshape(-1, seq_len + 1)

    def batch(self, rows: list[int], device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(self.toks[rows].astype(np.int64))
        m = torch.from_numpy(self.mask[rows].astype(np.bool_))
        x = t[:, :-1]
        y = t[:, 1:].masked_fill(~m[:, 1:], -100)
        return x.to(device), y.to(device)


class SFTLoader:
    def __init__(self, data_dir, split, seq_len, batch_size, rank=0, world_size=1, seed=0, device="cpu"):
        self.rows = _SFTRows(data_dir, split, seq_len)
        self.n = self.rows.toks.shape[0]
        self.batch_size, self.rank, self.world_size, self.seed = batch_size, rank, world_size, seed
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.samples_seen = 0

    def state_dict(self):
        return {"samples_seen": self.samples_seen, "seq_len": self.seq_len}

    def load_state_dict(self, st):
        self.samples_seen = int(st["samples_seen"])

    def _row(self, s: int) -> int:
        epoch, j = divmod(s, self.n)
        if getattr(self, "_perm_epoch", None) != epoch:
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.n)
            self._perm_epoch = epoch
        return int(self._perm[j])

    def next_batch(self):
        base = self.samples_seen + self.rank * self.batch_size
        rows = [self._row(base + b) for b in range(self.batch_size)]
        self.samples_seen += self.batch_size * self.world_size
        return self.rows.batch(rows, self.device)

    def tokens_per_step(self) -> int:
        return self.batch_size * self.world_size * self.seq_len


class SFTEvalLoader:
    bytes_per_token: dict = {}

    def __init__(self, data_dir, split, seq_len, device="cpu"):
        self.rows = _SFTRows(data_dir, split, seq_len)
        self.device = torch.device(device)

    def domains(self):
        return ["sft"]

    def batches(self, domain, max_windows, batch_size):
        n = min(self.rows.toks.shape[0], max_windows)
        for i in range(0, n, batch_size):
            yield self.rows.batch(list(range(i, min(n, i + batch_size))), self.device)
