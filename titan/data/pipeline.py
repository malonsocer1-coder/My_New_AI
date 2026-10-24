"""Pipeline reproducible de preparación de datos.

Fases (cada una deja artefactos verificables en ``output_dir``):
  1. Carga de fuentes con licencia declarada.
  2. Normalización, limpieza de PII, filtros de calidad e idioma.
  3. Deduplicación exacta y aproximada (MinHash-LSH) global.
  4. Descontaminación frente a los conjuntos de evaluación.
  5. Partición train/val determinista por hash del documento.
  6. Entrenamiento del tokenizador propio (si no existe).
  7. Tokenización a shards binarios + manifiesto con SHA-256.
  8. Ficha de datos (data_card.json) con estadísticas y licencias.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..tokenizer import TitanTokenizer, train_tokenizer
from ..utils.io import read_yaml, sha256_file, stable_hash, write_json
from .clean import FilterStats, QualityConfig, detect_language, normalize_text, quality_reason, scrub_pii
from .decontam import ContaminationIndex
from .dialect import classify_dialect
from .dedup import ExactDeduper, MinHashLSH, exact_key
from .sources import LICENSE_UNKNOWN, SourceSpec, iter_documents


@dataclass
class DataConfig:
    name: str
    output_dir: str
    sources: list[SourceSpec]
    cache_dir: str = "cache"
    val_fraction: float = 0.01
    seed: int = 1234
    tokenizer_dir: str | None = None
    vocab_size: int = 32768
    tokenizer_train_chars: int = 200_000_000
    shard_tokens: int = 100_000_000
    exact_dedup: bool = True
    minhash_dedup: bool = True
    minhash: dict[str, Any] = field(default_factory=lambda: {"num_perm": 128, "bands": 16, "ngram": 5})
    eval_files: list[str] = field(default_factory=list)
    decontam_ngram: int = 13
    allow_unknown_license: bool = False
    quality: dict[str, Any] = field(default_factory=dict)
    extra_domain_weights: dict[str, float] = field(default_factory=dict)  # p. ej. {"es_mx": 0.25}

    @classmethod
    def load(cls, path: str) -> "DataConfig":
        d = read_yaml(path)
        d["sources"] = [SourceSpec.from_dict(s) for s in d["sources"]]
        d["sources"] = [s for s in d["sources"] if s.enabled]
        return cls(**d)

    def fingerprint(self) -> str:
        d = dict(self.__dict__)
        d["sources"] = [s.__dict__ for s in self.sources]
        return stable_hash(d)


def _split_of(key: str, val_fraction: float) -> str:
    return "val" if int(key[:8], 16) % 100_000 < val_fraction * 100_000 else "train"


def _load_eval_texts(paths: list[str]) -> list[str]:
    texts = []
    for p in paths:
        for f in sorted(Path().glob(p)) if any(c in p for c in "*?[") else [Path(p)]:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for k in ("prompt", "question", "context", "answer", "text", "choices", "target"):
                        v = row.get(k)
                        if isinstance(v, str):
                            texts.append(v)
                        elif isinstance(v, list):
                            texts.extend(x for x in v if isinstance(x, str))
    return texts


# ----------------------------------------------------------------------
def clean_stage(cfg: DataConfig, log=print) -> dict:
    out = Path(cfg.output_dir)
    clean_dir = out / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    qcfg = QualityConfig(**cfg.quality)

    for s in cfg.sources:
        if s.license == LICENSE_UNKNOWN and not cfg.allow_unknown_license:
            raise ValueError(f"La fuente '{s.name}' no declara licencia; se rechaza por política de derechos de uso.")

    decon = ContaminationIndex(n=cfg.decontam_ngram)
    decon.add_eval_texts(_load_eval_texts(cfg.eval_files))
    log(f"[datos] índice de descontaminación: {len(decon)} n-gramas/textos de evaluación")

    exact = ExactDeduper() if cfg.exact_dedup else None
    lsh = MinHashLSH(**cfg.minhash) if cfg.minhash_dedup else None
    stats: dict[str, FilterStats] = defaultdict(FilterStats)
    langs: dict[str, Counter] = defaultdict(Counter)
    dialects: dict[str, Counter] = defaultdict(Counter)
    counts = defaultdict(lambda: {"docs": 0, "chars": 0, "bytes": 0})
    writers: dict[tuple[str, str], Any] = {}

    def writer(domain: str, split: str):
        k = (domain, split)
        if k not in writers:
            writers[k] = open(clean_dir / f"{domain}.{split}.jsonl", "w", encoding="utf-8")
        return writers[k]

    t0 = time.time()
    try:
        for spec in cfg.sources:
            st = stats[spec.name]
            for doc in iter_documents(spec, cfg.cache_dir):
                text = normalize_text(doc.text)
                if spec.scrub_pii:
                    text = scrub_pii(text)
                reason = quality_reason(text, spec.kind, qcfg)
                if reason is None and spec.kind == "prose" and spec.language not in ("multi", "unk"):
                    lang, _ = detect_language(text)
                    langs[spec.name][lang] += 1
                    if lang != spec.language:
                        reason = f"language_{lang}"
                if reason is None and exact is not None and exact.is_duplicate(text):
                    reason = "exact_duplicate"
                if reason is None and lsh is not None and not lsh.insert_if_new(text):
                    reason = "near_duplicate"
                if reason is None and decon.is_contaminated(text):
                    reason = "eval_contamination"
                st.add(reason)
                if reason is not None:
                    continue
                key = exact_key(text)
                split = _split_of(key, cfg.val_fraction)
                domain = spec.domain
                if spec.dialect_routing:
                    dia = classify_dialect(text)
                    dialects[spec.name][dia] += 1
                    if dia == "es-MX":
                        domain = f"{spec.domain}_mx"
                writer(domain, split).write(json.dumps(
                    {"text": text, "source": spec.name, "id": doc.doc_id}, ensure_ascii=False) + "\n")
                c = counts[(domain, split)]
                c["docs"] += 1
                c["chars"] += len(text)
                c["bytes"] += len(text.encode("utf-8"))
            log(f"[datos] {spec.name}: {st.to_dict()} ({time.time() - t0:.0f}s)")
    finally:
        for w in writers.values():
            w.close()

    return {
        "filter_stats": {k: v.to_dict() for k, v in stats.items()},
        "language_detection": {k: dict(v) for k, v in langs.items()},
        "dialects": {k: dict(v) for k, v in dialects.items()},
        "clean_counts": {f"{d}.{s}": v for (d, s), v in sorted(counts.items())},
    }


def _iter_clean_texts(clean_dir: Path, split: str, domains: list[str]):
    for d in domains:
        p = clean_dir / f"{d}.{split}.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                yield d, json.loads(line)["text"]


def tokenizer_stage(cfg: DataConfig, domains: list[str], log=print) -> TitanTokenizer:
    tok_dir = Path(cfg.tokenizer_dir or Path(cfg.output_dir) / "tokenizer")
    if (tok_dir / "tokenizer.json").exists():
        log(f"[tokenizador] reutilizando {tok_dir}")
        return TitanTokenizer.load(tok_dir)
    clean_dir = Path(cfg.output_dir) / "clean"
    # Muestra balanceada: el presupuesto de caracteres se reparte entre dominios
    per_domain = cfg.tokenizer_train_chars // max(1, len(domains))

    def sample():
        for d in domains:
            budget = per_domain
            for _, text in _iter_clean_texts(clean_dir, "train", [d]):
                if budget <= 0:
                    break
                budget -= len(text)
                yield text

    t0 = time.time()
    tok = train_tokenizer(sample(), cfg.vocab_size, tok_dir)
    log(f"[tokenizador] entrenado vocab={tok.vocab_size} en {time.time() - t0:.0f}s -> {tok_dir}")
    return tok


def tokenize_stage(cfg: DataConfig, tok: TitanTokenizer, domains: list[str], log=print) -> dict:
    out = Path(cfg.output_dir)
    tok_out = out / "tokens"
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    shards = []
    totals = {}
    for split in ("train", "val"):
        (tok_out / split).mkdir(parents=True, exist_ok=True)
        for d in domains:
            buf: list[np.ndarray] = []
            buf_len = 0
            idx = 0
            n_tokens = n_bytes = n_docs = 0

            def flush():
                nonlocal buf, buf_len, idx
                if not buf_len:
                    return
                arr = np.concatenate(buf).astype(dtype)
                path = tok_out / split / f"{d}_{idx:05d}.bin"
                arr.tofile(path)
                shards.append({"path": path.relative_to(out).as_posix(), "split": split, "domain": d,
                               "num_tokens": int(arr.size), "dtype": np.dtype(dtype).name,
                               "sha256": sha256_file(path)})
                buf, buf_len = [], 0
                idx += 1

            batch: list[str] = []

            def encode_batch():
                nonlocal buf_len, n_tokens, n_docs
                for ids in tok.encode_batch(batch):
                    ids.append(tok.eos_id)
                    a = np.asarray(ids, dtype=np.int64)
                    buf.append(a)
                    buf_len += a.size
                    n_tokens += a.size
                    n_docs += 1
                batch.clear()
                if buf_len >= cfg.shard_tokens:
                    flush()

            for _, text in _iter_clean_texts(out / "clean", split, [d]):
                batch.append(text)
                n_bytes += len(text.encode("utf-8"))
                if len(batch) >= 512:
                    encode_batch()
            if batch:
                encode_batch()
            flush()
            if n_docs:
                totals[f"{d}.{split}"] = {"docs": n_docs, "tokens": n_tokens, "bytes": n_bytes,
                                          "bytes_per_token": n_bytes / max(1, n_tokens - n_docs)}
                log(f"[tokens] {d}.{split}: {n_docs} docs, {n_tokens:,} tokens, "
                    f"{totals[f'{d}.{split}']['bytes_per_token']:.2f} bytes/token")
    return {"shards": shards, "totals": totals, "dtype": np.dtype(dtype).name}


def prepare(config_path: str, log=print) -> dict:
    cfg = DataConfig.load(config_path)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clean_info = clean_stage(cfg, log)
    domains = sorted({s.domain for s in cfg.sources} | {f"{s.domain}_mx" for s in cfg.sources if s.dialect_routing})
    domains = [d for d in domains if (out / "clean" / f"{d}.train.jsonl").exists()]
    tok = tokenizer_stage(cfg, domains, log)
    tok_info = tokenize_stage(cfg, tok, domains, log)
    weights = defaultdict(float)
    for s in cfg.sources:
        weights[s.domain] = max(weights[s.domain], s.weight)
    weights.update(cfg.extra_domain_weights)
    manifest = {
        "name": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "created_unix": int(time.time()),
        "tokenizer": {"path": tok.path, "sha256": tok.sha256, "vocab_size": tok.vocab_size,
                      "eos_id": tok.eos_id},
        "dtype": tok_info["dtype"],
        "domains": domains,
        "domain_weights": dict(weights),
        "shards": tok_info["shards"],
        "totals": tok_info["totals"],
    }
    write_json(out / "manifest.json", manifest)
    card = {
        "name": cfg.name,
        "sources": [{"name": s.name, "domain": s.domain, "license": s.license, "language": s.language,
                     "kind": s.kind, "loader": s.loader,
                     "origin": s.params.get("url") or s.params.get("path") or s.params.get("pattern")
                     or s.params.get("patterns") or s.params.get("generator")}
                    for s in cfg.sources],
        **clean_info,
        "token_totals": tok_info["totals"],
        "eval_files_for_decontamination": cfg.eval_files,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(out / "data_card.json", card)
    log(f"[datos] listo en {time.time() - t0:.0f}s -> {out / 'manifest.json'}")
    return manifest
