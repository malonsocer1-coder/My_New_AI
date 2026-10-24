"""Registro de fuentes de datos con licencia y cargadores.

Cada fuente declara dominio, idioma, tipo (prosa/código) y **licencia**. El
pipeline rechaza fuentes sin licencia declarada salvo que se autorice
explícitamente (``allow_unknown_license``), y escribe una ficha de datos
(DATA_CARD) con lo que realmente entró en el entrenamiento.

Cargadores disponibles:
  * ``zip_url``    archivo ZIP remoto (verificado por SHA-256) con miembros por patrón.
  * ``local_glob`` archivos locales por patrón glob.
  * ``jsonl``      archivos JSONL locales con un campo de texto.
  * ``hf``         conjuntos de Hugging Face en streaming (para los nodos en la nube).
  * ``synthetic``  generadores propios verificados (``titan.sft.synthetic``).
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import re
import sysconfig
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.io import sha256_file

LICENSE_UNKNOWN = "unknown"


@dataclass
class SourceSpec:
    name: str
    domain: str
    loader: str
    license: str = LICENSE_UNKNOWN
    language: str = "multi"
    kind: str = "prose"  # prose | code
    params: dict[str, Any] = field(default_factory=dict)
    max_docs: int | None = None
    detokenize: bool = False
    doc_chars: int | None = None  # trocear textos largos en documentos de ~N caracteres
    weight: float = 1.0  # peso de muestreo del dominio en el entrenamiento
    scrub_pii: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "SourceSpec":
        return cls(**d)


@dataclass
class Document:
    text: str
    source: str
    domain: str
    kind: str
    doc_id: str


# ----------------------------------------------------------------------
_DETOK_RULES = [
    (re.compile(r" ([,.;:!?%)\]}»])"), r"\1"),
    (re.compile(r"([(\[{«¿¡]) "), r"\1"),
    (re.compile(r" ' (s|t|re|ve|m|ll|d)\b"), r"'\1"),
    (re.compile(r" (n't)\b"), r"\1"),
    (re.compile(r'" ([^"]*?) "'), r'"\1"'),
    (re.compile(r" {2,}"), " "),
]


def detokenize(text: str) -> str:
    """Revierte la tokenización por espacios de corpus como Europarl."""
    for pat, rep in _DETOK_RULES:
        text = pat.sub(rep, text)
    return text


def chunk_text(text: str, target_chars: int) -> list[str]:
    """Agrupa párrafos/líneas en documentos de ~target_chars sin cortar líneas."""
    parts = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    docs, cur, size = [], [], 0
    for p in parts:
        cur.append(p)
        size += len(p) + 1
        if size >= target_chars:
            docs.append("\n".join(cur))
            cur, size = [], 0
    if cur:
        if docs and size < target_chars // 4:
            docs[-1] += "\n" + "\n".join(cur)
        else:
            docs.append("\n".join(cur))
    return docs


def _decode(raw: bytes, encoding: str | None) -> str:
    if encoding:
        return raw.decode(encoding, errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def download(url: str, dest: Path, sha256: str | None = None) -> Path:
    """Descarga con caché y verificación de integridad."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    got = sha256_file(tmp)
    if sha256 and got != sha256:
        tmp.unlink()
        raise ValueError(f"SHA-256 no coincide para {url}: {got} != {sha256}")
    os.replace(tmp, dest)
    return dest


# ----------------------------------------------------------------------
# Cargadores: cada uno produce (texto_crudo, identificador)
# ----------------------------------------------------------------------
def _iter_zip_url(spec: SourceSpec, cache_dir: Path) -> Iterator[tuple[str, str]]:
    p = spec.params
    fname = p.get("filename") or p["url"].rsplit("/", 1)[-1]
    path = download(p["url"], cache_dir / "downloads" / fname, p.get("sha256"))
    patterns = p.get("members", ["*"])
    if isinstance(patterns, str):
        patterns = [patterns]
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.endswith("/") or not any(fnmatch.fnmatch(name, pat) for pat in patterns):
                continue
            yield _decode(z.read(name), p.get("encoding")), name


def _iter_local_glob(spec: SourceSpec, cache_dir: Path) -> Iterator[tuple[str, str]]:
    p = spec.params
    pats = p["patterns"] if "patterns" in p else [p["pattern"]]
    excludes = p.get("exclude", [])
    max_bytes = p.get("max_file_bytes", 1_000_000)
    files: set[str] = set()
    stdlib = sysconfig.get_paths()["stdlib"]
    for pat in pats:
        files.update(glob.glob(os.path.expanduser(pat.replace("{stdlib}", stdlib)), recursive=True))
    for path in sorted(files):
        if any(fnmatch.fnmatch(path, ex.replace("{stdlib}", stdlib)) for ex in excludes):
            continue
        if not os.path.isfile(path) or os.path.getsize(path) > max_bytes:
            continue
        with open(path, "rb") as f:
            yield _decode(f.read(), p.get("encoding")), path


def _iter_jsonl(spec: SourceSpec, cache_dir: Path) -> Iterator[tuple[str, str]]:
    p = spec.params
    field_name = p.get("text_field", "text")
    for path in sorted(glob.glob(p["pattern"])):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    yield json.loads(line)[field_name], f"{path}:{i}"


def _iter_hf(spec: SourceSpec, cache_dir: Path) -> Iterator[tuple[str, str]]:
    try:
        from datasets import load_dataset  # dependencia opcional, solo en la nube
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("El cargador 'hf' requiere `pip install datasets` y acceso a huggingface.co") from e
    p = spec.params
    ds = load_dataset(p["path"], name=p.get("name"), split=p.get("split", "train"), streaming=True)
    field_name = p.get("text_field", "text")
    for i, row in enumerate(ds):
        yield row[field_name], f"{p['path']}:{i}"


def _iter_synthetic(spec: SourceSpec, cache_dir: Path) -> Iterator[tuple[str, str]]:
    from ..sft import synthetic
    gen = getattr(synthetic, spec.params["generator"])
    for i, text in enumerate(gen(**spec.params.get("kwargs", {}))):
        yield text, f"{spec.name}:{i}"


LOADERS = {
    "zip_url": _iter_zip_url,
    "local_glob": _iter_local_glob,
    "jsonl": _iter_jsonl,
    "hf": _iter_hf,
    "synthetic": _iter_synthetic,
}


def iter_documents(spec: SourceSpec, cache_dir: str | os.PathLike) -> Iterator[Document]:
    if spec.loader not in LOADERS:
        raise ValueError(f"Cargador desconocido: {spec.loader}")
    n = 0
    for raw, ident in LOADERS[spec.loader](spec, Path(cache_dir)):
        text = detokenize(raw) if spec.detokenize else raw
        pieces = chunk_text(text, spec.doc_chars) if spec.doc_chars else [text]
        for j, piece in enumerate(pieces):
            yield Document(piece, spec.name, spec.domain, spec.kind, f"{ident}#{j}")
            n += 1
            if spec.max_docs and n >= spec.max_docs:
                return
