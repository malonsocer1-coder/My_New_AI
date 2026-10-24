"""Utilidades de E/S: escritura atómica, hashes y YAML/JSON."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Hash estable de un objeto JSON-serializable (para versionar configs)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Escribe en un temporal del mismo directorio y renombra: nunca deja archivos a medias."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
