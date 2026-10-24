"""Memoria persistente con controles de privacidad (SQLite local).

  * Desactivada por defecto: el usuario la activa en la configuración.
  * Nunca guarda secretos: claves API, tokens y claves privadas se rechazan,
    y correos/teléfonos se enmascaran antes de guardar.
  * Caducidad configurable (``retention_days``), exportación completa y
    borrado total o selectivo por el usuario.
  * Todo queda en el disco del usuario (``D:\\TITAN_AI\\memory``); no se envía a ningún servicio.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from ..data.clean import _SECRET, scrub_pii

_WORD = re.compile(r"\w+", re.UNICODE)


class Memory:
    def __init__(self, path: str | Path, enabled: bool = False, retention_days: float | None = 365):
        self.enabled = enabled
        self.retention_days = retention_days
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL, text TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]', source TEXT NOT NULL DEFAULT 'agent')""")
        self.db.commit()
        self.purge_expired()

    def purge_expired(self) -> int:
        if not self.retention_days:
            return 0
        cut = time.time() - self.retention_days * 86400
        n = self.db.execute("DELETE FROM memories WHERE created < ?", (cut,)).rowcount
        self.db.commit()
        return n

    def save(self, text: str, tags: list[str] | None = None, source: str = "agent") -> dict:
        if not self.enabled:
            return {"ok": False, "error": "La memoria persistente está desactivada por el usuario."}
        if _SECRET.search(text):
            return {"ok": False, "error": "El texto parece contener una credencial; no se guarda."}
        clean = scrub_pii(text.strip())[:4000]
        cur = self.db.execute("INSERT INTO memories(created, text, tags, source) VALUES (?, ?, ?, ?)",
                              (time.time(), clean, json.dumps(tags or [], ensure_ascii=False), source))
        self.db.commit()
        return {"ok": True, "id": cur.lastrowid, "stored": clean}

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.enabled:
            return []
        q = set(w.lower() for w in _WORD.findall(query))
        rows = self.db.execute("SELECT id, created, text, tags FROM memories").fetchall()
        scored = []
        for rid, created, text, tags in rows:
            words = [w.lower() for w in _WORD.findall(text)]
            score = sum(1 for w in words if w in q) / (1 + len(words) ** 0.5)
            if score > 0:
                scored.append((score, {"id": rid, "created": created, "text": text, "tags": json.loads(tags)}))
        scored.sort(key=lambda t: -t[0])
        return [r for _, r in scored[:k]]

    def list(self) -> list[dict]:
        rows = self.db.execute("SELECT id, created, text, tags, source FROM memories ORDER BY id").fetchall()
        return [{"id": r[0], "created": r[1], "text": r[2], "tags": json.loads(r[3]), "source": r[4]} for r in rows]

    def delete(self, mem_id: int) -> bool:
        n = self.db.execute("DELETE FROM memories WHERE id = ?", (mem_id,)).rowcount
        self.db.commit()
        return n > 0

    def forget_all(self) -> int:
        n = self.db.execute("DELETE FROM memories").rowcount
        self.db.commit()
        return n

    def export(self, out: str | Path) -> int:
        items = self.list()
        Path(out).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(items)
