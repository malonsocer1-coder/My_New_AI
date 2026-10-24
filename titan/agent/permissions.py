"""Permisos explícitos del agente.

Principios:
  * Cada herramienta declara las capacidades que necesita.
  * La política la define el usuario en ``config/permissions.yaml``. Ninguna
    herramienta puede leer ni modificar ese archivo, y ningún texto procedente
    de páginas web, documentos o resultados de herramientas altera la política:
    el modelo solo puede *solicitar* acciones, nunca concederse permisos.
  * Las acciones costosas, irreversibles o sensibles (gastar dinero, crear
    recursos en la nube, borrar datos, publicar, usar credenciales) requieren
    autorización explícita en cada ocasión, aunque la política diga "allow".
  * Sin un aprobador interactivo, "ask" equivale a "deny".
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.io import append_jsonl, read_yaml

CAPABILITIES = {
    "file_read": "Leer archivos del espacio de trabajo",
    "file_write": "Crear o modificar archivos del espacio de trabajo",
    "file_delete": "Borrar archivos o datos",
    "code_exec": "Ejecutar código en el sandbox",
    "network": "Acceder a Internet",
    "credentials": "Usar credenciales o claves API",
    "install_deps": "Instalar dependencias",
    "external_tools": "Ejecutar programas externos",
    "cloud_resources": "Crear o modificar recursos en la nube",
    "spend_money": "Comprometer gasto económico",
    "publish": "Publicar cambios (git push, enviar mensajes, subir archivos)",
    "memory_read": "Leer la memoria persistente",
    "memory_write": "Guardar en la memoria persistente",
}

ALWAYS_ASK = frozenset({"file_delete", "credentials", "cloud_resources", "spend_money", "publish"})

DEFAULT_POLICY = {
    "file_read": "allow",
    "file_write": "ask",
    "file_delete": "ask",
    "code_exec": "ask",
    "network": "ask",
    "credentials": "ask",
    "install_deps": "ask",
    "external_tools": "ask",
    "cloud_resources": "ask",
    "spend_money": "ask",
    "publish": "ask",
    "memory_read": "allow",
    "memory_write": "ask",
}

LEVELS = ("allow", "ask", "deny")


@dataclass
class PermissionRequest:
    capability: str
    tool: str
    detail: str


@dataclass
class Decision:
    allowed: bool
    reason: str


Approver = Callable[[PermissionRequest], str]  # devuelve "allow_once" | "allow_session" | "deny"


@dataclass
class PermissionPolicy:
    rules: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_POLICY))
    approver: Approver | None = None
    audit_log: str | None = None
    _session: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: str | Path | None, approver: Approver | None = None,
             audit_log: str | None = None) -> "PermissionPolicy":
        rules = dict(DEFAULT_POLICY)
        if path and Path(path).exists():
            user = read_yaml(path).get("permissions", {})
            for k, v in user.items():
                if k not in CAPABILITIES:
                    raise ValueError(f"Capacidad desconocida en la política: {k}")
                if v not in LEVELS:
                    raise ValueError(f"Nivel inválido para {k}: {v}")
                rules[k] = v
        return cls(rules=rules, approver=approver, audit_log=audit_log)

    def effective_level(self, cap: str) -> str:
        level = self.rules.get(cap, "deny")
        if cap in ALWAYS_ASK and level == "allow":
            level = "ask"
        return level

    def check(self, cap: str, tool: str, detail: str = "") -> Decision:
        if cap not in CAPABILITIES:
            return self._log(Decision(False, f"capacidad desconocida: {cap}"), cap, tool, detail)
        level = self.effective_level(cap)
        if level == "deny":
            return self._log(Decision(False, "denegado por la política"), cap, tool, detail)
        if level == "allow":
            return self._log(Decision(True, "permitido por la política"), cap, tool, detail)
        if cap in self._session and cap not in ALWAYS_ASK:
            return self._log(Decision(True, "permitido en esta sesión"), cap, tool, detail)
        if self.approver is None:
            return self._log(Decision(False, "requiere aprobación y no hay aprobador interactivo"), cap, tool, detail)
        answer = self.approver(PermissionRequest(cap, tool, detail))
        if answer == "allow_session" and cap not in ALWAYS_ASK:
            self._session.add(cap)
            return self._log(Decision(True, "aprobado para la sesión"), cap, tool, detail)
        if answer in ("allow_once", "allow_session"):
            return self._log(Decision(True, "aprobado por el usuario"), cap, tool, detail)
        return self._log(Decision(False, "rechazado por el usuario"), cap, tool, detail)

    def _log(self, d: Decision, cap: str, tool: str, detail: str) -> Decision:
        if self.audit_log:
            append_jsonl(self.audit_log, {"unix": time.time(), "capability": cap, "tool": tool,
                                          "detail": detail[:500], "allowed": d.allowed, "reason": d.reason})
        return d


def console_approver(req: PermissionRequest) -> str:
    """Aprobador para la terminal."""
    print(f"\n[permiso] La herramienta '{req.tool}' solicita: {CAPABILITIES[req.capability]}")
    if req.detail:
        print(f"          Detalle: {req.detail[:400]}")
    extra = "" if req.capability in ALWAYS_ASK else " / s=sí para la sesión"
    ans = input(f"          ¿Autorizar? [y=sí una vez{extra} / N=no] ").strip().lower()
    if ans == "y":
        return "allow_once"
    if ans == "s" and req.capability not in ALWAYS_ASK:
        return "allow_session"
    return "deny"
