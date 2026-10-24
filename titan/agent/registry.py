"""Registro de herramientas: validación, permisos, límites de tiempo y auditoría."""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import time
from pathlib import Path

from ..utils.io import append_jsonl
from .protocol import ToolContext, ToolResult, ToolSpec, truncate_content, validate


class ToolRegistry:
    def __init__(self, audit_log: str | None = None):
        self.tools: dict[str, ToolSpec] = {}
        self.audit_log = audit_log
        self._pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="titan-tool")

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.tools:
            raise ValueError(f"Herramienta duplicada: {spec.name}")
        self.tools[spec.name] = spec

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        return [t.schema() for n, t in self.tools.items() if names is None or n in names]

    def call(self, name: str, args, ctx: ToolContext) -> ToolResult:
        t0 = time.time()
        spec = self.tools.get(name)
        if spec is None:
            res = ToolResult(False, error=f"Herramienta desconocida: {name}. Disponibles: {sorted(self.tools)}")
            return self._audit(name, args, res, t0)
        if not isinstance(args, dict):
            return self._audit(name, args, ToolResult(False, error="Los argumentos deben ser un objeto JSON"), t0)
        errs = validate(spec.parameters, args)
        if errs:
            return self._audit(name, args, ToolResult(False, error="Argumentos inválidos: " + "; ".join(errs)), t0)
        detail = ", ".join(f"{k}={str(v)[:120]}" for k, v in args.items())
        for cap in spec.capabilities:
            d = ctx.permissions.check(cap, name, detail)
            if not d.allowed:
                return self._audit(name, args, ToolResult(False, error=f"Permiso '{cap}' denegado: {d.reason}"), t0)
        fut = self._pool.submit(spec.func, args, ctx)
        try:
            res = fut.result(timeout=spec.timeout_s)
        except cf.TimeoutError:
            res = ToolResult(False, error=f"Tiempo límite de {spec.timeout_s}s agotado")
        except Exception as e:  # el error se devuelve al modelo para que se corrija
            res = ToolResult(False, error=f"{type(e).__name__}: {e}")
        if res.ok:
            res.content, res.truncated = truncate_content(res.content)
        if spec.untrusted_output:
            res.meta["untrusted"] = True
        return self._audit(name, args, res, t0)

    def _audit(self, name, args, res: ToolResult, t0: float) -> ToolResult:
        if self.audit_log:
            append_jsonl(self.audit_log, {"unix": t0, "tool": name, "args": str(args)[:500], "ok": res.ok,
                                          "error": res.error, "ms": round((time.time() - t0) * 1000)})
        return res

    def load_plugin(self, path: str | Path) -> list[str]:
        """Carga un plugin explícitamente autorizado en la configuración.

        Un plugin es un archivo .py con ``def register(registry): ...``. Solo se
        cargan los listados por el usuario; nunca se ejecutan archivos sin pedirlo.
        """
        path = Path(path)
        spec = importlib.util.spec_from_file_location(f"titan_plugin_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        before = set(self.tools)
        mod.register(self)
        return sorted(set(self.tools) - before)


def default_registry(audit_log: str | None = None, include: list[str] | None = None) -> ToolRegistry:
    from .tools.builtin import builtin_tools
    reg = ToolRegistry(audit_log)
    for t in builtin_tools():
        if include is None or t.name in include:
            reg.register(t)
    return reg
