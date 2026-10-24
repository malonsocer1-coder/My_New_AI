"""Protocolo de herramientas: esquemas, validación de argumentos y resultados estructurados."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

MAX_RESULT_CHARS = 8000


@dataclass
class ToolResult:
    ok: bool
    content: Any = None
    error: str | None = None
    truncated: bool = False
    meta: dict = field(default_factory=dict)

    def to_message(self, name: str) -> dict:
        return {"role": "tool", "name": name, "ok": self.ok,
                "content": self.content if self.ok else {"error": self.error}}


@dataclass
class ToolContext:
    workspace: str
    permissions: Any  # PermissionPolicy
    memory: Any = None
    settings: dict = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    func: Callable[[dict, ToolContext], ToolResult]
    capabilities: list[str] = field(default_factory=list)
    timeout_s: float = 30.0
    untrusted_output: bool = False  # contenido externo (web, documentos)

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


_TYPES = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "array": list, "object": dict, "null": type(None),
}


def validate(schema: dict, value: Any, path: str = "$") -> list[str]:
    """Subconjunto de JSON Schema: type, properties, required, additionalProperties,
    enum, minimum/maximum, minLength/maxLength, items, minItems/maxItems."""
    errs: list[str] = []
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = False
        for tt in types:
            py = _TYPES[tt]
            if tt in ("integer", "number") and isinstance(value, bool):
                continue
            if isinstance(value, py):
                ok = True
                break
        if not ok:
            return [f"{path}: se esperaba {t}, se recibió {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: valor fuera de {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: menor que {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: mayor que {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: longitud < {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longitud > {schema['maxLength']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: menos de {schema['minItems']} elementos")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: más de {schema['maxItems']} elementos")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += validate(schema["items"], v, f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in value:
                errs.append(f"{path}: falta '{r}'")
        if schema.get("additionalProperties", True) is False:
            for k in value:
                if k not in props:
                    errs.append(f"{path}: propiedad no permitida '{k}'")
        for k, v in value.items():
            if k in props:
                errs += validate(props[k], v, f"{path}.{k}")
    return errs


def truncate_content(content: Any, limit: int = MAX_RESULT_CHARS) -> tuple[Any, bool]:
    s = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    if len(s) <= limit:
        return content, False
    cut = s[:limit] + f"\n…[truncado: {len(s) - limit} caracteres omitidos]"
    return cut, True
