"""Herramientas integradas del agente TITAN.

Cada herramienta declara sus capacidades; el registro comprueba los permisos
antes de ejecutarla. Las rutas de archivo se confinan al espacio de trabajo:
no se puede salir con ``..`` ni con enlaces simbólicos, y las carpetas de
configuración, credenciales y memoria quedan fuera de su alcance.
"""

from __future__ import annotations

import ast
import ipaddress
import json
import math
import operator
import os
import re
import shutil
import socket
import subprocess
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from ..documents import read_document, summarize_table
from ..knowledge import KnowledgeBase
from ..protocol import ToolContext, ToolResult, ToolSpec
from ..sandbox import run_python

# ----------------------------------------------------------------------
# Rutas confinadas
# ----------------------------------------------------------------------
PROTECTED_NAMES = {"permissions.yaml", ".env", "credentials", "secrets", "memory.db", ".git"}


class PathError(ValueError):
    pass


def safe_path(ctx: ToolContext, rel: str, must_exist: bool = False) -> Path:
    root = Path(ctx.workspace).resolve()
    if os.path.isabs(rel) or rel.startswith(("\\", "/")) or re.match(r"^[a-zA-Z]:", rel):
        raise PathError("Solo se permiten rutas relativas al espacio de trabajo")
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise PathError("La ruta sale del espacio de trabajo")
    if any(part.lower() in PROTECTED_NAMES for part in p.relative_to(root).parts):
        raise PathError("Ruta protegida")
    if must_exist and not p.exists():
        raise PathError(f"No existe: {rel}")
    return p


# ----------------------------------------------------------------------
# Calculadora segura (sin eval)
# ----------------------------------------------------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUN = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
        "log10": math.log10, "exp": math.exp, "abs": abs, "round": round, "floor": math.floor,
        "ceil": math.ceil, "factorial": math.factorial}
_CONST = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float | int:
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 1000 or abs(a) > 1e6):
                raise ValueError("exponente demasiado grande")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords:
            args = [ev(a) for a in n.args]
            if n.func.id == "factorial" and (len(args) != 1 or args[0] > 1000):
                raise ValueError("factorial fuera de rango")
            return _FUN[n.func.id](*args)
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        raise ValueError(f"expresión no permitida: {ast.dump(n)[:60]}")

    if len(expr) > 500:
        raise ValueError("expresión demasiado larga")
    v = ev(ast.parse(expr, mode="eval"))
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        v = int(v)
    return v


def _calculator(args, ctx):
    try:
        return ToolResult(True, {"expression": args["expression"], "result": safe_eval(args["expression"])})
    except Exception as e:
        return ToolResult(False, error=f"No se pudo calcular: {e}")


# ----------------------------------------------------------------------
# Archivos
# ----------------------------------------------------------------------
def _list_dir(args, ctx):
    p = safe_path(ctx, args.get("path", "."), must_exist=True)
    items = []
    for c in sorted(p.iterdir()):
        if c.name.lower() in PROTECTED_NAMES:
            continue
        items.append({"name": c.name, "type": "dir" if c.is_dir() else "file",
                      "bytes": c.stat().st_size if c.is_file() else None})
    return ToolResult(True, items[:500])


def _read_file(args, ctx):
    p = safe_path(ctx, args["path"], must_exist=True)
    if p.is_dir():
        return ToolResult(False, error="Es un directorio; use list_dir")
    return ToolResult(True, read_document(p, max_chars=args.get("max_chars", 20000)))


def _write_file(args, ctx):
    p = safe_path(ctx, args["path"])
    if p.exists() and not args.get("overwrite", False):
        return ToolResult(False, error="El archivo ya existe; use overwrite=true para reemplazarlo")
    if p.exists():
        d = ctx.permissions.check("file_delete", "write_file", f"sobrescribir {args['path']}")
        if not d.allowed:
            return ToolResult(False, error=f"Permiso denegado para sobrescribir: {d.reason}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"], encoding="utf-8")
    return ToolResult(True, {"path": args["path"], "bytes": p.stat().st_size})


def _delete_file(args, ctx):
    p = safe_path(ctx, args["path"], must_exist=True)
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return ToolResult(True, {"deleted": args["path"]})


def _table_summary(args, ctx):
    return ToolResult(True, summarize_table(safe_path(ctx, args["path"], must_exist=True)))


# ----------------------------------------------------------------------
# Código
# ----------------------------------------------------------------------
def _python_exec(args, ctx):
    timeout = min(float(args.get("timeout_s", 10)), 60.0)
    res = run_python(args["code"], timeout_s=timeout, backend=ctx.settings.get("sandbox_backend", "auto"))
    ok = res.returncode == 0 and not res.timed_out
    content = {"stdout": res.stdout, "stderr": res.stderr[-4000:], "returncode": res.returncode,
               "timed_out": res.timed_out, "sandbox": res.backend}
    return ToolResult(ok, content, error=None if ok else (res.stderr[-2000:] or "tiempo agotado"))


# ----------------------------------------------------------------------
# Web (contenido no confiable)
# ----------------------------------------------------------------------
class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _check_public_host(url: str) -> None:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Solo URLs http(s)")
    for info in socket.getaddrinfo(u.hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("Destino en red privada o local bloqueado (protección SSRF)")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Revalida cada redirección: una web pública no puede redirigir a la red local."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _web_fetch(args, ctx):
    url = args["url"]
    _check_public_host(url)
    req = urllib.request.Request(url, headers={"User-Agent": "TITAN-AI/0.1 (+agente local)"})
    opener = urllib.request.build_opener(_SafeRedirect)
    with opener.open(req, timeout=20) as r:
        raw = r.read(2_000_000)
        ctype = r.headers.get("Content-Type", "")
    text = raw.decode("utf-8", "replace")
    if "html" in ctype or text.lstrip().startswith("<"):
        ex = _TextExtractor()
        ex.feed(text)
        text = re.sub(r"\n\s*\n+", "\n\n", "".join(ex.parts)).strip()
    return ToolResult(True, {"url": url, "text": text[: args.get("max_chars", 12000)]},
                      meta={"untrusted": True})


def _web_search(args, ctx):
    cfg = ctx.settings.get("web_search", {})
    backend = cfg.get("backend")
    q = args["query"]
    if backend == "searxng":
        url = cfg["url"].rstrip("/") + "/search?" + urllib.parse.urlencode({"q": q, "format": "json"})
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
        hits = [{"title": h.get("title"), "url": h.get("url"), "snippet": h.get("content")}
                for h in data.get("results", [])[: args.get("k", 5)]]
        return ToolResult(True, hits, meta={"untrusted": True})
    if backend == "brave":
        d = ctx.permissions.check("credentials", "web_search", "usar BRAVE_API_KEY")
        if not d.allowed:
            return ToolResult(False, error=f"Sin permiso para usar la clave: {d.reason}")
        key = os.environ.get("BRAVE_API_KEY")
        if not key:
            return ToolResult(False, error="Defina la variable de entorno BRAVE_API_KEY")
        req = urllib.request.Request("https://api.search.brave.com/res/v1/web/search?" +
                                     urllib.parse.urlencode({"q": q, "count": args.get("k", 5)}),
                                     headers={"X-Subscription-Token": key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        hits = [{"title": h.get("title"), "url": h.get("url"), "snippet": h.get("description")}
                for h in data.get("web", {}).get("results", [])]
        return ToolResult(True, hits, meta={"untrusted": True})
    return ToolResult(False, error="Búsqueda web no configurada (settings.web_search.backend: searxng | brave)")


# ----------------------------------------------------------------------
# Conocimiento local y memoria
# ----------------------------------------------------------------------
_KB_CACHE: dict[str, KnowledgeBase] = {}


def _kb_search(args, ctx):
    root = str(Path(ctx.workspace) / ctx.settings.get("knowledge_dir", "knowledge"))
    kb = _KB_CACHE.setdefault(root, KnowledgeBase(root))
    return ToolResult(True, kb.search(args["query"], int(args.get("k", 3))), meta={"untrusted": True})


def _memory_save(args, ctx):
    if ctx.memory is None:
        return ToolResult(False, error="Memoria no configurada")
    r = ctx.memory.save(args["text"], args.get("tags"))
    return ToolResult(r["ok"], r if r["ok"] else None, error=r.get("error"))


def _memory_search(args, ctx):
    if ctx.memory is None:
        return ToolResult(False, error="Memoria no configurada")
    return ToolResult(True, ctx.memory.search(args["query"], int(args.get("k", 5))))


# ----------------------------------------------------------------------
# Módulos auxiliares (visión, voz) mediante programas externos configurados
# ----------------------------------------------------------------------
def _external(kind: str):
    def run(args, ctx):
        cfg = ctx.settings.get("auxiliary", {}).get(kind)
        if not cfg:
            return ToolResult(False, error=f"Módulo auxiliar '{kind}' no configurado (settings.auxiliary.{kind}.command)")
        p = safe_path(ctx, args["path"], must_exist=True)
        cmd = [c.replace("{input}", str(p)).replace("{prompt}", args.get("prompt", "")) for c in cfg["command"]]
        r = subprocess.run(cmd, capture_output=True, timeout=cfg.get("timeout_s", 300))
        out = r.stdout.decode("utf-8", "replace").strip()
        return ToolResult(r.returncode == 0, out[:8000], error=None if r.returncode == 0 else r.stderr.decode()[-2000:])
    return run


def _s(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def builtin_tools() -> list[ToolSpec]:
    S, I, N, B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
    return [
        ToolSpec("calculator", "Evalúa una expresión aritmética exacta (+ - * / ** sqrt log ...).",
                 _s({"expression": {**S, "maxLength": 500}}, ["expression"]), _calculator, [], 5),
        ToolSpec("list_dir", "Lista archivos y carpetas del espacio de trabajo.",
                 _s({"path": S}, []), _list_dir, ["file_read"], 10),
        ToolSpec("read_file", "Lee un archivo (texto, PDF, Word, Excel, PowerPoint, CSV) del espacio de trabajo.",
                 _s({"path": S, "max_chars": {**I, "minimum": 100, "maximum": 100000}}, ["path"]),
                 _read_file, ["file_read"], 30, untrusted_output=True),
        ToolSpec("write_file", "Crea un archivo de texto en el espacio de trabajo.",
                 _s({"path": S, "content": S, "overwrite": B}, ["path", "content"]), _write_file, ["file_write"], 10),
        ToolSpec("delete_file", "Borra un archivo o carpeta del espacio de trabajo (siempre pide autorización).",
                 _s({"path": S}, ["path"]), _delete_file, ["file_delete"], 10),
        ToolSpec("table_summary", "Resume las columnas de un CSV (tipos, mínimos, máximos, medias).",
                 _s({"path": S}, ["path"]), _table_summary, ["file_read"], 30),
        ToolSpec("python_exec", "Ejecuta código Python en un sandbox aislado y devuelve stdout/stderr.",
                 _s({"code": {**S, "maxLength": 20000}, "timeout_s": {**N, "minimum": 1, "maximum": 60}}, ["code"]),
                 _python_exec, ["code_exec"], 70),
        ToolSpec("web_fetch", "Descarga una página web pública y devuelve su texto (contenido no confiable).",
                 _s({"url": S, "max_chars": I}, ["url"]), _web_fetch, ["network"], 30, untrusted_output=True),
        ToolSpec("web_search", "Busca en la web con el motor configurado.",
                 _s({"query": S, "k": {**I, "minimum": 1, "maximum": 10}}, ["query"]), _web_search, ["network"], 30,
                 untrusted_output=True),
        ToolSpec("kb_search", "Busca pasajes relevantes en la base de conocimiento local (carpeta knowledge).",
                 _s({"query": S, "k": {**I, "minimum": 1, "maximum": 10}}, ["query"]), _kb_search, ["file_read"], 30,
                 untrusted_output=True),
        ToolSpec("memory_save", "Guarda un dato en la memoria persistente (si el usuario la activó).",
                 _s({"text": {**S, "maxLength": 4000}, "tags": {"type": "array", "items": S}}, ["text"]),
                 _memory_save, ["memory_write"], 5),
        ToolSpec("memory_search", "Busca en la memoria persistente.",
                 _s({"query": S, "k": I}, ["query"]), _memory_search, ["memory_read"], 5),
        ToolSpec("describe_image", "Describe una imagen con el módulo de visión auxiliar configurado.",
                 _s({"path": S, "prompt": S}, ["path"]), _external("vision"), ["external_tools"], 300),
        ToolSpec("transcribe_audio", "Transcribe audio con el módulo de voz auxiliar configurado.",
                 _s({"path": S}, ["path"]), _external("speech"), ["external_tools"], 600),
    ]
