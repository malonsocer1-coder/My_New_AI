"""Servidor local de TITAN con interfaz web (sin dependencias externas).

  * Escucha solo en 127.0.0.1 por defecto.
  * Protección contra CSRF y DNS rebinding: toda llamada a la API exige la
    cabecera ``X-Titan-Token`` (secreto aleatorio por arranque, incrustado en
    la página servida) y un ``Host`` local. Una web ajena no puede usar el agente.
  * Modo chat (texto en streaming) y modo agente (herramientas con permisos).
    Las solicitudes de permiso aparecen en la interfaz y el usuario decide.
  * Backends de generación: motor PyTorch de TITAN o llama.cpp (GGUF, CUDA).
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..agent.loop import Agent
from ..agent.memory import Memory
from ..agent.permissions import CAPABILITIES, PermissionPolicy, PermissionRequest
from ..agent.protocol import ToolContext
from ..agent.registry import default_registry
from ..utils.io import read_yaml

STATIC = Path(__file__).parent / "static"


class Approvals:
    """Puente entre el agente (que espera) y la interfaz (que decide)."""

    def __init__(self, timeout_s: float = 180):
        self.pending: dict[str, dict] = {}
        self.timeout_s = timeout_s
        self.lock = threading.Lock()
        self.events: "queue.Queue[dict] | None" = None

    def approver(self, req: PermissionRequest) -> str:
        rid = uuid.uuid4().hex[:12]
        ev = threading.Event()
        item = {"id": rid, "capability": req.capability, "description": CAPABILITIES[req.capability],
                "tool": req.tool, "detail": req.detail[:600], "event": ev, "decision": "deny"}
        with self.lock:
            self.pending[rid] = item
        if self.events is not None:
            self.events.put({"type": "permission_request", **{k: v for k, v in item.items() if k != "event"}})
        ev.wait(self.timeout_s)
        with self.lock:
            self.pending.pop(rid, None)
        return item["decision"]

    def decide(self, rid: str, decision: str) -> bool:
        with self.lock:
            item = self.pending.get(rid)
        if not item or decision not in ("allow_once", "allow_session", "deny"):
            return False
        item["decision"] = decision
        item["event"].set()
        return True


class TitanApp:
    def __init__(self, model_path: str, device: str = "auto", agent_config: str | None = None):
        cfg = read_yaml(agent_config) if agent_config and Path(agent_config).exists() else {}
        self.cfg = cfg
        backend = cfg.get("backend", "titan")
        if backend == "llamacpp" or str(model_path).endswith(".gguf"):
            from ..infer.llamacpp import load_llamacpp_engine
            self.engine = load_llamacpp_engine(model_path, cfg.get("tokenizer"), cfg.get("n_gpu_layers", -1))
        else:
            from ..infer.engine import load_engine
            self.engine = load_engine(model_path, device=device)
        self.model_path = str(model_path)
        self.token = secrets.token_urlsafe(24)
        self.approvals = Approvals()
        root = Path(cfg.get("root", "."))
        self.workspace = Path(cfg.get("workspace", root / "workspace"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        logs = Path(cfg.get("logs_dir", root / "logs"))
        self.policy = PermissionPolicy.load(cfg.get("permissions"), approver=self.approvals.approver,
                                            audit_log=str(logs / "permissions_audit.jsonl"))
        mem_cfg = cfg.get("memory", {})
        self.memory = Memory(mem_cfg.get("path", root / "memory" / "memory.db"), enabled=mem_cfg.get("enabled", False),
                             retention_days=mem_cfg.get("retention_days", 365))
        self.registry = default_registry(audit_log=str(logs / "tools_audit.jsonl"), include=cfg.get("tools"))
        for plugin in cfg.get("plugins", []):
            self.registry.load_plugin(plugin)
        self.mcp_clients = []
        for srv in cfg.get("mcp_servers", []):
            from ..agent.mcp import MCPClient
            client = MCPClient(srv["name"], srv["command"])
            for spec in client.tool_specs():
                self.registry.register(spec)
            self.mcp_clients.append(client)
        self.ctx = ToolContext(workspace=str(self.workspace), permissions=self.policy, memory=self.memory,
                               settings=cfg.get("settings", {}))
        self.gen_lock = threading.Lock()

    def info(self) -> dict:
        m = getattr(self.engine, "model", None)
        return {"model": self.model_path, "params": m.num_params() if m is not None else None,
                "max_seq_len": self.engine.max_seq_len if hasattr(self.engine, "max_seq_len") else m.cfg.max_seq_len,
                "tools": [t["name"] for t in self.registry.schemas()], "workspace": str(self.workspace),
                "permissions": {k: self.policy.effective_level(k) for k in CAPABILITIES},
                "memory_enabled": self.memory.enabled}

    def stream_chat(self, body: dict, out: "queue.Queue[dict]") -> None:
        msgs = [m for m in body.get("messages", []) if m.get("role") in ("user", "assistant", "system")]
        sampling = {"temperature": float(body.get("temperature", 0.7)),
                    "max_new_tokens": min(int(body.get("max_new_tokens", 400)), 2048)}
        try:
            with self.gen_lock:
                if body.get("mode") == "agent":
                    self.approvals.events = out
                    agent = Agent(self.engine, self.registry, self.ctx, max_steps=int(self.cfg.get("max_steps", 8)),
                                  sampling=sampling)
                    res = agent.run(msgs, on_event=lambda e: out.put(_public_event(e)))
                    out.put({"type": "final", "content": res.final, "stop_reason": res.stop_reason})
                else:
                    fmt = self.engine.chat_format
                    ids = fmt.prompt_ids(msgs)
                    gen = []
                    shown = ""
                    for i in self.engine.stream_ids(ids, stop_ids=[fmt.im_end, self.engine.tok.eos_id], **sampling):
                        gen.append(i)
                        text = self.engine.tok.decode(gen, skip_special=True)
                        if not text.endswith("�"):
                            out.put({"type": "token", "text": text[len(shown):]})
                            shown = text
                    out.put({"type": "final", "content": shown})
        except Exception as e:  # el error llega a la interfaz en lugar de colgar la conexión
            out.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            self.approvals.events = None
            out.put({"type": "done"})


def _public_event(e: dict) -> dict:
    if e.get("type") == "assistant":
        return {"type": "assistant", "content": e.get("content"), "tool_calls": e.get("tool_calls")}
    if e.get("type") == "tool":
        c = e.get("content")
        return {"type": "tool_result", "name": e.get("name"), "ok": e.get("ok"),
                "content": c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)[:3000]}
    return e


def make_handler(app: TitanApp):
    allowed_hosts = {"127.0.0.1", "localhost", "[::1]"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "TITAN/0.1"

        def log_message(self, fmt, *args):  # sin ruido en consola
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in allowed_hosts

        def _auth(self) -> bool:
            return self._host_ok() and secrets.compare_digest(self.headers.get("X-Titan-Token", ""), app.token)

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "host no permitido"})
            if self.path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TITAN_TOKEN__", app.token)
                data = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; "
                                 "style-src 'unsafe-inline'; connect-src 'self'")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/info":
                if not self._auth():
                    return self._json(401, {"error": "token inválido"})
                self._json(200, app.info())
            else:
                self._json(404, {"error": "no encontrado"})

        def do_POST(self):
            if not self._auth():
                return self._json(401, {"error": "token inválido"})
            n = int(self.headers.get("Content-Length", "0"))
            if n > 1_000_000:
                return self._json(413, {"error": "petición demasiado grande"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "JSON inválido"})
            if self.path == "/api/approve":
                ok = app.approvals.decide(str(body.get("id")), str(body.get("decision")))
                return self._json(200 if ok else 404, {"ok": ok})
            if self.path == "/api/chat":
                q: "queue.Queue[dict]" = queue.Queue()
                threading.Thread(target=app.stream_chat, args=(body, q), daemon=True).start()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while True:
                    ev = q.get()
                    self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                    if ev["type"] == "done":
                        break
                return
            self._json(404, {"error": "no encontrado"})

    return Handler


def serve(model_path: str, host: str = "127.0.0.1", port: int = 8765, device: str = "auto",
          agent_config: str | None = None) -> None:
    app = TitanApp(model_path, device, agent_config)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("[aviso] El servidor escuchará fuera de localhost: cualquiera en tu red podría usarlo.")
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"TITAN listo en http://{host}:{port}  (Ctrl+C para salir)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
