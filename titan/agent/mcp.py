"""Cliente MCP (Model Context Protocol) por stdio.

Permite usar herramientas de servidores MCP compatibles como si fueran
herramientas de TITAN. Cada herramienta MCP requiere la capacidad
``external_tools`` (se pide autorización) y su salida se trata como no confiable.

Configuración (``config/agent.yaml``)::

    mcp_servers:
      - name: archivos
        command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "D:/TITAN_AI/app/workspace"]
"""

from __future__ import annotations

import json
import subprocess
import threading
from itertools import count

from .protocol import ToolResult, ToolSpec

PROTOCOL_VERSION = "2025-06-18"


class MCPClient:
    def __init__(self, name: str, command: list[str], env: dict | None = None, timeout_s: float = 60):
        self.name = name
        self.timeout_s = timeout_s
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env, text=True, encoding="utf-8", bufsize=1)
        self._ids = count(1)
        self._lock = threading.Lock()
        self.server_info = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "titan-ai", "version": "0.1.0"}})
        self._notify("notifications/initialized", {})

    def _send(self, msg: dict) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict) -> dict:
        with self._lock:
            rid = next(self._ids)
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"El servidor MCP '{self.name}' se cerró")
                msg = json.loads(line)
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError(msg["error"].get("message", str(msg["error"])))
                    return msg.get("result", {})

    def list_tools(self) -> list[dict]:
        return self._request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass

    def tool_specs(self) -> list[ToolSpec]:
        specs = []
        for t in self.list_tools():
            def run(args, ctx, _n=t["name"]):
                r = self.call_tool(_n, args)
                parts = [c.get("text", json.dumps(c)) for c in r.get("content", [])]
                return ToolResult(not r.get("isError", False), "\n".join(parts),
                                  error="\n".join(parts) if r.get("isError") else None)
            specs.append(ToolSpec(
                name=f"{self.name}.{t['name']}", description=t.get("description", ""),
                parameters=t.get("inputSchema", {"type": "object"}), func=run,
                capabilities=["external_tools"], timeout_s=self.timeout_s, untrusted_output=True))
        return specs
