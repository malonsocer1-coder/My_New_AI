"""Bucle del agente: planificar, llamar herramientas, interpretar resultados y corregir errores.

Cada iteración genera un turno del asistente. Si contiene llamadas a
herramientas, se validan y ejecutan (con permisos) y sus resultados se añaden
como mensajes ``tool``; si una llamada está mal formada o falla, el error se
devuelve al modelo para que lo corrija. El bucle termina con una respuesta sin
llamadas, al agotar ``max_steps`` o al superar ``max_seconds``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .protocol import ToolContext
from .registry import ToolRegistry


@dataclass
class AgentResult:
    final: str
    messages: list[dict]
    trace: list[dict] = field(default_factory=list)
    stop_reason: str = "answered"
    tool_calls: int = 0


class Agent:
    def __init__(self, engine, registry: ToolRegistry, ctx: ToolContext, max_steps: int = 8,
                 max_seconds: float = 300, tool_names: list[str] | None = None, system: str | None = None,
                 sampling: dict | None = None):
        self.engine, self.registry, self.ctx = engine, registry, ctx
        self.max_steps, self.max_seconds = max_steps, max_seconds
        self.tool_names, self.system = tool_names, system
        self.sampling = sampling or {"temperature": 0.2, "max_new_tokens": 512}

    def run(self, messages: list[dict], on_event: Callable[[dict], None] | None = None) -> AgentResult:
        msgs = list(messages)
        tools = self.registry.schemas(self.tool_names)
        trace: list[dict] = []
        n_calls = 0
        t0 = time.time()
        emit = on_event or (lambda e: None)
        for step in range(self.max_steps):
            if time.time() - t0 > self.max_seconds:
                return AgentResult("Se agotó el tiempo máximo de la tarea.", msgs, trace, "time_limit", n_calls)
            try:
                turn = self.engine.chat_turn(msgs, tools=tools, system=self.system, **self.sampling)
            except ValueError as e:  # ContextOverflow: no cabe más historial
                return AgentResult(f"No puedo continuar: {e}", msgs, trace, "context_overflow", n_calls)
            emit({"type": "assistant", "step": step, **turn})
            assistant = {"role": "assistant", "content": turn["content"], "tool_calls": turn["tool_calls"]}
            msgs.append(assistant)
            if turn["call_errors"]:
                for err in turn["call_errors"]:
                    msg = {"role": "tool", "name": "invalid_call", "ok": False,
                           "content": {"error": f"Llamada mal formada ({err['error']}). Emite JSON válido "
                                                '{"name": ..., "arguments": {...}}.'}}
                    msgs.append(msg)
                    trace.append({"step": step, "invalid_call": err})
                    emit({"type": "tool", **msg})
                continue
            if not turn["tool_calls"]:
                return AgentResult(turn["content"], msgs, trace, "answered", n_calls)
            for call in turn["tool_calls"]:
                n_calls += 1
                res = self.registry.call(call["name"], call.get("arguments", {}), self.ctx)
                msg = res.to_message(call["name"])
                msgs.append(msg)
                trace.append({"step": step, "call": call, "ok": res.ok, "error": res.error})
                emit({"type": "tool", **msg})
        return AgentResult("No llegué a una respuesta final dentro del límite de pasos.", msgs, trace,
                           "max_steps", n_calls)
