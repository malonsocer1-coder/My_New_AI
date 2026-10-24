"""Interfaz común de proveedores de GPU.

Reglas de seguridad aplicadas en todos los proveedores:
  * La clave API se lee de una variable de entorno del usuario; nunca se
    guarda en el repositorio, en registros ni en la configuración.
  * Las operaciones que crean recursos de pago exigen ``approved=True``, que
    solo se pasa tras la aprobación explícita del plan (``titan cloud launch --approve``).
  * ``dry_run`` construye y muestra la petición sin enviarla.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class ApprovalRequired(PermissionError):
    pass


@dataclass
class Offer:
    provider: str
    offer_id: str
    gpu: str
    num_gpus: int
    price_per_hour: float  # total de la máquina
    region: str = ""
    interconnect: str = ""
    reliability: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def price_per_gpu_hour(self) -> float:
        return self.price_per_hour / max(1, self.num_gpus)


@dataclass
class Instance:
    provider: str
    instance_id: str
    status: str
    price_per_hour: float | None = None
    ssh: str | None = None
    extra: dict = field(default_factory=dict)


class Provider:
    name = "base"
    api_key_env = ""
    base_url = ""

    def __init__(self, dry_run: bool = False, api_key: str | None = None, transport=None):
        self.dry_run = dry_run
        self._key = api_key
        self.transport = transport or self._urllib_transport
        self.sent: list[dict] = []

    @property
    def api_key(self) -> str:
        key = self._key or os.environ.get(self.api_key_env)
        if not key and not self.dry_run:
            raise RuntimeError(f"Defina la variable de entorno {self.api_key_env} con su clave de {self.name}")
        return key or "DRY-RUN"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "Accept": "application/json", "User-Agent": "titan-ai/0.1"}

    @staticmethod
    def _urllib_transport(method: str, url: str, headers: dict, body: dict | None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {e.read()[:500]!r}") from e
        return json.loads(raw) if raw else {}

    def request(self, method: str, path: str, body: dict | None = None, mutating: bool = False) -> Any:
        url = path if path.startswith("http") else self.base_url + path
        record = {"method": method, "url": url, "body": body}
        self.sent.append(record)
        if self.dry_run and mutating:
            return {"dry_run": True, **record}
        return self.transport(method, url, self._headers(), body)

    # -- a implementar por cada proveedor ---------------------------------
    def list_offers(self, gpu: str, num_gpus: int) -> list[Offer]:
        raise NotImplementedError

    def create(self, offer: Offer, spec: dict, approved: bool = False) -> Instance:
        raise NotImplementedError

    def get(self, instance_id: str) -> Instance:
        raise NotImplementedError

    def terminate(self, instance_id: str) -> dict:
        raise NotImplementedError

    def _require(self, approved: bool) -> None:
        if not approved and not self.dry_run:
            raise ApprovalRequired("Crear recursos de pago requiere aprobación explícita del plan (--approve)")
