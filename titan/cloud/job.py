"""Trabajos de entrenamiento en GPUs alquiladas: planificar, aprobar, lanzar, vigilar y detener.

Flujo seguro:
  1. ``plan``   busca ofertas, estima tiempo y coste y los compara con el presupuesto.
  2. ``launch`` solo con ``approve=True`` (``titan cloud launch --approve``), solo si
                el coste estimado cabe en el presupuesto, y deja constancia en el
                libro de gastos (``logs/cloud_ledger.jsonl``).
  3. En el nodo, el guardián (``titan.cloud.watchdog``) detiene el entrenamiento
     con checkpoint y apaga la máquina al alcanzar el límite de gasto o de horas.
  4. ``watch`` aplica además un segundo límite desde el equipo local.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..model import ModelConfig
from ..train.config import load_run_config
from ..utils.io import append_jsonl, read_jsonl, read_yaml
from .estimate import GPUS, estimate
from .providers.base import ApprovalRequired, Instance, Offer
from .providers.impl import get_provider

LEDGER = Path("logs/cloud_ledger.jsonl")
BOOTSTRAP = Path(__file__).resolve().parents[2] / "scripts" / "cloud" / "bootstrap.sh"


@dataclass
class JobSpec:
    name: str
    provider: str
    gpu: str = "H100"
    num_gpus: int = 8
    image: str = "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel"
    disk_gb: int = 500
    repo: str = "https://github.com/malonsocer1-coder/My_New_AI.git"
    branch: str = "claude/adoring-hypatia-oc7bdz"
    train_config: str = ""
    data_config: str | None = None
    run_dir: str | None = None
    budget_usd: float = 0.0
    max_hours: float | None = None
    max_price_per_gpu_hour: float | None = None
    remote_storage: str | None = None   # destino rclone, p. ej. "r2:titan-ai/runs"
    env_passthrough: list[str] = field(default_factory=lambda: ["HF_TOKEN"])
    cloud_type: str = "SECURE"
    region: str | None = None
    ssh_key_names: list[str] = field(default_factory=list)
    mfu: float | None = None

    @classmethod
    def load(cls, path: str | Path) -> "JobSpec":
        d = read_yaml(path)
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"Claves desconocidas en el trabajo: {sorted(bad)}")
        return cls(**d)


def _training_size(spec: JobSpec) -> tuple[ModelConfig, float]:
    model_cfg, tcfg = load_run_config(spec.train_config)
    tokens = tcfg.micro_batch_size * tcfg.grad_accum_steps * tcfg.seq_len * tcfg.max_steps * spec.num_gpus
    return model_cfg, float(tokens)


def plan(spec: JobSpec, provider=None, offers: list[Offer] | None = None) -> dict:
    prov = provider or get_provider(spec.provider, dry_run=True)
    offers = offers if offers is not None else prov.list_offers(spec.gpu, spec.num_gpus)
    if spec.max_price_per_gpu_hour:
        offers = [o for o in offers if 0 < o.price_per_gpu_hour <= spec.max_price_per_gpu_hour] or offers
    priced = [o for o in offers if o.price_per_hour > 0]
    offer = min(priced, key=lambda o: o.price_per_hour) if priced else (offers[0] if offers else None)
    model_cfg, tokens = _training_size(spec)
    rate_gpu = offer.price_per_gpu_hour if offer and offer.price_per_hour > 0 else GPUS[spec.gpu].price_high
    est = estimate(model_cfg, tokens, spec.gpu, spec.num_gpus, spec.mfu, rate_gpu)
    hours = est["hours"]
    if spec.max_hours:
        hours = min(hours, spec.max_hours)
    cost = hours * rate_gpu * spec.num_gpus
    warnings = []
    if not priced:
        warnings.append("Sin precio confirmado de la oferta: se usó el precio alto de referencia")
    if cost > spec.budget_usd:
        warnings.append(f"El coste estimado (${cost:,.0f}) supera el presupuesto (${spec.budget_usd:,.0f})")
    if spec.max_hours and est["hours"] > spec.max_hours:
        warnings.append(f"El entrenamiento completo necesita ~{est['hours']:.0f} h; se detendrá a las {spec.max_hours} h "
                        "con checkpoint (reanudable)")
    if not est["fits"]:
        warnings.append(f"Memoria estimada {est['mem_per_gpu_gb']} GB/GPU: use parallel=fsdp y/o recálculo de activaciones")
    return {"job": spec.name, "provider": spec.provider, "offer": asdict(offer) if offer else None,
            "rate_per_gpu_hour": round(rate_gpu, 3), "estimated_hours": round(hours, 2),
            "estimated_cost_usd": round(cost, 2), "budget_usd": spec.budget_usd,
            "within_budget": cost <= spec.budget_usd, "training_estimate": est, "warnings": warnings}


def startup_script(spec: JobSpec, rate_per_hour: float) -> str:
    env = {
        "TITAN_REPO": spec.repo, "TITAN_BRANCH": spec.branch, "TITAN_TRAIN_CONFIG": spec.train_config,
        "TITAN_DATA_CONFIG": spec.data_config or "", "TITAN_RUN_DIR": spec.run_dir or f"runs/{spec.name}",
        "TITAN_BUDGET_USD": str(spec.budget_usd), "TITAN_RATE_USD_H": str(rate_per_hour),
        "TITAN_MAX_HOURS": str(spec.max_hours or ""), "TITAN_REMOTE": spec.remote_storage or "",
        "TITAN_PROVIDER": spec.provider,
    }
    exports = "\n".join(f"export {k}={_shq(v)}" for k, v in env.items())
    return exports + "\n" + BOOTSTRAP.read_text()


def _shq(v: str) -> str:
    return "'" + str(v).replace("'", "'\"'\"'") + "'"


def launch(spec: JobSpec, approve: bool = False, provider=None, ledger: Path = LEDGER) -> Instance:
    if not approve:
        raise ApprovalRequired("Lanzar un trabajo de pago requiere --approve tras revisar el plan")
    prov = provider or get_provider(spec.provider)
    p = plan(spec, prov)
    if not p["within_budget"]:
        raise ApprovalRequired("Plan fuera de presupuesto: " + "; ".join(p["warnings"]))
    offer = Offer(**p["offer"])
    env = {k: os.environ[k] for k in spec.env_passthrough if k in os.environ}
    # La clave del proveedor viaja al nodo solo para que el guardián pueda apagar la máquina
    if prov.api_key_env in os.environ:
        env[prov.api_key_env] = os.environ[prov.api_key_env]
    rate = p["rate_per_gpu_hour"] * spec.num_gpus
    inst = prov.create(offer, {"name": spec.name, "image": spec.image, "disk_gb": spec.disk_gb, "env": env,
                               "startup_script": startup_script(spec, rate), "cloud_type": spec.cloud_type,
                               "region": spec.region, "ssh_key_names": spec.ssh_key_names}, approved=True)
    append_jsonl(ledger, {"unix": time.time(), "event": "launch", "job": spec.name, "provider": spec.provider,
                          "instance_id": inst.instance_id, "rate_per_hour": rate,
                          "estimated_cost_usd": p["estimated_cost_usd"], "budget_usd": spec.budget_usd,
                          "approved_by_user": True, "env_keys_sent": sorted(env)})
    return inst


def stop(spec: JobSpec, instance_id: str, provider=None, ledger: Path = LEDGER) -> dict:
    prov = provider or get_provider(spec.provider)
    r = prov.terminate(instance_id)
    append_jsonl(ledger, {"unix": time.time(), "event": "terminate", "job": spec.name, "instance_id": instance_id})
    return r


def spent_so_far(ledger: Path = LEDGER, now: float | None = None) -> dict:
    """Gasto estimado por instancia según el libro (desde el lanzamiento hasta su terminación o ahora)."""
    if not Path(ledger).exists():
        return {}
    now = now or time.time()
    out: dict[str, dict] = {}
    for e in read_jsonl(ledger):
        iid = e.get("instance_id")
        if e["event"] == "launch":
            out[iid] = {"job": e["job"], "start": e["unix"], "rate": e["rate_per_hour"], "end": None,
                        "budget": e["budget_usd"]}
        elif e["event"] == "terminate" and iid in out:
            out[iid]["end"] = e["unix"]
    for v in out.values():
        v["hours"] = ((v["end"] or now) - v["start"]) / 3600
        v["cost_usd"] = round(v["hours"] * v["rate"], 2)
    return out


def watch(spec: JobSpec, instance_id: str, provider=None, poll_s: int = 300, ledger: Path = LEDGER,
          margin: float = 1.05, max_polls: int | None = None, log=print) -> str:
    """Segundo guardián, desde el equipo local: apaga la instancia si el gasto supera el presupuesto."""
    prov = provider or get_provider(spec.provider)
    polls = 0
    while True:
        st = prov.get(instance_id)
        spent = spent_so_far(ledger).get(instance_id, {}).get("cost_usd", 0.0)
        log(f"[cloud] {instance_id}: {st.status} | gasto estimado ${spent:,.2f} / ${spec.budget_usd:,.2f}")
        if st.status.lower() in ("exited", "terminated", "stopped", "dead"):
            return "finished"
        if spent > spec.budget_usd * margin:
            log("[cloud] presupuesto superado: apagando la instancia")
            stop(spec, instance_id, prov, ledger)
            return "budget_stop"
        polls += 1
        if max_polls is not None and polls >= max_polls:
            return "polling"
        time.sleep(poll_s)
