"""Guardián que se ejecuta en el nodo GPU junto al entrenamiento.

Cada minuto:
  * Gasto: horas transcurridas × tarifa de la máquina. Al 90 % avisa; al 100 %
    envía SIGTERM al entrenamiento (que guarda checkpoint), sincroniza y apaga la máquina.
  * Límite de horas (``--max-hours``): igual que el gasto.
  * Latido: si ``heartbeat.json`` no se actualiza en ``--stale-min`` minutos,
    se considera colgado (fallo de GPU o de comunicación NCCL) y se reinicia
    el proceso para que reanude desde el último checkpoint.
  * Salud de GPU: temperatura, uso, memoria y errores ECC no corregibles (nvidia-smi).
  * Sincronización periódica de checkpoints al almacenamiento remoto.
Todo se registra en ``logs/watchdog.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import append_jsonl


@dataclass
class SpendGuard:
    budget_usd: float
    rate_per_hour: float
    start: float
    max_hours: float | None = None
    warn_frac: float = 0.9
    clock: Callable[[], float] = time.time

    def spent(self) -> float:
        return (self.clock() - self.start) / 3600 * self.rate_per_hour

    def check(self) -> str:
        hours = (self.clock() - self.start) / 3600
        s = self.spent()
        if (self.budget_usd and s >= self.budget_usd) or (self.max_hours and hours >= self.max_hours):
            return "stop"
        if (self.budget_usd and s >= self.warn_frac * self.budget_usd) or \
                (self.max_hours and hours >= self.warn_frac * self.max_hours):
            return "warn"
        return "ok"


def gpu_health() -> list[dict]:
    q = "index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,ecc.errors.uncorrected.volatile.total"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7:
            continue
        rows.append({"gpu": int(p[0]), "name": p[1], "temp_c": _num(p[2]), "util_pct": _num(p[3]),
                     "mem_used_mb": _num(p[4]), "mem_total_mb": _num(p[5]), "ecc_uncorrected": _num(p[6])})
    return rows


def _num(x: str):
    try:
        return float(x)
    except ValueError:
        return None


def heartbeat_age_s(run_dir: str | Path, now: float | None = None) -> float | None:
    hb = Path(run_dir) / "heartbeat.json"
    if not hb.exists():
        return None
    try:
        return (now or time.time()) - json.loads(hb.read_text())["unix"]
    except (ValueError, KeyError):
        return None


def _alive(pid: int) -> bool:
    """True si el proceso sigue ejecutándose (un zombi ya terminó aunque nadie lo haya recogido)."""
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line.split()[1]
    except OSError:
        pass
    return True


def stop_training(pid: int, grace_s: float = 900, log=print) -> bool:
    """SIGTERM al grupo de procesos del entrenamiento y espera a que guarde el checkpoint."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return True
    t0 = time.time()
    while time.time() - t0 < grace_s:
        if not _alive(pid):
            return True
        time.sleep(1)
    log("[watchdog] el entrenamiento no terminó a tiempo; SIGKILL")
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    return False


def terminate_self(provider: str | None) -> str:
    """Apaga la propia máquina usando la API del proveedor (o el sistema operativo como último recurso)."""
    from .providers.impl import get_provider
    ids = {"runpod": "RUNPOD_POD_ID", "vastai": "CONTAINER_ID"}
    iid = os.environ.get(ids.get(provider or "", ""), "")
    if provider and iid:
        try:
            get_provider(provider).terminate(iid)
            return f"terminado vía API ({provider}:{iid})"
        except Exception as e:  # se intenta el apagado del sistema
            err = repr(e)
    else:
        err = "sin identificador de instancia"
    subprocess.run(["shutdown", "-h", "now"], check=False)
    return f"apagado del sistema ({err})"


def run(args, clock=time.time, sleep=time.sleep) -> str:
    start_file = Path(args.run_dir) / "watchdog_start.json"
    start_file.parent.mkdir(parents=True, exist_ok=True)
    if start_file.exists():  # reanudaciones en la misma máquina conservan el gasto acumulado
        start = json.loads(start_file.read_text())["start"]
    else:
        start = clock()
        start_file.write_text(json.dumps({"start": start}))
    guard = SpendGuard(args.budget, args.rate, start, args.max_hours, clock=clock)
    log_path = Path(args.log)
    last_sync = clock()
    warned = False
    while True:
        state = guard.check()
        hb = heartbeat_age_s(args.run_dir, clock())
        rec = {"unix": clock(), "state": state, "spent_usd": round(guard.spent(), 2), "budget_usd": args.budget,
               "heartbeat_age_s": hb, "gpus": gpu_health()}
        append_jsonl(log_path, rec)
        pid = int(Path(args.pidfile).read_text()) if Path(args.pidfile).exists() else None
        if state == "warn" and not warned:
            print(f"[watchdog] aviso: gasto ${rec['spent_usd']} (90 % del límite)", flush=True)
            warned = True
        if state == "stop":
            print("[watchdog] límite alcanzado: deteniendo el entrenamiento con checkpoint", flush=True)
            if pid:
                stop_training(pid)
            if args.remote:
                _sync(args)
            if not args.no_terminate:
                append_jsonl(log_path, {"unix": clock(), "action": terminate_self(args.provider)})
            return "stopped"
        if hb is not None and hb > args.stale_min * 60 and pid:
            append_jsonl(log_path, {"unix": clock(), "action": "restart_hung_training", "heartbeat_age_s": hb})
            stop_training(pid, grace_s=120)  # torchrun --max-restarts relanza y reanuda
        if args.remote and clock() - last_sync > args.sync_min * 60:
            _sync(args)
            last_sync = clock()
        if pid is None and args.exit_when_done and Path(args.run_dir, "summary.json").exists():
            return "training_finished"
        sleep(args.interval)


def _sync(args) -> None:
    from .sync import push
    try:
        push(args.run_dir, args.remote)
    except Exception as e:
        print(f"[watchdog] fallo de sincronización: {e}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Guardián de gasto y salud para el nodo GPU")
    p.add_argument("--budget", type=float, required=True, help="USD máximos")
    p.add_argument("--rate", type=float, required=True, help="USD por hora de la máquina")
    p.add_argument("--max-hours", type=float)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pidfile", default="train.pid")
    p.add_argument("--remote", default="")
    p.add_argument("--provider", default=os.environ.get("TITAN_PROVIDER"))
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--sync-min", type=float, default=30)
    p.add_argument("--stale-min", type=float, default=20)
    p.add_argument("--log", default="logs/watchdog.jsonl")
    p.add_argument("--no-terminate", action="store_true")
    p.add_argument("--exit-when-done", action="store_true")
    print(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
