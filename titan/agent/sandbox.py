"""Ejecución aislada de código no confiable.

Backends, del más al menos aislado:
  * ``docker``   contenedor efímero sin red, con límites de CPU y memoria.
  * ``unshare``  (Linux) espacio de nombres de red vacío + límites de recursos.
  * ``process``  subproceso con límites de recursos (Linux) o solo con tiempo
                 límite (Windows). No aísla la red: por eso la política exige
                 aprobación explícita para ``code_exec``.

En todos los casos: directorio de trabajo temporal propio, entorno sin
variables sensibles (claves API, tokens), intérprete en modo aislado (-I) y
salida truncada.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SAFE_ENV_KEYS = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG", "LC_ALL", "PYTHONIOENCODING")


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    backend: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _clean_env() -> dict:
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _limits(cpu_s: int, mem_mb: int, fsize_mb: int):
    def apply():  # se ejecuta en el hijo antes de exec (solo POSIX)
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        os.setsid()
    return apply


def available_backend(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if sys.platform.startswith("linux") and shutil.which("unshare"):
        try:
            r = subprocess.run(["unshare", "--net", "--map-root-user", "true"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return "unshare"
        except Exception:
            pass
    return "process"


def run_python(code: str, timeout_s: float = 10.0, mem_mb: int = 512, workdir: str | None = None,
               backend: str = "auto", max_output: int = 20000, files: dict[str, str] | None = None) -> SandboxResult:
    backend = available_backend(backend)
    tmp = tempfile.mkdtemp(prefix="titan_sbx_")
    try:
        work = Path(workdir) if workdir else Path(tmp)
        for name, content in (files or {}).items():
            target = (work / name).resolve()
            if not str(target).startswith(str(work.resolve())):
                raise ValueError(f"Ruta fuera del sandbox: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        if backend == "docker":
            cmd = ["docker", "run", "--rm", "--network", "none", "--memory", f"{mem_mb}m", "--cpus", "1",
                   "--pids-limit", "64", "-v", f"{tmp}:/sbx:ro", "-v", f"{work}:/work", "-w", "/work",
                   "python:3.12-slim", "python", "-I", "/sbx/main.py"]
            preexec = None
        else:
            cmd = [sys.executable, "-I", str(script)]
            if backend == "unshare":
                cmd = ["unshare", "--net", "--map-root-user"] + cmd
            preexec = _limits(int(timeout_s) + 1, mem_mb, 64) if os.name == "posix" else None
        try:
            r = subprocess.run(cmd, cwd=str(work), env=_clean_env(), capture_output=True, timeout=timeout_s,
                               preexec_fn=preexec, stdin=subprocess.DEVNULL)
            out, err, rc, to = r.stdout, r.stderr, r.returncode, False
        except subprocess.TimeoutExpired as e:
            out, err, rc, to = e.stdout or b"", e.stderr or b"", -9, True
        dec = lambda b: (b.decode("utf-8", "replace") if isinstance(b, bytes) else b)[:max_output]
        return SandboxResult(dec(out), dec(err), rc, to, backend)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
