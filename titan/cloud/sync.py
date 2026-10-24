"""Sincronización de ejecuciones con almacenamiento remoto (rclone) y verificación de integridad.

``rclone`` admite S3, Cloudflare R2, Backblaze B2, Google Cloud Storage,
Google Drive y muchos más. Las credenciales se configuran con ``rclone
config`` o variables ``RCLONE_CONFIG_*`` en el equipo; nunca en el repositorio.

  push  copia (sin borrar nada en el destino) los checkpoints completos.
  pull  descarga y verifica cada checkpoint contra su MANIFEST.json (SHA-256).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _rclone(*args: str) -> None:
    if not shutil.which("rclone"):
        raise RuntimeError("rclone no está instalado (https://rclone.org/install/)")
    subprocess.run(["rclone", *args], check=True)


def push(local: str | Path, remote: str) -> None:
    # Se excluyen los checkpoints a medio escribir (*.tmp)
    _rclone("copy", str(local), remote, "--exclude", "*.tmp/**", "--transfers", "8", "--checksum")


def pull(remote: str, local: str | Path, verify: bool = True) -> list[str]:
    _rclone("copy", remote, str(local), "--transfers", "8", "--checksum")
    return verify_tree(local) if verify else []


def verify_tree(local: str | Path) -> list[str]:
    """Verifica todos los checkpoints y modelos exportados bajo ``local``. Devuelve los corruptos."""
    from ..train.checkpoint import CheckpointManager
    bad = []
    for man in Path(local).rglob("MANIFEST.json"):
        d = man.parent
        try:
            if d.name.startswith("step_"):
                if CheckpointManager.verify(d):
                    bad.append(str(d))
            else:
                from ..export.exporter import verify_export
                if (d / "MANIFEST.json").exists() and "info" in man.read_text():
                    if verify_export(d):
                        bad.append(str(d))
        except Exception as e:
            bad.append(f"{d}: {e}")
    return bad


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Sincronizar ejecuciones de TITAN con almacenamiento remoto")
    p.add_argument("action", choices=["push", "pull", "verify"])
    p.add_argument("--remote")
    p.add_argument("--local", required=True)
    a = p.parse_args(argv)
    if a.action == "push":
        push(a.local, a.remote)
    elif a.action == "pull":
        bad = pull(a.remote, a.local)
        print("Integridad verificada" if not bad else f"Archivos con problemas: {bad}")
    else:
        bad = verify_tree(a.local)
        print("Integridad verificada" if not bad else f"Archivos con problemas: {bad}")


if __name__ == "__main__":
    main()
