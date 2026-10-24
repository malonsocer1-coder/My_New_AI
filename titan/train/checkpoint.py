"""Checkpoints atómicos, verificables y reanudables.

Estructura de un checkpoint::

    checkpoints/step_00001000/
        model.safetensors      pesos (modo no fragmentado)
        dcp/                   estado fragmentado (FSDP, torch.distributed.checkpoint)
        optimizer.pt           estado de AdamW (modo no fragmentado)
        trainer_state.json     paso, tokens, tiempo, coste, estado del cargador...
        rng_rank{r}.pt         estados de los generadores aleatorios
        model_config.json      arquitectura
        MANIFEST.json          SHA-256 de cada archivo

El directorio se escribe como ``step_X.tmp`` y se renombra solo al terminar:
un corte de luz o una interrupción del proveedor nunca deja un checkpoint
corrupto marcado como válido. ``LATEST`` apunta al último completo.
"""

from __future__ import annotations

import json
import os
import time
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ..utils.io import atomic_write_text, read_json, sha256_file, write_json


def resolve_weights_dir(path: str | Path) -> Path:
    """Ejecución (runs/x) -> último checkpoint; directorio de checkpoints -> LATEST; si no, tal cual."""
    p = Path(path)
    if (p / "checkpoints").is_dir():
        p = p / "checkpoints"
    if (p / "LATEST").exists():
        p = p / (p / "LATEST").read_text().strip()
    return p


def consolidate_dcp(step_dir: str | Path) -> Path:
    """Convierte un checkpoint fragmentado (FSDP/DCP) en ``model.safetensors`` completo, sin GPUs."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    step_dir = Path(step_dir)
    out = step_dir / "model.safetensors"
    if out.exists():
        return out
    tmp = step_dir / "_consolidated.pt"
    dcp_to_torch_save(str(step_dir / "dcp"), str(tmp))
    full = torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.unlink()
    sd = {k: v.detach().contiguous() for k, v in full["model"].items()}
    cfg = json.loads((step_dir / "model_config.json").read_text())
    if cfg.get("tie_embeddings"):
        sd.pop("output.weight", None)
    save_file(sd, str(out))
    return out


def model_state_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    sd = model.state_dict()
    tied = getattr(getattr(model, "cfg", None), "tie_embeddings", False)
    out = {}
    for k, v in sd.items():
        if tied and k == "output.weight":
            continue
        out[k] = v.detach().contiguous().cpu()
    return out


def load_model_weights(model: torch.nn.Module, path: str | Path, strict: bool = True) -> None:
    path = resolve_weights_dir(path)
    if path.is_dir():
        if not (path / "model.safetensors").exists() and (path / "dcp").is_dir():
            consolidate_dcp(path)
        path = path / "model.safetensors"
    sd = load_file(str(path))
    tied = getattr(getattr(model, "cfg", None), "tie_embeddings", False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not (tied and m == "output.weight")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"Pesos incompatibles. Faltan: {missing[:8]} Sobran: {unexpected[:8]}")


def _rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


class CheckpointManager:
    def __init__(self, root: str | Path, rank: int = 0, world_size: int = 1, sharded: bool = False,
                 keep_last: int = 3, keep_every: int | None = None):
        self.root = Path(root)
        self.rank, self.world_size, self.sharded = rank, world_size, sharded
        self.keep_last, self.keep_every = keep_last, keep_every
        if rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _barrier(self) -> None:
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------------
    def save(self, step: int, model, optimizer, trainer_state: dict, model_config: dict,
             scaler=None) -> Path:
        final = self.root / f"step_{step:08d}"
        tmp = self.root / f"step_{step:08d}.tmp"
        if self.is_main:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
        self._barrier()
        if self.sharded:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict
            msd, osd = get_state_dict(model, optimizer)
            dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp / "dcp"))
        elif self.is_main:
            save_file(model_state_for_save(model), str(tmp / "model.safetensors"))
            opt = {"optimizer": optimizer.state_dict()}
            if scaler is not None:
                opt["scaler"] = scaler.state_dict()
            torch.save(opt, tmp / "optimizer.pt")
        torch.save(_rng_state(), tmp / f"rng_rank{self.rank}.pt")
        self._barrier()
        if self.is_main:
            write_json(tmp / "trainer_state.json", trainer_state)
            write_json(tmp / "model_config.json", model_config)
            files = {}
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    files[str(p.relative_to(tmp))] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            write_json(tmp / "MANIFEST.json", {"step": step, "files": files, "sharded": self.sharded,
                                               "world_size": self.world_size})
            if final.exists():
                shutil.rmtree(final)
            for attempt in range(5):  # en Windows el antivirus puede retener un archivo un instante
                try:
                    os.replace(tmp, final)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(1 + attempt)
            atomic_write_text(self.root / "LATEST", final.name + "\n")
            self._prune()
        self._barrier()
        return final

    def _prune(self) -> None:
        ckpts = self.list()
        protected = set(ckpts[-self.keep_last:]) if self.keep_last > 0 else set(ckpts)
        for c in ckpts:
            step = int(c.name.split("_")[1])
            if c in protected or (self.keep_every and step % self.keep_every == 0):
                continue
            shutil.rmtree(c, ignore_errors=True)

    def list(self) -> list[Path]:
        if not self.root.exists():
            return []
        out = [p for p in self.root.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")
               and (p / "MANIFEST.json").exists()]
        return sorted(out, key=lambda p: int(p.name.split("_")[1]))

    def latest(self) -> Path | None:
        ptr = self.root / "LATEST"
        if ptr.exists():
            p = self.root / ptr.read_text().strip()
            if (p / "MANIFEST.json").exists():
                return p
        ckpts = self.list()
        return ckpts[-1] if ckpts else None

    @staticmethod
    def verify(path: str | Path) -> list[str]:
        """Devuelve la lista de archivos corruptos o ausentes (vacía si está íntegro)."""
        path = Path(path)
        man = read_json(path / "MANIFEST.json")
        bad = []
        for rel, info in man["files"].items():
            f = path / rel
            if not f.exists() or sha256_file(f) != info["sha256"]:
                bad.append(rel)
        return bad

    # ------------------------------------------------------------------
    def load(self, path: str | Path, model, optimizer=None, scaler=None, verify: bool = True) -> dict:
        path = Path(path)
        if verify and self.is_main:
            bad = self.verify(path)
            if bad:
                raise RuntimeError(f"Checkpoint {path} corrupto: {bad}")
        self._barrier()
        man = read_json(path / "MANIFEST.json")
        if man.get("sharded"):
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
            msd, osd = get_state_dict(model, optimizer)
            state = {"model": msd, "optim": osd}
            dcp.load(state, checkpoint_id=str(path / "dcp"))
            set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
        else:
            load_model_weights(model, path / "model.safetensors")
            if optimizer is not None:
                opt = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
                optimizer.load_state_dict(opt["optimizer"])
                if scaler is not None and "scaler" in opt:
                    scaler.load_state_dict(opt["scaler"])
        rng = path / f"rng_rank{self.rank}.pt"
        if rng.exists():
            _set_rng_state(torch.load(rng, weights_only=False))
        return json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
