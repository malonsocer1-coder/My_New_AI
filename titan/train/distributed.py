"""Inicialización de procesos y paralelización (DDP, FSDP2 y HSDP).

Se lanza con ``torchrun`` (una o varias máquinas). Estrategias:
  * ``none``  un proceso, una GPU o CPU.
  * ``ddp``   réplica completa por GPU; para modelos que caben en una GPU.
  * ``fsdp``  FSDP2 (``fully_shard``): fragmenta parámetros, gradientes y
              estados del optimizador entre GPUs. Con ``fsdp_shard_size`` se
              usa HSDP: fragmentado dentro del nodo y réplica entre nodos,
              lo que reduce el tráfico por la red entre máquinas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def pick_device(requested: str = "auto", local_rank: int = 0) -> torch.device:
    if requested != "auto":
        dev = torch.device(requested)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", local_rank)
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed(requested_device: str = "auto", timeout_minutes: int = 30) -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = pick_device(requested_device, local_rank)
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world, device=device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def resolve_parallel(requested: str, world_size: int, n_params: int, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if world_size == 1:
        return "none"
    # Regla práctica: con AdamW y bf16 cada parámetro ocupa ~16 bytes de
    # estado (pesos fp32 + 2 momentos + gradiente + copia bf16). Si eso supera
    # ~40 % de la memoria de la GPU, conviene fragmentar.
    if device.type == "cuda":
        mem = torch.cuda.get_device_properties(device).total_memory
        return "fsdp" if n_params * 16 > 0.4 * mem else "ddp"
    return "ddp"


def apply_fsdp(model, world_size: int, device: torch.device, dtype: torch.dtype,
               shard_size: int | None = None):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dev_type = device.type
    if shard_size and shard_size < world_size:
        if world_size % shard_size:
            raise ValueError("world_size debe ser múltiplo de fsdp_shard_size")
        mesh = init_device_mesh(dev_type, (world_size // shard_size, shard_size),
                                mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh(dev_type, (world_size,), mesh_dim_names=("shard",))
    mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32) if dtype != torch.float32 else None
    kw = {"mesh": mesh}
    if mp is not None:
        kw["mp_policy"] = mp
    for layer in model.layers:
        fully_shard(layer, **kw)
    fully_shard(model, **kw)
    return model


def all_reduce_mean(x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1 and dist.is_initialized():
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= world_size
    return x
