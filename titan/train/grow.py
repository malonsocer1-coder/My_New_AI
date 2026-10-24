"""Entrenamiento por partes: crecimiento de modelos y unión de expertos.

Tres operaciones, todas sobre pesos propios de TITAN:

``grow_depth``
    Agranda un modelo añadiendo capas. Con ``method="copy_zero"`` (estilo
    LLaMA Pro) cada capa nueva es una copia de una existente con sus
    proyecciones de salida (``wo`` y ``w2``) a cero: la red resultante
    calcula exactamente lo mismo que la original y el entrenamiento posterior
    aprovecha la capacidad extra. Con ``method="stack"`` (G_stack) se
    duplica la pila completa; no preserva la función pero suele converger
    rápido.

``upcycle_to_moe``
    Convierte un modelo denso en Mixture of Experts copiando su FFN en N
    expertos ("sparse upcycling"). Como los expertos son idénticos y los pesos
    de enrutamiento se normalizan, la salida inicial es idéntica a la densa.

``merge_branches``
    Branch-Train-MiX: varias ramas entrenadas por separado desde la MISMA base
    (p. ej. español, código, matemáticas) se unen en un MoE. Cada experto es
    la FFN de una rama; atención, embeddings y normas se promedian; el
    enrutador se inicializa y se aprende en un ajuste conjunto corto.

Todas devuelven ``(ModelConfig, state_dict)`` y pueden guardarse con
``save_model_dir`` para usarse como ``init_from`` en el entrenador.
"""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..model import ModelConfig
from .checkpoint import resolve_weights_dir
from ..utils.io import read_json, sha256_file, write_json

_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")


def load_model_dir(path: str | Path) -> tuple[ModelConfig, dict[str, torch.Tensor]]:
    """Carga (config, pesos) de un checkpoint, una ejecución o un modelo exportado."""
    p = resolve_weights_dir(path)
    cfg_file = p / "model_config.json" if (p / "model_config.json").exists() else p / "config.json"
    data = read_json(cfg_file)
    data = data.get("titan_config", data)
    cfg = ModelConfig.from_dict({k: v for k, v in data.items() if k in ModelConfig.__dataclass_fields__})
    sd = load_file(str(p / "model.safetensors"))
    if cfg.tie_embeddings:
        sd.pop("output.weight", None)
    return cfg, sd


def save_model_dir(cfg: ModelConfig, sd: dict[str, torch.Tensor], out: str | Path, meta: dict | None = None,
                   tokenizer: str | Path | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: v.contiguous() for k, v in sd.items() if not (cfg.tie_embeddings and k == "output.weight")}
    save_file(clean, str(out / "model.safetensors"))
    write_json(out / "model_config.json", cfg.to_dict())
    if tokenizer:
        tp = Path(tokenizer)
        tp = tp / "tokenizer.json" if tp.is_dir() else tp
        (out / "tokenizer.json").write_bytes(tp.read_bytes())
    files = {f.name: sha256_file(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != "MANIFEST.json"}
    write_json(out / "MANIFEST.json", {"files": files, "operation": meta or {}})
    return out


def _split_layers(sd: dict[str, torch.Tensor]) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    layers: dict[int, dict[str, torch.Tensor]] = {}
    other = {}
    for k, v in sd.items():
        m = _LAYER.match(k)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = v
        else:
            other[k] = v
    return layers, other


def _zero_outputs(layer: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in layer.items():
        if k == "attention.wo.weight" or k.endswith("w2.weight"):
            out[k] = torch.zeros_like(v)
        else:
            out[k] = v.clone()
    return out


# ----------------------------------------------------------------------
def grow_depth(cfg: ModelConfig, sd: dict[str, torch.Tensor], new_n_layers: int,
               method: str = "copy_zero") -> tuple[ModelConfig, dict[str, torch.Tensor]]:
    L = cfg.n_layers
    if new_n_layers <= L:
        raise ValueError("new_n_layers debe ser mayor que el número actual de capas")
    layers, other = _split_layers(sd)
    if method == "stack":
        if new_n_layers % L:
            raise ValueError("method='stack' requiere un múltiplo del número de capas")
        order = [(i % L, False) for i in range(new_n_layers)]
    elif method == "copy_zero":
        # Reparte las capas nuevas uniformemente: cada copia va justo detrás de su original
        extra = new_n_layers - L
        order = []
        for i in range(L):
            order.append((i, False))
            copies = ((i + 1) * extra) // L - (i * extra) // L
            order += [(i, True)] * copies
    else:
        raise ValueError(f"Método desconocido: {method}")
    new_sd = dict(other)
    for new_i, (src, is_new) in enumerate(order):
        block = _zero_outputs(layers[src]) if is_new else {k: v.clone() for k, v in layers[src].items()}
        for k, v in block.items():
            new_sd[f"layers.{new_i}.{k}"] = v
    return replace(cfg, n_layers=new_n_layers), new_sd


def upcycle_to_moe(cfg: ModelConfig, sd: dict[str, torch.Tensor], n_experts: int, top_k: int = 2,
                   router_std: float = 0.02, seed: int = 0) -> tuple[ModelConfig, dict[str, torch.Tensor]]:
    if cfg.is_moe:
        raise ValueError("El modelo ya es MoE")
    gen = torch.Generator().manual_seed(seed)
    layers, other = _split_layers(sd)
    new_sd = dict(other)
    for i, block in layers.items():
        for k, v in block.items():
            if k.startswith("feed_forward."):
                w = k[len("feed_forward."):]
                for e in range(n_experts):
                    new_sd[f"layers.{i}.feed_forward.experts.{e}.{w}"] = v.clone()
            else:
                new_sd[f"layers.{i}.{k}"] = v.clone()
        new_sd[f"layers.{i}.feed_forward.gate.weight"] = torch.randn(
            n_experts, cfg.dim, generator=gen, dtype=torch.float32).mul_(router_std).to(block["attention.wq.weight"].dtype)
    return replace(cfg, n_experts=n_experts, n_experts_per_token=top_k), new_sd


def merge_branches(cfgs: list[ModelConfig], sds: list[dict[str, torch.Tensor]], top_k: int = 2,
                   shared_weights: list[float] | None = None, router_std: float = 0.02,
                   seed: int = 0) -> tuple[ModelConfig, dict[str, torch.Tensor]]:
    """Branch-Train-MiX: une N ramas densas (misma arquitectura y misma base) en un MoE de N expertos."""
    if len(cfgs) < 2:
        raise ValueError("Se necesitan al menos 2 ramas")
    base = cfgs[0]
    for c in cfgs[1:]:
        if replace(c, name=base.name) != base:
            raise ValueError("Todas las ramas deben tener la misma arquitectura (y venir de la misma base)")
    if base.is_moe:
        raise ValueError("Las ramas deben ser densas")
    n = len(sds)
    w = shared_weights or [1.0 / n] * n
    if abs(sum(w) - 1.0) > 1e-6:
        raise ValueError("shared_weights debe sumar 1")
    gen = torch.Generator().manual_seed(seed)
    new_sd: dict[str, torch.Tensor] = {}
    keys = sds[0].keys()
    for k in keys:
        m = _LAYER.match(k)
        if m and m.group(2).startswith("feed_forward."):
            i, rest = int(m.group(1)), m.group(2)[len("feed_forward."):]
            for e, sd in enumerate(sds):
                new_sd[f"layers.{i}.feed_forward.experts.{e}.{rest}"] = sd[k].clone()
        else:
            acc = sum(wi * sd[k].float() for wi, sd in zip(w, sds))
            new_sd[k] = acc.to(sds[0][k].dtype)
    for i in range(base.n_layers):
        new_sd[f"layers.{i}.feed_forward.gate.weight"] = torch.randn(
            n, base.dim, generator=gen).mul_(router_std).to(sds[0]["tok_embeddings.weight"].dtype)
    return replace(base, n_experts=n, n_experts_per_token=min(top_k, n)), new_sd


def describe(cfg: ModelConfig) -> dict:
    pc = cfg.param_counts()
    return {"n_layers": cfg.n_layers, "dim": cfg.dim, "n_experts": cfg.n_experts,
            "top_k": cfg.n_experts_per_token if cfg.is_moe else None,
            "params_total_M": round(pc["total"] / 1e6, 2), "params_active_M": round(pc["active"] / 1e6, 2)}


def clone_config(cfg: ModelConfig, **changes) -> ModelConfig:
    return replace(copy.deepcopy(cfg), **changes)
