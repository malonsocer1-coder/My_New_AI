"""Mide el rendimiento real de entrenamiento en el hardware disponible.

Con datos aleatorios del tamaño configurado ejecuta unos pasos completos
(forward, backward y AdamW), y devuelve tokens/s, memoria máxima y la
proyección de horas para los tokens objetivo. Sirve para sustituir las
estimaciones teóricas por números medidos en la GPU del usuario.
"""

from __future__ import annotations

import contextlib
import time

import torch

from ..model import Transformer
from .config import load_run_config
from .distributed import pick_device
from .trainer import DTYPES, build_optimizer, detect_peak_flops, resolve_compile


def _seqs_per_step(cfg) -> int:
    if cfg.batch_tokens:
        return cfg.batch_tokens // cfg.seq_len
    return cfg.micro_batch_size * cfg.grad_accum_steps


def find_batch(config_path: str, overrides: dict | None = None, target_tokens: float | None = None,
               log=print) -> dict:
    """Mide todos los micro-lotes posibles en esta GPU y proyecta las horas con el mejor
    (lo mismo que hace ``auto_batch: true`` al empezar a entrenar)."""
    from .autobatch import probe
    model_cfg, cfg = load_run_config(config_path, overrides)
    device = pick_device(cfg.device)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    seqs = _seqs_per_step(cfg)
    res = probe(model_cfg, cfg, device, DTYPES[cfg.dtype], seqs, cfg.auto_batch_max, cfg.auto_batch_margin_gb,
                log=log)
    tps = res["choice"]["tokens_per_s_eager"]
    total = target_tokens or seqs * cfg.seq_len * cfg.max_steps
    res.update({
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "params_M": round(model_cfg.param_counts()["total"] / 1e6, 2), "seq_len": cfg.seq_len,
        "tokens_per_step": seqs * cfg.seq_len, "target_tokens": int(total),
        "achieved_tflops_eager": round(tps * model_cfg.flops_per_token(cfg.seq_len) / 1e12, 2),
        "projected_hours_eager": round(total / tps / 3600, 2),
        "note": "medido sin torch.compile; con compile suele ser más rápido (mida con titan bench)",
    })
    return res


def bench(config_path: str, steps: int = 10, warmup: int = 3, overrides: dict | None = None,
          target_tokens: float | None = None) -> dict:
    model_cfg, cfg = load_run_config(config_path, overrides)
    device = pick_device(cfg.device)
    dtype = DTYPES[cfg.dtype]
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    if cfg.batch_tokens:
        cfg.grad_accum_steps = _seqs_per_step(cfg) // cfg.micro_batch_size
    model = Transformer(model_cfg).to(device)
    opt = build_optimizer(model, cfg, device)
    fwd = torch.compile(model) if resolve_compile(cfg, device) else model
    use_ac = dtype != torch.float32
    ctx = (lambda: torch.autocast(device.type, dtype=dtype)) if use_ac else contextlib.nullcontext
    B, T = cfg.micro_batch_size, cfg.seq_len
    x = torch.randint(0, model_cfg.vocab_size, (B, T + 1), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
        for _ in range(cfg.grad_accum_steps):
            with ctx():
                out = fwd(x[:, :-1], targets=x[:, 1:])
            (out.loss / cfg.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = (time.time() - t0) / steps
    tokens_step = B * T * cfg.grad_accum_steps
    tps = tokens_step / dt
    res = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "params_M": round(model.num_params() / 1e6, 2), "micro_batch": B, "seq_len": T,
        "grad_accum": cfg.grad_accum_steps, "sec_per_step": round(dt, 3), "tokens_per_s": round(tps),
        "achieved_tflops": round(tps * model_cfg.flops_per_token(T) / 1e12, 2),
    }
    peak = cfg.peak_flops_per_device or detect_peak_flops(device)
    if peak:
        res["mfu"] = round(tps * model_cfg.flops_per_token(T) / peak, 3)
    if device.type == "cuda":
        res["max_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        res["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1e9, 2)
    total = target_tokens or tokens_step * cfg.max_steps
    res["target_tokens"] = int(total)
    res["projected_hours"] = round(total / tps / 3600, 2)
    return res
