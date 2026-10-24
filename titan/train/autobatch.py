"""Elige el micro-lote más rápido que cabe en la GPU, midiéndolo en vez de estimarlo.

El tamaño del lote global (tokens por paso del optimizador) lo fija la config y
no cambia; solo se reparte de otra forma entre micro-lote y acumulación de
gradientes, así que la optimización es la misma con cualquier elección y un
entrenamiento puede reanudarse con otra. Para cada candidato se ejecutan pasos
reales (forward, backward y AdamW) con datos aleatorios y se mide:

* tokens/s del forward+backward;
* memoria libre que queda en la GPU tras el pico (``torch.cuda.mem_get_info``,
  que también cuenta lo que usan otros programas).

Se descartan los candidatos que agotan la memoria o dejan menos de
``margin_gb`` libres. En Windows esto importa: cuando se llena la VRAM, el
controlador puede desbordar a la RAM del sistema en lugar de fallar, y el
entrenamiento se vuelve varias veces más lento sin ningún error. Si nada cabe
sin recálculo de activaciones (*activation checkpointing*), se repite con él.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import time

import torch

from ..model import ModelConfig, Transformer


def candidate_micro_batches(seqs_per_rank: int, max_micro: int) -> list[int]:
    """Divisores de las secuencias por paso y rango (para que la acumulación sea exacta)."""
    return [b for b in range(1, min(max_micro, seqs_per_rank) + 1) if seqs_per_rank % b == 0]


def _free_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _try_micro(model, opt, b: int, seq_len: int, vocab: int, device, autocast_ctx, steps: int,
               margin_gb: float) -> dict:
    row = {"micro_batch": b, "activation_checkpointing": model.cfg.activation_checkpointing}
    x = torch.randint(0, vocab, (b, seq_len + 1), device=device)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for i in range(1 + steps):
            if i == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
            with autocast_ctx():
                out = model(x[:, :-1], targets=x[:, 1:])
            out.loss.backward()
            if i == 0:
                opt.step()  # reserva el estado de AdamW en el primer candidato
            opt.zero_grad(set_to_none=False)  # los gradientes viven todo el entrenamiento
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / steps
        row["tokens_per_s"] = round(b * seq_len / dt)
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            row["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            row["free_after_gb"] = round(free / 1e9, 2)
            row["gpu_total_gb"] = round(total / 1e9, 2)
            row["fits"] = free / 1e9 >= margin_gb
            if not row["fits"]:
                row["reason"] = f"quedan {free / 1e9:.2f} GB libres (< {margin_gb} GB de margen)"
        else:
            row["fits"] = True
    except torch.cuda.OutOfMemoryError:
        row.update(fits=False, reason="memoria agotada (OOM)")
    except RuntimeError as e:  # algunos backends informan el OOM como RuntimeError genérico
        if "out of memory" not in str(e).lower():
            raise
        row.update(fits=False, reason="memoria agotada (OOM)")
    finally:
        del x
        opt.zero_grad(set_to_none=False)
    return row


def probe(model_cfg: ModelConfig, cfg, device: torch.device, dtype: torch.dtype, seqs_per_rank: int,
          max_micro: int = 32, margin_gb: float = 0.7, steps: int = 3, log=print) -> dict:
    """Mide los candidatos y devuelve ``{"choice": {...}, "table": [...]}``."""
    from .trainer import build_optimizer
    use_ac = dtype != torch.float32
    autocast_ctx = (lambda: torch.autocast(device.type, dtype=dtype)) if use_ac else contextlib.nullcontext
    cands = candidate_micro_batches(seqs_per_rank, max_micro)
    table: list[dict] = []
    ckpt_options = [True] if model_cfg.activation_checkpointing else [False, True]
    for ac in ckpt_options:
        mc = dataclasses.replace(model_cfg, activation_checkpointing=ac)
        model = opt = None
        try:
            model = Transformer(mc)
            model.init_weights(seed=0)
            model.to(device).train()
            opt = build_optimizer(model, cfg, device)
            for b in cands:
                row = _try_micro(model, opt, b, cfg.seq_len, mc.vocab_size, device, autocast_ctx, steps, margin_gb)
                table.append(row)
                log(f"[autobatch] micro={b:<3} recálculo={'sí' if ac else 'no'} -> "
                    + (f"{row['tokens_per_s']:,} tok/s" if "tokens_per_s" in row else "—")
                    + (f", {row['free_after_gb']} GB libres" if "free_after_gb" in row else "")
                    + ("" if row["fits"] else f"  ✗ {row.get('reason', '')}"))
                if not row["fits"]:
                    break
                _free_cuda(device)
        except torch.cuda.OutOfMemoryError:
            table.append({"micro_batch": 0, "activation_checkpointing": ac, "fits": False,
                          "reason": "el modelo y el optimizador no caben"})
        finally:
            del model, opt
            _free_cuda(device)
        if any(r["fits"] for r in table):
            break
    ok = [r for r in table if r["fits"]]
    if not ok:
        raise RuntimeError("Ningún tamaño de micro-lote cabe en la memoria de la GPU; cierre otros programas "
                           "que usen la GPU o reduzca seq_len / el modelo.\n" + str(table))
    best = max(ok, key=lambda r: r["tokens_per_s"])
    choice = {"micro_batch_size": best["micro_batch"], "grad_accum_steps": seqs_per_rank // best["micro_batch"],
              "activation_checkpointing": best["activation_checkpointing"],
              "tokens_per_s_eager": best["tokens_per_s"]}
    return {"choice": choice, "table": table}
