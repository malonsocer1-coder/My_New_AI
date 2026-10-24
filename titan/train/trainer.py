"""Bucle de entrenamiento de TITAN.

Una misma implementación cubre: CPU (validación), una GPU, varias GPUs (DDP)
y modelos fragmentados (FSDP2/HSDP) en varias máquinas, tanto para
preentrenamiento como para SFT. Registra métricas en ``metrics.jsonl``,
escribe un latido (``heartbeat.json``) para el vigilante externo, aplica
límites de gasto y tiempo y guarda checkpoints reanudables.
"""

from __future__ import annotations

import contextlib
import math
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ..model import ModelConfig, Transformer
from ..utils.io import append_jsonl, write_json
from .checkpoint import CheckpointManager, load_model_weights
from .config import TrainConfig
from .distributed import (DistInfo, all_reduce_mean, apply_fsdp, cleanup_distributed, resolve_parallel,
                          setup_distributed)
from .schedule import lr_at

# FLOPs pico densos BF16 por dispositivo (sin dispersión), para estimar MFU.
PEAK_FLOPS = {
    "B300": 2.5e15, "B200": 2.25e15, "GB200": 2.5e15, "H200": 989e12, "H100": 989e12, "H800": 989e12,
    "A100": 312e12, "L40S": 362e12, "L40": 181e12, "A6000": 155e12, "RTX 6000 Ada": 364e12,
    "5090": 209e12, "4090": 165e12, "L4": 121e12, "A10": 125e12, "MI300X": 1.3e15,
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def detect_peak_flops(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    for k, v in PEAK_FLOPS.items():
        if k in name:
            return v
    return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).resolve().parents[2]).decode().strip()
    except Exception:
        return None


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig, device: torch.device):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = {}
    if device.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, **kw)


def build_loaders(cfg: TrainConfig, dinfo: DistInfo):
    if cfg.task == "pretrain":
        from ..data.loader import EvalLoader, PretrainLoader
        train = PretrainLoader(cfg.data, "train", cfg.seq_len, cfg.micro_batch_size, dinfo.rank,
                               dinfo.world_size, cfg.seed, cfg.domain_weights, dinfo.device)
        try:
            val = EvalLoader(cfg.data, "val", cfg.seq_len, dinfo.device)
        except Exception:
            val = None
        return train, val
    if cfg.task == "sft":
        from ..sft.data import SFTEvalLoader, SFTLoader
        train = SFTLoader(cfg.data, "train", cfg.seq_len, cfg.micro_batch_size, dinfo.rank,
                          dinfo.world_size, cfg.seed, dinfo.device)
        val = SFTEvalLoader(cfg.data, "val", cfg.seq_len, dinfo.device)
        return train, val
    raise ValueError(f"Tarea desconocida: {cfg.task}")


@torch.no_grad()
def evaluate(model, val_loader, cfg: TrainConfig, dinfo: DistInfo, autocast_ctx) -> dict:
    if val_loader is None:
        return {}
    was_training = model.training
    model.eval()
    results = {}
    bs = cfg.eval_batch_size or cfg.micro_batch_size
    for dom in val_loader.domains():
        tot = torch.zeros(2, device=dinfo.device, dtype=torch.float64)
        for i, (x, y) in enumerate(val_loader.batches(dom, cfg.eval_windows, bs)):
            if i % dinfo.world_size != dinfo.rank:
                continue
            with autocast_ctx():
                out = model(x, targets=y)
            n = (y != -100).sum()
            tot[0] += out.ce_loss.double() * n
            tot[1] += n
        if dinfo.distributed:
            dist.all_reduce(tot)
        if tot[1] == 0:
            continue
        loss = (tot[0] / tot[1]).item()
        r = {"loss": loss, "ppl": math.exp(min(loss, 50))}
        bpt = val_loader.bytes_per_token.get(dom) if hasattr(val_loader, "bytes_per_token") else None
        if bpt:
            r["bpb"] = loss / math.log(2) / bpt
        results[dom] = r
    if results:
        results["_mean"] = {"loss": sum(r["loss"] for r in results.values()) / len(results)}
    if was_training:
        model.train()
    return results


def _samples(model, tokenizer_path, prompts, max_new_tokens, device) -> list[dict]:
    if not prompts or not tokenizer_path:
        return []
    from ..infer.generate import generate_text
    from ..tokenizer import TitanTokenizer
    tok = TitanTokenizer.load(tokenizer_path)
    was = model.training
    model.eval()
    out = []
    for p in prompts:
        text = generate_text(model, tok, p, max_new_tokens=max_new_tokens, temperature=0.0)
        out.append({"prompt": p, "completion": text})
    if was:
        model.train()
    return out


def train(model_cfg: ModelConfig, cfg: TrainConfig, log=None) -> dict:
    dinfo = setup_distributed(cfg.device)
    is_main = dinfo.is_main

    def say(msg: str) -> None:
        if not is_main:
            return
        if log is None:
            print(msg, flush=True)
        else:
            log(msg)

    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    device = dinfo.device
    dtype = DTYPES[cfg.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 no está soportado en CPU; use fp32 o bf16")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    n_params = model_cfg.param_counts()["total"]
    parallel = resolve_parallel(cfg.parallel, dinfo.world_size, n_params, device)
    if parallel in ("ddp", "fsdp") and not dinfo.distributed:
        parallel = "none"

    # ---------------- modelo (pesos aleatorios propios) ----------------
    if parallel == "fsdp":
        with torch.device("meta"):
            model = Transformer(model_cfg)
        apply_fsdp(model, dinfo.world_size, device, dtype, cfg.fsdp_shard_size)
        model.to_empty(device=device)
        model.init_weights(seed=cfg.seed)
    else:
        model = Transformer(model_cfg)
        model.init_weights(seed=cfg.seed)
        model.to(device)
    raw_model = model

    if cfg.init_from:
        src = Path(cfg.init_from)
        if src.is_dir() and (src / "LATEST").exists():
            src = src / (src / "LATEST").read_text().strip()
        say(f"[init] cargando pesos base propios desde {src}")
        if parallel == "fsdp":
            from safetensors.torch import load_file
            from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
            sd = load_file(str(src / "model.safetensors" if src.is_dir() else src)) if is_main else {}
            set_model_state_dict(model, sd, options=StateDictOptions(full_state_dict=True,
                                                                     broadcast_from_rank0=True))
        else:
            load_model_weights(model, src)

    optimizer = build_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler("cuda") if (dtype == torch.float16 and parallel != "fsdp") else None
    use_autocast = parallel != "fsdp" and dtype != torch.float32
    autocast_ctx = (lambda: torch.autocast(device_type=device.type, dtype=dtype)) if use_autocast \
        else contextlib.nullcontext

    train_loader, val_loader = build_loaders(cfg, dinfo)
    ckpt = CheckpointManager(out_dir / "checkpoints", dinfo.rank, dinfo.world_size,
                             sharded=(parallel == "fsdp"), keep_last=cfg.keep_last, keep_every=cfg.keep_every)

    state = {"step": 0, "tokens": 0, "elapsed_s": 0.0, "best_val": None, "restarts": 0}
    latest = ckpt.latest() if cfg.resume else None
    if latest is not None:
        say(f"[reanudar] {latest}")
        state = ckpt.load(latest, raw_model, optimizer, scaler, verify=cfg.verify_checkpoints)
        train_loader.load_state_dict(state["loader"])
        state["restarts"] = state.get("restarts", 0) + 1
        if cfg.skip_samples_on_resume:
            train_loader.samples_seen += cfg.skip_samples_on_resume
            say(f"[reanudar] saltando {cfg.skip_samples_on_resume} muestras")

    fwd_model = model
    if parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        fwd_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if cfg.compile:
        fwd_model = torch.compile(fwd_model)

    tokens_per_step = train_loader.tokens_per_step() * cfg.grad_accum_steps
    peak = cfg.peak_flops_per_device or detect_peak_flops(device)
    flops_per_token = model_cfg.flops_per_token(cfg.seq_len)
    metrics_path = out_dir / "metrics.jsonl"

    if is_main:
        write_json(out_dir / "run_info.json", {
            "model_config": model_cfg.to_dict(), "train_config": cfg.to_dict(),
            "param_counts": model_cfg.param_counts(), "parallel": parallel,
            "world_size": dinfo.world_size, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__, "python": platform.python_version(), "git_commit": _git_commit(),
            "tokens_per_step": tokens_per_step, "started_unix": int(time.time()),
            "resumed_from": str(latest) if latest else None,
        })
    say(f"[modelo] {n_params / 1e6:.2f}M parámetros ({model_cfg.param_counts()['active'] / 1e6:.2f}M activos), "
        f"paralelismo={parallel}, mundo={dinfo.world_size}, dispositivo={device}, dtype={cfg.dtype}")
    say(f"[datos] {tokens_per_step:,} tokens/paso, {cfg.max_steps} pasos -> "
        f"{tokens_per_step * cfg.max_steps / 1e6:.1f}M tokens")

    stop = {"reason": None}

    def _on_signal(signum, frame):
        stop["reason"] = f"signal_{signum}"

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError):
            signal.signal(sig, _on_signal)

    def trainer_state() -> dict:
        return {**state, "loader": train_loader.state_dict()}

    def save(step: int) -> None:
        p = ckpt.save(step, raw_model, optimizer, trainer_state(), model_cfg.to_dict(), scaler)
        say(f"[checkpoint] {p}")

    model.train()
    start_step = state["step"]
    t_session = time.time()
    elapsed_before = state["elapsed_s"]
    loss_acc, n_acc, bad_steps = 0.0, 0, 0
    t_log = time.time()
    tokens_log = 0
    last_saved = start_step

    while state["step"] < cfg.max_steps:
        step = state["step"]
        lr = lr_at(step, cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps, cfg.schedule, cfg.wsd_decay_frac)
        for g in optimizer.param_groups:
            g["lr"] = lr
        step_loss = torch.zeros((), device=device)
        aux_loss = torch.zeros((), device=device)
        for micro in range(cfg.grad_accum_steps):
            last = micro == cfg.grad_accum_steps - 1
            x, y = train_loader.next_batch()
            sync_ctx = contextlib.nullcontext()
            if parallel == "ddp" and not last:
                sync_ctx = fwd_model.no_sync() if hasattr(fwd_model, "no_sync") else contextlib.nullcontext()
            if parallel == "fsdp":
                model.set_requires_gradient_sync(last)
            with sync_ctx:
                with autocast_ctx():
                    out = fwd_model(x, targets=y)
                loss = out.loss / cfg.grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            step_loss += out.ce_loss.detach() / cfg.grad_accum_steps
            if model_cfg.is_moe:
                aux_loss += out.aux_loss.detach() / cfg.grad_accum_steps
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        step_loss = all_reduce_mean(step_loss, dinfo.world_size)
        finite = bool(torch.isfinite(step_loss).item() and torch.isfinite(grad_norm).item())
        if finite:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            bad_steps = 0
        else:
            bad_steps += 1
            say(f"[aviso] paso {step}: pérdida/gradiente no finitos; se omite la actualización ({bad_steps})")
            if bad_steps >= cfg.max_bad_steps:
                stop["reason"] = "bad_steps"
        optimizer.zero_grad(set_to_none=True)

        state["step"] = step + 1
        state["tokens"] += tokens_per_step
        tokens_log += tokens_per_step
        if finite:
            loss_acc += step_loss.item()
            n_acc += 1
        elapsed = elapsed_before + (time.time() - t_session)
        state["elapsed_s"] = elapsed
        cost = elapsed / 3600 * dinfo.world_size * cfg.gpu_hour_usd

        if state["step"] % cfg.log_interval == 0 or state["step"] == cfg.max_steps:
            dt = time.time() - t_log
            tps = tokens_log / max(dt, 1e-9)
            steps_done = state["step"] - start_step
            sec_per_step = (time.time() - t_session) / max(1, steps_done)
            eta_s = (cfg.max_steps - state["step"]) * sec_per_step
            rec = {
                "type": "train", "step": state["step"], "loss": loss_acc / max(1, n_acc), "lr": lr,
                "grad_norm": float(grad_norm), "tokens": state["tokens"], "tokens_per_s": tps,
                "elapsed_h": elapsed / 3600, "eta_h": eta_s / 3600, "cost_usd": cost,
                "projected_cost_usd": cost + eta_s / 3600 * dinfo.world_size * cfg.gpu_hour_usd,
            }
            if model_cfg.is_moe:
                rec["moe_aux"] = float(aux_loss)
            if peak:
                rec["mfu"] = tps * flops_per_token / (peak * dinfo.world_size)
            if device.type == "cuda":
                rec["max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            if is_main:
                append_jsonl(metrics_path, rec)
                write_json(out_dir / "heartbeat.json", {"step": state["step"], "unix": time.time(),
                                                        "loss": rec["loss"], "cost_usd": cost})
                say(f"paso {state['step']:>6} | pérdida {rec['loss']:.4f} | lr {lr:.2e} | "
                    f"‖g‖ {rec['grad_norm']:.2f} | {tps:,.0f} tok/s | "
                    + (f"MFU {rec['mfu'] * 100:.1f}% | " if peak else "")
                    + f"ETA {rec['eta_h'] * 60:.1f} min | ${cost:.2f}")
            loss_acc, n_acc, tokens_log, t_log = 0.0, 0, 0, time.time()

        do_eval = cfg.eval_interval and (state["step"] % cfg.eval_interval == 0 or state["step"] == cfg.max_steps)
        if do_eval:
            ev = evaluate(fwd_model if parallel == "fsdp" else raw_model, val_loader, cfg, dinfo, autocast_ctx)
            if ev:
                mean = ev["_mean"]["loss"]
                if state["best_val"] is None or mean < state["best_val"]:
                    state["best_val"] = mean
                if is_main:
                    append_jsonl(metrics_path, {"type": "eval", "step": state["step"], "tokens": state["tokens"],
                                                **{k: v for k, v in ev.items()}})
                    say(f"[eval] paso {state['step']} | " + " | ".join(
                        f"{k}: {v['loss']:.4f}" + (f" ({v['bpb']:.3f} bpb)" if "bpb" in v else "")
                        for k, v in ev.items()))
            if parallel != "fsdp" and is_main and cfg.sample_prompts:
                samples = _samples(raw_model, cfg.tokenizer, cfg.sample_prompts, cfg.sample_max_new_tokens, device)
                for s in samples:
                    append_jsonl(out_dir / "samples.jsonl", {"step": state["step"], **s})

        # Límites de gasto y tiempo (comprobados por todos los rangos de forma consistente)
        if cfg.budget_usd is not None and cost >= cfg.budget_usd:
            stop["reason"] = stop["reason"] or "budget"
        if cfg.max_hours is not None and elapsed >= cfg.max_hours * 3600:
            stop["reason"] = stop["reason"] or "time_limit"
        if dinfo.distributed:
            flag = torch.tensor([1.0 if stop["reason"] else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item() > 0 and not stop["reason"]:
                stop["reason"] = "peer_stop"

        if state["step"] % cfg.ckpt_interval == 0 or stop["reason"]:
            save(state["step"])
            last_saved = state["step"]
        if stop["reason"]:
            say(f"[parada] motivo: {stop['reason']}")
            break

    if last_saved != state["step"]:
        save(state["step"])
    summary = {
        "stop_reason": stop["reason"] or "completed", "steps": state["step"], "tokens": state["tokens"],
        "elapsed_h": state["elapsed_s"] / 3600, "best_val_loss": state["best_val"],
        "cost_usd": state["elapsed_s"] / 3600 * dinfo.world_size * cfg.gpu_hour_usd,
        "checkpoint": str(ckpt.latest()) if is_main else None,
    }
    if is_main:
        write_json(out_dir / "summary.json", summary)
    cleanup_distributed()
    return summary


def main_from_yaml(path: str, overrides: dict | None = None) -> dict:
    from .config import load_run_config
    model_cfg, train_cfg = load_run_config(path, overrides)
    if os.environ.get("TITAN_OUT_DIR"):
        train_cfg.out_dir = os.environ["TITAN_OUT_DIR"]
    return train(model_cfg, train_cfg)
