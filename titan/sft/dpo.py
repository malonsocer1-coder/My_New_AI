"""Direct Preference Optimization (DPO) sobre un modelo TITAN ajustado con SFT.

Pérdida (Rafailov et al., 2023):
    L = -log σ( β · [ (log πθ(y⁺|x) − log πref(y⁺|x)) − (log πθ(y⁻|x) − log πref(y⁻|x)) ] )
con πref = copia congelada del modelo de partida. Solo cuentan los tokens del
último turno del asistente. Métricas: precisión de preferencia (fracción de
pares donde la recompensa implícita de la preferida supera a la rechazada) y
margen medio, en entrenamiento y en pares reservados.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..infer.engine import load_model, resolve_artifact
from ..tokenizer import TitanTokenizer
from ..train.checkpoint import model_state_for_save
from ..utils.io import append_jsonl, read_jsonl, read_yaml, write_json
from .chat import ChatFormat


@dataclass
class DPOConfig:
    init_from: str
    data: str
    eval_data: str | None = None
    out_dir: str = "runs/dpo"
    beta: float = 0.1
    lr: float = 2e-5
    steps: int = 200
    batch_size: int = 8
    warmup_steps: int = 20
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_interval: int = 10
    seed: int = 0
    device: str = "cpu"
    num_threads: int | None = None


def encode_pair(fmt: ChatFormat, pair: dict, max_len: int) -> tuple[list[int], list[int], list[int], list[int]] | None:
    prefix = fmt.prompt_ids(pair["messages"], pair.get("tools"))
    out = []
    for text in (pair["chosen"], pair["rejected"]):
        body = fmt.assistant_body({"content": text}) + [fmt.im_end]
        ids = prefix + body
        if len(ids) > max_len:
            return None
        out += [ids, [0] * len(prefix) + [1] * len(body)]
    return tuple(out)


def _batch_logps(model, seqs: list[list[int]], masks: list[list[int]], device, pad_id: int) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.float32)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s)
        m[i, :len(k)] = torch.tensor(k, dtype=torch.float32)
    ids, m = ids.to(device), m.to(device)
    logits = model(ids[:, :-1]).logits.float()
    logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (logp * m[:, 1:]).sum(-1)


def dpo_terms(policy, ref, batch, device, pad_id, beta):
    seqs = [b[0] for b in batch] + [b[2] for b in batch]
    masks = [b[1] for b in batch] + [b[3] for b in batch]
    n = len(batch)
    pol = _batch_logps(policy, seqs, masks, device, pad_id)
    with torch.no_grad():
        rf = _batch_logps(ref, seqs, masks, device, pad_id)
    r_c = beta * (pol[:n] - rf[:n])
    r_r = beta * (pol[n:] - rf[n:])
    loss = -F.logsigmoid(r_c - r_r).mean()
    return loss, (r_c > r_r).float().mean().item(), (r_c - r_r).mean().item()


@torch.no_grad()
def preference_accuracy(policy, ref, pairs, device, pad_id, beta, bs=8) -> dict:
    acc = marg = 0.0
    n = 0
    for i in range(0, len(pairs), bs):
        b = pairs[i:i + bs]
        _, a, m = dpo_terms(policy, ref, b, device, pad_id, beta)
        acc += a * len(b)
        marg += m * len(b)
        n += len(b)
    return {"pref_accuracy": round(acc / n, 4), "reward_margin": round(marg / n, 4), "n": n}


def train_dpo(cfg: DPOConfig, log=print) -> dict:
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)
    _, _, tok_path = resolve_artifact(cfg.init_from)
    tok = TitanTokenizer.load(tok_path)
    fmt = ChatFormat(tok)
    policy = load_model(cfg.init_from, device=cfg.device, dtype=torch.float32)
    policy.train()
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    max_len = policy.cfg.max_seq_len
    enc = [e for e in (encode_pair(fmt, p, max_len) for p in read_jsonl(cfg.data)) if e]
    ev = [e for e in (encode_pair(fmt, p, max_len) for p in read_jsonl(cfg.eval_data)) if e] if cfg.eval_data else []
    log(f"[dpo] {len(enc)} pares de entrenamiento, {len(ev)} de evaluación, β={cfg.beta}")
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = preference_accuracy(policy, ref, ev, device, tok.pad_id, cfg.beta) if ev else {}
    if before:
        log(f"[dpo] evaluación antes: {before}")
    t0 = time.time()
    order = list(range(len(enc)))
    rng = random.Random(cfg.seed)
    pos = len(order)
    for step in range(1, cfg.steps + 1):
        lr = cfg.lr * min(1.0, step / max(1, cfg.warmup_steps)) * 0.5 * (1 + math.cos(math.pi * step / cfg.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        if pos + cfg.batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        batch = [enc[i] for i in order[pos:pos + cfg.batch_size]]
        pos += cfg.batch_size
        loss, acc, margin = dpo_terms(policy, ref, batch, device, tok.pad_id, cfg.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_interval == 0 or step == cfg.steps:
            rec = {"step": step, "loss": round(loss.item(), 4), "pref_acc": acc, "margin": round(margin, 4),
                   "lr": lr, "elapsed_s": round(time.time() - t0, 1)}
            append_jsonl(out / "metrics.jsonl", rec)
            log(f"[dpo] paso {step} | pérdida {rec['loss']} | acierto {acc:.2f} | margen {rec['margin']}")
    policy.eval()
    after = preference_accuracy(policy, ref, ev, device, tok.pad_id, cfg.beta) if ev else {}
    from safetensors.torch import save_file
    save_file(model_state_for_save(policy), str(out / "model.safetensors"))
    write_json(out / "model_config.json", policy.cfg.to_dict())
    (out / "tokenizer.json").write_bytes(Path(tok_path).read_bytes())
    summary = {"config": asdict(cfg), "eval_before": before, "eval_after": after,
               "elapsed_h": round((time.time() - t0) / 3600, 3)}
    write_json(out / "summary.json", summary)
    log(f"[dpo] evaluación después: {after}")
    return summary


def main_from_yaml(path: str) -> dict:
    return train_dpo(DPOConfig(**read_yaml(path)))


if __name__ == "__main__":
    import sys
    print(json.dumps(main_from_yaml(sys.argv[1]), indent=2, ensure_ascii=False))
