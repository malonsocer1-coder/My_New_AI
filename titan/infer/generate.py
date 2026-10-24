"""Generación autorregresiva con KV cache y muestreo controlado."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from ..model import Transformer


@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0
    stop_ids: list[int] = field(default_factory=list)
    seed: int | None = None


def _sample(logits: torch.Tensor, p: SamplingParams, history: list[int], gen: torch.Generator | None) -> int:
    logits = logits.float()
    if p.repetition_penalty != 1.0 and history:
        idx = torch.tensor(sorted(set(history)), device=logits.device)
        vals = logits[idx]
        logits[idx] = torch.where(vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty)
    if p.temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / p.temperature
    if p.top_k:
        k = min(p.top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    if p.top_p is not None and p.top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[(cum - sp) > p.top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs.cpu(), 1, generator=gen))


@torch.no_grad()
def generate_ids(model: Transformer, prompt_ids: list[int], params: SamplingParams,
                 prefill_chunk: int = 512) -> Iterator[int]:
    """Genera IDs uno a uno (streaming). El prompt se procesa por fragmentos."""
    model.eval()
    device = next(model.parameters()).device
    max_len = model.cfg.max_seq_len
    if len(prompt_ids) >= max_len:
        prompt_ids = prompt_ids[-(max_len - 1):]
    total = min(max_len, len(prompt_ids) + params.max_new_tokens)
    cache = model.new_cache(1, total)
    gen = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    pos = 0
    logits = None
    for i in range(0, len(prompt_ids), prefill_chunk):
        chunk = torch.tensor([prompt_ids[i:i + prefill_chunk]], device=device)
        logits = model(chunk, start_pos=pos, cache=cache, last_only=True).logits
        pos += chunk.shape[1]
    history = list(prompt_ids)
    stop = set(params.stop_ids)
    for _ in range(total - len(prompt_ids)):
        nxt = _sample(logits[0, -1], params, history, gen)
        if nxt in stop:
            return
        yield nxt
        history.append(nxt)
        if pos >= total:
            return
        logits = model(torch.tensor([[nxt]], device=device), start_pos=pos, cache=cache).logits
        pos += 1


def generate_text(model: Transformer, tokenizer, prompt: str, max_new_tokens: int = 128,
                  temperature: float = 0.7, top_k: int | None = 50, top_p: float | None = 0.95,
                  repetition_penalty: float = 1.0, seed: int | None = 0, allow_special: bool = False,
                  stop_ids: list[int] | None = None) -> str:
    ids = tokenizer.encode(prompt, allow_special=allow_special)
    params = SamplingParams(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            stop_ids=stop_ids if stop_ids is not None else [tokenizer.eos_id], seed=seed)
    return tokenizer.decode(list(generate_ids(model, ids, params)))
