"""Implementación del transformer de TITAN (desde cero, sobre primitivas de PyTorch).

Los nombres de los módulos siguen la convención de Meta Llama
(tok_embeddings, layers.N.attention.wq, feed_forward.w1, ...). El módulo
``titan.export`` traduce estos nombres a los de Hugging Face y GGUF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


# ----------------------------------------------------------------------
# Componentes básicos
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


def rope_tables(head_dim: int, max_len: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D)
    xf = x.float()
    out = xf * cos + rotate_half(xf) * sin
    return out.type_as(x)


class KVCache:
    """Caché de claves/valores preasignada para generación incremental."""

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, device, dtype):
        shape = (batch_size, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.max_len = max_len

    def update(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor):
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache llena: {end} > {self.max_len}")
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin, start_pos: int = 0, cache: KVCache | None = None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer_id, start_pos, k, v)
        if self.n_rep > 1:
            # La cabeza de consulta h usa la cabeza kv h // n_rep (igual que Llama/HF)
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        S = k.shape[2]
        if T == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        elif S == T:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Prefill por fragmentos con prefijo ya cacheado
            i = torch.arange(T, device=x.device).unsqueeze(1) + (S - T)
            j = torch.arange(S, device=x.device).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=j <= i)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU: w2(silu(w1 x) * w3 x)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFeedForward(nn.Module):
    """Mixture of Experts con enrutamiento top-k (convención Mixtral).

    Devuelve la salida y las pérdidas auxiliares: balance de carga (Switch
    Transformer) y z-loss del router (ST-MoE) para estabilidad numérica.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_token
        self.gate = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(cfg.dim, cfg.ffn_dim) for _ in range(cfg.n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.gate(flat)
        probs = F.softmax(logits.float(), dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        top_w = (top_w / top_w.sum(-1, keepdim=True)).to(flat.dtype)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            # Se llama al experto aunque no reciba tokens: así todos los
            # parámetros participan en el grafo (necesario para DDP/FSDP).
            y = expert(flat[tok]) * top_w[tok, slot].unsqueeze(-1)
            out.index_add_(0, tok, y)
        frac = F.one_hot(top_i, self.n_experts).sum(1).float().mean(0) / self.top_k
        aux = self.n_experts * (frac * probs.mean(0)).sum()
        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        return out.view(B, T, D), aux, z, frac.detach()


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention = Attention(cfg, layer_id)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.is_moe = cfg.is_moe
        self.feed_forward = MoEFeedForward(cfg) if cfg.is_moe else FeedForward(cfg.dim, cfg.ffn_dim)

    def forward(self, x, cos, sin, start_pos: int = 0, cache: KVCache | None = None):
        h = x + self.attention(self.attention_norm(x), cos, sin, start_pos, cache)
        if self.is_moe:
            ff, aux, z, _ = self.feed_forward(self.ffn_norm(h))
        else:
            ff = self.feed_forward(self.ffn_norm(h))
            aux = z = h.new_zeros((), dtype=torch.float32)
        return h + ff, aux, z


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    ce_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.init_weights()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_weights(self, seed: int | None = None) -> None:
        """Inicialización aleatoria propia (no se cargan pesos externos).

        Puede llamarse de nuevo tras crear el modelo en el dispositivo "meta"
        y materializarlo con ``to_empty`` (modelos que no caben en una GPU).
        """
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
        std = self.cfg.init_std
        resid_std = std / math.sqrt(2 * self.cfg.n_layers)

        def normal_(t: torch.Tensor, s: float) -> None:
            if gen is None or t.device.type == "meta":
                nn.init.normal_(t, 0.0, s)
            else:
                t.copy_(torch.empty(t.shape, dtype=torch.float32).normal_(0.0, s, generator=gen).to(t))

        normal_(self.tok_embeddings.weight, std)
        for layer in self.layers:
            att = layer.attention
            for lin in (att.wq, att.wk, att.wv):
                normal_(lin.weight, std)
            normal_(att.wo.weight, resid_std)
            ffs = layer.feed_forward.experts if layer.is_moe else [layer.feed_forward]
            if layer.is_moe:
                normal_(layer.feed_forward.gate.weight, std)
            for ff in ffs:
                normal_(ff.w1.weight, std)
                normal_(ff.w3.weight, std)
                normal_(ff.w2.weight, resid_std)
            layer.attention_norm.reset_parameters()
            layer.ffn_norm.reset_parameters()
        self.norm.reset_parameters()
        if not self.cfg.tie_embeddings:
            normal_(self.output.weight, std)
        cos, sin = rope_tables(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta,
                               device=self.rope_cos.device if self.rope_cos.device.type != "meta" else None)
        if self.rope_cos.device.type != "meta":
            self.rope_cos.copy_(cos)
            self.rope_sin.copy_(sin)

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int = 0,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> ModelOutput:
        B, T = tokens.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"Secuencia de {start_pos + T} tokens supera max_seq_len={self.cfg.max_seq_len}")
        h = self.tok_embeddings(tokens)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        aux_total = h.new_zeros((), dtype=torch.float32)
        z_total = h.new_zeros((), dtype=torch.float32)
        use_ckpt = self.cfg.activation_checkpointing and self.training and cache is None
        for layer in self.layers:
            if use_ckpt:
                h, aux, z = checkpoint(layer, h, cos, sin, start_pos, None, use_reentrant=False)
            else:
                h, aux, z = layer(h, cos, sin, start_pos, cache)
            aux_total = aux_total + aux
            z_total = z_total + z
        h = self.norm(h)
        if last_only:
            h = h[:, -1:, :]
        logits = self.output(h)
        if targets is None:
            return ModelOutput(logits=logits)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        loss = ce
        if self.cfg.is_moe:
            n = self.cfg.n_layers
            aux_total = aux_total / n
            z_total = z_total / n
            loss = ce + self.cfg.moe_aux_loss_coef * aux_total + self.cfg.moe_z_loss_coef * z_total
        return ModelOutput(logits=logits, loss=loss, ce_loss=ce, aux_loss=aux_total, z_loss=z_total)

    # ------------------------------------------------------------------
    def new_cache(self, batch_size: int, max_len: int | None = None, dtype=None) -> KVCache:
        p = next(self.parameters())
        return KVCache(self.cfg, batch_size, max_len or self.cfg.max_seq_len, p.device, dtype or p.dtype)
