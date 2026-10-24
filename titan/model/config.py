"""Configuración de la arquitectura de TITAN.

La arquitectura es un transformer decodificador (autorregresivo) con:
  * Pre-normalización RMSNorm.
  * Embeddings posicionales rotatorios (RoPE, convención "rotate_half").
  * Atención multi-cabeza con grupos de claves/valores (GQA).
  * FFN SwiGLU, o capas Mixture of Experts con enrutamiento top-k.
  * Sin sesgos en proyecciones lineales.

Esta combinación es compatible tensor a tensor con los formatos Llama (densa)
y Mixtral (MoE), lo que permite exportar los pesos propios a GGUF/llama.cpp o
al formato de Hugging Face sin depender de pesos de terceros.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


def _round_up(x: int, multiple: int) -> int:
    return multiple * ((x + multiple - 1) // multiple)


@dataclass
class ModelConfig:
    name: str = "titan"
    vocab_size: int = 32768
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None
    ffn_dim: int | None = None
    multiple_of: int = 256
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    init_std: float = 0.02
    # Mixture of Experts (n_experts == 0 => modelo denso)
    n_experts: int = 0
    n_experts_per_token: int = 2
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.001
    # Recalcular activaciones en el backward para ahorrar memoria
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.ffn_dim is None:
            # 2/3 de 4*dim, redondeado (convención SwiGLU de Llama)
            self.ffn_dim = _round_up(int(2 * 4 * self.dim / 3), self.multiple_of)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim={self.dim} no es divisible por n_heads={self.n_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads debe ser múltiplo de n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim debe ser par para RoPE")
        if self.n_experts and not (1 <= self.n_experts_per_token <= self.n_experts):
            raise ValueError("n_experts_per_token fuera de rango")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    # ------------------------------------------------------------------
    # Conteo analítico de parámetros (debe coincidir con el modelo real)
    # ------------------------------------------------------------------
    def param_counts(self) -> dict[str, int]:
        d, hd = self.dim, self.head_dim
        attn = d * (self.n_heads * hd) + 2 * d * (self.n_kv_heads * hd) + (self.n_heads * hd) * d
        expert = 3 * d * self.ffn_dim
        if self.is_moe:
            ffn_total = self.n_experts * expert + d * self.n_experts
            ffn_active = self.n_experts_per_token * expert + d * self.n_experts
        else:
            ffn_total = ffn_active = expert
        norms = 2 * d
        per_layer_total = attn + ffn_total + norms
        per_layer_active = attn + ffn_active + norms
        embed = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        total = self.n_layers * per_layer_total + embed + head + d
        active = self.n_layers * per_layer_active + embed + head + d
        non_embedding = self.n_layers * per_layer_active + d
        return {
            "total": total,
            "active": active,
            "non_embedding_active": non_embedding,
            "embedding": embed,
            "lm_head": head,
            "per_layer_total": per_layer_total,
        }

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """FLOPs de entrenamiento (fwd+bwd) por token: 6*N_activos + término de atención."""
        seq = seq_len or self.max_seq_len
        n = self.param_counts()["non_embedding_active"] + (self.vocab_size * self.dim)
        attn = 12 * self.n_layers * self.dim * seq  # 6 * 2 * L * d * T (QK^T y AV)
        return 6.0 * n + attn

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Claves desconocidas en la config del modelo: {sorted(unknown)}")
        return cls(**data)
