"""Estimación de memoria, tiempo y coste de entrenamiento e inferencia.

Fórmulas:
  * Cómputo de entrenamiento ≈ FLOPs/token (6·N_activos + atención) × tokens.
  * Tiempo = cómputo / (n_gpus × pico_bf16 × MFU). MFU realista: 35–45 % denso,
    25–35 % MoE.
  * Memoria de entrenamiento con AdamW en precisión mixta ≈ 16 bytes/parámetro
    (pesos fp32 + gradiente + 2 momentos), repartidos entre GPUs con FSDP,
    más activaciones (≈ 34·d bytes/token/capa sin recálculo; ≈ 2·d con él).
  * Inferencia: pesos según el tipo de cuantización + caché KV. La velocidad de
    generación está limitada por el ancho de banda de memoria:
    tokens/s ≈ ancho_de_banda_efectivo / bytes leídos por token.

Los precios son referencias públicas de octubre de 2026 (rangos de mercado);
cambian a diario: verifique la oferta real antes de lanzar nada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..model import ModelConfig


@dataclass(frozen=True)
class GPU:
    name: str
    mem_gb: float
    bf16_tflops: float       # denso, sin dispersión
    bandwidth_gbs: float
    price_low: float         # USD por GPU-hora (mercado/comunidad)
    price_high: float        # USD por GPU-hora (bajo demanda, nube segura)


GPUS = {
    "H100": GPU("H100 SXM 80 GB", 80, 989, 3350, 1.50, 3.99),
    "H200": GPU("H200 141 GB", 141, 989, 4800, 3.59, 6.35),
    "B200": GPU("B200 192 GB", 192, 2250, 8000, 4.99, 8.41),
    "A100": GPU("A100 80 GB", 80, 312, 2039, 0.90, 1.89),
    "RTX4090": GPU("RTX 4090 24 GB", 24, 82.6, 1008, 0.30, 0.70),
    "RTX5070L": GPU("RTX 5070 Laptop 8 GB (estimado)", 8, 50, 384, 0.0, 0.0),
}

# Bytes por parámetro en inferencia (incluye escalas de cuantización)
QUANT_BYTES = {"BF16": 2.0, "Q8_0": 1.0625, "Q6_K": 0.82, "Q5_K_M": 0.71, "Q4_K_M": 0.606, "Q3_K_M": 0.49}

PRESETS: dict[str, dict] = {
    "titan-stage-a": dict(vocab_size=8192, dim=256, n_layers=6, n_heads=8, n_kv_heads=4, multiple_of=64, max_seq_len=256),
    "titan-100m": dict(vocab_size=32768, dim=768, n_layers=12, n_heads=12, n_kv_heads=4, multiple_of=256,
                       max_seq_len=1024, tie_embeddings=True),
    "titan-350m": dict(vocab_size=65536, dim=1024, n_layers=24, n_heads=16, n_kv_heads=4, multiple_of=256,
                       max_seq_len=4096, tie_embeddings=True),
    "titan-1b": dict(vocab_size=65536, dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn_dim=8192,
                     max_seq_len=4096, tie_embeddings=True, rope_theta=500000.0),
    "titan-3b": dict(vocab_size=65536, dim=3072, n_layers=28, n_heads=24, n_kv_heads=8, ffn_dim=8192,
                     max_seq_len=8192, tie_embeddings=True, rope_theta=500000.0),
    "titan-8b": dict(vocab_size=65536, dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, ffn_dim=14336,
                     max_seq_len=8192, rope_theta=500000.0),
    "titan-moe-7b-a2b": dict(vocab_size=65536, dim=2048, n_layers=24, n_heads=16, n_kv_heads=4, ffn_dim=5632,
                             n_experts=8, n_experts_per_token=2, max_seq_len=8192, rope_theta=500000.0),
    "titan-30b": dict(vocab_size=65536, dim=6656, n_layers=60, n_heads=52, n_kv_heads=4, ffn_dim=17920,
                      max_seq_len=8192, rope_theta=500000.0),
    "titan-70b": dict(vocab_size=65536, dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, ffn_dim=28672,
                      max_seq_len=8192, rope_theta=500000.0),
    "titan-100b": dict(vocab_size=65536, dim=9216, n_layers=88, n_heads=72, n_kv_heads=8, ffn_dim=32768,
                       max_seq_len=8192, rope_theta=500000.0),
}

# Tokens de entrenamiento recomendados (más allá de Chinchilla ≈ 20/parámetro:
# entrenar de más abarata la inferencia, que es lo que importa en un PC).
RECOMMENDED_TOKENS = {
    "titan-stage-a": 2.0e7, "titan-100m": 2.0e9, "titan-350m": 2.0e10, "titan-1b": 1.0e11, "titan-3b": 5.0e11,
    "titan-8b": 1.5e12, "titan-moe-7b-a2b": 1.0e12, "titan-30b": 1.5e12, "titan-70b": 2.0e12, "titan-100b": 2.0e12,
}


def preset_config(name: str) -> ModelConfig:
    return ModelConfig(name=name, **PRESETS[name])


def train_memory_gb(cfg: ModelConfig, n_gpus: int, micro_tokens: int, act_ckpt: bool = True,
                    bytes_per_param: float = 16.0) -> dict:
    n = cfg.param_counts()["total"]
    states = n * bytes_per_param / n_gpus
    per_tok_layer = (2 if act_ckpt else 34) * cfg.dim
    acts = micro_tokens * cfg.n_layers * per_tok_layer + (micro_tokens * 34 * cfg.dim if act_ckpt else 0)
    ce_tokens = min(micro_tokens, cfg.ce_chunk_size) if cfg.ce_chunk_size else micro_tokens
    logits = ce_tokens * cfg.vocab_size * 10  # bf16 + copia fp32 + gradiente (por fragmento si se trocea)
    total = states + acts + logits + 1.5e9  # contexto CUDA, buffers de comunicación, fragmentación
    return {"states_gb": states / 1e9, "activations_gb": acts / 1e9, "logits_gb": logits / 1e9, "total_gb": total / 1e9}


def estimate(cfg: ModelConfig, tokens: float, gpu: str = "H100", n_gpus: int = 8, mfu: float | None = None,
             price: float | None = None, seq_len: int | None = None) -> dict:
    g = GPUS[gpu]
    mfu = mfu or (0.30 if cfg.is_moe else 0.40)
    seq = seq_len or min(cfg.max_seq_len, 4096)
    flops = cfg.flops_per_token(seq) * tokens
    hours = flops / (n_gpus * g.bf16_tflops * 1e12 * mfu) / 3600
    pc = cfg.param_counts()
    mem = train_memory_gb(cfg, n_gpus, micro_tokens=seq * 2)
    lo, hi = (price, price) if price else (g.price_low, g.price_high)
    min_gpus = 1
    while train_memory_gb(cfg, min_gpus, seq * 1)["total_gb"] > 0.9 * g.mem_gb and min_gpus < 4096:
        min_gpus *= 2
    return {
        "model": cfg.name, "params_total_B": round(pc["total"] / 1e9, 3), "params_active_B": round(pc["active"] / 1e9, 3),
        "tokens_B": round(tokens / 1e9, 1), "tokens_per_param": round(tokens / pc["total"], 1),
        "train_flops": f"{flops:.2e}", "gpu": g.name, "n_gpus": n_gpus, "mfu": mfu,
        "hours": round(hours, 1), "days": round(hours / 24, 2), "gpu_hours": round(hours * n_gpus),
        "cost_usd_low": round(hours * n_gpus * lo), "cost_usd_high": round(hours * n_gpus * hi),
        "min_gpus_to_fit": min_gpus, "mem_per_gpu_gb": round(mem["total_gb"], 1),
        "fits": mem["total_gb"] < 0.9 * g.mem_gb,
        "weights_bf16_gb": round(pc["total"] * 2 / 1e9, 2),
        "checkpoint_gb": round(pc["total"] * 16 / 1e9, 1),
    }


def inference(cfg: ModelConfig, quant: str = "Q4_K_M", ctx: int = 4096, vram_gb: float = 8.0,
              ram_gb: float = 16.0, gpu_bw: float = 384, ram_bw: float = 60, efficiency: float = 0.6) -> dict:
    pc = cfg.param_counts()
    weights = pc["total"] * QUANT_BYTES[quant] / 1e9
    active_bytes = pc["active"] * QUANT_BYTES[quant] / 1e9
    kv = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * ctx * 2 / 1e9
    overhead = 0.6
    need = weights + kv + overhead
    gpu_frac = min(1.0, max(0.0, (vram_gb - kv - overhead) / weights)) if weights else 1.0
    # Tiempo por token: bytes leídos en GPU / ancho de banda GPU + bytes en RAM / ancho de banda RAM
    t = active_bytes * gpu_frac / (gpu_bw * efficiency) + active_bytes * (1 - gpu_frac) / ram_bw
    t += 0.002  # sobrecoste fijo por token (lanzamiento de kernels, muestreo): techo ~500 tok/s
    return {"model": cfg.name, "quant": quant, "weights_gb": round(weights, 2), "kv_cache_gb": round(kv, 2),
            "total_gb": round(need, 2), "fits_vram": need <= vram_gb, "fits_vram_plus_ram": need <= vram_gb + ram_gb * 0.6,
            "gpu_layers_fraction": round(gpu_frac, 2), "est_tok_s": round(1 / t, 1) if t > 0 else None}


def scaling_table(gpu: str = "H100", n_gpus: int = 8) -> list[dict]:
    rows = []
    for name in PRESETS:
        cfg = preset_config(name)
        e = estimate(cfg, RECOMMENDED_TOKENS[name], gpu, n_gpus)
        best_q = next((q for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M") if inference(cfg, q)["fits_vram"]), None)
        inf = inference(cfg, best_q or "Q4_K_M")
        e.update({"local_quant": best_q or "no cabe en 8 GB", "local_total_gb": inf["total_gb"],
                  "local_est_tok_s": inf["est_tok_s"], "local_fits_with_ram_offload": inf["fits_vram_plus_ram"]})
        rows.append(e)
    return rows


def estimate_from_args(args) -> None:
    from ..utils.io import read_yaml
    if args.table:
        rows = scaling_table(args.gpu, args.num_gpus)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return
        print(f"{'modelo':<18}{'params':>9}{'tokens':>9}{'GPU-h':>9}{'días':>7}{'USD bajo':>10}{'USD alto':>10}"
              f"{'pesos bf16':>11}  {'local (8 GB)':>15}{'tok/s':>7}")
        for r in rows:
            print(f"{r['model']:<18}{r['params_total_B']:>8.2f}B{r['tokens_B']:>8.0f}B{r['gpu_hours']:>9}{r['days']:>7.1f}"
                  f"{r['cost_usd_low']:>10,}{r['cost_usd_high']:>10,}{r['weights_bf16_gb']:>10.1f}G"
                  f"  {r['local_quant']:>15}{(r['local_est_tok_s'] or 0):>7.0f}")
        return
    if args.preset:
        cfg = preset_config(args.preset)
    elif args.config:
        cfg = ModelConfig.from_dict(read_yaml(args.config))
    else:
        raise SystemExit("Indique --preset, --config o --table")
    tokens = args.tokens or RECOMMENDED_TOKENS.get(cfg.name, 20 * cfg.param_counts()["total"])
    res = {"train": estimate(cfg, tokens, args.gpu, args.num_gpus, args.mfu, args.price),
           "inference_rtx5070_8gb": {q: inference(cfg, q) for q in QUANT_BYTES}}
    print(json.dumps(res, indent=2, ensure_ascii=False))
