"""Cuantización GGUF con llama.cpp y medición de velocidad de inferencia.

Tipos recomendados para la RTX 5070 Laptop (8 GB):
  * Q8_0    casi sin pérdida; ~1,06 bytes/parámetro.
  * Q5_K_M  muy buena calidad; ~0,69 bytes/parámetro.
  * Q4_K_M  el equilibrio habitual; ~0,60 bytes/parámetro.
Los tipos "K" exigen filas múltiplo de 256; si un tensor no cumple, llama.cpp
usa automáticamente un tipo compatible para ese tensor.
"""

from __future__ import annotations

import time
from pathlib import Path

FTYPES = {"F16": 1, "Q4_0": 2, "Q8_0": 7, "Q5_0": 8, "Q2_K": 10, "Q3_K_M": 12, "Q4_K_S": 14, "Q4_K_M": 15,
          "Q5_K_S": 16, "Q5_K_M": 17, "Q6_K": 18}


def quantize_gguf(src: str | Path, dst: str | Path, qtype: str = "Q4_K_M", threads: int = 0) -> dict:
    import llama_cpp
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = FTYPES[qtype]
    params.nthread = threads
    t0 = time.time()
    rc = llama_cpp.llama_model_quantize(str(src).encode(), str(dst).encode(), params)
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize devolvió {rc}")
    return {"type": qtype, "src_mb": round(src.stat().st_size / 1e6, 2), "dst_mb": round(dst.stat().st_size / 1e6, 2),
            "seconds": round(time.time() - t0, 1), "path": str(dst)}


def bench_gguf(path: str | Path, prompt_tokens: int = 128, gen_tokens: int = 128, n_gpu_layers: int = -1,
               n_threads: int | None = None, n_ctx: int | None = None) -> dict:
    """Mide tokens/s de procesamiento del prompt y de generación (voraz)."""
    import llama_cpp
    llm = llama_cpp.Llama(model_path=str(path), n_gpu_layers=n_gpu_layers, n_threads=n_threads,
                          n_ctx=n_ctx or 0, verbose=False)
    ctx = llm.n_ctx()
    prompt_tokens = min(prompt_tokens, ctx // 2)
    gen_tokens = min(gen_tokens, ctx - prompt_tokens - 1)
    prompt = [(i * 7919) % 1000 + 100 for i in range(prompt_tokens)]
    llm.reset()
    t0 = time.time()
    llm.eval(prompt)
    t_prompt = time.time() - t0
    n = 0
    t1 = time.time()
    for _ in range(gen_tokens):  # decodificación token a token sobre el prompt ya evaluado
        tok = llm.sample(temp=0.0)
        llm.eval([tok])
        n += 1
    t_gen = time.time() - t1
    return {"model": str(path), "mb": round(Path(path).stat().st_size / 1e6, 2), "n_gpu_layers": n_gpu_layers,
            "prompt_tokens": prompt_tokens, "prompt_tok_s": round(prompt_tokens / t_prompt, 1),
            "gen_tokens": n, "gen_tok_s": round(n / t_gen, 1)}
