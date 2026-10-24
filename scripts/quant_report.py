"""Informe de cuantización: tamaño, perplejidad en validación y velocidad de cada variante.

Uso:
  python scripts/quant_report.py --model models/base/titan-stage-a --data data/stage_a/manifest.json \
      --out reports/quantization_stage_a.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from titan.data.loader import EvalLoader  # noqa: E402
from titan.export.gguf_tools import bench_gguf, quantize_gguf  # noqa: E402
from titan.export.quantize import quantize_model_dir  # noqa: E402
from titan.infer.engine import load_model  # noqa: E402


def windows(manifest, seq_len, n, domains=("es", "en", "code")):
    ev = EvalLoader(manifest, "val", seq_len)
    out = []
    for d in domains:
        if d in ev.streams:
            for x, y in ev.batches(d, n, 1):
                out.append(torch.cat([x[0], y[0, -1:]]).tolist())
    return out


@torch.no_grad()
def nll_torch(model, wins):
    tot = cnt = 0.0
    for w in wins:
        t = torch.tensor([w])
        o = model(t[:, :-1], targets=t[:, 1:])
        tot += o.ce_loss.item() * (len(w) - 1)
        cnt += len(w) - 1
    return tot / cnt


def nll_gguf(path, wins, n_ctx):
    import llama_cpp
    llm = llama_cpp.Llama(model_path=str(path), n_ctx=n_ctx, logits_all=True, verbose=False, n_gpu_layers=0)
    tot = cnt = 0.0
    for w in wins:
        llm.reset()
        llm.eval(w[:-1])
        logits = np.array(llm.scores[: len(w) - 1], dtype=np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        tot -= logp[np.arange(len(w) - 1), w[1:]].sum()
        cnt += len(w) - 1
    return tot / cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, default=16)
    p.add_argument("--qdir", default="models/quantized")
    a = p.parse_args()
    model_dir = Path(a.model)
    cfg = json.loads((model_dir / "model_config.json").read_text())
    seq = cfg["max_seq_len"]
    wins = windows(a.data, seq, a.windows)
    name = cfg["name"]
    qdir = Path(a.qdir)
    rows = []

    ref = load_model(model_dir, device="cpu", dtype=torch.float32)
    base_nll = nll_torch(ref, wins)
    size = (model_dir / "model.safetensors").stat().st_size
    rows.append({"variant": "TITAN bf16 (original)", "engine": "pytorch", "mb": round(size / 1e6, 2),
                 "nll": round(base_nll, 4), "ppl": round(math.exp(base_nll), 3)})
    for bits in (8, 4):
        out = qdir / f"{name}-int{bits}"
        info = quantize_model_dir(model_dir, out, bits=bits, group_size=64)
        nll = nll_torch(load_model(out, device="cpu", dtype=torch.float32), wins)
        rows.append({"variant": f"TITAN int{bits} (propia)", "engine": "pytorch", "mb": info["quantized_mb"],
                     "nll": round(nll, 4), "ppl": round(math.exp(nll), 3)})

    f16 = next((model_dir / "gguf").glob("*.gguf"))
    variants = [("F16", f16)]
    for q in ("Q8_0", "Q5_K_M", "Q4_K_M"):
        dst = qdir / f"{name}-{q}.gguf"
        quantize_gguf(f16, dst, q)
        variants.append((q, dst))
    for q, path in variants:
        nll = nll_gguf(path, wins, seq)
        b = bench_gguf(path, prompt_tokens=min(128, seq // 2), gen_tokens=min(96, seq // 2), n_gpu_layers=0,
                       n_ctx=seq)
        rows.append({"variant": f"GGUF {q}", "engine": "llama.cpp (CPU)", "mb": b["mb"], "nll": round(nll, 4),
                     "ppl": round(math.exp(nll), 3), "prompt_tok_s": b["prompt_tok_s"], "gen_tok_s": b["gen_tok_s"]})

    # Velocidad del motor PyTorch (generación voraz con caché KV)
    from titan.infer.generate import SamplingParams, generate_ids
    prompt = wins[0][:64]
    t0 = time.time()
    n = len(list(generate_ids(ref, prompt, SamplingParams(max_new_tokens=96, temperature=0.0, stop_ids=[]))))
    rows[0]["gen_tok_s"] = round(n / (time.time() - t0), 1)

    for r in rows:
        r["ppl_increase_pct"] = round((r["ppl"] / rows[0]["ppl"] - 1) * 100, 2)
    report = {"model": str(model_dir), "eval_tokens": sum(len(w) - 1 for w in wins), "rows": rows,
              "hardware": "4 vCPU Xeon de la sesión en la nube (sin GPU); no representa la RTX 5070",
              "unix": time.time()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
