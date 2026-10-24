"""Cuantización de pesos (solo pesos) para el motor PyTorch de TITAN.

  * int8: simétrica por canal de salida (una escala por fila).
  * int4: simétrica por grupos de ``group_size`` columnas, dos valores por byte.

Las normas (vectores 1D) se conservan en fp16. Los pesos originales no se
tocan: el resultado es un artefacto derivado en otro directorio con su
MANIFEST. Al cargar se descuantiza a la precisión de cómputo: reduce el disco
y la descarga, pero no la RAM en ejecución. Para memoria y velocidad reales en
la RTX 5070 use GGUF (Q8_0 / Q5_K_M / Q4_K_M) con llama.cpp (ver
``titan.export.gguf_tools``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..utils.io import sha256_file, write_json


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()


def quantize_int4(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = w.shape
    g = group_size if cols % group_size == 0 else cols
    wg = w.float().reshape(rows, cols // g, g)
    scale = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.round(wg / scale).clamp(-8, 7).to(torch.int8).reshape(rows, cols)
    u = (q + 8).to(torch.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(-1).to(torch.float16)


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, cols: int) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = (packed >> 4).to(torch.int8) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], cols).float()
    groups = scale.shape[1]
    return (q.reshape(q.shape[0], groups, cols // groups) * scale.float().unsqueeze(-1)).reshape(q.shape[0], cols)


def quantize_model_dir(model_dir: str | Path, out: str | Path, bits: int = 8, group_size: int = 128) -> dict:
    src, out = Path(model_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src / "model.safetensors"))
    qsd: dict[str, torch.Tensor] = {}
    shapes = {}
    orig_bytes = q_bytes = 0
    for k, v in sd.items():
        orig_bytes += v.numel() * v.element_size()
        if v.ndim == 2:
            q, s = quantize_int8(v) if bits == 8 else quantize_int4(v, group_size)
            qsd[k + ".q"], qsd[k + ".scale"] = q, s
            shapes[k] = list(v.shape)
        else:
            qsd[k] = v.to(torch.float16)
    for t in qsd.values():
        q_bytes += t.numel() * t.element_size()
    save_file(qsd, str(out / "model.safetensors"))
    for f in ("model_config.json", "tokenizer.json", "generation_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    meta = {"bits": bits, "group_size": group_size if bits == 4 else None, "shapes": shapes,
            "source": str(src), "source_sha256": sha256_file(src / "model.safetensors")}
    write_json(out / "quantization.json", meta)
    files = {f.name: sha256_file(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != "MANIFEST.json"}
    write_json(out / "MANIFEST.json", {"files": files, "derived_from": str(src)})
    return {"out": str(out), "bits": bits, "original_mb": round(orig_bytes / 1e6, 2),
            "quantized_mb": round(q_bytes / 1e6, 2), "ratio": round(orig_bytes / max(1, q_bytes), 2)}


def dequantized_state_dict(qdir: str | Path) -> dict[str, torch.Tensor]:
    qdir = Path(qdir)
    meta = json.loads((qdir / "quantization.json").read_text())
    qsd = load_file(str(qdir / "model.safetensors"))
    sd = {}
    for k, v in qsd.items():
        if k.endswith(".scale"):
            continue
        if k.endswith(".q"):
            name = k[:-2]
            s = qsd[name + ".scale"]
            sd[name] = dequantize_int8(v, s) if meta["bits"] == 8 else dequantize_int4(v, s, meta["shapes"][name][1])
        else:
            sd[k] = v.float()
    return sd


def load_quantized_into(model: torch.nn.Module, qdir: str | Path) -> None:
    sd = dequantized_state_dict(qdir)
    tied = getattr(model.cfg, "tie_embeddings", False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not (tied and m == "output.weight")]
    if missing or unexpected:
        raise RuntimeError(f"Pesos cuantizados incompatibles: faltan {missing[:5]}, sobran {unexpected[:5]}")


@torch.no_grad()
def compare_logits(ref_model, other_model, tokens: torch.Tensor) -> dict:
    """Divergencia KL media y coincidencia del token más probable entre dos modelos."""
    a = torch.log_softmax(ref_model(tokens).logits.float(), -1)
    b = torch.log_softmax(other_model(tokens).logits.float(), -1)
    kl = (a.exp() * (a - b)).sum(-1).mean().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return {"kl_divergence": round(kl, 6), "top1_agreement": round(top1, 4)}
