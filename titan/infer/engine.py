"""Motor de inferencia: carga pesos propios desde cualquier artefacto de TITAN.

Acepta:
  * un directorio de ejecución (``runs/x``: usa el último checkpoint),
  * un checkpoint (``runs/x/checkpoints/step_N``),
  * un modelo exportado (``models/base/x`` con config.json + model.safetensors),
  * un modelo cuantizado con ``titan.export.quantize`` (int8/int4).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch

from ..model import ModelConfig, Transformer
from ..sft.chat import ChatFormat
from ..tokenizer import TitanTokenizer
from ..train.checkpoint import load_model_weights
from ..train.distributed import pick_device
from .generate import SamplingParams, generate_ids


def resolve_artifact(path: str | Path) -> tuple[Path, Path, Path | None]:
    """Devuelve (directorio_de_pesos, config_json, tokenizador)."""
    p = Path(path)
    if (p / "checkpoints").is_dir():
        latest = (p / "checkpoints" / "LATEST")
        name = latest.read_text().strip() if latest.exists() else sorted((p / "checkpoints").glob("step_*"))[-1].name
        p = p / "checkpoints" / name
    if (p / "model_config.json").exists():
        cfg = p / "model_config.json"
    elif (p / "config.json").exists():
        cfg = p / "config.json"
    else:
        raise FileNotFoundError(f"No encuentro config del modelo en {p}")
    tok = None
    for cand in (p / "tokenizer.json", p.parent.parent / "tokenizer.json"):
        if cand.exists():
            tok = cand
            break
    if tok is None:
        info = p.parent.parent / "run_info.json"
        if info.exists():
            t = json.loads(info.read_text())["train_config"].get("tokenizer")
            if t:
                tok = Path(t) / "tokenizer.json" if Path(t).is_dir() else Path(t)
    return p, cfg, tok


def load_model(path: str | Path, device: str = "auto", dtype: torch.dtype | None = None) -> Transformer:
    wdir, cfg_path, _ = resolve_artifact(path)
    data = json.loads(cfg_path.read_text())
    data = data.get("titan_config", data)
    cfg = ModelConfig.from_dict({k: v for k, v in data.items() if k in ModelConfig.__dataclass_fields__})
    dev = pick_device(device)
    if dtype is None:
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = Transformer(cfg)
    qmeta = wdir / "quantization.json"
    if qmeta.exists():
        from ..export.quantize import load_quantized_into
        load_quantized_into(model, wdir)
    else:
        load_model_weights(model, wdir)
    model.to(device=dev, dtype=dtype).eval()
    return model


class Engine:
    def __init__(self, model: Transformer, tokenizer: TitanTokenizer):
        self.model = model
        self.tok = tokenizer
        self.chat_format = ChatFormat(tokenizer)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def stream_ids(self, prompt_ids: list[int], **kw) -> Iterator[int]:
        params = SamplingParams(**kw)
        return generate_ids(self.model, prompt_ids, params)

    def complete(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, **kw) -> str:
        ids = self.tok.encode(prompt)
        out = list(self.stream_ids(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                                   stop_ids=kw.pop("stop_ids", [self.tok.eos_id]), **kw))
        return self.tok.decode(out)

    def chat_ids(self, messages: list[dict], tools: list[dict] | None = None, system: str | None = None,
                 max_new_tokens: int = 512, temperature: float = 0.7, **kw) -> list[int]:
        prompt = self.chat_format.prompt_ids(messages, tools, system)
        out = []
        for i in self.stream_ids(prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                                 stop_ids=[self.chat_format.im_end, self.tok.eos_id], **kw):
            out.append(i)
            if i == self.chat_format.call_end:
                break  # pausa para ejecutar la herramienta
        return out

    def chat_turn(self, messages: list[dict], tools: list[dict] | None = None, **kw) -> dict:
        return self.chat_format.parse_assistant(self.chat_ids(messages, tools, **kw))

    def chat(self, messages: list[dict], **kw) -> str:
        return self.chat_turn(messages, **kw)["content"]

    @torch.no_grad()
    def loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        """log P(continuation | context) y si es la continuación voraz (para opción múltiple)."""
        ctx = self.tok.encode(context)
        cont = self.tok.encode(continuation)
        ids = (ctx + cont)[-self.model.cfg.max_seq_len - 1:]
        n = min(len(cont), len(ids) - 1)
        x = torch.tensor([ids[:-1]], device=self.device)
        logits = self.model(x).logits[0, -n:].float()
        target = torch.tensor(ids[-n:], device=self.device)
        logp = torch.log_softmax(logits, -1)
        ll = logp.gather(1, target[:, None]).sum().item()
        greedy = bool((logits.argmax(-1) == target).all().item())
        return ll, greedy


def load_engine(path: str | Path, device: str = "auto", tokenizer: str | Path | None = None) -> Engine:
    _, _, tok_path = resolve_artifact(path)
    tok_path = Path(tokenizer) if tokenizer else tok_path
    if tok_path is None:
        raise FileNotFoundError("No encuentro el tokenizador; páselo con tokenizer=...")
    return Engine(load_model(path, device), TitanTokenizer.load(tok_path))
