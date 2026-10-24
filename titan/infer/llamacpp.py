"""Backend de inferencia con llama.cpp (GGUF): CUDA en la RTX, cuantización y descarga parcial a RAM.

Expone la misma interfaz que ``titan.infer.engine.Engine`` (stream_ids,
chat_turn, complete...), así que el agente y el servidor funcionan igual con
cualquiera de los dos backends. ``n_gpu_layers`` controla cuántas capas van a
la GPU: -1 = todas; un número menor deja el resto en RAM cuando el modelo no
cabe en los 8 GB de VRAM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..sft.chat import ChatFormat
from ..tokenizer import TitanTokenizer


class LlamaCppEngine:
    def __init__(self, gguf_path: str | Path, tokenizer: TitanTokenizer, n_gpu_layers: int = -1,
                 n_ctx: int | None = None, n_threads: int | None = None):
        import llama_cpp
        self.llm = llama_cpp.Llama(model_path=str(gguf_path), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx or 0,
                                   n_threads=n_threads, verbose=False, logits_all=False)
        self.tok = tokenizer
        self.chat_format = ChatFormat(tokenizer)
        self.max_seq_len = self.llm.n_ctx()
        self.model = None

    def stream_ids(self, prompt_ids: list[int], max_new_tokens: int = 256, temperature: float = 0.7,
                   top_k: int | None = 50, top_p: float | None = 0.95, repetition_penalty: float = 1.0,
                   stop_ids: list[int] | None = None, seed: int | None = None) -> Iterator[int]:
        stop = set(stop_ids or [self.tok.eos_id])
        prompt_ids = prompt_ids[-(self.max_seq_len - 1):]
        budget = min(max_new_tokens, self.max_seq_len - len(prompt_ids))
        self.llm.reset()
        if seed is not None:
            self.llm.set_seed(seed)
        n = 0
        for t in self.llm.generate(prompt_ids, top_k=top_k or 0, top_p=top_p or 1.0, temp=max(temperature, 0.0),
                                   repeat_penalty=repetition_penalty, reset=True):
            if t in stop or n >= budget:
                return
            yield t
            n += 1

    def complete(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, **kw) -> str:
        return self.tok.decode(list(self.stream_ids(self.tok.encode(prompt), max_new_tokens, temperature, **kw)))

    def chat_ids(self, messages, tools=None, system=None, max_new_tokens=512, temperature=0.7, **kw) -> list[int]:
        prompt = self.chat_format.prompt_ids(messages, tools, system)
        out = []
        for i in self.stream_ids(prompt, max_new_tokens, temperature,
                                 stop_ids=[self.chat_format.im_end, self.tok.eos_id], **kw):
            out.append(i)
            if i == self.chat_format.call_end:
                break
        return out

    def chat_turn(self, messages, tools=None, **kw) -> dict:
        return self.chat_format.parse_assistant(self.chat_ids(messages, tools, **kw))

    def chat(self, messages, **kw) -> str:
        return self.chat_turn(messages, **kw)["content"]


def load_llamacpp_engine(path: str | Path, tokenizer: str | Path | None = None, n_gpu_layers: int = -1,
                         n_ctx: int | None = None) -> LlamaCppEngine:
    p = Path(path)
    gguf = p if p.suffix == ".gguf" else sorted(p.rglob("*.gguf"))[0]
    tok_path = Path(tokenizer) if tokenizer else None
    if tok_path is None:
        for cand in (gguf.parent / "tokenizer.json", gguf.parent.parent / "tokenizer.json"):
            if cand.exists():
                tok_path = cand
                break
    if tok_path is None:
        raise FileNotFoundError("No encuentro tokenizer.json junto al GGUF; páselo con tokenizer=...")
    return LlamaCppEngine(gguf, TitanTokenizer.load(tok_path), n_gpu_layers, n_ctx)
