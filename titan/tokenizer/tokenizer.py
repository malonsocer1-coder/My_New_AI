"""Tokenizador BPE a nivel de bytes, entrenado con los datos del proyecto.

Decisiones técnicas:
  * BPE a nivel de bytes: cualquier texto UTF-8 (español, inglés, código,
    emojis) es representable sin token desconocido.
  * Pre-tokenización con la expresión regular de Qwen2/Llama-3 pero con
    dígitos individuales (\\p{N}): mejora la aritmética y es reconocida por
    llama.cpp como pre-tokenizador "qwen2", lo que permite exportar a GGUF.
  * Tokens especiales reservados para chat y herramientas. El texto no
    confiable (páginas web, resultados de herramientas, documentos) se
    codifica con ``allow_special=False`` para que nunca pueda inyectar
    marcadores de rol o de llamada a herramienta.
  * Con vocabularios <= 65 536 los IDs caben en uint16 (mitad de disco).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from ..utils.io import sha256_file, write_json

PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# El orden fija los IDs 0..N-1. No reordenar: rompería checkpoints existentes.
SPECIAL_TOKENS = [
    "<|endoftext|>",   # separador de documentos / fin de generación
    "<|pad|>",
    "<|im_start|>",    # inicio de mensaje de chat (seguido del rol)
    "<|im_end|>",      # fin de mensaje
    "<|tool_call|>",   # inicio de llamada estructurada (JSON)
    "<|/tool_call|>",
    "<|tool_result|>", # inicio de resultado de herramienta
    "<|/tool_result|>",
    "<|think|>",       # razonamiento intermedio opcional
    "<|/think|>",
    "<|image|>",       # marcador para módulos multimodales auxiliares
    "<|audio|>",
] + [f"<|reserved_{i}|>" for i in range(20)]


def _build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    out_dir: str | os.PathLike,
    min_frequency: int = 2,
    length: int | None = None,
) -> "TitanTokenizer":
    """Entrena el BPE desde cero sobre ``texts`` y lo guarda en ``out_dir``."""
    tok = _build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    t = TitanTokenizer.load(out)
    write_json(out / "tokenizer_meta.json", {
        "vocab_size": t.vocab_size,
        "requested_vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "pretokenizer": "qwen2-style regex, single digits",
        "sha256": t.sha256,
    })
    return t


class TitanTokenizer:
    def __init__(self, tok: Tokenizer, path: str | None = None):
        self._tok = tok
        self.path = path
        self.special_ids = {s: tok.token_to_id(s) for s in SPECIAL_TOKENS}
        missing = [s for s, i in self.special_ids.items() if i is None]
        if missing:
            raise ValueError(f"Faltan tokens especiales en el tokenizador: {missing}")
        self.eos_id = self.special_ids["<|endoftext|>"]
        self.pad_id = self.special_ids["<|pad|>"]
        self.sha256 = sha256_file(path) if path else None

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TitanTokenizer":
        p = Path(path)
        if p.is_dir():
            p = p / "tokenizer.json"
        return cls(Tokenizer.from_file(str(p)), str(p))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def raw(self) -> Tokenizer:
        return self._tok

    def token_id(self, token: str) -> int:
        i = self._tok.token_to_id(token)
        if i is None:
            raise KeyError(token)
        return i

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Codifica texto. Con ``allow_special=False`` los marcadores especiales
        presentes en el texto se tratan como texto normal (protección contra
        inyección de roles/herramientas)."""
        self._tok.encode_special_tokens = not allow_special
        try:
            return self._tok.encode(text, add_special_tokens=False).ids
        finally:
            self._tok.encode_special_tokens = False

    def encode_batch(self, texts: list[str], allow_special: bool = False) -> list[list[int]]:
        self._tok.encode_special_tokens = not allow_special
        try:
            return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]
        finally:
            self._tok.encode_special_tokens = False

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    def iter_decode(self, ids: Iterable[int]) -> Iterator[str]:
        """Decodificación incremental segura para UTF-8 (para streaming)."""
        buf: list[int] = []
        emitted = ""
        for i in ids:
            buf.append(i)
            text = self.decode(buf)
            if text.endswith("�"):
                continue  # carácter multibyte incompleto
            yield text[len(emitted):]
            emitted = text

    def vocab(self) -> dict[str, int]:
        return self._tok.get_vocab()

    def merges(self) -> list[str]:
        data = json.loads(self._tok.to_str())
        merges = data["model"]["merges"]
        return [m if isinstance(m, str) else " ".join(m) for m in merges]

    def save(self, out_dir: str | os.PathLike) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(out / "tokenizer.json"))
