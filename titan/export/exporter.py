"""Exportación de los pesos propios de TITAN a formatos de inferencia.

Formatos:
  * ``titan``  pesos nativos (safetensors) + config + tokenizador. Los carga
               ``titan.infer.engine`` y el servidor local.
  * ``hf``     formato Hugging Face (LlamaForCausalLM / MixtralForCausalLM):
               permite usar transformers, vLLM y otras herramientas.
  * ``gguf``   formato de llama.cpp (arquitectura "llama", con expertos si es
               MoE). Es el formato recomendado para Windows/RTX: llama.cpp,
               LM Studio y Ollama lo ejecutan con CUDA y cuantización.

Los pesos originales nunca se modifican: cada exportación es un artefacto
derivado con su propio MANIFEST.json (SHA-256 de cada archivo).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from ..model import ModelConfig
from ..tokenizer import SPECIAL_TOKENS, TitanTokenizer
from ..train.grow import load_model_dir
from ..utils.io import sha256_file, write_json

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

def _chat_template() -> str:
    """Plantilla Jinja equivalente a ``titan.sft.chat.ChatFormat`` (verificado en tests).

    Nota de seguridad: las herramientas externas tokenizan el texto renderizado
    y pueden interpretar marcadores especiales escritos en el contenido; para
    el agente use siempre el motor de TITAN, que los neutraliza.
    """
    from ..sft.chat import DEFAULT_SYSTEM
    sys_prompt = DEFAULT_SYSTEM.replace("'", "\\'")
    return (
        "{%- set default_system = '" + sys_prompt + "' -%}"
        "{%- if messages[0]['role'] != 'system' -%}"
        "{{- '<|im_start|>system\\n' + default_system + '<|im_end|>\\n' -}}"
        "{%- endif -%}"
        "{%- for m in messages -%}"
        "{{- '<|im_start|>' + m['role'] + '\\n' -}}"
        "{%- if m['role'] == 'tool' -%}"
        "{{- '<|tool_result|>' + ({'name': m.get('name'), 'ok': m.get('ok', true), 'content': m['content']} | tojson)"
        " + '<|/tool_result|>' -}}"
        "{%- else -%}"
        "{{- m['content'] or '' -}}"
        "{%- for c in (m.get('tool_calls') or []) -%}"
        "{{- '<|tool_call|>' + ({'name': c['name'], 'arguments': c['arguments']} | tojson) + '<|/tool_call|>' -}}"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{{- '<|im_end|>\\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
    )


CHAT_TEMPLATE = _chat_template()


def _resolve_tokenizer(checkpoint: str | Path, explicit: str | Path | None) -> Path:
    from ..infer.engine import resolve_artifact
    if explicit:
        p = Path(explicit)
        return p / "tokenizer.json" if p.is_dir() else p
    _, _, tok = resolve_artifact(checkpoint)
    if tok is None:
        raise FileNotFoundError("No encuentro el tokenizador; páselo con --tokenizer")
    return Path(tok)


# ----------------------------------------------------------------------
# Hugging Face
# ----------------------------------------------------------------------
def hf_state_dict(cfg: ModelConfig, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {"model.embed_tokens.weight": sd["tok_embeddings.weight"], "model.norm.weight": sd["norm.weight"]}
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = sd["output.weight"]
    for i in range(cfg.n_layers):
        p, q = f"layers.{i}.", f"model.layers.{i}."
        out[q + "self_attn.q_proj.weight"] = sd[p + "attention.wq.weight"]
        out[q + "self_attn.k_proj.weight"] = sd[p + "attention.wk.weight"]
        out[q + "self_attn.v_proj.weight"] = sd[p + "attention.wv.weight"]
        out[q + "self_attn.o_proj.weight"] = sd[p + "attention.wo.weight"]
        out[q + "input_layernorm.weight"] = sd[p + "attention_norm.weight"]
        out[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]
        if cfg.is_moe:
            out[q + "block_sparse_moe.gate.weight"] = sd[p + "feed_forward.gate.weight"]
            for e in range(cfg.n_experts):
                for w in ("w1", "w2", "w3"):
                    out[q + f"block_sparse_moe.experts.{e}.{w}.weight"] = sd[p + f"feed_forward.experts.{e}.{w}.weight"]
        else:
            out[q + "mlp.gate_proj.weight"] = sd[p + "feed_forward.w1.weight"]
            out[q + "mlp.up_proj.weight"] = sd[p + "feed_forward.w3.weight"]
            out[q + "mlp.down_proj.weight"] = sd[p + "feed_forward.w2.weight"]
    return out


def hf_config(cfg: ModelConfig, tok: TitanTokenizer, dtype: str) -> dict:
    base = {
        "vocab_size": cfg.vocab_size, "hidden_size": cfg.dim, "intermediate_size": cfg.ffn_dim,
        "num_hidden_layers": cfg.n_layers, "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_seq_len, "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta, "rope_parameters": {"rope_type": "default", "rope_theta": cfg.rope_theta},
        "tie_word_embeddings": cfg.tie_embeddings, "hidden_act": "silu", "attention_bias": False,
        "attention_dropout": 0.0, "initializer_range": cfg.init_std, "use_cache": True,
        "bos_token_id": tok.eos_id, "eos_token_id": [tok.special_ids["<|im_end|>"], tok.eos_id],
        "pad_token_id": tok.pad_id, "torch_dtype": {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype],
        "titan_config": cfg.to_dict(),
    }
    if cfg.is_moe:
        base.update({"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
                     "num_local_experts": cfg.n_experts, "num_experts_per_tok": cfg.n_experts_per_token,
                     "router_aux_loss_coef": cfg.moe_aux_loss_coef, "output_router_logits": False})
    else:
        base.update({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "mlp_bias": False,
                     "pretraining_tp": 1})
    return base


def export_hf(cfg, sd, tok: TitanTokenizer, tok_path: Path, out: Path, dtype: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    t = DTYPES[dtype]
    hsd = {k: v.to(t).contiguous() for k, v in hf_state_dict(cfg, sd).items()}
    save_file(hsd, str(out / "model.safetensors"), metadata={"format": "pt"})
    write_json(out / "config.json", hf_config(cfg, tok, dtype))
    (out / "tokenizer.json").write_bytes(tok_path.read_bytes())
    write_json(out / "tokenizer_config.json", {
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": None, "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>", "unk_token": None, "add_bos_token": False, "add_eos_token": False,
        "model_max_length": cfg.max_seq_len, "chat_template": CHAT_TEMPLATE,
        "additional_special_tokens": SPECIAL_TOKENS})
    write_json(out / "special_tokens_map.json", {"eos_token": "<|im_end|>", "pad_token": "<|pad|>"})
    write_json(out / "generation_config.json", {"eos_token_id": [tok.special_ids["<|im_end|>"], tok.eos_id],
                                                "pad_token_id": tok.pad_id, "do_sample": True,
                                                "temperature": 0.7, "top_p": 0.95})
    return [out / f for f in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")]


# ----------------------------------------------------------------------
# GGUF (llama.cpp)
# ----------------------------------------------------------------------
def _permute_qk(w: torch.Tensor, n_head: int) -> torch.Tensor:
    """De la convención rotate_half (HF/TITAN) a la de pares intercalados de llama.cpp."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def export_gguf(cfg: ModelConfig, sd: dict[str, torch.Tensor], tok: TitanTokenizer, out_file: Path,
                dtype: str = "fp16") -> Path:
    import gguf

    out_file.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out_file), arch="llama")
    w.add_name(cfg.name)
    w.add_context_length(cfg.max_seq_len)
    w.add_embedding_length(cfg.dim)
    w.add_block_count(cfg.n_layers)
    w.add_feed_forward_length(cfg.ffn_dim)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(cfg.n_kv_heads)
    w.add_rope_dimension_count(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_vocab_size(cfg.vocab_size)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if dtype == "fp16" else gguf.LlamaFileType.ALL_F32)
    if cfg.is_moe:
        w.add_expert_count(cfg.n_experts)
        w.add_expert_used_count(cfg.n_experts_per_token)

    # Tokenizador BPE a nivel de bytes con el pre-tokenizador "qwen2" (dígitos individuales)
    vocab = tok.vocab()
    tokens = [None] * tok.vocab_size
    for t, i in vocab.items():
        tokens[i] = t
    special = set(SPECIAL_TOKENS)
    types = [gguf.TokenType.CONTROL if t in special else gguf.TokenType.NORMAL for t in tokens]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(tok.merges())
    w.add_bos_token_id(tok.eos_id)
    w.add_eos_token_id(tok.special_ids["<|im_end|>"])
    w.add_pad_token_id(tok.pad_id)
    w.add_add_bos_token(False)
    w.add_chat_template(CHAT_TEMPLATE)

    fdt = np.float16 if dtype == "fp16" else np.float32

    def add(name: str, t: torch.Tensor, force_f32: bool = False) -> None:
        arr = t.detach().float().numpy()
        w.add_tensor(name, arr.astype(np.float32 if force_f32 or arr.ndim == 1 else fdt))

    add("token_embd.weight", sd["tok_embeddings.weight"])
    add("output_norm.weight", sd["norm.weight"], True)
    if not cfg.tie_embeddings:
        add("output.weight", sd["output.weight"])
    for i in range(cfg.n_layers):
        p, b = f"layers.{i}.", f"blk.{i}."
        add(b + "attn_norm.weight", sd[p + "attention_norm.weight"], True)
        add(b + "attn_q.weight", _permute_qk(sd[p + "attention.wq.weight"], cfg.n_heads))
        add(b + "attn_k.weight", _permute_qk(sd[p + "attention.wk.weight"], cfg.n_kv_heads))
        add(b + "attn_v.weight", sd[p + "attention.wv.weight"])
        add(b + "attn_output.weight", sd[p + "attention.wo.weight"])
        add(b + "ffn_norm.weight", sd[p + "ffn_norm.weight"], True)
        if cfg.is_moe:
            add(b + "ffn_gate_inp.weight", sd[p + "feed_forward.gate.weight"], True)
            for gname, wn in (("ffn_gate_exps", "w1"), ("ffn_down_exps", "w2"), ("ffn_up_exps", "w3")):
                stacked = torch.stack([sd[p + f"feed_forward.experts.{e}.{wn}.weight"] for e in range(cfg.n_experts)])
                add(b + gname + ".weight", stacked)
        else:
            add(b + "ffn_gate.weight", sd[p + "feed_forward.w1.weight"])
            add(b + "ffn_up.weight", sd[p + "feed_forward.w3.weight"])
            add(b + "ffn_down.weight", sd[p + "feed_forward.w2.weight"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out_file


# ----------------------------------------------------------------------
def model_card(cfg: ModelConfig, info: dict) -> str:
    pc = cfg.param_counts()
    return f"""# {cfg.name}

Modelo de lenguaje **entrenado desde pesos aleatorios** por el proyecto TITAN AI.
No deriva de pesos preentrenados de terceros.

| | |
|---|---|
| Arquitectura | Transformer decodificador {'Mixture of Experts' if cfg.is_moe else 'denso'} (RMSNorm, RoPE, GQA, SwiGLU) |
| Parámetros | {pc['total'] / 1e6:.2f} M totales, {pc['active'] / 1e6:.2f} M activos por token |
| Capas / dimensión / cabezas | {cfg.n_layers} / {cfg.dim} / {cfg.n_heads} (kv {cfg.n_kv_heads}) |
| Contexto | {cfg.max_seq_len} tokens |
| Vocabulario | {cfg.vocab_size} (BPE propio a nivel de bytes) |
| Checkpoint de origen | `{info['source']}` |
| Paso / tokens vistos | {info.get('step')} / {info.get('tokens')} |
| Exportado | {info['exported_utc']} |

Consulte `MANIFEST.json` para verificar la integridad (SHA-256) y
`docs/EVALUACION.md` del repositorio para las métricas medidas.
"""


def export_checkpoint(checkpoint: str | Path, out: str | Path, formats: list[str] | tuple = ("titan", "hf", "gguf"),
                      dtype: str = "bf16", tokenizer: str | Path | None = None) -> dict:
    from ..train.checkpoint import resolve_weights_dir
    src = resolve_weights_dir(checkpoint)
    cfg, sd = load_model_dir(src)
    if cfg.tie_embeddings:
        sd = {**sd, "output.weight": sd["tok_embeddings.weight"]}
    tok_path = _resolve_tokenizer(checkpoint, tokenizer)
    tok = TitanTokenizer.load(tok_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = {}
    if (src / "trainer_state.json").exists():
        state = json.loads((src / "trainer_state.json").read_text())
    info = {"source": str(src), "step": state.get("step"), "tokens": state.get("tokens"),
            "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "formats": list(formats)}
    files: list[Path] = []
    if "titan" in formats:
        t = DTYPES[dtype]
        nsd = {k: v.to(t).contiguous() for k, v in sd.items() if not (cfg.tie_embeddings and k == "output.weight")}
        save_file(nsd, str(out / "model.safetensors"))
        write_json(out / "model_config.json", cfg.to_dict())
        (out / "tokenizer.json").write_bytes(tok_path.read_bytes())
        write_json(out / "generation_config.json", {"temperature": 0.7, "top_p": 0.95, "top_k": 50,
                                                    "max_new_tokens": 512})
        (out / "MODEL_CARD.md").write_text(model_card(cfg, info), encoding="utf-8")
        files += [out / "model.safetensors", out / "model_config.json", out / "tokenizer.json"]
    if "hf" in formats:
        files += export_hf(cfg, sd, tok, tok_path, out / "hf", dtype)
    if "gguf" in formats:
        g_dtype = "fp32" if dtype == "fp32" else "fp16"
        files.append(export_gguf(cfg, sd, tok, out / "gguf" / f"{cfg.name}-{g_dtype}.gguf", g_dtype))
    manifest = {"info": info, "model_config": cfg.to_dict(),
                "files": {str(f.relative_to(out)): {"sha256": sha256_file(f), "bytes": f.stat().st_size} for f in files}}
    write_json(out / "MANIFEST.json", manifest)
    return {"out": str(out), "files": {k: v["bytes"] for k, v in manifest["files"].items()}, **info}


def verify_export(out: str | Path) -> list[str]:
    """Comprueba los SHA-256 de un modelo exportado. Devuelve los archivos con problemas."""
    out = Path(out)
    man = json.loads((out / "MANIFEST.json").read_text())
    return [f for f, meta in man["files"].items()
            if not (out / f).exists() or sha256_file(out / f) != meta["sha256"]]
