import json

import numpy as np
import pytest
import torch

from titan.export.exporter import export_checkpoint, verify_export
from titan.infer.engine import load_model
from titan.model import ModelConfig, Transformer
from titan.train import grow
from titan.train.checkpoint import model_state_for_save
from titan.tokenizer import train_tokenizer

TEXT = ["El sistema de entrenamiento de TITAN aprende desde cero. " * 20, "def f(x):\n    return x * 2\n" * 20,
        "Pregunta: ¿Cuánto es 12 + 30?\nRespuesta: 42\n" * 20]


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    train_tokenizer(TEXT * 5, 320, d)
    return d


def _make(tmp_path, tok_dir, **kw):
    cfg = ModelConfig(vocab_size=320, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64,
                      multiple_of=32, **kw)
    m = Transformer(cfg)
    m.init_weights(seed=5)
    src = grow.save_model_dir(cfg, model_state_for_save(m), tmp_path / "src", tokenizer=tok_dir)
    return m.eval(), src


def _ids():
    return torch.randint(0, 320, (1, 20), generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("kw", [{}, {"tie_embeddings": True}, {"n_experts": 4, "n_experts_per_token": 2}])
def test_titan_and_hf_exports_match(tmp_path, tok_dir, kw):
    transformers = pytest.importorskip("transformers")
    m, src = _make(tmp_path, tok_dir, **kw)
    out = tmp_path / "exp"
    export_checkpoint(src, out, formats=["titan", "hf"], dtype="fp32")
    assert verify_export(out) == []
    x = _ids()
    with torch.no_grad():
        ref = m(x).logits
        ours = load_model(out, device="cpu")(x).logits
        hf = transformers.AutoModelForCausalLM.from_pretrained(str(out / "hf"), dtype=torch.float32).eval()
        theirs = hf(x).logits
    torch.testing.assert_close(ours, ref)
    torch.testing.assert_close(theirs, ref, atol=2e-4, rtol=2e-4)


def test_hf_chat_template_matches_chat_format(tmp_path, tok_dir):
    transformers = pytest.importorskip("transformers")
    from titan.sft.chat import ChatFormat
    from titan.tokenizer import TitanTokenizer
    m, src = _make(tmp_path, tok_dir)
    export_checkpoint(src, tmp_path / "exp", formats=["hf"], dtype="fp32")
    hf_tok = transformers.AutoTokenizer.from_pretrained(str(tmp_path / "exp" / "hf"))
    cf = ChatFormat(TitanTokenizer.load(tok_dir))
    convs = [
        [{"role": "system", "content": "Sé breve."}, {"role": "user", "content": "Hola"}],
        [{"role": "user", "content": "¿Cuánto es 2+2?"},  # sin system: se usa el de TITAN
         {"role": "assistant", "content": "Calculo.", "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}]},
         {"role": "tool", "name": "calculator", "ok": True, "content": {"result": 4}},
         {"role": "assistant", "content": "Son 4."}, {"role": "user", "content": "Gracias"}],
    ]
    for msgs in convs:
        theirs = hf_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(theirs, "keys"):  # transformers v5 devuelve BatchEncoding
            theirs = theirs["input_ids"]
        assert list(theirs) == cf.prompt_ids(msgs)


@pytest.mark.parametrize("kw", [{}, {"n_experts": 4, "n_experts_per_token": 2}])
def test_gguf_export(tmp_path, tok_dir, kw):
    gguf = pytest.importorskip("gguf")
    m, src = _make(tmp_path, tok_dir, **kw)
    out = tmp_path / "exp"
    info = export_checkpoint(src, out, formats=["gguf"], dtype="fp32")
    path = next((out / "gguf").glob("*.gguf"))
    r = gguf.GGUFReader(str(path))
    names = {t.name for t in r.tensors}
    assert "token_embd.weight" in names and "blk.1.attn_q.weight" in names
    if kw:
        assert "blk.0.ffn_gate_exps.weight" in names
    try:
        import llama_cpp
    except ImportError:
        pytest.skip("llama-cpp-python no instalado: solo se verificó la estructura del GGUF")
    llm = llama_cpp.Llama(model_path=str(path), n_ctx=64, logits_all=True, verbose=False, n_threads=2)
    ids = _ids()[0].tolist()
    llm.eval(ids)
    theirs = np.array(llm.scores[: len(ids)])
    with torch.no_grad():
        ref = m(torch.tensor([ids])).logits[0].numpy()
    np.testing.assert_allclose(theirs, ref, atol=2e-3, rtol=2e-3)
    # llama.cpp debe tokenizar exactamente igual que el tokenizador propio
    from titan.tokenizer import TitanTokenizer
    tok = TitanTokenizer.load(tok_dir)
    for text in ["Pregunta: ¿Cuánto es 1234 + 56?", "def f(x):\n    return x * 2", "Ñandú café 2026 ☕"]:
        assert llm.tokenize(text.encode("utf-8"), add_bos=False, special=False) == tok.encode(text)


def test_verify_detects_corruption(tmp_path, tok_dir):
    _, src = _make(tmp_path, tok_dir)
    out = tmp_path / "exp"
    export_checkpoint(src, out, formats=["titan"], dtype="bf16")
    with open(out / "model.safetensors", "r+b") as f:
        f.seek(200)
        f.write(b"\x00\x01\x02\x03")
    assert verify_export(out) == ["model.safetensors"]


@pytest.mark.parametrize("bits", [8, 4])
def test_quantization_roundtrip(tmp_path, tok_dir, bits):
    from titan.export.quantize import compare_logits, dequantize_int4, quantize_int4, quantize_model_dir
    w = torch.randn(16, 256)
    p, s = quantize_int4(w, 64)
    assert p.shape == (16, 128) and (dequantize_int4(p, s, 256) - w).abs().max() < s.float().max() * 0.6
    m, src = _make(tmp_path, tok_dir)
    out = tmp_path / "exp"
    export_checkpoint(src, out, formats=["titan"], dtype="fp32")
    info = quantize_model_dir(out, tmp_path / f"q{bits}", bits=bits, group_size=32)
    assert info["ratio"] > (3.0 if bits == 8 else 5.0)
    qm = load_model(tmp_path / f"q{bits}", device="cpu")
    cmp = compare_logits(m, qm, _ids())
    print(bits, cmp)
    # Modelo sin entrenar: logits casi uniformes, así que se exige KL pequeña
    assert cmp["kl_divergence"] < (1e-3 if bits == 8 else 2e-2)
    assert cmp["top1_agreement"] >= (0.9 if bits == 8 else 0.6)


def test_llamacpp_engine_greedy_matches_titan(tmp_path, tok_dir):
    pytest.importorskip("llama_cpp")
    from titan.infer.engine import load_engine
    from titan.infer.llamacpp import load_llamacpp_engine
    _, src = _make(tmp_path, tok_dir)
    out = tmp_path / "exp"
    export_checkpoint(src, out, formats=["titan", "gguf"], dtype="fp32")
    a = load_engine(out, device="cpu").complete("El sistema de", max_new_tokens=12, temperature=0.0)
    b = load_llamacpp_engine(out / "gguf").complete("El sistema de", max_new_tokens=12, temperature=0.0)
    assert a == b
