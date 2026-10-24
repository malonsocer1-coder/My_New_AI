import json

import torch

from titan.model import ModelConfig, Transformer
from titan.sft.dpo import DPOConfig, train_dpo
from titan.sft.synthetic_prefs import PREF_EVAL_SEED_BASE, preference_pairs
from titan.sft.synthetic_tools import write_jsonl
from titan.tokenizer import train_tokenizer
from titan.train import grow
from titan.train.checkpoint import model_state_for_save


def test_preference_pairs_are_verified_and_disjoint():
    train = preference_pairs(30, seed=2)
    ev = preference_pairs(10, seed=0, eval_split=True)
    for p in train + ev:
        assert p["chosen"] != p["rejected"]
        assert p["messages"][0]["role"] == "user"
    assert PREF_EVAL_SEED_BASE > 2 * 1_000_003 + 30
    kinds = {p["kind"] for p in train}
    assert "missing_file" in kinds and "calculator" in kinds


def test_dpo_increases_preference_margin(tmp_path):
    pairs = preference_pairs(24, seed=5)
    write_jsonl(pairs, tmp_path / "train.jsonl")
    write_jsonl(pairs[:8], tmp_path / "eval.jsonl")
    tok = train_tokenizer([json.dumps(p, ensure_ascii=False) for p in pairs] * 4, 500, tmp_path / "tok")
    cfg = ModelConfig(vocab_size=500, dim=48, n_layers=2, n_heads=4, max_seq_len=2048, multiple_of=16)
    m = Transformer(cfg)
    m.init_weights(seed=0)
    src = grow.save_model_dir(cfg, model_state_for_save(m), tmp_path / "sft", tokenizer=tmp_path / "tok")
    res = train_dpo(DPOConfig(init_from=str(src), data=str(tmp_path / "train.jsonl"),
                              eval_data=str(tmp_path / "eval.jsonl"), out_dir=str(tmp_path / "dpo"),
                              steps=30, batch_size=4, lr=3e-3, warmup_steps=2, log_interval=10),
                    log=lambda *_: None)
    assert res["eval_before"]["reward_margin"] == 0.0  # política = referencia al empezar
    assert res["eval_after"]["reward_margin"] > 0 and res["eval_after"]["pref_accuracy"] > 0.5
    assert (tmp_path / "dpo" / "model.safetensors").exists()
