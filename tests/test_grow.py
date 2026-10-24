import json

import numpy as np
import pytest
import torch
import yaml

from titan.model import ModelConfig, Transformer
from titan.train import grow
from titan.train.checkpoint import load_model_weights, model_state_for_save
from titan.train.plan import Plan

TINY = dict(vocab_size=101, dim=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64, multiple_of=16)


def _model(seed=0, **kw):
    m = Transformer(ModelConfig(**{**TINY, **kw}))
    m.init_weights(seed=seed)
    return m.eval()


def _logits(cfg, sd, x):
    m = Transformer(cfg)
    m.load_state_dict(sd, strict=not cfg.tie_embeddings)
    return m.eval()(x).logits


def _x():
    return torch.randint(0, 101, (2, 24), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("new_layers", [4, 6, 8])
def test_grow_depth_copy_zero_preserves_function(new_layers):
    m = _model()
    sd = model_state_for_save(m)
    cfg2, sd2 = grow.grow_depth(m.cfg, sd, new_layers, "copy_zero")
    assert cfg2.n_layers == new_layers
    with torch.no_grad():
        torch.testing.assert_close(_logits(cfg2, sd2, _x()), m(_x()).logits, atol=1e-5, rtol=1e-5)
    assert sum(v.numel() for v in sd2.values()) == cfg2.param_counts()["total"]


def test_grow_stack_doubles_layers_and_trains():
    m = _model()
    cfg2, sd2 = grow.grow_depth(m.cfg, model_state_for_save(m), 6, "stack")
    big = Transformer(cfg2)
    big.load_state_dict(sd2)
    out = big(_x(), targets=_x())
    out.loss.backward()
    assert torch.isfinite(out.loss)


def test_upcycle_preserves_function():
    m = _model()
    cfg2, sd2 = grow.upcycle_to_moe(m.cfg, model_state_for_save(m), n_experts=4, top_k=2)
    assert cfg2.is_moe and cfg2.n_experts == 4
    with torch.no_grad():
        torch.testing.assert_close(_logits(cfg2, sd2, _x()), m(_x()).logits, atol=1e-5, rtol=1e-5)


def test_merge_identical_branches_equals_dense():
    m = _model()
    sd = model_state_for_save(m)
    cfg2, sd2 = grow.merge_branches([m.cfg, m.cfg], [sd, sd], top_k=2)
    with torch.no_grad():
        torch.testing.assert_close(_logits(cfg2, sd2, _x()), m(_x()).logits, atol=1e-5, rtol=1e-5)


def test_merge_distinct_branches():
    a, b, c = _model(1), _model(2), _model(3)
    sds = [model_state_for_save(x) for x in (a, b, c)]
    cfg2, sd2 = grow.merge_branches([a.cfg] * 3, sds, top_k=2)
    assert cfg2.n_experts == 3
    # cada experto es exactamente la FFN de su rama
    for e, sd in enumerate(sds):
        torch.testing.assert_close(sd2[f"layers.1.feed_forward.experts.{e}.w1.weight"], sd["layers.1.feed_forward.w1.weight"])
    # lo compartido es el promedio
    avg = sum(sd["layers.0.attention.wq.weight"] for sd in sds) / 3
    torch.testing.assert_close(sd2["layers.0.attention.wq.weight"], avg)
    moe = Transformer(cfg2)
    moe.load_state_dict(sd2)
    assert torch.isfinite(moe(_x(), targets=_x()).loss)


def test_merge_rejects_mismatched_architectures():
    a, b = _model(), _model(dim=96, n_heads=4)
    with pytest.raises(ValueError):
        grow.merge_branches([a.cfg, b.cfg], [model_state_for_save(a), model_state_for_save(b)])


def test_save_and_load_model_dir(tmp_path):
    m = _model(tie_embeddings=True)
    out = grow.save_model_dir(m.cfg, model_state_for_save(m), tmp_path / "m")
    cfg, sd = grow.load_model_dir(out)
    assert cfg == m.cfg
    m2 = Transformer(cfg)
    load_model_weights(m2, out)
    with torch.no_grad():
        torch.testing.assert_close(m2.eval()(_x()).logits, m(_x()).logits)


def _manifest(tmp_path):
    (tmp_path / "tokens").mkdir()
    shards = []
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for d in ("es", "code"):
            # secuencias con estructura aprendible: patrón periódico por dominio
            base = np.tile(np.arange(10 if d == "es" else 13) + (20 if d == "code" else 30), 800)
            arr = (base + (rng.random(base.size) < 0.05) * 1).astype(np.uint16)
            p = tmp_path / "tokens" / f"{d}_{split}.bin"
            arr.tofile(p)
            shards.append({"path": f"tokens/{d}_{split}.bin", "split": split, "domain": d,
                           "num_tokens": int(arr.size), "dtype": "uint16"})
    man = {"dtype": "uint16", "shards": shards, "domain_weights": {"es": 1.0, "code": 1.0},
           "totals": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return tmp_path / "manifest.json"


def test_plan_end_to_end_btx(tmp_path):
    man = _manifest(tmp_path)
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(yaml.safe_dump({
        "model": {**TINY, "vocab_size": 64},
        "train": {"data": str(man), "seq_len": 16, "micro_batch_size": 4, "max_steps": 6, "lr": 3e-3,
                  "warmup_steps": 1, "dtype": "fp32", "device": "cpu", "eval_interval": 6, "eval_windows": 4,
                  "log_interval": 3, "ckpt_interval": 3, "parallel": "none"},
    }))
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({
        "name": "mini", "workdir": str(tmp_path / "runs"),
        "phases": [
            {"id": "base", "type": "train", "config": str(train_yaml)},
            {"id": "exp_es", "type": "train", "config": str(train_yaml), "init_from": "@base",
             "set": {"domain_weights": {"es": 1.0, "code": 0.0}, "max_steps": 4}},
            {"id": "exp_code", "type": "train", "config": str(train_yaml), "init_from": "@base",
             "set": {"domain_weights": {"es": 0.0, "code": 1.0}, "max_steps": 4}},
            {"id": "merge", "type": "merge_experts", "branches": ["@exp_es", "@exp_code"], "top_k": 1},
            {"id": "tune", "type": "train", "config": str(train_yaml), "init_from": "@merge",
             "model_from": "@merge", "set": {"max_steps": 4}},
        ],
    }))
    plan = Plan(plan_yaml)
    state = plan.run(log=lambda *_: None)
    assert all(v["status"] == "done" for v in state["phases"].values())
    info = json.loads((tmp_path / "runs" / "tune" / "run_info.json").read_text())
    assert info["model_config"]["n_experts"] == 2
    # volver a ejecutar no repite nada
    before = (tmp_path / "runs" / "tune" / "summary.json").stat().st_mtime
    plan.run(log=lambda *_: None)
    assert (tmp_path / "runs" / "tune" / "summary.json").stat().st_mtime == before
