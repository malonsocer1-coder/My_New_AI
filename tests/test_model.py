import math

import pytest
import torch

from titan.model import ModelConfig, Transformer

TINY = dict(vocab_size=97, dim=64, n_layers=2, n_heads=4, max_seq_len=64, multiple_of=16)


def make(**kw):
    cfg = ModelConfig(**{**TINY, **kw})
    torch.manual_seed(0)
    return Transformer(cfg)


@pytest.mark.parametrize("kw", [{}, {"n_kv_heads": 2}, {"n_kv_heads": 1, "tie_embeddings": True},
                                {"n_experts": 4, "n_experts_per_token": 2}])
def test_param_count_matches_analytic(kw):
    m = make(**kw)
    assert m.num_params() == m.cfg.param_counts()["total"]


def test_initial_loss_near_uniform():
    m = make()
    x = torch.randint(0, 97, (4, 32))
    out = m(x, targets=x)
    assert abs(out.loss.item() - math.log(97)) < 0.5


def test_seeded_init_is_deterministic():
    a, b = make(), make()
    a.init_weights(seed=7)
    b.init_weights(seed=7)
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n


@pytest.mark.parametrize("kw", [{}, {"n_kv_heads": 2}, {"n_experts": 4}])
def test_kv_cache_matches_full_forward(kw):
    m = make(**kw).eval()
    x = torch.randint(0, 97, (1, 20))
    with torch.no_grad():
        full = m(x).logits
        cache = m.new_cache(1, 32)
        # prefill por fragmentos (7 + 5) y luego token a token
        a = m(x[:, :7], start_pos=0, cache=cache).logits
        b = m(x[:, 7:12], start_pos=7, cache=cache).logits
        steps = [m(x[:, i:i + 1], start_pos=i, cache=cache).logits for i in range(12, 20)]
    inc = torch.cat([a, b] + steps, dim=1)
    torch.testing.assert_close(inc, full, atol=1e-4, rtol=1e-4)


def test_causality():
    m = make().eval()
    x = torch.randint(0, 97, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 97
    with torch.no_grad():
        lx, ly = m(x).logits, m(y).logits
    torch.testing.assert_close(lx[:, :10], ly[:, :10])
    assert not torch.allclose(lx[:, 10:], ly[:, 10:])


def test_activation_checkpointing_same_grads():
    x = torch.randint(0, 97, (2, 16))
    grads = []
    for ckpt in (False, True):
        m = make(activation_checkpointing=ckpt)
        m.init_weights(seed=3)
        m(x, targets=x).loss.backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], atol=1e-6, rtol=1e-5)


def test_meta_init_then_materialize():
    cfg = ModelConfig(**TINY)
    with torch.device("meta"):
        m = Transformer(cfg)
    m.to_empty(device="cpu")
    m.init_weights(seed=1)
    out = m(torch.randint(0, 97, (1, 8)))
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("kw", [{}, {"n_experts": 4}])
def test_overfits_small_batch(kw):
    m = make(**kw)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 97, (4, 24))
    first = None
    for _ in range(150):
        out = m(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        first = first or out.ce_loss.item()
    assert out.ce_loss.item() < 0.2 * first


def test_moe_routes_to_all_experts_and_grads_flow():
    m = make(n_experts=4, n_experts_per_token=1)
    x = torch.randint(0, 97, (8, 32))
    out = m(x, targets=x)
    out.loss.backward()
    for layer in m.layers:
        for e in layer.feed_forward.experts:
            assert e.w1.weight.grad is not None
    assert out.aux_loss.item() > 0
