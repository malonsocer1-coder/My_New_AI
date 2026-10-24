"""Plan de 118 M para el PC: micro-lote automático, referencias entre fases, piloto de LR,
importadores de chat y el plan completo (de pesos aleatorios a GGUF) con un modelo diminuto."""

import importlib.util
import json

import numpy as np
import pytest
import torch
import yaml

from titan.model import ModelConfig
from titan.sft.importers import eval_index, import_chats
from titan.sft.synthetic_tools import tool_conversations
from titan.tokenizer import train_tokenizer
from titan.train.autobatch import candidate_micro_batches, probe
from titan.train.config import TrainConfig, load_run_config
from titan.train.distributed import DistInfo
from titan.train.plan import Plan, PlanError, _resolve
from titan.train.trainer import resolve_batch, resolve_compile


def test_titan_120m_config_is_in_requested_range():
    m, t = load_run_config("configs/train/pc_120m_base.yaml")
    total = m.param_counts()["total"]
    assert 110e6 <= total <= 120e6
    assert m.dim // m.n_heads == 64 and m.n_heads % m.n_kv_heads == 0
    assert t.batch_tokens % t.seq_len == 0 and t.auto_batch
    plan = Plan("configs/plans/pc_120m.yaml")
    assert [p["id"] for p in plan.phases][:3] == ["lr_pilot", "base", "long_ctx"]


def test_candidate_micro_batches_divide_the_step():
    assert candidate_micro_batches(256, 32) == [1, 2, 4, 8, 16, 32]
    assert candidate_micro_batches(12, 32) == [1, 2, 3, 4, 6, 12]
    assert all(256 % b == 0 for b in candidate_micro_batches(256, 100))


def _dinfo():
    return DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))


def test_resolve_batch_keeps_global_batch():
    cfg = TrainConfig(seq_len=64, batch_tokens=64 * 24, micro_batch_size=4)
    mc = ModelConfig(vocab_size=64, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, multiple_of=16, max_seq_len=64)
    resolve_batch(mc, cfg, _dinfo(), torch.device("cpu"), torch.float32, None)
    assert cfg.micro_batch_size * cfg.grad_accum_steps == 24
    with pytest.raises(ValueError):
        resolve_batch(mc, TrainConfig(seq_len=64, batch_tokens=64 * 24, micro_batch_size=5), _dinfo(),
                      torch.device("cpu"), torch.float32, None)
    with pytest.raises(ValueError):
        resolve_batch(mc, TrainConfig(seq_len=64, batch_tokens=100), _dinfo(), torch.device("cpu"), torch.float32, None)


def test_probe_measures_every_candidate_on_cpu():
    mc = ModelConfig(vocab_size=128, dim=32, n_layers=2, n_heads=2, n_kv_heads=1, multiple_of=16, max_seq_len=32)
    cfg = TrainConfig(seq_len=32, dtype="fp32", device="cpu")
    res = probe(mc, cfg, torch.device("cpu"), torch.float32, seqs_per_rank=8, max_micro=8, steps=1,
                log=lambda *_: None)
    assert [r["micro_batch"] for r in res["table"]] == [1, 2, 4, 8]
    ch = res["choice"]
    assert ch["micro_batch_size"] * ch["grad_accum_steps"] == 8
    assert ch["tokens_per_s_eager"] == max(r["tokens_per_s"] for r in res["table"])


def test_compile_auto_is_off_without_gpu():
    assert resolve_compile(TrainConfig(compile="auto"), torch.device("cpu"), lambda *_: None) is False
    with pytest.raises(ValueError):
        resolve_compile(TrainConfig(compile="siempre"), torch.device("cpu"), lambda *_: None)


def test_result_references(tmp_path):
    st = {"pilot": {"status": "done", "result": {"best": {"lr": 2e-3}}}, "x": {"status": "running"}}
    assert _resolve("@result:pilot.best.lr", tmp_path, st) == 2e-3
    assert _resolve({"min_lr": "@result:pilot.best.lr*0.1"}, tmp_path, st)["min_lr"] == pytest.approx(2e-4)
    assert _resolve("@base/gguf", tmp_path, st) == str(tmp_path / "base" / "gguf")
    with pytest.raises(PlanError):
        _resolve("@result:x.best.lr", tmp_path, st)
    with pytest.raises(PlanError):
        _resolve("@result:pilot.best.wd", tmp_path, st)


# ---------------------------------------------------------------------------
def _oasst_rows():
    return [
        {"message_id": "r1", "parent_id": None, "text": "¿Cuál es la capital de Jalisco?", "role": "prompter",
         "lang": "es"},
        {"message_id": "a1", "parent_id": "r1", "text": "La capital de Jalisco es Guadalajara.",
         "role": "assistant", "lang": "es", "rank": 0},
        {"message_id": "a2", "parent_id": "r1", "text": "Es Monterrey.", "role": "assistant", "lang": "es", "rank": 1},
        {"message_id": "p2", "parent_id": "a1", "text": "¿Y cuántos habitantes tiene? Escribe a ana@correo.mx",
         "role": "prompter", "lang": "es"},
        {"message_id": "a3", "parent_id": "p2", "text": "Alrededor de 1,4 millones en el municipio.",
         "role": "assistant", "lang": "es", "rank": 0},
        {"message_id": "a4", "parent_id": "p2", "text": "Respuesta borrada", "role": "assistant", "lang": "es",
         "rank": None, "deleted": True},
        {"message_id": "r2", "parent_id": None, "text": "What is the capital of France?", "role": "prompter",
         "lang": "en"},
        {"message_id": "b1", "parent_id": "r2", "text": "Paris.", "role": "assistant", "lang": "en", "rank": 0},
        {"message_id": "r3", "parent_id": None, "text": "Pregunta sin respuesta", "role": "prompter", "lang": "es"},
    ]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_oasst_import_follows_best_ranked_thread(tmp_path):
    f = _write_jsonl(tmp_path / "oasst.jsonl", _oasst_rows())
    convs, stats = import_chats({"name": "oasst_es", "format": "oasst", "loader": "jsonl",
                                 "params": {"path": str(f)}, "langs": ["es"]}, log=lambda *_: None)
    assert len(convs) == 1 and stats["kept"] == 1
    msgs = convs[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["content"] == "La capital de Jalisco es Guadalajara."
    assert "ana@correo.mx" not in msgs[2]["content"]  # datos personales eliminados
    assert msgs[3]["content"].startswith("Alrededor")


def test_pairs_import_filters_dedups_and_decontaminates(tmp_path):
    eval_q = json.loads(open("titan/eval/data/mc_general_es.jsonl", encoding="utf-8").readline())["question"]
    rows = [
        {"inputs": "Explica qué es la fotosíntesis.", "targets": "Es el proceso por el que las plantas…",
         "language_code": "spa"},
        {"inputs": "Explica qué es la fotosíntesis.", "targets": "Es el proceso por el que las plantas…",
         "language_code": "spa"},
        {"inputs": "Explain photosynthesis.", "targets": "It is…", "language_code": "eng"},
        {"inputs": eval_q + " " + eval_q, "targets": "respuesta", "language_code": "spa"},
    ]
    f = _write_jsonl(tmp_path / "aya.jsonl", rows)
    spec = {"name": "aya_es", "format": "pairs", "loader": "jsonl", "params": {"path": str(f)},
            "filter": {"language_code": ["spa"]}, "prompt_field": "inputs", "response_field": "targets"}
    convs, stats = import_chats(spec, eval_index(["titan/eval/data/*.jsonl"]), log=lambda *_: None)
    assert stats["kept"] == 1 and stats["duplicates"] == 1
    assert stats["contaminated"] == 1
    with pytest.raises(KeyError):  # columnas equivocadas: error claro, no datos vacíos
        import_chats({**spec, "prompt_field": "instruction"}, log=lambda *_: None)


# ---------------------------------------------------------------------------
def _pretrain_manifest(root, vocab):
    (root / "tokens").mkdir(parents=True)
    rng = np.random.default_rng(0)
    shards = []
    for split in ("train", "val"):
        for d in ("es", "code"):
            base = np.tile(np.arange(10 if d == "es" else 13) + (40 if d == "code" else 60), 1500)
            arr = (base + (rng.random(base.size) < 0.05)).astype(np.uint16) % vocab
            p = root / "tokens" / f"{d}_{split}.bin"
            arr.tofile(p)
            shards.append({"path": f"tokens/{d}_{split}.bin", "split": split, "domain": d,
                           "num_tokens": int(arr.size), "dtype": "uint16"})
    man = {"dtype": "uint16", "shards": shards, "domain_weights": {"es": 1.0, "code": 1.0}, "totals": {}}
    (root / "manifest.json").write_text(json.dumps(man))
    return root / "manifest.json"


def test_full_plan_from_random_weights_to_gguf(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_CACHE", str(tmp_path / "cache"))
    texts = [json.dumps(c, ensure_ascii=False) for c in tool_conversations(40, seed=5)]
    tok = train_tokenizer(texts * 3, 600, tmp_path / "tok")
    man = _pretrain_manifest(tmp_path / "data", tok.vocab_size)
    model = dict(vocab_size=tok.vocab_size, dim=32, n_layers=2, n_heads=2, n_kv_heads=1, multiple_of=16,
                 max_seq_len=1024, tie_embeddings=True, ce_chunk_size=64)
    common = dict(tokenizer=str(tmp_path / "tok"), dtype="fp32", device="cpu", parallel="none", log_interval=2,
                  eval_windows=4, eval_batch_size=2, warmup_steps=1, keep_last=1)
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"model": model, "train": {
        **common, "data": str(man), "seq_len": 32, "batch_tokens": 32 * 4, "micro_batch_size": 2,
        "max_steps": 6, "lr": 3e-3, "schedule": "wsd", "eval_interval": 6, "ckpt_interval": 3}}))
    (tmp_path / "long.yaml").write_text(yaml.safe_dump({"model": model, "train": {
        **common, "data": str(man), "seq_len": 64, "batch_tokens": 64 * 2, "micro_batch_size": 1,
        "max_steps": 3, "lr": 1e-3, "eval_interval": 3, "ckpt_interval": 3}}))
    (tmp_path / "sft.yaml").write_text(yaml.safe_dump({"model": model, "train": {
        **common, "task": "sft", "data": "pendiente", "seq_len": 1024, "batch_tokens": 1024 * 2,
        "micro_batch_size": 1, "max_epochs": 0.5, "lr": 1e-3, "weight_decay": 0.0, "eval_interval": 4,
        "ckpt_interval": 4}}))
    oasst = _write_jsonl(tmp_path / "oasst.jsonl", _oasst_rows())
    has_llama = importlib.util.find_spec("llama_cpp") is not None
    phases = [
        {"id": "lr_pilot", "type": "sweep", "config": str(tmp_path / "base.yaml"),
         "set": {"max_steps": 4, "eval_interval": 4},
         "variants": [{"lr": 1e-3, "min_lr": 1e-4}, {"lr": 3e-3, "min_lr": 3e-4}], "tie_tolerance": 0.0},
        {"id": "base", "type": "train", "config": str(tmp_path / "base.yaml"),
         "set": {"lr": "@result:lr_pilot.best.lr", "min_lr": "@result:lr_pilot.best.lr*0.1"}},
        {"id": "long_ctx", "type": "train", "config": str(tmp_path / "long.yaml"), "init_from": "@base"},
        {"id": "sft_data", "type": "sft_data", "tokenizer": str(tmp_path / "tok"), "seq_len": 1024,
         "synthetic": {"tools": 30, "arith": 10, "seed": 1},
         "imports": [
             {"name": "oasst_es", "format": "oasst", "loader": "jsonl", "params": {"path": str(oasst)},
              "license": "Apache-2.0"},
             {"name": "roto", "format": "pairs", "loader": "jsonl", "optional": True,
              "params": {"path": str(oasst)}, "prompt_field": "no_existe"},
         ]},
        {"id": "sft", "type": "train", "config": str(tmp_path / "sft.yaml"), "init_from": "@long_ctx",
         "set": {"data": "@sft_data"}},
        {"id": "dpo", "type": "dpo", "init_from": "@sft", "pairs": {"train": 12, "eval": 4},
         "set": {"steps": 2, "batch_size": 2, "warmup_steps": 1, "device": "cpu", "lr": 1e-5}},
        {"id": "export", "type": "export", "checkpoint": "@dpo", "formats": ["titan", "gguf"], "dtype": "fp32"},
        {"id": "gguf", "type": "gguf_quantize", "model": "@export/gguf", "types": ["Q8_0"],
         "enabled": has_llama},
        {"id": "eval", "type": "eval", "checkpoint": "@export", "suite": "bpb,mc", "limit": 3,
         "data": str(man), "device": "cpu"},
    ]
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump({"name": "mini120", "workdir": str(tmp_path / "runs"),
                                                        "phases": phases}))
    plan = Plan(tmp_path / "plan.yaml")
    state = plan.run(log=lambda *_: None)
    ph = state["phases"]
    assert all(ph[p["id"]]["status"] == "done" for p in phases if p.get("enabled", True)), ph
    pilot = ph["lr_pilot"]["result"]
    best = min(v["final_val_loss"] for v in pilot["variants"])
    assert next(v for v in pilot["variants"] if v["values"] == pilot["best"])["final_val_loss"] == best
    base_info = json.loads((tmp_path / "runs" / "base" / "run_info.json").read_text())
    assert base_info["train_config"]["lr"] == pilot["best"]["lr"]
    assert base_info["train_config"]["min_lr"] == pytest.approx(pilot["best"]["lr"] * 0.1)
    assert base_info["train_config"]["grad_accum_steps"] == 2  # 4 secuencias / micro-lote 2
    sft = ph["sft_data"]["result"]
    assert [s["name"] for s in sft["skipped"]] == ["roto"]
    assert any(s["name"] == "oasst_es" and s["kept"] == 1 for s in sft["sources"])
    sft_info = json.loads((tmp_path / "runs" / "sft" / "run_info.json").read_text())
    rows = json.loads((tmp_path / "runs" / "sft_data" / "meta.json").read_text())["splits"]["train"]["rows"]
    assert sft_info["train_config"]["max_steps"] == max(1, -(-rows // 4))  # ceil(0.5 épocas * filas / 2)
    assert "pref_accuracy" in ph["dpo"]["result"]["eval_after"]
    assert (tmp_path / "runs" / "export" / "model.safetensors").exists()
    assert list((tmp_path / "runs" / "export" / "gguf").glob("*.gguf"))
    if has_llama:
        q8 = list((tmp_path / "runs" / "gguf").glob("*-Q8_0.gguf"))
        assert q8 and (tmp_path / "runs" / "gguf" / "tokenizer.json").exists()
        from titan.infer.llamacpp import load_llamacpp_engine
        eng = load_llamacpp_engine(q8[0], n_gpu_layers=0)  # lo que usa la interfaz web con el GGUF
        assert isinstance(eng.complete("Hola", max_new_tokens=4, temperature=0.0), str)
    assert "bpb" in ph["eval"]["result"]["summary"]
    # volver a ejecutar no repite nada
    mtime = (tmp_path / "runs" / "plan_state.json").stat().st_mtime
    plan.run(log=lambda *_: None)
    assert json.loads((tmp_path / "runs" / "plan_state.json").read_text())["phases"] == ph
    assert (tmp_path / "runs" / "plan_state.json").stat().st_mtime >= mtime


def test_session_hours_are_shared_by_all_phases(tmp_path, monkeypatch):
    import titan.train.plan as plan_mod
    clock = iter([1000.0] + [1000.0 + 7200.0] * 10)  # la primera fase consumió 2 h de una sesión de 1 h
    monkeypatch.setattr(plan_mod.time, "time", lambda: next(clock))
    (tmp_path / "p.yaml").write_text(yaml.safe_dump({"name": "t", "workdir": str(tmp_path / "w"), "phases": [
        {"id": "a", "type": "train", "config": "configs/train/pc_120m_base.yaml"}]}))
    plan = Plan(tmp_path / "p.yaml")
    plan.run(session_max_hours=1.0, log=lambda *_: None)  # sin tiempo restante: no empieza la fase
    assert plan.status()[0]["status"] == "pending"
