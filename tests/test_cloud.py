import argparse
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

from titan.cloud import job as jobmod
from titan.cloud.estimate import PRESETS, estimate, inference, preset_config, scaling_table
from titan.cloud.providers.base import ApprovalRequired, Offer
from titan.cloud.providers.impl import Lambda, RunPod, VastAI
from titan.cloud.watchdog import SpendGuard, run as watchdog_run, stop_training


def test_presets_param_counts():
    assert 7.0e9 < preset_config("titan-8b").param_counts()["total"] < 8.0e9
    big = preset_config("titan-100b").param_counts()["total"]
    assert big * 2 / 1e9 > 190  # ~195 GB en bf16: cubre el objetivo de almacenamiento de 200 GB
    moe = preset_config("titan-moe-7b-a2b").param_counts()
    assert moe["active"] < moe["total"] / 2


def test_estimate_scales_and_local_fit():
    small = estimate(preset_config("titan-1b"), 1e11, "H100", 8)
    large = estimate(preset_config("titan-8b"), 1e11, "H100", 8)
    assert large["hours"] > 5 * small["hours"] and small["cost_usd_low"] <= small["cost_usd_high"]
    assert inference(preset_config("titan-1b"), "Q4_K_M")["fits_vram"]
    assert not inference(preset_config("titan-70b"), "Q4_K_M")["fits_vram"]
    assert len(scaling_table()) == len(PRESETS)


class FakeTransport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        for key, resp in self.responses.items():
            if key in url:
                return resp
        return {}


def test_create_requires_approval_and_key_not_logged(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "secreto-123")
    t = FakeTransport({"/pods": {"id": "pod1", "desiredStatus": "RUNNING"}})
    rp = RunPod(transport=t)
    offer = Offer("runpod", "NVIDIA H100 80GB HBM3", "H100", 8, 24.0)
    spec = {"name": "x", "image": "img", "startup_script": "echo hola"}
    with pytest.raises(ApprovalRequired):
        rp.create(offer, spec)
    inst = rp.create(offer, spec, approved=True)
    assert inst.instance_id == "pod1"
    method, url, headers, body = t.calls[-1]
    assert body["gpuCount"] == 8 and body["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"]
    assert "secreto-123" not in json.dumps(rp.sent)  # la clave solo viaja en la cabecera


def test_dry_run_sends_nothing():
    v = VastAI(dry_run=True, transport=lambda *a: pytest.fail("no debe enviar"))
    r = v.terminate("123")
    assert r["dry_run"] and v.sent[0]["method"] == "DELETE"


def test_vast_and_lambda_parse_offers(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k")
    monkeypatch.setenv("LAMBDA_API_KEY", "k")
    v = VastAI(transport=FakeTransport({"/bundles": {"offers": [{"id": 7, "num_gpus": 8, "dph_total": 15.2}]}}))
    assert v.list_offers("H100", 8)[0].price_per_gpu_hour == pytest.approx(1.9)
    lam = Lambda(transport=FakeTransport({"/instance-types": {"data": {"gpu_8x_h100_sxm5": {
        "instance_type": {"price_cents_per_hour": 2392, "specs": {"gpus": 8}},
        "regions_with_capacity_available": [{"name": "us-east-1"}]}}}}))
    o = lam.list_offers("H100", 8)[0]
    assert o.price_per_hour == pytest.approx(23.92) and o.region == "us-east-1"


@pytest.fixture
def job_spec(tmp_path):
    tc = tmp_path / "train.yaml"
    tc.write_text(yaml.safe_dump({"model_config": "configs/model/titan_1b.yaml",
                                  "train": {"seq_len": 2048, "micro_batch_size": 8, "grad_accum_steps": 4,
                                            "max_steps": 1000}}))
    return jobmod.JobSpec(name="t", provider="vastai", train_config=str(tc), budget_usd=100.0, max_hours=10)


def test_plan_budget_and_launch_gating(job_spec, tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k")
    offers = [Offer("vastai", "1", "H100", 8, 16.0), Offer("vastai", "2", "H100", 8, 20.0)]
    p = jobmod.plan(job_spec, VastAI(dry_run=True), offers)
    assert p["offer"]["offer_id"] == "1" and p["rate_per_gpu_hour"] == 2.0
    with pytest.raises(ApprovalRequired):
        jobmod.launch(job_spec, approve=False, ledger=tmp_path / "l.jsonl")

    class Prov(VastAI):
        def list_offers(self, gpu, n):
            return offers

    t = FakeTransport({"/asks/": {"new_contract": 99}})
    ledger = tmp_path / "ledger.jsonl"
    job_spec.budget_usd = 10_000
    inst = jobmod.launch(job_spec, approve=True, provider=Prov(transport=t), ledger=ledger)
    assert inst.instance_id == "99"
    entry = json.loads(ledger.read_text().splitlines()[0])
    assert entry["approved_by_user"] and entry["rate_per_hour"] == 16.0
    job_spec.budget_usd = 1
    with pytest.raises(ApprovalRequired):  # fuera de presupuesto: no se lanza aunque se apruebe
        jobmod.launch(job_spec, approve=True, provider=Prov(transport=t), ledger=ledger)


def test_spend_guard_with_fake_clock():
    now = [0.0]
    g = SpendGuard(budget_usd=100, rate_per_hour=20, start=0.0, clock=lambda: now[0])
    assert g.check() == "ok"
    now[0] = 4.6 * 3600  # 92 $
    assert g.check() == "warn"
    now[0] = 5.0 * 3600
    assert g.check() == "stop"


def test_watchdog_stops_training_gracefully(tmp_path):
    child = tmp_path / "child.py"
    flag = tmp_path / "saved.txt"
    child.write_text(f"import signal, sys, time\n"
                     f"def h(s, f):\n    open({str(flag)!r}, 'w').write('checkpoint')\n    sys.exit(0)\n"
                     f"signal.signal(signal.SIGTERM, h)\nwhile True: time.sleep(0.1)\n")
    p = subprocess.Popen([sys.executable, str(child)], start_new_session=True)
    time.sleep(0.5)
    (tmp_path / "train.pid").write_text(str(p.pid))
    now = [1000.0]
    args = argparse.Namespace(budget=10.0, rate=10.0, max_hours=None, run_dir=str(tmp_path / "run"),
                              pidfile=str(tmp_path / "train.pid"), remote="", provider=None, interval=1,
                              sync_min=30, stale_min=20, log=str(tmp_path / "wd.jsonl"), no_terminate=True,
                              exit_when_done=False)

    def fake_sleep(_):
        now[0] += 3600  # una hora por iteración: a la segunda se agota el presupuesto

    assert watchdog_run(args, clock=lambda: now[0], sleep=fake_sleep) == "stopped"
    p.wait(10)
    assert flag.read_text() == "checkpoint"
