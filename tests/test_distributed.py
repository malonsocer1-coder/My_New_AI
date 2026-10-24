"""Entrenamiento distribuido real con 2 procesos (torchrun + gloo en CPU): DDP y FSDP2."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _manifest(tmp: Path) -> Path:
    (tmp / "tokens").mkdir()
    shards = []
    for split in ("train", "val"):
        arr = np.tile(np.arange(40, dtype=np.uint16), 400)
        arr.tofile(tmp / "tokens" / f"x_{split}.bin")
        shards.append({"path": f"tokens/x_{split}.bin", "split": split, "domain": "x", "num_tokens": int(arr.size),
                       "dtype": "uint16"})
    (tmp / "manifest.json").write_text(json.dumps({"dtype": "uint16", "shards": shards, "domain_weights": {"x": 1.0},
                                                   "totals": {}}))
    return tmp / "manifest.json"


@pytest.mark.parametrize("mode", ["ddp", "fsdp"])
def test_two_process_training_and_resume(tmp_path, mode):
    man = _manifest(tmp_path)
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({
        "model": {"vocab_size": 64, "dim": 32, "n_layers": 2, "n_heads": 2, "max_seq_len": 32, "multiple_of": 16},
        "train": {"data": str(man), "seq_len": 32, "micro_batch_size": 4, "grad_accum_steps": 2, "max_steps": 8,
                  "lr": 3e-3, "warmup_steps": 1, "dtype": "fp32", "device": "cpu", "num_threads": 1,
                  "eval_interval": 8, "eval_windows": 4, "log_interval": 4, "ckpt_interval": 4, "parallel": mode,
                  "out_dir": str(tmp_path / "run")}}))
    torchrun = Path(sys.executable).parent / "torchrun"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}

    def run(steps):
        cmd = [str(torchrun), "--standalone", "--nproc_per_node=2", "-m", "titan.cli", "train", "--config", str(cfg),
               "--set", f"max_steps={steps}"]
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r.stdout

    run(4)
    out = run(8)
    assert "[reanudar]" in out
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["steps"] == 8 and summary["stop_reason"] == "completed"
    losses = [json.loads(l)["loss"] for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
              if '"train"' in l]
    assert losses[-1] < losses[0]
    from titan.train.grow import load_model_dir
    cfg_m, sd = load_model_dir(tmp_path / "run")  # consolida DCP si hace falta
    assert sum(v.numel() for v in sd.values()) == cfg_m.param_counts()["total"]
