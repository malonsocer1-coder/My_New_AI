"""Mini estudio de escalado: varios tamaños con los mismos tokens y datos.

Comprueba con datos propios que la pérdida de validación baja al aumentar los
parámetros (la base empírica para decidir cuándo escalar) y ajusta una ley de
potencias L(N) = a·N^(-b) + c de forma aproximada.

Uso: python scripts/scaling_sweep.py --steps 1000 --out reports/scaling_sweep.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from titan.model import ModelConfig  # noqa: E402
from titan.train.config import TrainConfig  # noqa: E402
from titan.train.trainer import train  # noqa: E402

SIZES = {
    "xs": dict(dim=128, n_layers=4, n_heads=4, n_kv_heads=2),
    "s": dict(dim=256, n_layers=6, n_heads=8, n_kv_heads=4),
    "m": dict(dim=384, n_layers=8, n_heads=6, n_kv_heads=2),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--out", default="reports/scaling_sweep.json")
    p.add_argument("--data", default="data/stage_a/manifest.json")
    p.add_argument("--sizes", nargs="*", default=list(SIZES), help="subconjunto de tamaños")
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--tag", default="", help="sufijo para las carpetas de ejecución")
    a = p.parse_args()
    rows = []
    for name, dims in SIZES.items():
        if name not in a.sizes:
            continue
        mc = ModelConfig(name=f"sweep-{name}", vocab_size=8192, multiple_of=64, max_seq_len=256, **dims)
        run = f"sweep_{name}{a.tag}"
        tc = TrainConfig(run_name=run, out_dir=f"runs/{run}", data=a.data, seq_len=256,
                         micro_batch_size=16, max_steps=a.steps, lr=a.lr, min_lr=a.lr / 10,
                         warmup_steps=min(200, a.steps // 10), dtype="fp32", device="cpu", num_threads=4,
                         compile=True, eval_interval=a.steps, eval_windows=64, eval_batch_size=32,
                         log_interval=100, ckpt_interval=a.steps, keep_last=1, seed=1234)
        s = train(mc, tc, log=lambda m: print(m, flush=True))
        ev = [json.loads(l) for l in open(f"runs/{run}/metrics.jsonl") if '"eval"' in l][-1]
        info = json.loads(open(f"runs/{run}/run_info.json").read())
        rows.append({"size": name, "lr": a.lr, "params": mc.param_counts()["total"],
                     "non_embedding_params": mc.param_counts()["non_embedding_active"],
                     "tokens": s["tokens"], "hours": round(s["elapsed_h"], 3), "val_mean_loss": round(ev["_mean"]["loss"], 4),
                     "val_bpb": {k: round(v["bpb"], 4) for k, v in ev.items() if isinstance(v, dict) and "bpb" in v},
                     "tokens_per_step": info["tokens_per_step"]})
        print(rows[-1], flush=True)
    n = np.array([r["non_embedding_params"] for r in rows], dtype=float)
    loss = np.array([r["val_mean_loss"] for r in rows])
    # pendiente log-log (sin término constante); requiere al menos dos tamaños
    slope = float(np.polyfit(np.log(n), np.log(loss), 1)[0]) if len(rows) > 1 else None
    report = {"rows": rows, "loglog_slope_loss_vs_nonembedding_params": round(slope, 4) if slope is not None else None,
              "note": "mismos datos, tokens y programación de LR; CPU de la sesión en la nube"}
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
