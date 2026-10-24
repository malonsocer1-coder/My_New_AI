"""Compara base, ramas expertas, MoE unido, MoE ajustado y control denso (bits por byte por dominio).

Uso: python scripts/btx_report.py --workdir runs/stage_a_btx --base runs/stage_a --out reports/btx_stage_a.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.eval.harness import eval_bpb  # noqa: E402
from titan.infer.engine import load_engine  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="runs/stage_a_btx")
    p.add_argument("--base", default="runs/stage_a")
    p.add_argument("--data", default="data/stage_a/manifest.json")
    p.add_argument("--tokenizer", default="data/stage_a/tokenizer/tokenizer.json")
    p.add_argument("--out", default="reports/btx_stage_a.json")
    p.add_argument("--windows", type=int, default=64)
    a = p.parse_args()
    w = Path(a.workdir)
    models = {"base": a.base, "expert_es": w / "expert_es", "expert_en": w / "expert_en",
              "expert_code": w / "expert_code", "moe_unido": w / "merge", "moe_ajustado": w / "moe_tune",
              "control_denso": w / "dense_control"}
    rows = {}
    for name, path in models.items():
        if not Path(path).exists():
            continue
        eng = load_engine(path, device="cpu", tokenizer=a.tokenizer)
        r = eval_bpb(eng, a.data, windows=a.windows)
        pc = eng.model.cfg.param_counts()
        rows[name] = {"params_total": pc["total"], "params_active": pc["active"],
                      "bpb": {d: v["bpb"] for d, v in r.items() if "bpb" in v}}
        rows[name]["bpb_mean"] = round(sum(rows[name]["bpb"].values()) / len(rows[name]["bpb"]), 4)
        print(name, rows[name]["params_total"], rows[name]["bpb_mean"], rows[name]["bpb"], flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
