"""Genera pares de preferencia verificados para DPO (entrenamiento y evaluación reservada).

Uso: python scripts/make_pref_data.py --out data/dpo_raw --train 2000 --eval 200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.sft.synthetic_prefs import preference_pairs  # noqa: E402
from titan.sft.synthetic_tools import write_jsonl  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/dpo_raw")
p.add_argument("--train", type=int, default=2000)
p.add_argument("--eval", type=int, default=200)
p.add_argument("--seed", type=int, default=3)
a = p.parse_args()
write_jsonl(preference_pairs(a.train, seed=a.seed), Path(a.out) / "train.jsonl")
write_jsonl(preference_pairs(a.eval, seed=0, eval_split=True), Path(a.out) / "eval.jsonl")
print(f"escrito en {a.out}")
