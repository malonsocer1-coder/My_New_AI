"""Genera los datos SFT sintéticos verificados (uso de herramientas + aritmética conversacional).

Uso: python scripts/make_sft_data.py --out data/sft_raw --tools 3000 --arith 2000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.sft.synthetic import arithmetic_chats  # noqa: E402
from titan.sft.synthetic_tools import tool_conversations, write_jsonl  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/sft_raw")
p.add_argument("--tools", type=int, default=3000)
p.add_argument("--arith", type=int, default=2000)
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
out = Path(a.out)
write_jsonl(tool_conversations(a.tools, seed=a.seed), out / "tools.jsonl")
write_jsonl(arithmetic_chats(a.arith, seed=100 + a.seed, exclude_file="titan/eval/data/arith_es.jsonl"),
            out / "arith.jsonl")
print(f"escrito en {out}")
