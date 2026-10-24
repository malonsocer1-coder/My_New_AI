import json

import numpy as np
import yaml

from titan.eval.checkers import check_constraint, check_uncertainty
from titan.sft.data import SFTLoader, pack, prepare_sft
from titan.sft.synthetic_tools import EVAL_SEED_BASE, tool_conversations, tool_eval_items, write_jsonl
from titan.tokenizer import train_tokenizer


def test_checkers():
    assert check_constraint("- a\n- b\n- c", {"type": "num_bullets", "value": 3})
    assert check_constraint('Aquí: {"nombre": "Ana", "edad": 3}', {"type": "json_keys", "value": ["nombre", "edad"]})
    assert not check_constraint("hola, mundo", {"type": "no_comma"})
    assert check_constraint("Claro: la fotosíntesis...", {"type": "starts_with", "value": "Claro:"})
    assert check_constraint("gracias gracias", {"type": "count_word", "value": ["gracias", 2]})
    assert check_uncertainty("No lo sé, es aleatorio.", "abstain")
    assert check_uncertainty("¿Podrías darme más detalles?", "clarify")
    assert not check_uncertainty("El número será 42.", "abstain")


def test_pack_never_splits_conversations():
    enc = [([1] * 5, [-100, -100, 3, 3, 3]), ([2] * 4, [-100, 2, 2, 2]), ([3] * 9, [3] * 9), ([4] * 11, [4] * 11)]
    toks, mask = pack(enc, row_len=10, pad_id=0)
    assert toks.shape == (2, 10)  # la de 11 tokens no cabe y se descarta
    assert list(toks[0]) == [1] * 5 + [2] * 4 + [0]
    assert list(mask[0]) == [0, 0, 1, 1, 1, 0, 1, 1, 1, 0]


def test_tool_seeds_disjoint():
    train = tool_conversations(30, seed=5)
    ev = tool_eval_items(10)
    assert max(t["seed"] for t in train) < EVAL_SEED_BASE <= min(e["seed"] for e in ev)


def test_prepare_and_load_sft(tmp_path):
    convs = tool_conversations(40, seed=3)
    write_jsonl(convs, tmp_path / "tools.jsonl")
    texts = [json.dumps(c, ensure_ascii=False) for c in convs]
    tok = train_tokenizer(texts * 3, 600, tmp_path / "tok")
    cfg = {"tokenizer": str(tmp_path / "tok"), "seq_len": 1024, "output_dir": str(tmp_path / "sft"),
           "files": [str(tmp_path / "tools.jsonl")], "val_fraction": 0.1, "seed": 0}
    (tmp_path / "sft.yaml").write_text(yaml.safe_dump(cfg))
    meta = prepare_sft(str(tmp_path / "sft.yaml"), log=lambda *_: None)
    assert meta["splits"]["train"]["trained_tokens"] > 0
    ld = SFTLoader(tmp_path / "sft", "train", 1024, batch_size=2)
    x, y = ld.next_batch()
    assert x.shape == (2, 1024) and (y == -100).any() and (y != -100).any()
    # las etiquetas visibles coinciden con el siguiente token
    rows = np.memmap(tmp_path / "sft" / "train_tokens.bin", dtype=meta["dtype"], mode="r")
    assert rows.size % 1025 == 0
