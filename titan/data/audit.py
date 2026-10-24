"""Auditoría de contaminación: comprueba a posteriori que ningún texto de evaluación
está en los datos limpios de entrenamiento (útil al añadir benchmarks nuevos)."""

from __future__ import annotations

import json
from pathlib import Path

from .decontam import ContaminationIndex
from .pipeline import _load_eval_texts


def audit_contamination(data_dir: str, eval_files: list[str], n: int = 13) -> dict:
    idx = ContaminationIndex(n=n)
    idx.add_eval_texts(_load_eval_texts(eval_files))
    report = {"eval_items_indexed": len(idx), "files": {}}
    for p in sorted(Path(data_dir, "clean").glob("*.jsonl")):
        hits = total = 0
        examples = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                total += 1
                row = json.loads(line)
                if idx.is_contaminated(row["text"]):
                    hits += 1
                    if len(examples) < 3:
                        examples.append(row["id"])
        report["files"][p.name] = {"docs": total, "contaminated": hits, "examples": examples}
    report["total_contaminated"] = sum(v["contaminated"] for v in report["files"].values())
    return report
