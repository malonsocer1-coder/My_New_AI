"""Planes de entrenamiento por fases ("entrenar por partes").

Un plan es una lista de fases que se ejecutan en orden. Cada fase deja sus
resultados en ``workdir/<id>`` y su estado en ``workdir/plan_state.json``,
de modo que el plan puede detenerse en cualquier momento (o limitarse a unas
horas por sesión) y continuar después sin repetir lo terminado.

Tipos de fase:
  * ``train``          entrena (preentrenamiento o SFT) con un YAML de config.
  * ``grow_depth``     agranda un modelo añadiendo capas (preserva la función).
  * ``upcycle``        convierte un modelo denso en MoE copiando la FFN.
  * ``merge_experts``  une ramas entrenadas por separado en un MoE (BTX).

Las referencias ``@fase`` apuntan al resultado de una fase anterior.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..utils.io import read_json, read_yaml, write_json
from . import grow


def _resolve(ref, workdir: Path):
    if isinstance(ref, str) and ref.startswith("@"):
        return str(workdir / ref[1:])
    if isinstance(ref, list):
        return [_resolve(r, workdir) for r in ref]
    return ref


class Plan:
    def __init__(self, path: str | Path, workdir: str | None = None):
        self.path = Path(path)
        self.spec = read_yaml(self.path)
        self.name = self.spec.get("name", self.path.stem)
        self.workdir = Path(workdir or self.spec.get("workdir", f"runs/{self.name}"))
        self.phases = self.spec["phases"]
        ids = [p["id"] for p in self.phases]
        if len(ids) != len(set(ids)):
            raise ValueError("Los id de fase deben ser únicos")
        self.state_path = self.workdir / "plan_state.json"

    def state(self) -> dict:
        if self.state_path.exists():
            return read_json(self.state_path)
        return {"plan": self.name, "phases": {}}

    def _save_state(self, st: dict) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        write_json(self.state_path, st)

    def status(self) -> list[dict]:
        st = self.state()["phases"]
        return [{"id": p["id"], "type": p["type"], **st.get(p["id"], {"status": "pending"})} for p in self.phases]

    # ------------------------------------------------------------------
    def run(self, only: list[str] | None = None, session_max_hours: float | None = None, log=print) -> dict:
        st = self.state()
        for phase in self.phases:
            pid = phase["id"]
            if only and pid not in only:
                continue
            cur = st["phases"].get(pid, {})
            if cur.get("status") == "done":
                log(f"[plan] {pid}: ya completada, se omite")
                continue
            log(f"[plan] ▶ fase {pid} ({phase['type']})")
            st["phases"][pid] = {**cur, "status": "running", "started_unix": cur.get("started_unix", time.time())}
            self._save_state(st)
            try:
                result = self._run_phase(phase, session_max_hours, log)
            except Exception as e:
                st["phases"][pid] = {**st["phases"][pid], "status": "failed", "error": repr(e)}
                self._save_state(st)
                raise
            done = result.get("stop_reason", "completed") == "completed"
            st["phases"][pid] = {**st["phases"][pid], "status": "done" if done else "interrupted",
                                 "result": result, "updated_unix": time.time()}
            if done:
                st["phases"][pid]["finished_unix"] = time.time()
            self._save_state(st)
            if not done:
                log(f"[plan] fase {pid} detenida ({result.get('stop_reason')}); "
                    "vuelva a ejecutar el plan para continuar desde el último checkpoint")
                break
        return self.state()

    def _run_phase(self, phase: dict, session_max_hours: float | None, log) -> dict:
        out = self.workdir / phase["id"]
        kind = phase["type"]
        if kind == "train":
            from ..model import ModelConfig
            from .config import load_run_config
            from .trainer import train
            overrides = dict(phase.get("set", {}))
            model_cfg, train_cfg = load_run_config(phase["config"], overrides)
            if "model_from" in phase:
                src_cfg, _ = grow.load_model_dir(_resolve(phase["model_from"], self.workdir))
                model_cfg = ModelConfig.from_dict({**src_cfg.to_dict(), **phase.get("model_set", {})})
            train_cfg.out_dir = str(out)
            train_cfg.run_name = f"{self.name}/{phase['id']}"
            if "init_from" in phase:
                train_cfg.init_from = _resolve(phase["init_from"], self.workdir)
            if session_max_hours is not None:
                train_cfg.session_max_hours = session_max_hours
            return train(model_cfg, train_cfg, log=log)

        if kind == "grow_depth":
            cfg, sd = grow.load_model_dir(_resolve(phase["source"], self.workdir))
            new_cfg, new_sd = grow.grow_depth(cfg, sd, phase["new_n_layers"], phase.get("method", "copy_zero"))
            grow.save_model_dir(new_cfg, new_sd, out, {"op": "grow_depth", **phase}, phase.get("tokenizer"))
            info = grow.describe(new_cfg)
            log(f"[plan] modelo agrandado: {info}")
            return info

        if kind == "upcycle":
            cfg, sd = grow.load_model_dir(_resolve(phase["source"], self.workdir))
            new_cfg, new_sd = grow.upcycle_to_moe(cfg, sd, phase["n_experts"], phase.get("top_k", 2))
            grow.save_model_dir(new_cfg, new_sd, out, {"op": "upcycle", **phase}, phase.get("tokenizer"))
            info = grow.describe(new_cfg)
            log(f"[plan] MoE por upcycling: {info}")
            return info

        if kind == "merge_experts":
            loaded = [grow.load_model_dir(p) for p in _resolve(phase["branches"], self.workdir)]
            new_cfg, new_sd = grow.merge_branches([c for c, _ in loaded], [s for _, s in loaded],
                                                  top_k=phase.get("top_k", 2),
                                                  shared_weights=phase.get("shared_weights"))
            grow.save_model_dir(new_cfg, new_sd, out, {"op": "merge_experts", **phase}, phase.get("tokenizer"))
            info = grow.describe(new_cfg)
            log(f"[plan] expertos unidos: {info}")
            return info

        raise ValueError(f"Tipo de fase desconocido: {kind}")
