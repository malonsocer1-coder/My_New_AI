"""Planes de entrenamiento por fases ("entrenar por partes").

Un plan es una lista de fases que se ejecutan en orden. Cada fase deja sus
resultados en ``workdir/<id>`` y su estado en ``workdir/plan_state.json``,
de modo que el plan puede detenerse en cualquier momento (o limitarse a unas
horas por sesión) y continuar después sin repetir lo terminado.

Tipos de fase:
  * ``train``          entrena (preentrenamiento o SFT) con un YAML de config.
  * ``sweep``          varias ejecuciones cortas (p. ej. tasas de aprendizaje) y elige la mejor.
  * ``grow_depth``     agranda un modelo añadiendo capas (preserva la función).
  * ``upcycle``        convierte un modelo denso en MoE copiando la FFN.
  * ``merge_experts``  une ramas entrenadas por separado en un MoE (BTX).
  * ``sft_data``       genera y tokeniza los datos de SFT (sintéticos verificados + importados).
  * ``dpo``            genera pares de preferencia verificados y ajusta con DPO.
  * ``export``         exporta pesos verificados (TITAN, Hugging Face, GGUF).
  * ``gguf_quantize``  cuantiza el GGUF exportado con llama.cpp.
  * ``eval``           ejecuta los benchmarks propios y guarda el informe.

Referencias en los valores de la fase:
  * ``@fase`` o ``@fase/ruta`` -> ruta dentro del resultado de una fase anterior.
  * ``@result:fase.clave.subclave`` -> un valor del resultado de una fase terminada
    (p. ej. la mejor tasa de aprendizaje de un ``sweep``); admite ``*factor``.

El límite de horas de una sesión (``--hours``) es para todo el plan: si una fase
termina antes, la siguiente usa solo el tiempo que queda.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..utils.io import read_json, read_yaml, write_json
from . import grow


class PlanError(RuntimeError):
    pass


def _lookup(expr: str, phases_state: dict):
    factor = 1.0
    if "*" in expr:
        expr, f = expr.split("*", 1)
        factor = float(f)
    pid, *keys = expr.split(".")
    ph = phases_state.get(pid)
    if not ph or ph.get("status") != "done":
        raise PlanError(f"@result:{expr}: la fase '{pid}' no ha terminado")
    val = ph.get("result", {})
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            raise PlanError(f"@result:{expr}: no existe la clave '{k}' en el resultado de '{pid}'")
        val = val[k]
    return val * factor if factor != 1.0 else val


def _resolve(ref, workdir: Path, phases_state: dict | None = None):
    if isinstance(ref, str) and ref.startswith("@result:"):
        return _lookup(ref[len("@result:"):], phases_state or {})
    if isinstance(ref, str) and ref.startswith("@"):
        return str(workdir / ref[1:])
    if isinstance(ref, list):
        return [_resolve(r, workdir, phases_state) for r in ref]
    if isinstance(ref, dict):
        return {k: _resolve(v, workdir, phases_state) for k, v in ref.items()}
    return ref


def _tag(values: dict) -> str:
    return "_".join(f"{k}-{v}" for k, v in sorted(values.items()))


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
        self._deadline: float | None = None

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

    def _hours_left(self) -> float | None:
        if self._deadline is None:
            return None
        return (self._deadline - time.time()) / 3600

    # ------------------------------------------------------------------
    def run(self, only: list[str] | None = None, session_max_hours: float | None = None, log=print) -> dict:
        st = self.state()
        self._deadline = time.time() + session_max_hours * 3600 if session_max_hours else None
        for phase in self.phases:
            pid = phase["id"]
            if only and pid not in only:
                continue
            cur = st["phases"].get(pid, {})
            if cur.get("status") == "done":
                log(f"[plan] {pid}: ya completada, se omite")
                continue
            left = self._hours_left()
            if left is not None and left <= 0:
                log("[plan] se agotó el tiempo de esta sesión; vuelva a ejecutar el plan para continuar")
                break
            if phase.get("enabled", True) is False:
                log(f"[plan] {pid}: desactivada en el plan, se omite")
                continue
            log(f"[plan] ▶ fase {pid} ({phase['type']})")
            st["phases"][pid] = {**cur, "status": "running", "started_unix": cur.get("started_unix", time.time())}
            self._save_state(st)
            try:
                result = self._run_phase(phase, log)
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

    # ------------------------------------------------------------------
    def _train(self, config: str, overrides: dict, out: Path, run_name: str, phase: dict, log) -> dict:
        from ..model import ModelConfig
        from .config import load_run_config
        from .trainer import train
        model_cfg, train_cfg = load_run_config(config, overrides)
        if "model_from" in phase:
            src_cfg, _ = grow.load_model_dir(self._r(phase["model_from"]))
            model_cfg = ModelConfig.from_dict({**src_cfg.to_dict(), **phase.get("model_set", {})})
        train_cfg.out_dir = str(out)
        train_cfg.run_name = run_name
        if "init_from" in phase:
            train_cfg.init_from = self._r(phase["init_from"])
        left = self._hours_left()
        if left is not None:
            train_cfg.session_max_hours = max(left, 1e-3)
        return train(model_cfg, train_cfg, log=log)

    def _r(self, ref):
        return _resolve(ref, self.workdir, self.state()["phases"])

    def _run_phase(self, phase: dict, log) -> dict:
        out = self.workdir / phase["id"]
        kind = phase["type"]
        r = self._r
        if kind == "train":
            return self._train(phase["config"], r(dict(phase.get("set", {}))), out,
                               f"{self.name}/{phase['id']}", phase, log)

        if kind == "sweep":
            return self._sweep(phase, out, log)

        if kind == "grow_depth":
            cfg, sd = grow.load_model_dir(r(phase["source"]))
            new_cfg, new_sd = grow.grow_depth(cfg, sd, phase["new_n_layers"], phase.get("method", "copy_zero"))
            grow.save_model_dir(new_cfg, new_sd, out, {"op": "grow_depth", **phase}, phase.get("tokenizer"))
            info = grow.describe(new_cfg)
            log(f"[plan] modelo agrandado: {info}")
            return info

        if kind == "upcycle":
            cfg, sd = grow.load_model_dir(r(phase["source"]))
            new_cfg, new_sd = grow.upcycle_to_moe(cfg, sd, phase["n_experts"], phase.get("top_k", 2))
            grow.save_model_dir(new_cfg, new_sd, out, {"op": "upcycle", **phase}, phase.get("tokenizer"))
            info = grow.describe(new_cfg)
            log(f"[plan] MoE por upcycling: {info}")
            return info

        if kind == "merge_experts":
            loaded = [grow.load_model_dir(p) for p in r(phase["branches"])]
            new_cfg, new_sd = grow.merge_branches([c for c, _ in loaded], [s for _, s in loaded],
                                                  top_k=phase.get("top_k", 2),
                                                  shared_weights=phase.get("shared_weights"))
            grow.save_model_dir(new_cfg, new_sd, out, {"op": "merge_experts", **phase}, phase.get("tokenizer"))
            info = grow.describe(new_cfg)
            log(f"[plan] expertos unidos: {info}")
            return info

        if kind == "sft_data":
            return self._sft_data(phase, out, log)

        if kind == "dpo":
            from ..sft.dpo import DPOConfig, train_dpo
            from ..sft.synthetic_prefs import preference_pairs
            from ..sft.synthetic_tools import write_jsonl
            pairs = phase.get("pairs", {})
            data = out / "data"
            write_jsonl(preference_pairs(int(pairs.get("train", 2000)), seed=int(pairs.get("seed", 3))),
                        data / "train.jsonl")
            write_jsonl(preference_pairs(int(pairs.get("eval", 200)), seed=0, eval_split=True), data / "eval.jsonl")
            cfg = DPOConfig(init_from=r(phase["init_from"]), data=str(data / "train.jsonl"),
                            eval_data=str(data / "eval.jsonl"), out_dir=str(out), **r(dict(phase.get("set", {}))))
            summ = train_dpo(cfg, log=log)
            return {"stop_reason": "completed", "eval_before": summ["eval_before"], "eval_after": summ["eval_after"],
                    "elapsed_h": summ["elapsed_h"]}

        if kind == "export":
            from ..export.exporter import export_checkpoint
            info = export_checkpoint(r(phase["checkpoint"]), out, formats=phase.get("formats", ["titan", "hf", "gguf"]),
                                     dtype=phase.get("dtype", "bf16"), tokenizer=r(phase.get("tokenizer")))
            return {"stop_reason": "completed", "out": info["out"], "files": info["files"]}

        if kind == "gguf_quantize":
            from ..export.gguf_tools import quantize_gguf
            src = Path(r(phase["model"]))
            if src.is_dir():
                found = sorted(src.rglob("*.gguf"))
                if len(found) != 1:
                    raise PlanError(f"{src}: se esperaba un único .gguf y hay {len(found)}; indique el archivo")
                src = found[0]
            types = phase.get("types", ["Q8_0"])
            base = src.stem.rsplit("-", 1)[0]
            res = {t: quantize_gguf(src, out / f"{base}-{t}.gguf", t) for t in types}
            for cand in (src.parent / "tokenizer.json", src.parent.parent / "tokenizer.json"):
                if cand.exists():  # el motor llama.cpp de TITAN busca el tokenizador junto al GGUF
                    (out / "tokenizer.json").write_bytes(cand.read_bytes())
                    break
            return {"stop_reason": "completed", "source": str(src), "quantized": res}

        if kind == "eval":
            from ..eval.harness import run_suite
            out.mkdir(parents=True, exist_ok=True)
            res = run_suite(r(phase["checkpoint"]), phase.get("suite", "all"), out_path=str(out / "eval.json"),
                            device=phase.get("device", "auto"), limit=phase.get("limit"), data=r(phase.get("data")))
            return {"stop_reason": "completed", "report": str(out / "eval.json"), "summary": res["summary"]}

        raise ValueError(f"Tipo de fase desconocido: {kind}")

    # ------------------------------------------------------------------
    def _sweep(self, phase: dict, out: Path, log) -> dict:
        """Ejecuta cada variante (reanudable) y elige la de menor métrica.

        Con ``tie_tolerance`` (relativa) se elige la primera variante de la lista que
        quede dentro de ese margen del mínimo: ordenadas de menor a mayor tasa de
        aprendizaje, compensa que las ejecuciones cortas favorecen tasas altas.
        """
        metric = phase.get("metric", "final_val_loss")
        base_set = self._r(dict(phase.get("set", {})))
        results = []
        for values in phase["variants"]:
            vout = out / _tag(values)
            summ_path = vout / "summary.json"
            summ = read_json(summ_path) if summ_path.exists() else None
            if not summ or summ.get("stop_reason") not in ("completed", "bad_steps"):
                log(f"[plan] variante {values}")
                summ = self._train(phase["config"], {**base_set, **values}, vout,
                                   f"{self.name}/{phase['id']}/{_tag(values)}", phase, log)
                if summ.get("stop_reason") not in ("completed", "bad_steps"):
                    return {"stop_reason": summ["stop_reason"], "partial": results}
            if summ["stop_reason"] == "bad_steps":  # divergió: cuenta como la peor
                log(f"[plan] variante {values}: divergió (pérdida no finita)")
                results.append({"values": values, metric: float("inf"), "diverged": True})
                continue
            if summ.get(metric) is None:
                raise PlanError(f"la variante {values} no tiene '{metric}' (¿eval_interval desactivado?)")
            results.append({"values": values, metric: summ[metric], "tokens": summ.get("tokens"),
                            "elapsed_h": summ.get("elapsed_h")})
        best_val = min(x[metric] for x in results)
        if best_val == float("inf"):
            raise PlanError(f"{phase['id']}: todas las variantes divergieron; pruebe tasas más bajas")
        tol = float(phase.get("tie_tolerance", 0.0))
        chosen = next(x for x in results if x[metric] <= best_val * (1 + tol))
        log(f"[plan] {phase['id']}: elegida {chosen['values']} ({metric} {chosen[metric]:.4f}; mínimo {best_val:.4f})")
        return {"stop_reason": "completed", "best": chosen["values"], "metric": metric, "variants": results}

    def _sft_data(self, phase: dict, out: Path, log) -> dict:
        from ..sft.data import prepare_sft
        from ..sft.synthetic import arithmetic_chats
        from ..sft.synthetic_tools import tool_conversations, write_jsonl
        raw = out / "raw"
        syn = phase.get("synthetic", {})
        seed = int(syn.get("seed", 1))
        card = {"sources": [], "skipped": []}
        if syn.get("tools"):
            write_jsonl(tool_conversations(int(syn["tools"]), seed=seed), raw / "tools.jsonl")
            card["sources"].append({"name": "synthetic_tools", "kept": int(syn["tools"]),
                                    "license": "Propio (generado y verificado ejecutando las herramientas)"})
        if syn.get("arith"):
            write_jsonl(arithmetic_chats(int(syn["arith"]), seed=100 + seed,
                                         exclude_file="titan/eval/data/arith_es.jsonl"), raw / "arith.jsonl")
            card["sources"].append({"name": "synthetic_arith", "kept": int(syn["arith"]),
                                    "license": "Propio (generado y verificado)"})
        imports = [s for s in phase.get("imports", []) if s.get("enabled", True)]
        if imports:
            from ..sft.importers import eval_index, import_chats
            idx = eval_index(phase.get("eval_files", ["titan/eval/data/*.jsonl"]))
            for spec in imports:
                try:
                    convs, stats = import_chats(spec, idx, log)
                except Exception as e:
                    if not spec.get("optional"):
                        raise
                    log(f"[sft] aviso: no se pudo importar '{spec['name']}' ({e!r}); se continúa sin esa fuente")
                    card["skipped"].append({"name": spec["name"], "error": repr(e)})
                    continue
                write_jsonl(convs, raw / f"{spec['name']}.jsonl")
                card["sources"].append(stats)
        meta = prepare_sft({"tokenizer": self._r(phase["tokenizer"]), "seq_len": int(phase["seq_len"]),
                            "output_dir": str(out), "files": [str(raw / "*.jsonl")],
                            "val_fraction": phase.get("val_fraction", 0.03), "seed": phase.get("seed", 0)}, log)
        write_json(out / "sft_data_card.json", {**card, "splits": meta["splits"]})
        return {"stop_reason": "completed", "splits": meta["splits"], **card}
