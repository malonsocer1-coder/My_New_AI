"""Configuración de entrenamiento (preentrenamiento y SFT)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from ..model.config import ModelConfig
from ..utils.io import read_yaml


@dataclass
class TrainConfig:
    run_name: str = "titan-run"
    out_dir: str = "runs/titan-run"
    task: str = "pretrain"  # pretrain | sft
    data: str = ""          # manifest.json (pretrain) o directorio SFT tokenizado
    tokenizer: str | None = None
    seq_len: int = 1024
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    # Lote global fijo en tokens por paso del optimizador. Si se define, la acumulación
    # se calcula sola (batch_tokens / (seq_len * micro_batch_size * GPUs)).
    batch_tokens: int | None = None
    # Elegir el micro-lote midiendo tokens/s y memoria en la GPU (ver train/autobatch.py).
    auto_batch: bool = False
    auto_batch_max: int = 32
    auto_batch_margin_gb: float = 0.7   # memoria de GPU que debe quedar libre en el pico
    max_steps: int = 1000
    max_epochs: float | None = None  # SFT: calcula max_steps a partir del tamaño de los datos
    # Optimización
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    schedule: str = "cosine"  # cosine | wsd | linear | constant
    wsd_decay_frac: float = 0.2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    # Numérica y rendimiento
    dtype: str = "bf16"  # bf16 | fp16 | fp32
    compile: bool | str = False  # True | False | "auto" (solo si hay GPU y Triton instalado)
    parallel: str = "auto"  # auto | none | ddp | fsdp
    fsdp_shard_size: int | None = None  # HSDP: tamaño del grupo de sharding (p. ej. GPUs por nodo)
    device: str = "auto"
    num_threads: int | None = None
    # Evaluación, registro y checkpoints
    eval_interval: int = 200
    eval_windows: int = 64
    eval_batch_size: int | None = None
    log_interval: int = 10
    ckpt_interval: int = 200
    keep_last: int = 3
    keep_every: int | None = None
    resume: bool = True
    verify_checkpoints: bool = True
    init_from: str | None = None  # checkpoint/pesos base para SFT o continuación
    freeze_patterns: list[str] = field(default_factory=list)  # p. ej. ["layers.*.attention.*"]
    skip_samples_on_resume: int = 0  # saltar datos tras un pico de pérdida
    max_bad_steps: int = 10
    sample_prompts: list[str] = field(default_factory=list)
    sample_max_new_tokens: int = 48
    seed: int = 1234
    domain_weights: dict[str, float] | None = None
    # Presupuesto y límites
    gpu_hour_usd: float = 0.0
    budget_usd: float | None = None
    max_hours: float | None = None          # tiempo total acumulado del entrenamiento
    session_max_hours: float | None = None  # tiempo de esta sesión (entrenar "por ratos")
    peak_flops_per_device: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Claves desconocidas en la config de entrenamiento: {sorted(unknown)}")
        return cls(**d)


def load_run_config(path: str, overrides: dict[str, Any] | None = None) -> tuple[ModelConfig, TrainConfig]:
    """Lee un YAML con secciones ``model`` y ``train`` (o ``model_config: ruta``)."""
    d = read_yaml(path)
    model_d = d.get("model")
    if model_d is None and "model_config" in d:
        model_d = read_yaml(d["model_config"])
    if model_d is None:
        raise ValueError(f"{path}: falta la sección 'model' o 'model_config'")
    train_d = dict(d.get("train", {}))
    overrides = dict(overrides or {})
    if "model_config" in overrides:  # cambiar la arquitectura sin editar el YAML
        model_d = read_yaml(overrides.pop("model_config"))
    for k, v in overrides.items():
        if k.startswith("model."):
            model_d[k[6:]] = v
        else:
            train_d[k] = v
    return ModelConfig.from_dict(model_d), TrainConfig.from_dict(train_d)
