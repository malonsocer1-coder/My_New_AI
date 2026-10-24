# Entrenamiento por etapas

Cada etapa produce artefactos verificables. Si faltan recursos para una etapa,
el trabajo previo queda guardado y el siguiente paso viable queda documentado.

| Etapa | Objetivo | Dónde | Estado |
|---|---|---|---|
| A — Validación técnica | modelo pequeño desde cero: aprende, generaliza a validación, genera, guarda/carga, reanuda | CPU de la sesión | **hecha** (ver `reports/INFORME_ETAPA_A.md`) |
| B — Prototipo | datos reales en español (~3 mil millones de tokens), 118 M parámetros, plan completo hasta GGUF, evaluación automática | tu PC (RTX 5070) | configurada (`configs/plans/pc_120m.yaml`, guía en `docs/PC_MODELO_120M.md`), pendiente de ejecutar |
| C — Escalado | DDP/FSDP2 en GPUs alquiladas, 350 M–1 B | nube | código verificado con 2 procesos; pendiente de presupuesto |
| D — Entrenamiento principal | el tamaño elegido según presupuesto (1 B–8 B o MoE) | nube | pendiente de presupuesto y autorización |
| E — Capacidades | SFT de instrucciones y herramientas, ajuste de preferencias | PC o nube | pipeline hecho y probado; SFT de demostración sobre la Etapa A |
| F — Despliegue | exportación, GGUF, cuantización, servidor local | PC | hecho y verificado contra llama.cpp |

## Comandos

```bash
titan data prepare --config configs/data/stage_a.yaml          # datos + tokenizador + shards
titan train --config configs/train/stage_a.yaml                # preentrenamiento (reanuda solo)
titan train --config configs/train/stage_a.yaml --set max_steps=100 lr=1e-3   # sobrescribir claves
torchrun --nproc_per_node=8 -m titan.cli train --config configs/train/titan_1b.yaml   # 8 GPUs
python scripts/make_sft_data.py && titan sft prepare --config configs/sft/stage_a.yaml
titan train --config configs/train/sft_stage_a.yaml            # SFT desde el modelo base
python scripts/make_pref_data.py && titan dpo --config configs/train/dpo_stage_a.yaml   # preferencias
titan plan run --plan configs/plans/pc_120m.yaml --hours 8     # plan completo de 118 M por sesiones
titan plan status --plan configs/plans/pc_120m.yaml
titan bench --config configs/train/pc_120m_base.yaml --find-batch   # micro-lotes que caben y su velocidad
titan bench --config configs/train/pc_120m_base.yaml --steps 10     # velocidad con la config tal cual
```

## Fases de capacidad y su justificación

1. **Preentrenamiento desde pesos aleatorios** — imprescindible: es donde el
   modelo aprende lengua y conocimiento. Consume > 95 % del cómputo.
2. **SFT (ajuste supervisado)** — necesario para conversar, seguir
   instrucciones y emitir llamadas a herramientas con el formato del protocolo.
   Barato (decenas a cientos de millones de tokens). Pérdida solo en los turnos del asistente.
3. **Ajuste de preferencias (DPO)** — implementado (`titan dpo`, `titan/sft/dpo.py`)
   con pares verificados por programa que castigan los fallos observados
   (inventar resultados tras un error, números falsos, afirmar acciones
   denegadas, falsa seguridad). Rinde de verdad a partir de ~1 B parámetros;
   en la Etapa A se ejecutó como demostración del circuito.
4. **Evaluación de seguridad y fiabilidad** — en cada etapa (`titan eval`), y
   con más peso antes de exponer el agente a herramientas con efectos.
5. **Optimización para inferencia** — al final, sin tocar los pesos originales.

## Hiperparámetros por escala (puntos de partida)

| Escala | LR pico | Tokens/paso | Calentamiento | Programación |
|---|---|---|---|---|
| ≤ 10 M | 1,5e-3 | 4 k | 200 pasos | coseno |
| 100 M | 6e-4 | 131 k | 500 | WSD |
| 118 M profundo (28 × 576) | elegida por el piloto entre 8e-4, 1,6e-3 y 3,2e-3 | 262 k | 500 | WSD |
| 1 B | 3e-4 | 524 k | 2 000 | WSD |
| 8 B | 1,5e-4–3e-4 | 2–4 M | 2 000 | WSD |

WSD (calentamiento-meseta-caída) permite continuar desde un checkpoint de la
meseta con más datos sin rehacer la programación, encaja con el entrenamiento
por partes.

## Lote, memoria y velocidad

| Clave | Qué hace |
|---|---|
| `batch_tokens` | lote global fijo (tokens por paso del optimizador); la acumulación se calcula sola |
| `auto_batch: true` | al empezar mide en la GPU cada micro-lote posible (pasos reales con AdamW) y elige el de más tokens/s que deja al menos `auto_batch_margin_gb` de VRAM libre; si nada cabe, activa el recálculo de activaciones. La medición se guarda en `cache/auto_batch/` y se reutiliza |
| `compile: auto` | `torch.compile` solo si hay GPU y Triton; si la compilación falla, sigue sin compilar |
| `max_epochs` (SFT) | calcula `max_steps` a partir del número real de filas de los datos |

Cambiar el reparto micro-lote × acumulación entre sesiones no altera el
entrenamiento: el orden de los datos depende solo del número de muestras vistas.

## Planes por fases

`titan plan run` ejecuta un YAML de fases reanudables (`titan/train/plan.py`):
`train`, `sweep` (ejecuciones cortas que eligen, p. ej., la tasa de
aprendizaje), `grow_depth`, `upcycle`, `merge_experts`, `sft_data`, `dpo`,
`export`, `gguf_quantize` y `eval`. Los valores pueden referirse a fases
anteriores: `@base` (su carpeta) o `@result:lr_pilot.best.lr` (un valor de su
resultado, admite `*0.1`). `--hours` limita la sesión entera, no cada fase.

## Métricas que se registran

`runs/<n>/metrics.jsonl`: pérdida, LR, norma del gradiente, tokens, tokens/s,
MFU (si se conoce el pico de la GPU), memoria máxima, horas, ETA, coste
acumulado y proyectado; evaluaciones por dominio (pérdida, perplejidad y bits
por byte, comparables entre tokenizadores). `samples.jsonl`: muestras
generadas en cada evaluación. `run_info.json`: configs, commit de git,
hardware y versiones.
