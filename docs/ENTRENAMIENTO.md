# Entrenamiento por etapas

Cada etapa produce artefactos verificables. Si faltan recursos para una etapa,
el trabajo previo queda guardado y el siguiente paso viable queda documentado.

| Etapa | Objetivo | Dónde | Estado |
|---|---|---|---|
| A — Validación técnica | modelo pequeño desde cero: aprende, generaliza a validación, genera, guarda/carga, reanuda | CPU de la sesión | **hecha** (ver `reports/INFORME_ETAPA_A.md`) |
| B — Prototipo | datos reales en español (2–3 mil millones de tokens), 100 M parámetros, plan por partes, evaluación automática | tu PC (RTX 5070) | configurada (`configs/plans/pc_btx_100m.yaml`), pendiente de ejecutar |
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
titan plan run --plan configs/plans/pc_btx_100m.yaml --hours 4 # entrenamiento por partes
titan bench --config configs/train/pc_base_100m.yaml           # medir el hardware
```

## Fases de capacidad y su justificación

1. **Preentrenamiento desde pesos aleatorios** — imprescindible: es donde el
   modelo aprende lengua y conocimiento. Consume > 95 % del cómputo.
2. **SFT (ajuste supervisado)** — necesario para conversar, seguir
   instrucciones y emitir llamadas a herramientas con el formato del protocolo.
   Barato (decenas a cientos de millones de tokens). Pérdida solo en los turnos del asistente.
3. **Ajuste de preferencias (DPO u otros)** — útil a partir de ~1 B parámetros,
   cuando el modelo ya produce respuestas competentes entre las que elegir; en
   modelos pequeños aporta poco. Pendiente: se añadirá sobre el mismo
   formato de chat cuando haya un modelo base de ese tamaño.
4. **Evaluación de seguridad y fiabilidad** — en cada etapa (`titan eval`), y
   con más peso antes de exponer el agente a herramientas con efectos.
5. **Optimización para inferencia** — al final, sin tocar los pesos originales.

## Hiperparámetros por escala (puntos de partida)

| Escala | LR pico | Tokens/paso | Calentamiento | Programación |
|---|---|---|---|---|
| ≤ 10 M | 1,5e-3 | 4 k | 200 pasos | coseno |
| 100 M | 6e-4 | 131 k | 500 | WSD |
| 1 B | 3e-4 | 524 k | 2 000 | WSD |
| 8 B | 1,5e-4–3e-4 | 2–4 M | 2 000 | WSD |

WSD (calentamiento-meseta-caída) permite continuar desde un checkpoint de la
meseta con más datos sin rehacer la programación, encaja con el entrenamiento
por partes.

## Métricas que se registran

`runs/<n>/metrics.jsonl`: pérdida, LR, norma del gradiente, tokens, tokens/s,
MFU (si se conoce el pico de la GPU), memoria máxima, horas, ETA, coste
acumulado y proyectado; evaluaciones por dominio (pérdida, perplejidad y bits
por byte, comparables entre tokenizadores). `samples.jsonl`: muestras
generadas en cada evaluación. `run_info.json`: configs, commit de git,
hardware y versiones.
