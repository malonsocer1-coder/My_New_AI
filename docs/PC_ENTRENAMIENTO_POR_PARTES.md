# Entrenar TITAN por partes en tu PC (RTX 5070 Laptop, 8 GB)

Esta guía explica cómo entrenar en tu propio equipo un modelo TITAN **desde
pesos aleatorios**, por fases que se guardan en el disco y se pueden hacer en
días distintos.

## Qué se puede esperar (y qué no)

| | |
|---|---|
| Modelo final | Mixture of Experts de **214 M parámetros totales, 157 M activos** por token |
| Datos | ~3 300 M tokens de entrenamiento, mayoritariamente español |
| Tiempo total estimado | **32–67 h** de GPU (≈ 45 h en el caso central); mídelo con `titan bench` |
| Calidad esperable | Texto en español con cierta fluidez, conocimientos básicos. **No** razonará como un modelo grande ni será un agente fiable: es un prototipo propio que valida todo el sistema con una GPU real |

La capacidad de un modelo de lenguaje depende sobre todo de los parámetros y
de los tokens de entrenamiento. Por encima de ~350 M parámetros la VRAM de 8 GB
ya no alcanza para entrenar y el tiempo se mide en semanas; para eso está el
plan de nube (ver `docs/ESCALADO_Y_PRESUPUESTO.md`).

## Por qué "por partes" funciona así

Entrenar dos modelos de 100 M por separado desde cero y "pegarlos" **no**
funciona: cada modelo organiza su conocimiento de forma distinta y sus pesos
no encajan. Lo que sí funciona, y está implementado y probado en
`titan/train/grow.py`:

1. **Branch-Train-MiX (plan por defecto)**: se entrena una base común; se
   copian sus pesos en varias ramas que se especializan **una a la vez**
   (español, inglés, matemáticas); al final se unen en un solo modelo
   Mixture of Experts con un enrutador que aprende a qué experto enviar cada
   token. Como todas las ramas salen de la misma base, se pueden unir.
2. **Crecimiento progresivo (alternativa)**: se entrena la base de 100 M y se
   agranda a 24 capas (176 M). Las capas nuevas empiezan como copias con la
   salida a cero, de modo que el modelo agrandado calcula **exactamente lo
   mismo** que el original (verificado en `tests/test_grow.py`) y el
   entrenamiento posterior aprovecha la capacidad extra.

Las piezas ocupan poco disco (100 M parámetros ≈ 200 MB en bf16); lo que
limita es la potencia y la memoria de la GPU, no el almacenamiento.

## Fases del plan `configs/plans/pc_btx_100m.yaml`

| Fase | Qué hace | Tokens | Memoria GPU estimada | Tiempo estimado* |
|---|---|---|---|---|
| `base` | Base densa de 100,7 M, datos mezclados | 1 966 M | ~5–6 GB | 16–33 h |
| `expert_es` | Rama: 80 % español | 498 M | ~5–6 GB | 4–8 h |
| `expert_en` | Rama: 80 % inglés | 498 M | ~5–6 GB | 4–8 h |
| `expert_math` | Rama: 70 % matemáticas | 498 M | ~5–6 GB | 4–8 h |
| `merge` | Une las 3 ramas en un MoE (214 M / 157 M activos) | — | CPU, segundos | < 1 min |
| `moe_tune` | Ajuste conjunto del MoE y su enrutador | 301 M | ~6–7 GB | 5–10 h |

\* Rango para 25–12 TFLOPS efectivos. La cifra real de tu GPU depende del
límite de potencia de la laptop y de su refrigeración: **mídela** antes de
empezar (paso 2).

## Paso a paso

### 1. Instalar (una vez)

Requisitos: Windows 10/11, controlador NVIDIA reciente, Git y Python 3.12
(`winget install --id Git.Git -e` y `winget install --id Python.Python.3.12 -e`).

```powershell
# Descarga el instalador del repositorio y ejecútalo:
powershell -ExecutionPolicy Bypass -File install.ps1 -NoSleep
```

Crea `D:\TITAN_AI` con su estructura, clona el código en `D:\TITAN_AI\app`,
instala PyTorch con CUDA 12.8 (necesario para las RTX 50) y comprueba que la
GPU es visible. `-NoSleep` evita que la laptop se suspenda con el cargador
conectado durante entrenamientos largos.

### 2. Diagnosticar y medir tu GPU

```powershell
cd D:\TITAN_AI\app
.\.venv\Scripts\Activate.ps1
titan doctor                                   # GPU, CUDA, bf16, VRAM, RAM, disco, acceso a Hugging Face
titan bench --config configs/train/pc_base_100m.yaml --steps 10
```

Devuelve tokens/s, memoria máxima (`max_mem_gb`) y `projected_hours` para la
fase base. Si se queda sin memoria (*CUDA out of memory*):

```powershell
titan bench --config configs/train/pc_base_100m.yaml --set micro_batch_size=2 grad_accum_steps=64
```

y aplica el mismo cambio en `configs/train/pc_base_100m.yaml` (el producto
`micro_batch_size × grad_accum_steps` debe quedar igual: 128).

### 3. Preparar los datos

```powershell
titan data prepare --config configs/data/pc_es.yaml
```

Lee en streaming FineWeb-2 (español), Wikipedia en español, FineWeb-Edu,
OpenWebMath, aritmética sintética verificada y código de la biblioteca
estándar de Python. Limpia, elimina datos personales y duplicados, descarta
textos que coincidan con los benchmarks, entrena el tokenizador propio
(32 768 tokens) y escribe los shards. Necesita varias horas y ~25–35 GB de
disco. Al terminar revisa `data\pc_es\data_card.json`: lista las licencias y
los tokens reales por dominio.

### 4. Entrenar por ratos

```powershell
powershell -File scripts\windows\train_plan.ps1 -Hours 4    # 4 horas y se detiene
powershell -File scripts\windows\train_plan.ps1 -Status     # ver el avance
```

Cada vez que lo ejecutas continúa donde se quedó. También puedes detenerlo
con **Ctrl+C**: termina el paso en curso, guarda un checkpoint y sale. Si se
va la luz, se pierde como máximo el trabajo desde el último checkpoint (cada
500 pasos, ≈ 30–60 min).

### Alternativa: crecimiento progresivo

```powershell
powershell -File scripts\windows\train_plan.ps1 -Plan configs\plans\pc_grow_100m.yaml -Hours 4
```

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `CUDA disponible: False` | Actualiza el controlador NVIDIA; reinstala PyTorch con `-TorchIndex https://download.pytorch.org/whl/cu128` |
| *CUDA out of memory* | Baja `micro_batch_size` y sube `grad_accum_steps` en la misma proporción |
| La velocidad cae tras unos minutos | La laptop se calienta y baja la frecuencia: usa el modo de rendimiento del fabricante y una base refrigerante |
| `Checkpoint ... corrupto` | Borra esa carpeta `step_*`; el entrenamiento usará el checkpoint anterior |
| Pérdida `nan` repetida | El entrenador omite esos pasos; si persiste, reanuda con `--set skip_samples_on_resume=50000` |

## Seguir trabajando con Claude Code en el PC

La sesión de Claude Code en la nube no puede ejecutar nada en tu equipo. Para
que Claude trabaje directamente en el PC, abre `D:\TITAN_AI\app` con la app
Claude Desktop, o ejecuta `claude remote-control` en una terminal dentro de esa
carpeta (la sesión aparecerá en la app de Claude Code).
