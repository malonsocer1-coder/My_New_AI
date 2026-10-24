# Entrenar TITAN por partes en tu PC (RTX 5070 Laptop, 8 GB)

Esta guía explica cómo entrenar en tu propio equipo un modelo TITAN **desde
pesos aleatorios**, por fases que se guardan en el disco y se pueden hacer en
días distintos.

## Qué se puede esperar (y qué no)

| | |
|---|---|
| Modelo final | ~100 M parámetros densos (plan recomendado) o 176 M (crecimiento) o MoE 214 M / 157 M activos (BTX) |
| Datos | ~3 300–3 800 M tokens de entrenamiento, mayoritariamente español (con dominio propio para el español de México) |
| Tiempo total estimado | **32–67 h** de GPU (≈ 45 h en el caso central); mídelo con `titan bench` |
| Calidad esperable | Texto en español con cierta fluidez, conocimientos básicos. **No** razonará como un modelo grande ni será un agente fiable: es un prototipo propio que valida todo el sistema con una GPU real |

La capacidad de un modelo de lenguaje depende sobre todo de los parámetros y
de los tokens de entrenamiento. Por encima de ~350 M parámetros la VRAM de 8 GB
ya no alcanza para entrenar y el tiempo se mide en semanas; para eso está el
plan de nube (ver `docs/ESCALADO_Y_PRESUPUESTO.md`).

## Tres formas de entrenar por partes (todas guardan cada fase en disco)

Entrenar dos modelos de 100 M por separado desde cero y "pegarlos" **no**
funciona: cada modelo organiza su conocimiento de forma distinta y sus pesos
no encajan. Estas tres sí funcionan (implementadas en `titan/train/grow.py` y
`titan/train/plan.py`):

| Plan | Qué hace | Archivo |
|---|---|---|
| **Denso por sesiones (recomendado)** | base de 100 M y después se sigue entrenando el mismo modelo, por ratos | `configs/plans/pc_dense_100m.yaml` |
| Crecimiento | base de 100 M → se agranda a 24 capas (176 M) sin perder nada → se sigue entrenando | `configs/plans/pc_grow_100m.yaml` |
| Expertos (Branch-Train-MiX) | base → ramas especializadas una a una → unión en MoE → ajuste conjunto | `configs/plans/pc_btx_100m.yaml` |

### Evidencia real (prueba a pequeña escala en esta sesión)

Sobre el modelo de la Etapa A (8,6 M), con el **mismo cómputo extra** para las
tres opciones y midiendo bits por byte en validación (menos es mejor;
`reports/btx_stage_a.json`):

| Opción | Media (5 dominios) | Español |
|---|---|---|
| Base de partida | 1,3458 | 1,2779 |
| **Seguir entrenando el denso** | **1,3377** | 1,2849 |
| Agrandar 6 → 12 capas y seguir | 1,3434 | **1,2770** |
| Unir 3 expertos (BTX), mejor variante | 1,3553 | 1,2824 |

* Los expertos **sí** se especializan (inglés 1,639 → 1,606; código 1,810 → 1,767),
  pero al unirlos se pierde esa ventaja y el ajuste corto no la recupera.
* El modelo agrandado sin entrenar da **exactamente** la misma calidad que el
  original (1,3458): crecer no pierde nada, como predicen los tests.
* Limitaciones de la prueba: modelo diminuto, datos escasos (el español ya se
  había repetido ~8 veces) y ramas cortas. En la literatura, BTX gana a escalas
  mucho mayores con ramas largas; a 100 M con datos nuevos podría cambiar. Por
  eso el plan BTX se mantiene como **experimento**, y `scripts/btx_report.py`
  permite repetir esta comparación con tus datos reales.

**Recomendación:** empezar con `pc_dense_100m.yaml` (la fase `base` es común a
los tres planes, así que no se pierde nada si después quieres probar otro a
partir de ella).

Las piezas ocupan poco disco (100 M parámetros ≈ 200 MB en bf16); lo que
limita es la potencia y la memoria de la GPU, no el almacenamiento.

## Fases del plan de expertos `configs/plans/pc_btx_100m.yaml` (experimental)

| Fase | Qué hace | Tokens | Memoria GPU estimada | Tiempo estimado* |
|---|---|---|---|---|
| `base` | Base densa de 100,7 M, datos mezclados | 1 966 M | ~4–5 GB (entropía cruzada por fragmentos) | 16–33 h |
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
powershell -File scripts\windows\train_plan.ps1 -Hours 4    # 4 horas y se detiene (plan denso recomendado)
powershell -File scripts\windows\train_plan.ps1 -Status     # ver el avance
```

Cada vez que lo ejecutas continúa donde se quedó. También puedes detenerlo
con **Ctrl+C**: termina el paso en curso, guarda un checkpoint y sale. Si se
va la luz, se pierde como máximo el trabajo desde el último checkpoint (cada
500 pasos, ≈ 30–60 min).

### Alternativas: crecimiento o expertos

```powershell
powershell -File scripts\windows\train_plan.ps1 -Plan configs\plans\pc_grow_100m.yaml -Hours 4
powershell -File scripts\windows\train_plan.ps1 -Plan configs\plans\pc_btx_100m.yaml -Hours 4
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
