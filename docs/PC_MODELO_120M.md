# TITAN 120M en tu PC (RTX 5070 Laptop, 8 GB): el plan que exprime la GPU

Plan completo para entrenar **en tu equipo**, desde pesos aleatorios, un modelo
propio de **118,0 M de parámetros** y terminar con un modelo de chat con
herramientas exportado a GGUF. Se ejecuta por sesiones (p. ej. 8 h cada noche)
y cada sesión sigue donde se quedó la anterior.

| | |
|---|---|
| Modelo | **118,0 M** parámetros (99,1 M sin embeddings): 28 capas, d = 576, 9 cabezas de atención (3 KV), FFN 1 536, vocabulario 32 768, contexto 2 048 |
| Datos | ~2 900 M tokens de entrenamiento (≈ 23 por parámetro, el óptimo de Chinchilla es ~20), mayoritariamente español y con dominio propio para el español de México |
| Tiempo estimado | **32–67 h de GPU** en total (rango para 25–12 TFLOPS efectivos); mídelo en tu equipo antes de empezar |
| Memoria | ~6 GB de VRAM con micro-lote 4 (estimación); el ajuste automático lo **mide** y elige |
| Disco | ~7 GB de datos + ~1,4 GB por checkpoint (se conservan pocos) + ~1 GB de exportaciones |
| Coste | 0 USD (solo electricidad) |
| Plan | `configs/plans/pc_120m.yaml` (plan por defecto de `scripts\windows\train_plan.ps1`) |

## Por qué 118 M y por qué «profundo y estrecho»

* Pediste 110–120 M. Con vocabulario de 32 768 y 28 capas de 576 quedan 118,0 M.
* A menos de 1 000 M de parámetros, **más capas finas** rinden mejor que pocas
  capas anchas con los mismos parámetros (MobileLLM, Liu et al. 2024). Los
  mejores modelos abiertos de este tamaño usan exactamente esta anchura
  (MobileLLM-125M y SmolLM2-135M: d = 576, 9 cabezas, 3 KV, FFN 1 536, ~30 capas).
* Contrapartida: las matrices pequeñas aprovechan peor la GPU, así que una
  versión ancha (`configs/model/titan_120m_ancho.yaml`, 768 × 15 capas) puede
  dar algo más de tokens/s. Puedes comparar ambas en tu GPU (paso 2); si la
  ancha resultara mucho más rápida, cambia `model_config` en
  `configs/train/pc_120m_*.yaml`.

Por encima de ~150–200 M en esta GPU el tiempo crece a semanas y la memoria se
vuelve justa; la escala siguiente (350 M a 8 B) está en el plan de nube
(`docs/ESCALADO_Y_PRESUPUESTO.md`).

## Qué hace para exprimir la GPU

| Técnica | Efecto |
|---|---|
| bf16 con autocast + TF32 | usa los núcleos tensoriales; pesos y AdamW en fp32 para estabilidad |
| **Micro-lote automático medido** (`auto_batch`) | prueba cada micro-lote posible con pasos reales, mide tokens/s y la VRAM que queda libre, y elige el más rápido que cabe; el lote global (262 144 tokens) no cambia |
| Margen de VRAM (`auto_batch_margin_gb: 0.7`) | evita que Windows desborde la VRAM a la RAM del sistema, que no da error pero vuelve el entrenamiento varias veces más lento |
| `torch.compile` automático (`compile: auto`) | fusiona operaciones si Triton está instalado (`install.ps1 -Triton`); si falla, sigue sin compilar |
| AdamW «fused» y atención SDPA (FlashAttention) | menos lanzamientos de kernels y atención sin matriz T × T en memoria |
| Entropía cruzada por fragmentos | no materializa los logits de 32 768 por token: ahorra ~1 GB |
| Contexto 1 024 en la base y 2 048 solo al final | la base cuesta un 18 % menos por token; la fase `long_ctx` adapta el modelo a 2 048 |
| Programa WSD (meseta + caída) | se puede alargar o acortar el entrenamiento sin reiniciar (ver abajo) |
| Piloto de tasa de aprendizaje | 3 ejecuciones cortas eligen la mejor tasa con tus datos y tu lote antes de gastar ~40 h |
| Checkpoints cada ~45 min, atómicos y con SHA-256 | un corte de luz o Ctrl+C pierde como máximo eso |

Ajustes de Windows que también cuentan (no los puede hacer el programa):

1. **Panel de control de NVIDIA → Administrar la configuración 3D → Configuración
   de programa → `python.exe` de `D:\TITAN_AI\app\.venv\Scripts` → «Política de
   respaldo de memoria del sistema CUDA» → «Preferir sin respaldo de memoria del
   sistema».** Así, si algo no cabe, falla con un error claro en vez de ir lento.
2. Portátil **conectado a la corriente**, modo de energía «Máximo rendimiento»
   (y el modo de rendimiento del fabricante, si lo tiene), buena ventilación o
   base refrigerante: una laptop caliente baja la frecuencia de la GPU.
3. Cierra navegadores, juegos y programas que usen la GPU mientras entrena.
4. `install.ps1 -NoSleep` evita la suspensión con el cargador conectado.

## Fases del plan

| Fase | Qué hace | Tokens | Tiempo* |
|---|---|---|---|
| `lr_pilot` | 3 ejecuciones de 250 pasos (LR 8e-4, 1,6e-3 y 3,2e-3) con calentamiento, meseta y caída; gana la menor pérdida de validación final (empates < 0,5 % → la más baja) | 197 M | 2,0–4,1 h |
| `base` | preentrenamiento a contexto 1 024 con la LR elegida | 2 753 M | 27,7–57,7 h |
| `long_ctx` | continúa a contexto 2 048 | 131 M | 1,6–3,3 h |
| `sft_data` | conversaciones sintéticas verificadas (8 000 de herramientas ejecutadas de verdad + 3 000 de aritmética) + conversaciones abiertas en español con licencia Apache-2.0 (OASST2 y Aya), limpias, sin datos personales y descontaminadas | — | minutos (descarga) |
| `sft` | 3 épocas, solo cuentan los tokens del asistente | ~20–40 M | 0,4–1 h |
| `dpo` | 3 000 pares verificados: fidelidad a las herramientas, honestidad ante errores, archivos inexistentes, permisos denegados, preguntas sin respuesta | — | minutos |
| `export` · `gguf` · `eval` | exporta (TITAN, Hugging Face, GGUF), cuantiza a Q8_0 y Q4_K_M y ejecuta los benchmarks propios | — | < 30 min |

\* Rango para 25–12 TFLOPS efectivos (0,906 GFLOP por token a contexto 1 024).
El pico bf16 de una RTX 5070 Laptop ronda 50 TFLOPS (derivado de las
especificaciones publicadas, depende de la potencia de cada portátil), así que
el rango supone un aprovechamiento del 24–50 %. **Mídelo** en el paso 2.

## Paso a paso

### Atajo: todo en uno

Con Git y Python 3.12 instalados, dos comandos hacen los pasos 1 a 4 (cada paso
se omite si ya está hecho, así que se puede repetir sin miedo):

```powershell
git clone -b claude/adoring-hypatia-oc7bdz https://github.com/malonsocer1-coder/My_New_AI.git D:\TITAN_AI\app
powershell -ExecutionPolicy Bypass -File D:\TITAN_AI\app\scripts\windows\entrenar_todo.ps1 -Horas 8 -Programar -Hora 23:00
```

`-Programar` crea la tarea de Windows `TITAN_Entrenamiento`, que continúa el
plan cada día a las 23:00 durante 8 h (despierta el equipo si está suspendido)
sin que tengas que estar delante; se quita con
`Unregister-ScheduledTask -TaskName TITAN_Entrenamiento -Confirm:$false`.
El portátil debe quedarse encendido o suspendido (no apagado) y con el cargador.

### 1. Instalar (una vez)

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -NoSleep -Triton
```

Crea `D:\TITAN_AI`, clona el código, instala PyTorch con CUDA 12.8 (necesario
para las RTX 50), `triton-windows` para `torch.compile` (lo prueba y lo
desinstala si no funciona) y ejecuta `titan doctor`.

### 2. Medir tu GPU (5–10 minutos)

```powershell
cd D:\TITAN_AI\app
.\.venv\Scripts\Activate.ps1
titan bench --config configs/train/pc_120m_base.yaml --find-batch --out reports\bench_pc_120m.json
# opcional: la versión ancha, para comparar tokens/s
titan bench --config configs/train/pc_120m_base.yaml --find-batch --set model_config=configs/model/titan_120m_ancho.yaml
# velocidad con torch.compile y el micro-lote elegido (sustituya 8 por el que salió arriba)
titan bench --config configs/train/pc_120m_base.yaml --steps 5 --set auto_batch=false micro_batch_size=8
```

`--find-batch` imprime, para cada micro-lote, tokens/s y GB libres, y
`projected_hours_eager` para la fase base. Si quieres, envíame el JSON y ajusto
el plan a tu cifra real.

### 3. Preparar los datos (varias horas, sobre todo de descarga)

```powershell
titan data prepare --config configs/data/pc_es.yaml
```

Al terminar revisa `data\pc_es\data_card.json` (licencias y tokens reales por
dominio). El objetivo es ~3 300 M tokens únicos; si salen bastantes menos, sube
`max_docs` de FineWeb-2.

### 4. Entrenar por sesiones

```powershell
powershell -File scripts\windows\train_plan.ps1 -Hours 8     # 8 h y se detiene guardando checkpoint
powershell -File scripts\windows\train_plan.ps1 -Status      # en qué fase va
```

El límite de horas vale para toda la sesión: si una fase termina antes, la
siguiente usa el tiempo que queda. Ctrl+C termina el paso en curso, guarda y sale.

### 5. Usar el resultado

| Archivo | Qué es |
|---|---|
| `runs\pc_120m\export\` | modelo final (TITAN bf16 + Hugging Face + GGUF) con `MANIFEST.json` y SHA-256 |
| `runs\pc_120m\gguf\titan-120m-Q8_0.gguf` | el recomendado para el PC (~125 MB) |
| `runs\pc_120m\eval\eval.json` | resultados de los benchmarks propios |
| `runs\pc_120m\plan_state.json` | estado y resultados de cada fase (incluida la LR elegida) |

```powershell
powershell -File scripts\windows\run_server.ps1   # abre la interfaz web en http://127.0.0.1:8765
titan gguf bench --model runs\pc_120m\gguf\titan-120m-Q8_0.gguf   # mide tokens/s en tu equipo
```

`run_server.ps1` busca solo el resultado del plan (primero el GGUF Q8_0, si no
la exportación TITAN). La interfaz tiene modo chat y modo agente; en modo
agente el modelo puede usar las herramientas con las que se entrenó en el SFT:
calculadora, leer archivos, listar carpetas, escribir archivos, ejecutar Python
en el sandbox (sin red ni secretos) y buscar en tus documentos locales. Las
acciones sensibles (borrar, escribir fuera del espacio de trabajo, credenciales,
gastar, publicar) siempre piden tu aprobación en un diálogo. El agente tiene más
herramientas (Word/Excel/PowerPoint, web, MCP…), pero conviene ofrecerle solo
aquellas con las que se entrenó: con más, el prompt no cabe o el modelo se
confunde (ver `docs/EVALUACION.md`).

Velocidad de uso (estimación, sin medir todavía): en Q8_0 el modelo ocupa
~0,8 GB con su caché, cabe entero en la GPU y debería generar **varios cientos
de tokens/s** (~400 según `titan estimate`; en un modelo tan pequeño manda la
sobrecarga fija por token, no la memoria). Aun solo con la CPU, llama.cpp
debería superar con holgura los 10 tokens/s. Mídelo con `titan gguf bench`.

## Alargar o acortar el entrenamiento

La fase `base` usa WSD: LR constante hasta el 80 % de `max_steps` y caída al
final. Mientras el paso actual esté por debajo del 80 % del nuevo valor puedes
cambiar `max_steps` sin perder nada, añadiendo `max_steps` al `set` de la fase
`base` en `configs/plans/pc_120m.yaml`:

* **Acortar:** `max_steps: 6000` (1 573 M tokens, ~13 por parámetro) ahorra ~40 % del tiempo.
* **Alargar:** `max_steps: 14000` (3 670 M tokens) si tienes tiempo y datos;
  más tokens siguen mejorando un modelo pequeño.

Para omitir el piloto (ahorra 2–4 h), pon `enabled: false` en `lr_pilot` y
fija `lr: 1.6e-3` y `min_lr: 1.6e-4` en el `set` de `base`.

## Qué está verificado y qué no

Verificado en la sesión de desarrollo (CPU, sin GPU):

| Prueba | Resultado |
|---|---|
| Tamaño | 117 997 632 parámetros (`tests/test_pc120m.py` comprueba que está entre 110 y 120 M) |
| Aprende con datos reales | 40 pasos sobre los datos de la Etapa A (vocabulario 8 192, 103,8 M): pérdida 9,0 → 7,04 |
| Exportación del modelo de 118 M | logits idénticos a `transformers` (diferencia 0,0) y a llama.cpp (diferencia máxima 4·10⁻⁵), misma predicción en cada posición |
| Plan completo | piloto → base → contexto largo → datos SFT (con una importación rota que se omite) → SFT por épocas → DPO → exportación → GGUF Q8_0 → evaluación, con un modelo diminuto (`tests/test_pc120m.py`) |
| Micro-lote automático | mide todos los candidatos y elige el más rápido (probado en CPU) |
| Velocidad en CPU | 536 tokens/s → la base tardaría ~1 430 h en CPU: **hace falta la GPU** |

Sin verificar (no hay acceso a tu equipo ni a Hugging Face desde aquí):

* Velocidad y memoria reales en la RTX 5070 (paso 2).
* Los scripts de Windows (pueden necesitar ajustes menores).
* Los importadores de OASST2 y Aya: escritos según las columnas documentadas
  de esos conjuntos y probados con datos locales del mismo formato; son
  opcionales, así que si fallan el SFT continúa con los datos sintéticos y el
  motivo queda en `runs\pc_120m\sft_data\sft_data_card.json`.

Una observación de la prueba corta: con un lote 64 veces más pequeño que el del
plan y LR 1,6e-3, la pérdida tuvo un repunte (6,89 → 7,59) antes de seguir
bajando. Con el lote real es menos probable, pero es justo el motivo de que la
LR se elija con el piloto y no a ojo.

## Qué esperar del modelo

Un modelo propio de 118 M bien entrenado escribe español con fluidez razonable,
conoce hechos frecuentes y aprende el formato de chat y de llamadas a
herramientas. **No** razona de forma fiable, comete errores factuales y fuera de
las tareas entrenadas se equivoca de herramienta. Es el prototipo útil más
grande que tu laptop entrena en días; para un asistente sólido el siguiente
escalón es 1 B en la nube.
