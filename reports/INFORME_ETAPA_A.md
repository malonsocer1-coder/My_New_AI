# Informe de la Etapa A — validación técnica

**Fecha:** 24 de octubre de 2026 · **Hardware:** sesión de Claude Code en la nube,
4 vCPU Intel Xeon @ 2,8 GHz, 15 GB de RAM, **sin GPU** · **PyTorch** 2.14.1 · **Coste:** 0 USD.

Todas las cifras de este informe se midieron en esta sesión; los archivos de
origen están en `reports/stage_a/` (métricas, muestras, ficha de datos,
manifiesto) y en `reports/*.json`.

## 1. Modelo

| | |
|---|---|
| Arquitectura | transformer decodificador denso (RMSNorm, RoPE, GQA 8/4 cabezas, SwiGLU), sin sesgos |
| Dimensiones | 6 capas, d = 256, FFN = 704, contexto 256 |
| Parámetros | **8 621 312** (4,43 M sin embeddings), inicializados **aleatoriamente** (semilla 1234) |
| Tokenizador | BPE propio a nivel de bytes, 8 192 tokens, entrenado sobre el corpus limpio |
| Pesos publicados | `releases/titan-stage-a/` (bf16, 17 MB; GGUF Q8_0, 9 MB; SHA-256 en MANIFEST.json) |

## 2. Datos

11,24 M tokens de entrenamiento y 0,38 M de validación en cinco dominios
(español, inglés, otras lenguas romances, código Python y aritmética sintética
verificada). Todas las fuentes tienen la licencia declarada (ver `docs/DATOS.md`
y `reports/stage_a/data_card.json`). Duplicados eliminados: 1 exacto y 54
casi duplicados. Contaminación con los benchmarks: **0** documentos (auditado
con `titan data audit`).

## 3. Entrenamiento

5 000 pasos × 4 096 tokens = **20,5 M tokens** (≈ 1,8 épocas), AdamW
(β = 0,9/0,95; wd 0,1), LR pico 1,5e-3 con coseno, `torch.compile`, fp32.
Duración: **1,59 h**, con una mediana de 4 777 tokens/s.

| Paso | Pérdida train | Val es (bpb) | Val en (bpb) | Val code (bpb) | Val math (bpb) | Val media (nats) |
|---|---|---|---|---|---|---|
| 25 | 8,72 | — | — | — | — | — |
| 500 | — | 1,577 | 2,140 | 2,341 | 0,586 | 4,083 |
| 1 000 | — | 1,447 | 1,979 | 2,135 | 0,569 | 3,741 |
| 2 000 | — | 1,353 | 1,827 | 1,979 | 0,486 | 3,439 |
| 3 000 | — | 1,287 | 1,728 | 1,872 | 0,428 | 3,233 |
| 4 000 | — | 1,272 | 1,666 | 1,820 | 0,404 | 3,136 |
| 5 000 | 2,78 (media últimos 200 pasos: 2,82) | 1,278 | 1,639 | 1,810 | 0,389 | **3,100** |

Lectura:
* **Aprende:** la pérdida de entrenamiento baja de 8,72 a ~2,8 (el azar con 8 192 tokens es ln 8192 ≈ 9,01).
* **Generaliza:** la pérdida de validación baja en todos los dominios hasta el paso ~4 000.
* **Se satura por falta de datos:** el español (0,87 M tokens, ≈ 8 pasadas) y el
  código dejan de mejorar desde el paso 4 000 y empiezan a sobreajustar ligeramente.
  Es la señal esperada de que el límite son los datos, no el código: la Etapa B
  usa ~2 500 M tokens.

### Generación (voraz, mismo prompt)

| Prompt | Paso 500 | Paso 5 000 |
|---|---|---|
| «El Parlamento Europeo» | «, la Comisión, la Comisión, la Comisión…» | «no es un asunto que no se ha hecho. En este momento no hay ninguna duda de que la Conferencia Intergubernamental se haya adoptado una decisión…» |
| «def suma(a, b):» | «Asuntos, y, en el Parlamento Europeo…» | «return\n    return b\n\ndef _get_imp(b, b, b):\n    """Return a big b…» |
| «Pregunta: ¿Cuánto es 25 + 17?» | «2» | «25» (incorrecto; imita el formato) |

El modelo pasa de repetir palabras a producir español gramatical (sin
significado fiable) y estructura de código Python. No calcula bien.

## 4. Robustez del sistema

| Prueba | Resultado |
|---|---|
| Interrupción real del entrenamiento (SIGTERM en el paso 2 011) | terminó el paso, guardó checkpoint íntegro y salió con `signal_15` |
| Verificación SHA-256 del checkpoint | 0 archivos corruptos |
| Reanudación | continuó en el paso 2 011, con el estado de AdamW y del cargador; pérdida continua (3,40 → 3,38) |
| Entrenamiento distribuido (2 procesos, torchrun) | DDP y FSDP2 con checkpoint fragmentado y reanudación: correctos (`tests/test_distributed.py`) |
| Exportación | los logits coinciden con `transformers` (Llama y Mixtral) y con llama.cpp (GGUF); llama.cpp tokeniza igual y genera idéntico |
| Suite de tests | ver `docs/ESTADO_DEL_PROYECTO.md` |

## 5. Evaluación del modelo base (`reports/eval_stage_a_base.json`)

| Suite | Resultado | Referencia |
|---|---|---|
| Opción múltiple (60 preguntas) | exactitud 20,0 %; normalizada 31,7 % | azar 25 % (IC 95 % ≈ ±11 pp): **no distinguible del azar** |
| Calibración (ECE) | 0,18 | |
| Consistencia al reordenar opciones | 100 % | trivial: sin conocimiento, el sesgo es estable |
| Aritmética (300) | 6,7 % de coincidencia exacta | |
| Código (20, tests reales) | 0 % pass@1 | las soluciones de referencia pasan el 100 % |
| Instrucciones, incertidumbre, herramientas | no aplicables | el contexto de 256 tokens no admite el prompt de chat; se evalúan tras el SFT (contexto 768) |
| Recuperación en contexto | 0 % | |

Conclusión honesta: a 8,6 M de parámetros y 20 M de tokens el modelo **no
tiene capacidades útiles**; la etapa cumple su objetivo, que era validar el
sistema completo de extremo a extremo con un modelo propio.

## 6. Cuantización (`reports/quantization_stage_a.json`)

Perplejidad sobre 12 288 tokens de validación (es, en, code). Velocidades en
la CPU de la sesión, compartida con otros procesos: son indicativas.

| Variante | Tamaño | Perplejidad | Δ vs. original |
|---|---|---|---|
| TITAN bf16 (original) | 17,3 MB | 33,88 | — |
| TITAN int8 (propia) | 8,7 MB | 33,91 | +0,09 % |
| TITAN int4 (propia, grupos de 64) | 4,6 MB | 34,96 | +3,17 % |
| GGUF F16 (llama.cpp) | 17,5 MB | 33,88 | −0,01 % |
| GGUF Q8_0 | 9,4 MB | 33,89 | +0,02 % |
| GGUF Q5_K_M | 6,7 MB | 34,01 | +0,36 % |
| GGUF Q4_K_M | 6,0 MB | 34,02 | +0,39 % |

## 7. SFT de demostración (`reports/eval_stage_a_sft.json`, `releases/titan-stage-a-sft/`)

800 pasos sobre el modelo base con 4 750 conversaciones sintéticas verificadas
(herramientas y aritmética), contexto ampliado a 768 tokens; pérdida de
validación SFT 0,035. En 64 tareas de herramientas con semillas reservadas:
herramienta correcta 100 %, llamadas válidas 100 %, respuesta correcta 82,8 %
(dentro de la distribución de las plantillas; ver `docs/EVALUACION.md`). La
interfaz web funciona de extremo a extremo con este modelo
(`reports/ui_agente.png`).

## 8. Experimentos adicionales sobre la Etapa A

| Experimento | Resultado | Detalle |
|---|---|---|
| Escalado con 4,1 M tokens fijos (2,9 / 8,6 / 18,9 M) | 3,813 / **3,668** / 3,711 (3,704 con LR menor) | más parámetros sin más datos no mejoran; `reports/scaling_sweep*.json` |
| Entrenamiento por partes a igual cómputo | denso continuado **1,338** < crecimiento 1,343 < base 1,346 < BTX 1,355 (bpb medio) | el crecimiento preserva la calidad exacta; BTX no gana a esta escala; `reports/btx_stage_a.json` |
| DPO sobre el SFT | 94,5 % de preferencia en pares reservados | mueve todas las preferencias, no crea conductas ausentes; `reports/dpo_stage_a*.json` |
| SFT v2 con el escenario «archivo inexistente» | respuestas honestas 0/3 → **3/3** | ver `docs/EVALUACION.md` |

## 9. Siguiente paso viable

Etapa B en el PC (RTX 5070): `configs/plans/pc_btx_100m.yaml`, ~2 500 M tokens en
español y otros dominios, 100 M parámetros y entrenamiento por partes. No
requiere gasto. Ver `docs/PC_ENTRENAMIENTO_POR_PARTES.md`.
