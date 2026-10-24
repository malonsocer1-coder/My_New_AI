# Escalado, presupuesto y capacidad máxima viable

Todas las cifras salen de `titan estimate` (`titan/cloud/estimate.py`), con
estos supuestos: MFU del 40 % en modelos densos y del 30 % en MoE, 8×H100 SXM,
precios de octubre de 2026 entre ~1,5 USD/GPU-h (mercado) y ~4 USD/GPU-h (nube
segura bajo demanda), y la arquitectura de TITAN (GQA, SwiGLU, vocabulario de
65 536). Son **estimaciones**: el coste real se mide con `titan bench`, con
`metrics.jsonl` y con el libro de gastos.

## Tabla de escalas (8×H100)

| Modelo | Parámetros | Tokens | GPU-h | Días (8 GPU) | USD (bajo–alto) | Pesos bf16 | En la RTX 5070 (8 GB) | tok/s est. |
|---|---|---|---|---|---|---|---|---|
| titan-stage-a | 8,6 M | 20 M | — | — | 0 (CPU, hecho) | 17 MB | Q8_0 | — |
| titan-100m | 0,10 B | 2 B | 1 | < 0,1 | 2–4 | 0,2 GB | Q8_0 | ~400 |
| titan-350m | 0,34 B | 20 B | 45 | 0,2 | 68–181 | 0,7 GB | Q8_0 | ~280 |
| titan-1b | 1,11 B | 100 B | 580 | 3,0 | 869–2 313 | 2,2 GB | Q8_0 | ~140 |
| titan-3b | 3,02 B | 500 B | 7 846 | 41 | 11 769–31 306 | 6,0 GB | Q8_0 | ~63 |
| titan-moe-7b-a2b | 7,2 B (1,9 B activos) | 1 000 B | 13 762 | 72 | 20 643–54 912 | 14,3 GB | Q5_K_M (5,9 GB) | ~115 |
| titan-8b | 7,5 B | 1 500 B | 52 590 | 274 | 78 884–209 832 | 15,0 GB | Q5_K_M (6,5 GB) | ~40 |
| titan-30b | 28 B | 1 500 B | 195 294 | 1 017 | 0,29–0,78 M | 56 GB | no cabe (18 GB en Q4) | ~5 con RAM |
| titan-70b | 70 B | 2 000 B | 626 541 | 3 263 | 0,94–2,5 M | 139 GB | no cabe | ~2 |
| titan-100b | 98 B | 2 000 B | 872 802 | 4 546 | 1,3–3,5 M | **195 GB** | no cabe | ~1 |

* Los días se dividen casi linealmente al añadir nodos (con red InfiniBand):
  titan-3b en 32 GPUs ≈ 10 días; titan-8b en 128 GPUs ≈ 17 días.
* Con B200 el tiempo baja a ~45 % y el coste total es similar o algo menor.
* Memoria de entrenamiento: 1 B y 3 B caben en una H100 de 80 GB; 8 B y el
  MoE necesitan ≥ 2 GPUs con FSDP; 70–100 B necesitan ≥ 32 GPUs.
* Un checkpoint completo (pesos fp32 + AdamW) ocupa ~16 bytes/parámetro: 18 GB
  para 1 B, 120 GB para 8 B y 1,6 TB para 100 B. El almacenamiento remoto debe
  dimensionarse para `keep_last` copias.
* El modelo de 100 B ocupa ~195 GB en bf16: el proyecto y su estructura en
  `D:\TITAN_AI` admiten modelos de 200 GB o más, pero **el tamaño no es una
  meta en sí**. A igual presupuesto, un modelo más pequeño entrenado con más
  tokens suele rendir mejor y es el único que se puede usar en el PC.

## Disponibilidad de datos

A partir de ~3 B parámetros el límite deja de ser solo el dinero: hacen falta
cientos de miles de millones de tokens **de calidad**. El español abierto de
calidad (FineWeb-2 `spa_Latn` + Wikipedia) no alcanza por sí solo el billón de
tokens; el resto de la mezcla debe ser inglés educativo, código y matemáticas,
lo cual es habitual y también mejora el razonamiento en español.

## Recomendación

Sin conocer tu presupuesto, recomiendo una **escalera con validación en cada peldaño**:

| Nivel | Qué | Coste | Por qué |
|---|---|---|---|
| 0 | Etapa A (hecho) + plan por partes de 100 M en tu PC | 0 USD (≈ 45 h de laptop) | Valida datos en español reales, GPU y despliegue local |
| 1 | titan-350m con 20 B tokens en 8×H100 (~6 h) | ~70–180 USD | Prueba de extremo a extremo del circuito de nube (aprobación, guardián, sincronización) con riesgo bajo |
| 2 | titan-1b con 100 B tokens (~3 días en 8×H100) | ~0,9–2,3 mil USD | Primer modelo con utilidad real en español; corre a >100 tok/s en tu laptop |
| 3 | titan-moe-7b-a2b con 1 T tokens o titan-3b con 500 B | ~12–55 mil USD | **Máxima capacidad viable para tu RTX 5070**: el MoE cabe en Q5_K_M (5,9 GB) y, al activar solo 1,9 B parámetros por token, es rápido |
| 4 | titan-8b con 1,5 T tokens | ~79–210 mil USD | Calidad claramente superior; cabe en 8 GB solo en Q4/Q5 (~40 tok/s estimado) |
| 5 | 30–100 B | 0,3–3,5 millones USD | Solo con presupuesto institucional; no cabe en el PC: requiere la versión remota |

Para pasar de un nivel al siguiente: la pérdida de validación y los benchmarks
del nivel actual deben mejorar como predice el escalado (ver
`docs/EVALUACION.md`), y el gasto real debe coincidir con la estimación ±20 %.

## Requisito de 10 tokens/s en el PC

Según las estimaciones (ancho de banda de memoria de ~384 GB/s, eficiencia del
60 %), todos los modelos hasta titan-8b superan con holgura 10 tok/s si caben
en la VRAM. Por encima de ~12 GB de pesos hay que dejar capas en la RAM y la
velocidad cae por debajo de 10 tok/s. Hay que confirmarlo midiendo en la
laptop con `titan gguf bench --model ... --gpu-layers -1`.
