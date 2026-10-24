# Informe de rendimiento y costes

Medido en la sesión de desarrollo (4 vCPU Xeon @ 2,8 GHz, 15 GB de RAM, sin
GPU) el 24 de octubre de 2026. **Coste real hasta hoy: 0 USD** (no se ha
alquilado ninguna GPU).

## Entrenamiento (medido)

| Ejecución | Parámetros | Tokens | Duración | Rendimiento | Resultado |
|---|---|---|---|---|---|
| Etapa A (preentrenamiento) | 8,6 M | 20,5 M | 1,59 h | 4 777 tok/s (mediana) ≈ 0,25 TFLOPS efectivos | val 3,100 nats; es 1,278 bpb |
| SFT de demostración | 8,6 M | 4,9 M | 0,71 h* | ~4 200 tok/s | val SFT 0,035 |
| DPO de demostración | 8,6 M | 150 pasos × 8 pares | 0,22 h | — | 94,5 % de preferencia en pares reservados |
| Escalado xs / s / m | 2,9 / 8,6 / 18,9 M | 4,1 M c/u | 0,14 / 0,31 / 0,51 h | 11 k / 4,2 k / 2,2 k tok/s | ver abajo |
| DDP y FSDP2 (2 procesos) | 1,1 M | — | segundos | — | correcto con reanudación |

\* Incluye 20 min en pausa (SIGSTOP) mientras se ejecutaba otra tarea.

### Escalado con datos y tokens fijos (`reports/scaling_sweep*.json`)

| Modelo | Parámetros (sin embeddings) | LR | Pérdida de validación |
|---|---|---|---|
| xs | 2,9 M (0,8 M) | 1,5e-3 | 3,813 |
| s | 8,6 M (4,4 M) | 1,5e-3 | **3,668** |
| m | 18,9 M (12,6 M) | 1,5e-3 | 3,711 |
| m | 18,9 M (12,6 M) | 6e-4 | 3,704 |

**Lectura:** con solo 4,1 M tokens, el modelo de 18,9 M no supera al de 8,6 M
ni bajando el LR (3,711 → 3,704). La causa dominante es el presupuesto de
datos: un modelo más grande necesita proporcionalmente más tokens para
aprovechar su capacidad (con este presupuesto, incluso el de 8,6 M está
muy por encima del tamaño óptimo según Chinchilla, ~20 tokens/parámetro). Es
la regla que guía el plan de escalado: **subir parámetros solo junto con los
tokens, y comprobarlo con un barrido corto antes de gastar.**

## Inferencia (medido en CPU sin otras cargas, modelo de 8,6 M)

`reports/inference_speed_cpu.json` (velocidad) y `reports/quantization_stage_a.json` (calidad).

| Variante | Tamaño | Δ perplejidad | Prompt | Generación |
|---|---|---|---|---|
| TITAN PyTorch fp32 (caché KV) | 17,3 MB | — | — | 221 tok/s |
| GGUF F16 (llama.cpp) | 17,5 MB | −0,01 % | 6 368 tok/s | 931 tok/s |
| GGUF Q8_0 | 9,4 MB | +0,02 % | 5 967 tok/s | 1 027 tok/s |
| GGUF Q5_K_M | 6,7 MB | +0,36 % | 3 869 tok/s | 855 tok/s |
| GGUF Q4_K_M | 6,0 MB | +0,39 % | 7 097 tok/s | 965 tok/s |

Con un modelo tan pequeño la velocidad la limita la sobrecarga fija por token,
no la memoria: por eso la cuantización apenas la cambia aquí. En modelos de
1–8 B, limitados por el ancho de banda de memoria, la cuantización sí acelera
de forma casi proporcional al tamaño (ver las estimaciones para la RTX 5070 en
`docs/DESPLIEGUE_WINDOWS.md`). llama.cpp es ~4× más rápido que el motor
PyTorch en CPU: es el backend recomendado para el PC.

## Costes proyectados (no gastados)

Ver `docs/ESCALADO_Y_PRESUPUESTO.md`: de ~70–180 USD (350 M) a ~0,9–2,3 mil
USD (1 B), ~12–55 mil USD (3 B o MoE 7B-A2B) y ~79–210 mil USD (8 B), con
8×H100 a 1,5–4 USD/GPU-h y MFU del 40 %. En el PC, el plan por partes de 100 M
cuesta 0 USD y ~32–67 h de GPU.
