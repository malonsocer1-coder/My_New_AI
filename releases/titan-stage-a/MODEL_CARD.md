# titan-stage-a

Modelo de lenguaje **entrenado desde pesos aleatorios** por el proyecto TITAN AI.
No deriva de pesos preentrenados de terceros.

| | |
|---|---|
| Arquitectura | Transformer decodificador denso (RMSNorm, RoPE, GQA, SwiGLU) |
| Parámetros | 8.62 M totales, 8.62 M activos por token |
| Capas / dimensión / cabezas | 6 / 256 / 8 (kv 4) |
| Contexto | 256 tokens |
| Vocabulario | 8192 (BPE propio a nivel de bytes) |
| Checkpoint de origen | `runs/stage_a/checkpoints/step_00005000` |
| Paso / tokens vistos | 5000 / 20480000 |
| Exportado | 2026-10-24T06:43:17Z |

Consulte `MANIFEST.json` para verificar la integridad (SHA-256) y
`docs/EVALUACION.md` del repositorio para las métricas medidas.
