# titan-stage-a-sft-v2

Modelo de lenguaje **entrenado desde pesos aleatorios** por el proyecto TITAN AI.
No deriva de pesos preentrenados de terceros.

| | |
|---|---|
| Arquitectura | Transformer decodificador denso (RMSNorm, RoPE, GQA, SwiGLU) |
| Parámetros | 8.62 M totales, 8.62 M activos por token |
| Capas / dimensión / cabezas | 6 / 256 / 8 (kv 4) |
| Contexto | 768 tokens |
| Vocabulario | 8192 (BPE propio a nivel de bytes) |
| Checkpoint de origen | `runs/dpo_stage_a_v2` |
| Paso / tokens vistos | None / None |
| Exportado | 2026-10-24T11:13:07Z |

Consulte `MANIFEST.json` para verificar la integridad (SHA-256) y
`docs/EVALUACION.md` del repositorio para las métricas medidas.
