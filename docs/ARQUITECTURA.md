# Arquitectura de TITAN AI

## Visión general

```
                    ┌────────────────────────────── Datos ──────────────────────────────┐
 fuentes con        │ normalización → PII → calidad → idioma → dedup exacta + MinHash → │
 licencia  ───────▶ │ descontaminación → partición train/val → tokenizador BPE propio → │──▶ shards uint16/32
 (HF, ZIP, local,   │ shards + manifiesto SHA-256 + ficha de datos                     │    + manifest.json
  sintéticos)       └───────────────────────────────────────────────────────────────────┘
                                                   │
                    ┌──────────────────────── Entrenamiento ───────────────────────────┐
                    │ pesos aleatorios propios → AdamW + WSD/coseno → bf16 →            │
                    │ none / DDP / FSDP2 / HSDP (torchrun) → checkpoints atómicos       │──▶ runs/<ejecución>
                    │ → reanudación exacta → límites de gasto/tiempo → métricas JSONL   │
                    │ planes por fases: base → ramas → MoE (BTX) / crecimiento          │
                    └──────────────────────────────────────────────────────────────────┘
                                                   │
             ┌──────────── SFT ────────────┐       │       ┌──────────── Evaluación ──────────────┐
             │ chat con tokens especiales  │◀──────┼──────▶│ bpb, opción múltiple, aritmética,    │
             │ máscara de pérdida          │       │       │ código con tests, instrucciones,     │
             │ datos sintéticos verificados│       │       │ incertidumbre, herramientas, contexto│
             └─────────────────────────────┘       │       └──────────────────────────────────────┘
                                                   ▼
                    ┌──────────────────────── Exportación ─────────────────────────────┐
                    │ TITAN safetensors · Hugging Face (Llama/Mixtral) · GGUF (llama.cpp)│
                    │ cuantización int8/int4 propia y GGUF Q8_0…Q4_K_M · MANIFEST SHA-256│
                    └──────────────────────────────────────────────────────────────────┘
                                                   ▼
                    ┌───────────────────── Inferencia y agente ────────────────────────┐
                    │ motor PyTorch o llama.cpp (CUDA, descarga parcial a RAM)          │
                    │ servidor web local · bucle de agente · registro de herramientas   │
                    │ permisos · sandbox · memoria · RAG local · MCP · plugins          │
                    └──────────────────────────────────────────────────────────────────┘
```

## Modelo (`titan/model`)

Transformer decodificador autorregresivo implementado sobre primitivas de
PyTorch, con **inicialización aleatoria propia** (`Transformer.init_weights`):
ningún peso procede de modelos de terceros.

| Componente | Elección | Motivo |
|---|---|---|
| Normalización | RMSNorm previa (pre-norm) | estabilidad en profundidad, estándar actual |
| Posiciones | RoPE (`rotate_half`), θ configurable | extrapolación razonable; la tabla no tiene pesos y puede ampliarse |
| Atención | GQA (`n_kv_heads` < `n_heads`) con SDPA | caché KV 2–8× más pequeña: clave para 8 GB de VRAM |
| FFN | SwiGLU (`w2(silu(w1 x) · w3 x)`) | mejor calidad por FLOP que GELU |
| MoE | top-k (Mixtral), pérdida de balance + z-loss | más parámetros a igual coste por token |
| Sesgos | ninguno | menos parámetros, compatible con Llama |
| Embeddings | atados u opcionales | a escala pequeña ahorran ~25 % de parámetros |
| Inicialización | normal(0, 0,02); salidas residuales /√(2L) | control de la varianza del residuo |

La disposición de tensores coincide con Llama (denso) y Mixtral (MoE), por lo
que la exportación a Hugging Face y GGUF es directa y está verificada numéricamente.

## Tokenizador (`titan/tokenizer`)

BPE a nivel de bytes entrenado con los datos del proyecto (biblioteca
`tokenizers`, algoritmo estándar; el vocabulario y las fusiones son propios).

* Pre-tokenización estilo Qwen2: dígitos **individuales** (mejor aritmética) y
  reconocida por llama.cpp → GGUF sin parches.
* 32 tokens especiales reservados (chat, herramientas, razonamiento, multimodal y reserva).
* El texto no confiable se codifica con `allow_special=False`: una página web
  no puede fabricar `<|im_start|>system`.

## Datos (`titan/data`)

Ver `docs/DATOS.md`. Cargador determinista: el estado es un único entero y las
permutaciones por época se calculan con una red de Feistel (memoria O(1)).

## Entrenamiento (`titan/train`)

* `trainer.py`: un bucle para CPU, 1 GPU, DDP y FSDP2/HSDP; bf16 con autocast
  o política mixta de FSDP; acumulación de gradiente; recorte; omisión de
  pasos no finitos; señales SIGTERM/SIGINT → checkpoint y salida limpia.
* `checkpoint.py`: escritura en `step_X.tmp` + renombrado atómico, `LATEST`,
  SHA-256 por archivo, rotación; DCP fragmentado para FSDP con consolidación offline.
* `grow.py` y `plan.py`: entrenamiento por partes (crecimiento con función
  preservada, upcycling a MoE, Branch-Train-MiX) con fases reanudables.

## Agente (`titan/agent`)

Ver `docs/AGENTE_Y_HERRAMIENTAS.md`. El modelo central es TITAN; los módulos
de visión y voz son **componentes auxiliares independientes** y opcionales
(programas externos configurados por el usuario), documentados como tales.

## Formatos en disco

| Artefacto | Contenido |
|---|---|
| `data/<n>/manifest.json` | shards, SHA-256, dtype, tokens por dominio, hash del tokenizador |
| `data/<n>/data_card.json` | fuentes, licencias, estadísticas de filtrado |
| `runs/<n>/run_info.json` | configs completas, commit de git, versión de PyTorch, hardware |
| `runs/<n>/metrics.jsonl` | pérdida, LR, normas, tokens/s, MFU, coste, ETA, evaluaciones |
| `runs/<n>/checkpoints/step_N/` | pesos, optimizador, estado, RNG, MANIFEST |
| `models/base/<n>/` | exportación con MODEL_CARD, `hf/`, `gguf/` |
