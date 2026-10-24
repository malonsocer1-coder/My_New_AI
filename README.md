# TITAN AI

Modelo de lenguaje **propio, entrenado desde pesos aleatorios**, con todo el
sistema alrededor: datos con licencia, tokenizador propio, entrenamiento
reanudable y distribuido, evaluación objetiva, ajuste de instrucciones y uso
de herramientas, agente con permisos y sandbox, exportación verificada (GGUF /
Hugging Face), cuantización, interfaz web local e integración con GPUs en la
nube con límites de gasto.

> **Estado (24 oct. 2026):** el sistema está completo y verificado de extremo a
> extremo con un modelo propio de 8,6 M de parámetros (Etapa A), entrenado en
> CPU porque esta sesión no tiene GPU. Ese modelo **no es útil todavía**: su
> función era validar el pipeline. El siguiente paso es la Etapa B en tu PC (100 M,
> gratis) y, con tu autorización y un presupuesto, las escalas de 350 M a 8 B
> en la nube. Ver [docs/ESTADO_DEL_PROYECTO.md](docs/ESTADO_DEL_PROYECTO.md).

## Qué está verificado

| Pieza | Evidencia |
|---|---|
| Aprendizaje desde cero | pérdida 8,72 → 2,78; validación por dominio en [reports/INFORME_ETAPA_A.md](reports/INFORME_ETAPA_A.md) |
| Reanudación tras interrupción | SIGTERM real en el paso 2 011 → checkpoint íntegro → continuación exacta |
| Entrenamiento distribuido | DDP y FSDP2 con torchrun (2 procesos), checkpoint fragmentado y reanudación |
| Entrenamiento por partes | crecimiento y unión de expertos que preservan exactamente la función; plan por fases de extremo a extremo |
| Exportación | logits idénticos a `transformers` (Llama/Mixtral) y a llama.cpp; tokenización y generación idénticas en llama.cpp |
| Cuantización | Q8_0 +0,02 %, Q4_K_M +0,39 % de perplejidad; int8 propia +0,09 % |
| SFT de uso de herramientas | 100 % de llamadas válidas y 82,8 % de respuestas correctas en 64 tareas reservadas (dentro de la distribución de las plantillas) |
| DPO | pares verificados contra inventar resultados y falsa seguridad; margen de preferencia creciente (test) |
| Español de México | clasificador de variante y dominio `es_mx` con peso propio en la mezcla |
| Agente | 15 herramientas (incluye crear Word/Excel/PowerPoint), sandbox sin red ni secretos, permisos con aprobación obligatoria para acciones sensibles, rutas confinadas, protección contra inyección, MCP |
| Servidor web | token anti-CSRF, validación de `Host`, streaming y diálogos de permisos |
| Nube | planes con presupuesto, `--approve` obligatorio, guardián de gasto probado con un proceso real |

## Inicio rápido

```bash
# Linux / sesión de desarrollo
uv venv .venv && uv pip install -e ".[dev,export]"
titan data prepare --config configs/data/stage_a.yaml
titan train --config configs/train/stage_a.yaml
titan eval --checkpoint runs/stage_a --suite all
titan export --checkpoint runs/stage_a --out models/base/titan-stage-a
titan serve --model models/base/titan-stage-a          # http://127.0.0.1:8765
titan doctor                                            # diagnóstico del equipo
python -m pytest                                        # suite de tests
```

Windows (D:\TITAN_AI): `scripts/windows/install.ps1` → ver
[docs/PC_ENTRENAMIENTO_POR_PARTES.md](docs/PC_ENTRENAMIENTO_POR_PARTES.md) y
[docs/DESPLIEGUE_WINDOWS.md](docs/DESPLIEGUE_WINDOWS.md).

Probar el modelo ya entrenado sin entrenar nada:
`titan generate --checkpoint releases/titan-stage-a --prompt "El Parlamento Europeo"`.

## Documentación

| Documento | Contenido |
|---|---|
| [ESTADO_DEL_PROYECTO](docs/ESTADO_DEL_PROYECTO.md) | resumen para retroalimentación, qué falta y decisiones pendientes |
| [ARQUITECTURA](docs/ARQUITECTURA.md) | modelo, tokenizador, formatos y flujo completo |
| [DATOS](docs/DATOS.md) | pipeline, fuentes, licencias, descontaminación |
| [ENTRENAMIENTO](docs/ENTRENAMIENTO.md) | etapas A–F, comandos, hiperparámetros |
| [PC_ENTRENAMIENTO_POR_PARTES](docs/PC_ENTRENAMIENTO_POR_PARTES.md) | plan para la RTX 5070: base → expertos → MoE |
| [ESCALADO_Y_PRESUPUESTO](docs/ESCALADO_Y_PRESUPUESTO.md) | tabla de 100 M a 100 B: tiempo, coste, memoria, cabe o no en 8 GB |
| [NUBE](docs/NUBE.md) | proveedores comparados, flujo de lanzamiento, guardián de gasto |
| [AGENTE_Y_HERRAMIENTAS](docs/AGENTE_Y_HERRAMIENTAS.md) | protocolo, herramientas, MCP, multimodal |
| [EVALUACION](docs/EVALUACION.md) | benchmarks, metodología y resultados reales |
| [DESPLIEGUE_WINDOWS](docs/DESPLIEGUE_WINDOWS.md) | exportar, cuantizar, ejecutar en D:\TITAN_AI |
| [SEGURIDAD](docs/SEGURIDAD.md) | permisos, sandbox, credenciales, gasto |
| [RECUPERACION](docs/RECUPERACION.md) | guía ante fallos |

## Estructura del código

```
titan/
  model/      transformer denso y MoE (Llama/Mixtral-compatible)
  tokenizer/  BPE propio
  data/       fuentes, limpieza, dedup, descontaminación, shards, cargador
  train/      entrenador, checkpoints, distribuido, crecimiento/BTX, planes, bench
  sft/        formato de chat, datos sintéticos verificados, empaquetado SFT
  eval/       benchmarks propios y verificadores
  infer/      generación, motor PyTorch, backend llama.cpp
  export/     TITAN/HF/GGUF, cuantización int8/int4 y GGUF
  agent/      protocolo, registro, permisos, sandbox, memoria, RAG, MCP, herramientas
  cloud/      estimador, proveedores, trabajos, guardián, sincronización
  ui/         servidor HTTP + interfaz web
configs/  scripts/  tests/  docs/  reports/  releases/
```

Licencia del código: Apache-2.0. Los datos conservan las licencias de sus fuentes (ver `docs/DATOS.md`).
