# Evaluación objetiva

`titan eval --checkpoint <ejecución|modelo> --suite all --out informe.json`

Principios:
* Ninguna capacidad se declara sin una medición reproducible: cada suite tiene
  datos fijos versionados en `titan/eval/data/`, verificación por programa y su
  línea base (azar o referencia).
* Separación estricta: los textos de evaluación se indexan antes de limpiar
  los datos y los documentos coincidentes se descartan (`titan data audit` lo
  comprueba a posteriori); las semillas de las tareas sintéticas de evaluación
  están reservadas y nunca se usan para entrenar.
* No se usa ningún modelo externo como juez.
* El número de parámetros, el tamaño de los archivos o la apariencia de las
  respuestas no cuentan como prueba de capacidad.

## Suites

| Suite | Mide | Datos | Métrica | Línea base |
|---|---|---|---|---|
| `bpb` | comprensión/modelado de texto por dominio | shards de validación | pérdida, perplejidad, **bits por byte** (comparables entre tokenizadores) | — |
| `mc` | conocimiento general | 60 preguntas propias en español (geografía, ciencia, matemáticas, lengua, historia, tecnología) | exactitud (bruta y normalizada por longitud), **ECE**, consistencia al reordenar opciones, robustez sin acentos/minúsculas | 25 % |
| `arith` | matemáticas | 300 problemas generados con semilla reservada | coincidencia exacta | — |
| `code` | programación | 20 funciones con tests ejecutados en el sandbox | pass@1 (voraz) | soluciones de referencia: 100 % |
| `instructions` | seguimiento de instrucciones | 20 consignas con restricciones verificables (longitud, formato, JSON, idioma…) | exactitud por consigna y por restricción | — |
| `uncertainty` | reconocimiento de incertidumbre, premisas falsas, ambigüedad | 16 consignas | marcadores léxicos (indicativo) | — |
| `tool_use` | uso de herramientas | 64 tareas con herramientas reales y semillas reservadas | selección correcta, llamadas válidas, respuesta correcta | — |
| `retrieval` | recuperación en contexto largo | dato insertado al 25/50/90 % del contexto | exactitud | — |

Si el prompt de una suite de chat no cabe en el contexto del modelo, se informa
`context_overflow` en lugar de contarlo como fallo.

Benchmarks estándar (MMLU, Global-MMLU en español, ARC, HellaSwag, GSM8K,
HumanEval, MBPP, Belebele `spa_Latn`, XNLI-es) se añadirán como suites cuando
haya acceso a Hugging Face y un modelo que supere el azar; antes de usarlos
deben entrar en `eval_files` para la descontaminación. Solo se compararán con
modelos de referencia en condiciones equivalentes (mismo número de ejemplos,
mismo formato de prompt, misma métrica).

## Resultados medidos

### Etapa A, modelo base (8,6 M parámetros, 20,5 M tokens)

| Suite | Resultado |
|---|---|
| bpb (validación) | es 1,278 · en 1,639 · code 1,810 · math 0,389 · multi 1,613 |
| mc | 20,0 % (normalizada 31,7 %); azar 25 %, IC 95 % ≈ ±11 pp: **no supera al azar** |
| arith | 6,7 % |
| code | 0 % |
| retrieval | 0 % |
| instructions / uncertainty / tool_use | no aplicables (contexto de 256 tokens) |

### Etapa A + SFT de demostración

Ver la sección siguiente (se completa con `reports/eval_stage_a_sft.json`).
