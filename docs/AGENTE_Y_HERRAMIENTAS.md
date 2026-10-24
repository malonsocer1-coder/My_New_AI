# Agente y herramientas

TITAN se convierte en agente mediante un **bucle explícito** (`titan/agent/loop.py`):

1. El modelo genera un turno; si contiene `<|tool_call|>{"name", "arguments"}<|/tool_call|>`,
   la generación se pausa.
2. El registro (`registry.py`) valida los argumentos contra el JSON Schema de la
   herramienta, comprueba los permisos, la ejecuta con límite de tiempo,
   trunca la salida y la audita.
3. El resultado vuelve al modelo como mensaje `tool` (`<|tool_result|>...`).
4. Llamadas mal formadas o fallidas se devuelven como error para que el modelo
   se corrija; el bucle termina con una respuesta sin llamadas, o al agotar
   `max_steps` o `max_seconds`.

## Herramientas integradas

| Herramienta | Capacidad | Descripción |
|---|---|---|
| `calculator` | — | aritmética exacta con un evaluador de AST (sin `eval`) |
| `list_dir`, `read_file` | file_read | archivos del espacio de trabajo; texto, CSV, JSON, PDF, Word, Excel, PowerPoint |
| `write_file` | file_write (+ file_delete para sobrescribir) | crear archivos de texto |
| `write_document` | file_write (+ file_delete para sobrescribir) | crear Word (.docx), Excel (.xlsx), PowerPoint (.pptx), Markdown, CSV o JSON |
| `delete_file` | file_delete (siempre pregunta) | borrar |
| `table_summary` | file_read | análisis de columnas de un CSV |
| `python_exec` | code_exec | Python en el sandbox (análisis de datos, depuración, cálculos) |
| `web_fetch` | network | texto de una página pública (anti-SSRF), marcado como no confiable |
| `web_search` | network (+ credentials con Brave) | SearXNG propio o Brave Search API |
| `kb_search` | file_read | búsqueda BM25 en la base de conocimiento local (`workspace/knowledge`) |
| `memory_save`, `memory_search` | memory_write / memory_read | memoria persistente con privacidad |
| `describe_image`, `transcribe_audio` | external_tools | módulos auxiliares configurados por el usuario |

Ampliaciones:
* **MCP**: `mcp_servers` en `config/agent.yaml` conecta servidores Model Context
  Protocol por stdio; sus herramientas aparecen como `servidor.herramienta`
  (capacidad `external_tools`, salida no confiable).
* **Plugins**: archivos `.py` con `register(registry)`, cargados solo si se listan.

## Composición multimodal

El núcleo es el modelo TITAN entrenado desde cero. Visión y voz se resuelven
con **componentes auxiliares independientes** (por ejemplo whisper.cpp para
transcribir o un modelo de visión GGUF con llama.cpp) que TITAN invoca como
herramientas. No sustituyen al modelo central y se documentan como de terceros.
Los tokens `<|image|>` y `<|audio|>` quedan reservados para una futura
integración nativa.

## Entrenamiento del comportamiento de agente

Un modelo entrenado desde cero **no** aprende a usar herramientas por sí solo.
`titan/sft/synthetic_tools.py` genera conversaciones con ejecución real:

| Tipo | Qué enseña |
|---|---|
| calculator | delegar cálculos exactos |
| read_file / write | operar con archivos y reportar resultados |
| error_recovery | ante un error, listar el directorio, encontrar el archivo correcto y reintentar |
| python | escribir y ejecutar código, usar su salida |
| kb | buscar en documentos locales y citar la fuente |
| permission_denied | aceptar una negativa y explicarla sin insistir |
| no_tool | no usar herramientas cuando no hacen falta; reconocer lo que no se puede saber |

La evaluación `tool_use` usa semillas reservadas (≥ 9·10⁹) que nunca se usan
para entrenar, y mide selección de herramienta, validez de las llamadas y
exactitud de la respuesta. Es una evaluación **dentro de la distribución** de
las plantillas: mide si el modelo aprendió el protocolo, no su capacidad
general. Para capacidad general hacen falta más datos y escala (ver
`docs/ESCALADO_Y_PRESUPUESTO.md`).

## Tareas largas

* Límite de pasos y de tiempo por tarea; registro de cada paso (`trace`).
* Los resultados se truncan (8 000 caracteres) para no desbordar el contexto.
* Para tareas de varias sesiones, la memoria persistente y los archivos del
  espacio de trabajo conservan el estado entre peticiones.
