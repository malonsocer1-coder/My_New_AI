# Seguridad y autonomía

## Permisos explícitos

Cada herramienta declara sus capacidades; la política del usuario
(`D:\TITAN_AI\config\permissions.yaml`, plantilla en
`configs/agent/permissions.example.yaml`) decide `allow`, `ask` o `deny`.

| Capacidad | Por defecto | Siempre pregunta |
|---|---|---|
| file_read, memory_read | allow | |
| file_write, code_exec, network, install_deps, external_tools, memory_write | ask | |
| file_delete, credentials, cloud_resources, spend_money, publish | ask | **sí**, aunque la política diga allow |

* Sin aprobador interactivo (scripts, tareas desatendidas), `ask` = `deny`.
* Sobrescribir un archivo existente exige además `file_delete`.
* Toda decisión queda en `logs/permissions_audit.jsonl`; toda llamada, en `logs/tools_audit.jsonl`.

## Inyección de instrucciones

* El modelo **solo puede solicitar** acciones; ninguna herramienta modifica la
  política ni puede leer `permissions.yaml`, `.env`, `credentials`, `secrets`,
  `memory.db` ni `.git` (rutas protegidas).
* Los resultados de herramientas, documentos y páginas web se insertan como
  datos dentro de `<|tool_result|>` y se codifican sin tokens especiales: no
  pueden simular un cambio de rol ni una llamada a herramienta (verificado en
  `tests/test_agent.py::test_chat_format_roundtrip_and_injection`).
* El prompt de sistema indica que ese contenido son datos, no instrucciones.

## Ejecución de código no confiable

`titan/agent/sandbox.py`:
* Linux: espacio de nombres de red vacío (`unshare --net`) + límites de CPU,
  memoria, tamaño de archivo y procesos; o Docker con `--network none`.
* Windows: subproceso con tiempo límite (no aísla la red): `code_exec` requiere
  aprobación. Para aislamiento fuerte use Docker Desktop o WSL2.
* Intérprete aislado (`-I`), directorio temporal propio y entorno sin variables
  sensibles: las claves API del usuario no llegan al código ejecutado.

## Red

* `web_fetch` bloquea destinos privados, de bucle local y de enlace local, y
  revalida cada redirección (protección SSRF). Riesgo residual: DNS rebinding
  entre la comprobación y la conexión; por eso `network` pide aprobación.
* El servidor web escucha en 127.0.0.1, exige el token `X-Titan-Token`
  (aleatorio por arranque) y un `Host` local: una página web ajena no puede usar
  el agente (CSRF / DNS rebinding). Cabeceras CSP y `X-Frame-Options: DENY`.

## Credenciales

* Las claves (`RUNPOD_API_KEY`, `VAST_API_KEY`, `LAMBDA_API_KEY`, `HF_TOKEN`,
  `BRAVE_API_KEY`) se leen de variables de entorno; nunca se escriben en el
  repositorio, en configuraciones ni en registros (`.gitignore` excluye `.env`,
  `secrets/` y `config/credentials*`).
* Al lanzar un trabajo en la nube solo viajan al nodo las variables listadas en
  `env_passthrough` y, para que el guardián pueda apagar la máquina, la clave del
  proveedor; el libro de gastos registra **qué** variables se enviaron, no sus valores.

## Gasto

* Ningún recurso de pago se crea sin `--approve` y sin que el coste estimado
  quepa en el presupuesto del trabajo.
* El guardián del nodo detiene el entrenamiento con checkpoint y apaga la
  máquina al alcanzar el presupuesto o las horas máximas; `titan cloud watch`
  aplica un segundo límite desde el equipo local.

## Memoria

Desactivada por defecto; rechaza credenciales, enmascara datos personales,
caduca a los `retention_days` y se puede exportar o borrar por completo.
