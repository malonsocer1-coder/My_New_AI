# Estado del proyecto TITAN AI y puntos para tu retroalimentación

*Actualizado: 24 de octubre de 2026.*

## Dónde estamos

| Etapa | Estado | Evidencia |
|---|---|---|
| A — Validación técnica | ✅ **completada** | modelo propio de 8,6 M entrenado desde cero; [informe](../reports/INFORME_ETAPA_A.md) |
| B — Prototipo (PC, 100 M, por partes) | 🟡 **lista para ejecutar en tu PC** | configs, instalador y guía; falta ejecutarla en la RTX 5070 |
| C — Escalado en la nube | 🟡 **código verificado**, sin lanzar | DDP/FSDP2 probados; proveedores en modo de prueba; falta presupuesto, claves y red |
| D — Entrenamiento principal | ⏸️ pendiente de tu decisión de escala y presupuesto | [tabla de escalado](ESCALADO_Y_PRESUPUESTO.md) |
| E — SFT / herramientas | ✅ pipeline completo; SFT de demostración: 100 % de llamadas válidas y 82,8 % de respuestas correctas en tareas de herramientas reservadas | [EVALUACION.md](EVALUACION.md) |
| F — Despliegue | ✅ exportación, GGUF, cuantización y servidor web verificados | falta medir en la RTX 5070 |

## Qué funciona hoy (verificado)

1. **Modelo propio desde cero:** arquitectura Llama/Mixtral-compatible, inicialización aleatoria propia, tokenizador BPE propio.
2. **Datos:** pipeline con licencias, PII, calidad, deduplicación, descontaminación (0 contaminados), fichas de datos y **dominio propio para el español de México** (`es_mx`, con más peso en la mezcla).
3. **Entrenamiento:** checkpoints atómicos con SHA-256, reanudación exacta tras SIGTERM real, DDP y FSDP2, límites de gasto y tiempo, métricas y ETA, entropía cruzada por fragmentos para ahorrar VRAM.
4. **Entrenamiento por partes:** agrandar el modelo y unir expertos preservando lo aprendido; planes por fases reanudables.
5. **Evaluación:** 8 suites reproducibles sin jueces externos.
6. **Exportación:** formatos TITAN, Hugging Face y GGUF con equivalencia numérica probada contra `transformers` y llama.cpp.
7. **Cuantización:** Q8_0 a Q4_K_M con pérdida medida (< 0,4 % de perplejidad en la Etapa A).
8. **Agente:** 15 herramientas (incluida la creación de Word/Excel/PowerPoint), permisos, sandbox, memoria con privacidad, búsqueda en documentos locales, MCP y plugins.
9. **Ajuste por preferencias (DPO):** con pares verificados que castigan inventar resultados, números falsos y falsa seguridad.
10. **Interfaz web local:** chat y agente con diálogos de permisos y protección anti-CSRF ([captura](../reports/ui_agente.png)).
11. **Nube:** estimación de coste, `--approve` obligatorio, rechazo por presupuesto y guardián en el nodo.
12. **Diagnóstico:** `titan doctor` revisa GPU, CUDA, VRAM, disco, paquetes y red en tu PC.

## Qué NO está hecho y por qué

| Pendiente | Motivo | Qué se necesita |
|---|---|---|
| Entrenar en GPU | esta sesión no tiene GPU | ejecutar el plan en tu PC o autorizar la nube |
| Datos reales en español a escala | la red de esta sesión bloquea huggingface.co | ejecutar `titan data prepare --config configs/data/pc_es.yaml` en tu PC |
| Lanzar trabajos en la nube | sin autorización de gasto ni claves; la red bloquea las API de los proveedores | tu decisión de presupuesto + clave del proveedor |
| Probar los scripts de Windows | esta máquina es Linux | ejecutarlos en tu PC (pueden necesitar ajustes menores) |
| Velocidad real en la RTX 5070 | sin acceso a tu equipo | `titan bench` y `titan gguf bench` en el PC |
| DPO a escala útil | implementado y probado, pero aporta poco por debajo de ~1 B parámetros | un modelo base ≥ 1 B |
| Benchmarks estándar (MMLU, GSM8K…) | necesitan Hugging Face y un modelo por encima del azar | Etapa B/C |
| Código abierto a gran escala | The Stack exige aceptar sus términos | aceptarlos en Hugging Face y definir `HF_TOKEN` |

## Decisiones que necesito de ti

1. **Presupuesto para la nube** (ver la escalera en [ESCALADO_Y_PRESUPUESTO.md](ESCALADO_Y_PRESUPUESTO.md)):
   * 0 USD → solo PC (100 M).
   * ~100–200 USD → titan-350m (validación de la nube).
   * ~1–2,5 mil USD → titan-1b (primer modelo útil en español).
   * ~12–55 mil USD → titan-3b o MoE 7B-A2B (máxima capacidad que cabe bien en tu laptop).
2. **Proveedor:** Vast.ai (más barato, fiabilidad variable), RunPod (equilibrado) o Lambda (nodos 8×H100 con NVLink).
3. **Almacenamiento remoto para checkpoints** (R2, B2, S3, Google Drive vía rclone) y
   si quieres que los pesos grandes se suban a algún sitio además de tu disco.
4. **Prioridades de capacidad:** ¿más peso en español de México, en
   programación o en uso de herramientas? Cambia la mezcla de datos.
5. **¿Quieres que Claude trabaje directamente en tu PC** (app Claude Desktop o
   `claude remote-control`) para ejecutar la Etapa B y ajustar los scripts de Windows?

## Preguntas concretas para tu retroalimentación

* ¿El plan por partes (base de 100 M → 3 expertos → MoE de 214 M) es lo que
  imaginabas, o prefieres la alternativa de agrandar el modelo de 100 M a 176 M?
* ¿Te parece bien que la memoria persistente del agente venga desactivada por defecto?
* ¿Qué herramientas te importan más en el día a día (archivos, código, web, documentos, voz)?
* ¿Cuánto tiempo seguido puede estar tu laptop entrenando (horas por día)?
