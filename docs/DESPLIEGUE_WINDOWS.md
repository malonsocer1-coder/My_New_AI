# Despliegue local en Windows (D:\TITAN_AI)

Equipo de destino: RTX 5070 Laptop (8 GB VRAM), i7-14700HX, 16 GB RAM, SSD de 1 TB.

## Estructura

```
D:\TITAN_AI
├─ app\                código (este repositorio) + .venv; app\data, app\runs, app\models,
│                       app\cache y app\logs son uniones a las carpetas de abajo
├─ models\
│  ├─ base\            pesos ORIGINALES exportados (nunca se modifican)
│  ├─ checkpoints\     checkpoints descargados de la nube o del PC
│  ├─ quantized\       GGUF Q8_0 / Q5_K_M / Q4_K_M e int8/int4 (artefactos derivados)
│  └─ distilled\       modelos destilados o reducidos (derivados)
├─ datasets\           datos preparados (shards + manifiestos)
├─ tokenizers\         copias de los tokenizadores
├─ training\           ejecuciones de entrenamiento (runs)
├─ evaluation\         informes de evaluación
├─ tools\              binarios externos (llama.cpp, whisper.cpp)
├─ plugins\            plugins autorizados del agente
├─ memory\             memoria persistente (memory.db)
├─ config\             permissions.yaml, agent.yaml (fuera del alcance del agente)
├─ logs\  cache\  backups\  documentation\
```

El tamaño puede crecer sin límite impuesto por la aplicación: el único límite
es el disco.

## Instalación

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -NoSleep
```

Ver `docs/PC_ENTRENAMIENTO_POR_PARTES.md` para los requisitos.

## Del checkpoint al modelo local

```powershell
cd D:\TITAN_AI\app; .\.venv\Scripts\Activate.ps1
# 1. Exportar los pesos originales (TITAN + Hugging Face + GGUF f16), con SHA-256
titan export --checkpoint runs\pc_btx\moe_tune --out models\base\titan-pc
# 2. Cuantizar para la GPU (artefactos derivados)
titan gguf quantize --model models\base\titan-pc\gguf\titan-100m-fp16.gguf --out models\quantized\titan-pc-Q8_0.gguf --type Q8_0
titan gguf quantize --model models\base\titan-pc\gguf\titan-100m-fp16.gguf --out models\quantized\titan-pc-Q4_K_M.gguf --type Q4_K_M
# 3. Medir velocidad real en la RTX 5070
titan gguf bench --model models\quantized\titan-pc-Q4_K_M.gguf --gpu-layers -1
# 4. Interfaz web con agente
powershell -File scripts\windows\run_server.ps1 -Model models\quantized\titan-pc-Q4_K_M.gguf
```

Para que llama.cpp use la GPU, `llama-cpp-python` debe compilarse con CUDA
(requiere CUDA Toolkit 12.8+ y Visual Studio Build Tools):

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --force-reinstall --no-cache-dir llama-cpp-python
```

Alternativa sin compilar: usar el motor PyTorch de TITAN con CUDA (`backend:
titan` en `config\agent.yaml`, modelo en `models\base\...`); es más lento que
llama.cpp, pero funciona tal cual con la instalación de PyTorch.

## Qué cabe en 8 GB (estimación, verificar con `titan gguf bench`)

| Modelo | Cuantización | Memoria total | ¿Cabe en VRAM? | tok/s estimado |
|---|---|---|---|---|
| titan-1b | Q8_0 | 1,9 GB | sí | ~140 |
| titan-3b | Q8_0 | 4,3 GB | sí | ~63 |
| titan-moe-7b-a2b | Q5_K_M | 5,9 GB | sí | ~115 |
| titan-8b | Q5_K_M / Q4_K_M | 6,5 / 5,7 GB | sí | ~40 / ~46 |
| titan-8b | Q8_0 | 9,1 GB | no (86 % de capas en GPU) | ~20 |
| titan-30b | Q4_K_M | 18 GB | no (41 % en GPU, resto en RAM) | ~5 |

* **Descarga parcial a RAM:** `n_gpu_layers` en `config\agent.yaml` (o
  `--gpu-layers` en `titan gguf bench`) deja parte de las capas en la RAM.
  Funciona, pero por debajo de ~85 % de capas en GPU la velocidad cae bajo 10 tok/s.
* **Versión remota opcional:** si el modelo completo no cabe, ejecute el
  servidor (`titan serve`) en una máquina con GPU grande y conéctese desde el
  navegador del PC (túnel SSH: `ssh -L 8765:127.0.0.1:8765 usuario@servidor`).
  El modo local sigue funcionando sin conexión con el modelo cuantizado.
* **Sin conexión:** el chat, el agente, la calculadora, los archivos, Python
  en sandbox, la base de conocimiento local y la memoria funcionan sin
  Internet; solo `web_fetch`/`web_search` y los proveedores de nube la necesitan.

## Resultados medidos (Etapa A, en la CPU de la sesión en la nube)

Ver `reports/quantization_stage_a.json` y `reports/INFORME_ETAPA_A.md`. Son
cifras de un modelo de 8,6 M parámetros en 4 vCPU sin GPU: validan el circuito
de exportación y cuantización, no la velocidad de la RTX 5070.
