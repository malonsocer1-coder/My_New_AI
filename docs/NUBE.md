# Entrenamiento en la nube

## Qué infraestructura hay realmente

| Lugar | Recursos | Uso |
|---|---|---|
| Sesión de Claude Code (nube) | 4 vCPU Xeon, 15 GB RAM, ~25 GB libres, **sin GPU** | desarrollo, tests, Etapa A en CPU |
| Tu PC | RTX 5070 Laptop 8 GB, i7-14700HX, 16 GB RAM | entrenamiento por partes hasta ~350 M; inferencia local |
| GPUs alquiladas | H100/H200/B200 por horas | entrenamiento principal ≥ 1 B |

La sesión de Claude Code **no** es un clúster de GPU ni da acceso a
infraestructura de Anthropic. Además, su política de red bloquea
`rest.runpod.io`, `console.vast.ai`, `api.lambdalabs.com` y `huggingface.co`:
los trabajos se lanzan **desde tu PC** (o desde esta sesión si añades esos
dominios en *Network access* del entorno).

## Comparación de proveedores (referencias de octubre de 2026)

| Proveedor | H100 USD/GPU-h | Ventajas | Inconvenientes |
|---|---|---|---|
| Vast.ai | ~1,50–2,30 | lo más barato; mercado con muchas ofertas | máquinas de terceros: fiabilidad e interconexión variables; filtrar `reliability ≥ 0,98` y `verified` |
| RunPod | ~2,00–3,99 (subió en oct. 2026) | API sencilla, facturación por segundo, nube segura | precio Secure mayor; disponibilidad de 8×H100 variable |
| Lambda | ~2,69–3,49 (8×H100 SXM) | nodos 8×H100 con NVLink, imagen con CUDA lista | a veces sin capacidad; solo nodos completos |
| AWS (p5) | ~6,9 bajo demanda; ~3–4,5 spot | fiabilidad, cuotas empresariales | caro bajo demanda; spot interrumpible; cuotas a solicitar |
| Google Cloud (a3) | ~11 bajo demanda; ~3,7 spot | ecosistema, almacenamiento | lo más caro bajo demanda |

Fuentes: índices de precios de [Ornn](https://data.ornn.com/gpu-cloud-price-comparison),
[gpuperhour.com](https://gpuperhour.com/), [CloudZero](https://www.cloudzero.com/blog/cloud-gpu-pricing-comparison/),
[pricingsaas (RunPod, oct. 2026)](https://pricingsaas.com/companies/runpod/diffs/2026W42),
[Lambda](https://www.lambda.ai/pricing), [Holori (AWS p5)](https://calculator.holori.com/aws/ec2/p5.48xlarge/us-east-1)
y [Google Cloud](https://cloud.google.com/products/compute/gpus-pricing). Los
precios cambian a diario: `titan cloud offers` consulta la oferta real.

**Recomendación:** para ≤ 8 GPUs, Vast.ai (filtrando fiabilidad) o RunPod
Secure por precio; Lambda si se necesita un nodo 8×H100 con NVLink garantizado.
Para varios nodos (≥ 16 GPUs) hace falta red InfiniBand: Lambda 1-Click
Clusters o hiperescaladores. Elegir por **coste total del entrenamiento
completado**, no por precio por hora.

## Flujo de un trabajo

```bash
# En tu PC (con las claves en variables de entorno, nunca en archivos del repo):
setx RUNPOD_API_KEY "..."                    # Windows; en Linux: export RUNPOD_API_KEY=...
titan cloud providers                        # qué claves están definidas
titan cloud offers --provider vastai --gpu H100 --num-gpus 8
titan cloud plan   --job configs/cloud/titan_1b_runpod.yaml          # coste y avisos
titan cloud launch --job configs/cloud/titan_1b_runpod.yaml --approve # GASTA DINERO
titan cloud watch  --job configs/cloud/titan_1b_runpod.yaml --instance <id>
titan cloud spend                            # gasto estimado según el libro
titan cloud stop   --job ... --instance <id>
python -m titan.cloud.sync pull --remote r2:titan-ai/runs --local runs   # descarga verificada
```

`launch` se niega si falta `--approve` o si el coste estimado supera
`budget_usd`. En el nodo, `scripts/cloud/bootstrap.sh`:
1. clona el código e instala dependencias;
2. restaura y verifica checkpoints previos desde el almacenamiento remoto;
3. prepara datos si faltan (mejor llevarlos ya preparados);
4. arranca el **guardián** (`titan.cloud.watchdog`): gasto, horas, latido, salud de GPU, sincronización;
5. lanza `torchrun --max-restarts=3` (DDP/FSDP2 según la config);
6. sincroniza al terminar y el guardián apaga la máquina al llegar al límite.

## Varios nodos

`torchrun --nnodes=N --node_rank=i --rdzv-backend=c10d --rdzv-endpoint=<ip>:29500`
con `parallel: fsdp` y `fsdp_shard_size: 8` (HSDP: fragmentado dentro del nodo
y réplica entre nodos). Para > 8 B parámetros o > 64 GPUs, valorar
paralelismo tensorial/de tubería (Megatron-Core/TorchTitan) manteniendo el
mismo formato de pesos.

## Antes del primer lanzamiento real

1. Revisar `titan cloud offers --dry-run` y compararlo con la documentación vigente de la API.
2. Verificar que la imagen Docker de `image` existe.
3. Ejecutar un trabajo corto (`max_steps` pequeño, `budget_usd` de 5–10 USD) para validar todo el circuito.
