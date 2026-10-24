# Guía de recuperación ante fallos

| Situación | Qué ocurre | Qué hacer |
|---|---|---|
| Ctrl+C, cierre de sesión, `SIGTERM` del proveedor | El paso en curso termina, se guarda un checkpoint y el proceso sale con `stop_reason: signal_N` | Volver a lanzar el mismo comando: reanuda desde `LATEST` |
| Corte de luz / apagado brusco | Se pierde como máximo el trabajo desde el último checkpoint; un checkpoint a medias queda como `step_X.tmp` y se ignora | Relanzar; borrar a mano los `*.tmp` si se quiere liberar espacio |
| Checkpoint corrupto (`Checkpoint ... corrupto`) | La verificación SHA-256 lo detecta al cargar | Borrar esa carpeta `step_*` y editar `LATEST` al anterior (o borrarlo: se usa el más reciente íntegro) |
| Pérdida `nan`/`inf` | El paso se omite sin actualizar pesos; tras `max_bad_steps` seguidos se detiene | Reanudar con `--set skip_samples_on_resume=50000` (salta los datos del pico) y/o bajar `lr` |
| Picos de pérdida repetidos | | Bajar `lr`, subir `warmup_steps`, comprobar los datos del dominio afectado |
| *CUDA out of memory* | | Reducir `micro_batch_size` y subir `grad_accum_steps` en la misma proporción; activar `model.activation_checkpointing=true`; en varias GPUs usar `parallel: fsdp` |
| GPU caída o NCCL colgado (nube) | El guardián detecta el latido parado (`--stale-min`) y reinicia; `torchrun --max-restarts=3` relanza y reanuda | Revisar `logs/watchdog.jsonl` (temperatura, ECC); si se repite, cambiar de máquina |
| El proveedor recupera la máquina (spot/comunidad) | Los checkpoints sincronizados siguen en el almacenamiento remoto | Lanzar de nuevo el trabajo: `bootstrap.sh` descarga y verifica los checkpoints y reanuda |
| Presupuesto agotado | El guardián detiene con checkpoint, sincroniza y apaga | Ampliar `budget_usd` conscientemente y relanzar: continúa donde quedó |
| Fase de un plan interrumpida | `plan_state.json` la marca `interrupted` | `titan plan run --plan ...` la continúa; las fases `done` no se repiten |
| Datos modificados tras entrenar | El manifiesto cambia de huella | No mezclar: preparar en otro `output_dir` y empezar otra ejecución o reanudar conscientemente |

## Verificar integridad

```bash
python -m titan.cloud.sync verify --local runs          # todos los checkpoints
python -c "from titan.export.exporter import verify_export; print(verify_export('models/base/titan'))"
```

## Copias de seguridad

`D:\TITAN_AI\backups`: copie periódicamente `models\base` (pesos originales),
el último checkpoint de cada ejecución y `datasets\<n>\manifest.json` +
`tokenizer`. Con rclone: `rclone copy D:\TITAN_AI\models remoto:titan-backup\models --checksum`.
Los pesos originales nunca se sobrescriben: cuantizaciones y destilados son
artefactos derivados en carpetas propias con su MANIFEST.
