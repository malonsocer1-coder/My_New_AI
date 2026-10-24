# Arranque del nodo GPU (lo antepone titan.cloud.job con las variables TITAN_*).
# Pasos: código -> dependencias -> restaurar ejecuciones previas -> datos ->
# guardián de gasto -> entrenamiento con reinicio automático -> sincronización final.
set -euo pipefail
: "${TITAN_REPO:?}" "${TITAN_BRANCH:?}" "${TITAN_TRAIN_CONFIG:?}" "${TITAN_BUDGET_USD:?}" "${TITAN_RATE_USD_H:?}"
WORK="${TITAN_WORKDIR:-/workspace/titan}"
mkdir -p "$WORK" && cd "$WORK"
exec > >(tee -a bootstrap.log) 2>&1
echo "[bootstrap] $(date -u) inicio en $(hostname)"

if [ ! -d app/.git ]; then git clone --branch "$TITAN_BRANCH" "$TITAN_REPO" app; else git -C app pull --ff-only; fi
cd app
python -m pip install --upgrade pip
python -m pip install -e ".[cloud]"
command -v rclone >/dev/null || (curl -fsSL https://rclone.org/install.sh | bash) || echo "[bootstrap] sin rclone"

mkdir -p runs logs
if [ -n "${TITAN_REMOTE:-}" ]; then
  # Tras una interrupción del proveedor, se recuperan los checkpoints ya sincronizados
  python -m titan.cloud.sync pull --remote "$TITAN_REMOTE" --local runs || echo "[bootstrap] nada que restaurar"
fi

if [ -n "${TITAN_DATA_CONFIG:-}" ]; then
  OUT_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$TITAN_DATA_CONFIG")
  if [ ! -f "$OUT_DIR/manifest.json" ]; then titan data prepare --config "$TITAN_DATA_CONFIG"; fi
fi

NGPU=$(nvidia-smi -L | wc -l)
echo "[bootstrap] GPUs: $NGPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m titan.cloud.watchdog --budget "$TITAN_BUDGET_USD" --rate "$TITAN_RATE_USD_H" \
  ${TITAN_MAX_HOURS:+--max-hours "$TITAN_MAX_HOURS"} --run-dir "$TITAN_RUN_DIR" \
  --remote "${TITAN_REMOTE:-}" --pidfile train.pid --exit-when-done &
WATCHDOG=$!

# setsid: el entrenamiento tiene su propio grupo de procesos (el guardián lo detiene limpiamente)
TITAN_OUT_DIR="$TITAN_RUN_DIR" setsid torchrun --standalone --nproc_per_node="$NGPU" --max-restarts=3 \
  -m titan.cli train --config "$TITAN_TRAIN_CONFIG" &
echo $! > train.pid
wait "$(cat train.pid)" || echo "[bootstrap] el entrenamiento terminó con error"
rm -f train.pid

if [ -n "${TITAN_REMOTE:-}" ]; then python -m titan.cloud.sync push --local runs --remote "$TITAN_REMOTE" || true; fi
wait "$WATCHDOG" || true
echo "[bootstrap] $(date -u) fin"
