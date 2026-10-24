<#
.SYNOPSIS
  Todo en uno: instala, mide la GPU, prepara los datos y entrena el plan de 118 M (pc_120m).

.DESCRIPTION
  Cada paso se omite si ya está hecho, así que se puede ejecutar las veces que haga falta:
    1. Instalación (install.ps1 -NoSleep -Triton), solo si no existe el entorno virtual.
    2. Comprobación de que PyTorch ve la GPU.
    3. Medición de la GPU (titan bench --find-batch -> reports\bench_pc_120m.json), una vez.
    4. Preparación de los datos (titan data prepare), si no existe data\pc_es\manifest.json.
    5. Con -Programar: tarea de Windows que continúa el entrenamiento cada día a la hora -Hora
       durante -Horas (despierta el equipo si está suspendido y solo arranca con corriente).
    6. Entrenamiento (train_plan.ps1), salvo con -SoloProgramar.

  Requisitos: Git y Python 3.12 instalados, portátil conectado a la corriente.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\entrenar_todo.ps1
  powershell -ExecutionPolicy Bypass -File scripts\windows\entrenar_todo.ps1 -Horas 8 -Programar -Hora 23:00
#>
param(
    [string]$Root = "D:\TITAN_AI",
    [double]$Horas = 0,        # 0 = sin límite (hasta terminar o Ctrl+C)
    [switch]$Programar,
    [string]$Hora = "23:00",
    [switch]$SoloProgramar
)

$ErrorActionPreference = "Stop"
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

$App = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VPy = Join-Path $App ".venv\Scripts\python.exe"
$TrainScript = Join-Path $PSScriptRoot "train_plan.ps1"

# 1. Instalación -----------------------------------------------------------
if (-not (Test-Path $VPy)) {
    Step "Instalando TITAN AI (solo la primera vez)"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "install.ps1") -Root $Root -NoSleep -Triton
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VPy)) { throw "La instalación falló; revise los mensajes de arriba." }
}
Set-Location $App
# Python escribe avisos en stderr; con "Stop" PowerShell 5.1 los trataría como errores fatales.
$ErrorActionPreference = "Continue"

# 2. GPU -------------------------------------------------------------------
Step "Comprobando la GPU"
& $VPy -c "import torch, sys; ok = torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else 'sin GPU CUDA'); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch no ve la GPU. Actualice el controlador NVIDIA y ejecute de nuevo install.ps1 (ver docs\PC_MODELO_120M.md)."
}

# 3. Medición --------------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Join-Path $App "reports") | Out-Null
$Bench = Join-Path $App "reports\bench_pc_120m.json"
if (-not (Test-Path $Bench)) {
    Step "Midiendo la GPU (5-10 minutos)"
    & $VPy -m titan.cli bench --config configs/train/pc_120m_base.yaml --find-batch --out $Bench
    if ($LASTEXITCODE -ne 0) { Write-Warning "La medición falló; el entrenamiento la repetirá al empezar." }
}

# 4. Datos -----------------------------------------------------------------
$Manifest = Join-Path $App "data\pc_es\manifest.json"
if (-not (Test-Path $Manifest)) {
    Step "Preparando los datos (varias horas, necesita Internet)"
    & $VPy -m titan.cli data prepare --config configs/data/pc_es.yaml
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Manifest)) {
        throw "La preparación de datos falló; revise el registro de arriba y vuelva a ejecutar este script."
    }
}

# 5. Tarea programada --------------------------------------------------------
if ($Programar -or $SoloProgramar) {
    $h = if ($Horas -gt 0) { $Horas } else { 8 }
    Step "Programando el entrenamiento cada día a las $Hora durante $h h"
    $ErrorActionPreference = "Stop"
    try {
        $action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $App `
            -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$TrainScript`" -Hours $h"
        $trigger = New-ScheduledTaskTrigger -Daily -At $Hora
        $settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Hours ([math]::Ceiling($h) + 1))
        Register-ScheduledTask -TaskName "TITAN_Entrenamiento" -Action $action -Trigger $trigger `
            -Settings $settings -Description "Continúa el plan pc_120m de TITAN AI" -Force | Out-Null
        Write-Host "  Tarea 'TITAN_Entrenamiento' creada. Se quita con:"
        Write-Host "  Unregister-ScheduledTask -TaskName TITAN_Entrenamiento -Confirm:`$false"
    } catch {
        Write-Warning "No se pudo crear la tarea programada: $($_.Exception.Message)"
    }
    $ErrorActionPreference = "Continue"
}

# 6. Entrenamiento -----------------------------------------------------------
if (-not $SoloProgramar) {
    Step "Entrenando (plan pc_120m). Ctrl+C guarda un checkpoint y sale; volver a ejecutar continúa."
    if ($Horas -gt 0) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $TrainScript -Hours $Horas
    } else {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $TrainScript
    }
}
