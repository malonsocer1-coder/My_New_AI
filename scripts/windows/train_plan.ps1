<#
.SYNOPSIS
  Ejecuta (o continúa) el plan de entrenamiento por partes en el PC.

.DESCRIPTION
  Cada ejecución avanza el plan desde donde se quedó. Con -Hours limita la
  sesión: al cumplirse el tiempo guarda un checkpoint y se detiene; la
  siguiente ejecución reanuda exactamente en ese punto.

  Plan por defecto: configs\plans\pc_120m.yaml (modelo de ~118 M parámetros, desde
  pesos aleatorios hasta el GGUF final). Mantenga el portátil conectado a la corriente.

.EXAMPLE
  powershell -File scripts\windows\train_plan.ps1 -Hours 8
  powershell -File scripts\windows\train_plan.ps1 -Plan configs\plans\pc_dense_100m.yaml
  powershell -File scripts\windows\train_plan.ps1 -Status
#>
param(
    [string]$Plan = "configs\plans\pc_120m.yaml",
    [double]$Hours = 0,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$App = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $App
$VPy = Join-Path $App ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) { throw "No se encontró el entorno virtual. Ejecute primero scripts\windows\install.ps1" }

if ($Status) {
    & $VPy -m titan.cli plan status --plan $Plan
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path (Join-Path $App "logs") | Out-Null
$log = Join-Path $App ("logs\plan_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$cmdArgs = @("-m", "titan.cli", "plan", "run", "--plan", $Plan)
if ($Hours -gt 0) { $cmdArgs += @("--hours", "$Hours") }

# Python escribe avisos en stderr; con "Stop" PowerShell 5.1 los trataría como errores fatales.
$ErrorActionPreference = "Continue"
Write-Host "Registro: $log"
Write-Host "Pulse Ctrl+C para detener: se guardará un checkpoint al terminar el paso en curso."
& $VPy @cmdArgs 2>&1 | Tee-Object -FilePath $log
& $VPy -m titan.cli plan status --plan $Plan
