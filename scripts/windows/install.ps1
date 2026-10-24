<#
.SYNOPSIS
  Instala TITAN AI en Windows (por defecto en D:\TITAN_AI) con soporte para GPU NVIDIA.

.DESCRIPTION
  1. Crea la estructura de carpetas de D:\TITAN_AI.
  2. Clona (o actualiza) el código en D:\TITAN_AI\app.
  3. Crea un entorno virtual de Python e instala PyTorch con CUDA y TITAN.
  4. Enlaza app\data, app\runs, app\models, app\cache y app\logs con las carpetas de D:\TITAN_AI
     (uniones de directorio: no requieren permisos de administrador).
  5. Comprueba que la GPU es visible para PyTorch.

  Las GPU RTX 50 (Blackwell) necesitan PyTorch compilado con CUDA 12.8 o superior.
  Con -Triton instala además triton-windows (compatible con su versión de PyTorch) para
  que torch.compile acelere el entrenamiento; si la prueba falla, lo desinstala y se
  entrena sin compilar (más lento, igual de correcto).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install.ps1
  powershell -ExecutionPolicy Bypass -File install.ps1 -Root E:\TITAN_AI -NoSleep
  powershell -ExecutionPolicy Bypass -File install.ps1 -Triton
#>
param(
    [string]$Root = "D:\TITAN_AI",
    [string]$Repo = "https://github.com/malonsocer1-coder/My_New_AI.git",
    [string]$Branch = "claude/adoring-hypatia-oc7bdz",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipTorch,
    [switch]$Triton,
    [switch]$NoSleep
)

$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# 1. Estructura de carpetas ------------------------------------------------
Step "Creando estructura en $Root"
$dirs = @(
    "models\base", "models\checkpoints", "models\quantized", "models\distilled",
    "datasets", "tokenizers", "training", "evaluation", "tools", "plugins", "memory",
    "config", "logs", "cache", "backups", "app", "documentation"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }

# 2. Código ---------------------------------------------------------------
Step "Obteniendo el código"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "No se encontró Git. Instálelo con: winget install --id Git.Git -e"
}
$App = Join-Path $Root "app"
if (Test-Path (Join-Path $App ".git")) {
    git -C $App fetch origin $Branch
    git -C $App checkout $Branch
    git -C $App pull --ff-only origin $Branch
} else {
    if ((Get-ChildItem $App -Force | Measure-Object).Count -gt 0) {
        throw "$App no está vacío y no es un repositorio git. Muévalo o bórrelo antes de continuar."
    }
    git clone --branch $Branch $Repo $App
}

# 3. Python y dependencias -------------------------------------------------
Step "Buscando Python 3.12 / 3.11"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "No se encontró el lanzador 'py'. Instale Python 3.12 con: winget install --id Python.Python.3.12 -e"
}
$pyver = $null
# En Windows PowerShell 5.1, redirigir stderr de un programa con "Stop" aborta el script:
# se relaja solo durante la sonda de versiones.
$ErrorActionPreference = "Continue"
foreach ($v in @("3.12", "3.11")) {
    & py "-$v" -c "import sys" *> $null
    if ($LASTEXITCODE -eq 0) { $pyver = $v; break }
}
$ErrorActionPreference = "Stop"
if (-not $pyver) {
    throw "No se encontró Python 3.12 ni 3.11. Instálelo con: winget install --id Python.Python.3.12 -e"
}
Write-Host "Usando Python $pyver"

$Venv = Join-Path $App ".venv"
$VPy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VPy)) { & py "-$pyver" -m venv $Venv }
& $VPy -m pip install --upgrade pip wheel

if (-not $SkipTorch) {
    Step "Instalando PyTorch con CUDA desde $TorchIndex"
    & $VPy -m pip install torch --index-url $TorchIndex
}
Step "Instalando TITAN AI"
& $VPy -m pip install -e "$($App)[cloud,export,docs,dev]"

# llama.cpp (cuantización GGUF e inferencia rápida). Solo ruedas precompiladas: compilarlo en
# Windows exige Visual Studio; si no hay rueda, se sigue sin él (todo lo demás funciona).
Step "Instalando llama-cpp-python (precompilado)"
$ErrorActionPreference = "Continue"
& $VPy -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
if ($LASTEXITCODE -ne 0) {
    Write-Warning "No hay rueda precompilada de llama-cpp-python para este Python; se omite (la cuantización GGUF del plan se saltará)."
}
$ErrorActionPreference = "Stop"

if ($Triton) {
    Step "Instalando triton-windows para torch.compile"
    # triton-windows 3.N corresponde a PyTorch 2.(N+4): 2.8 -> 3.4, 2.9 -> 3.5, ...
    $bound = & $VPy -c "import torch; v=torch.__version__.split('+')[0].split('.'); print('3.' + str(int(v[1]) - 3))"
    & $VPy -m pip install -U "triton-windows<$bound"
    $ErrorActionPreference = "Continue"
    & $VPy -c "import torch; f=torch.compile(lambda x: x*2+1); assert f(torch.ones(8, device='cuda')).sum().item() == 24; print('torch.compile funciona')"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "torch.compile no funciona con este Triton; se desinstala y se entrenará sin compilar."
        & $VPy -m pip uninstall -y triton-windows
    }
    $ErrorActionPreference = "Stop"
}

# 4. Enlaces a las carpetas de datos ---------------------------------------
Step "Enlazando carpetas de datos"
function Link-Dir($name, $target) {
    $p = Join-Path $App $name
    if (-not (Test-Path $p)) {
        New-Item -ItemType Junction -Path $p -Target $target | Out-Null
        Write-Host "  $p -> $target"
    }
}
Link-Dir "data"   (Join-Path $Root "datasets")
Link-Dir "runs"   (Join-Path $Root "training")
Link-Dir "models" (Join-Path $Root "models")
Link-Dir "cache"  (Join-Path $Root "cache")
Link-Dir "logs"   (Join-Path $Root "logs")

$perm = Join-Path $Root "config\permissions.yaml"
if (-not (Test-Path $perm)) {
    Copy-Item (Join-Path $App "configs\agent\permissions.example.yaml") $perm
    Write-Host "  Política de permisos creada en $perm (edítela usted; el agente no puede modificarla)"
}

# 5. Verificación ----------------------------------------------------------
Step "Diagnóstico del equipo (titan doctor)"
$ErrorActionPreference = "Continue"
& $VPy -m titan.cli doctor
$ErrorActionPreference = "Stop"

if ($NoSleep) {
    Step "Desactivando la suspensión con corriente (para entrenamientos largos)"
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
}

Write-Host "`nInstalación terminada." -ForegroundColor Green
Write-Host "Siguientes pasos (desde $App):"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  titan bench --config configs/train/pc_120m_base.yaml --find-batch --out reports\bench_pc.json"
Write-Host "  titan data prepare --config configs/data/pc_es.yaml"
Write-Host "  powershell -File scripts\windows\train_plan.ps1 -Hours 8"
