<#
.SYNOPSIS
  Inicia la interfaz web local de TITAN y abre el navegador.
.EXAMPLE
  powershell -File scripts\windows\run_server.ps1
  powershell -File scripts\windows\run_server.ps1 -Model models\base\titan -Port 8765
#>
param(
    [string]$Model = "",
    [int]$Port = 8765,
    [string]$Config = ""
)
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Root = Split-Path -Parent $App
Set-Location $App
$VPy = Join-Path $App ".venv\Scripts\python.exe"
if (-not $Config) { $Config = Join-Path $Root "config\agent.yaml" }
if (-not (Test-Path $Config)) {
    Copy-Item (Join-Path $App "configs\agent\agent.example.yaml") $Config
    Write-Host "Configuración creada en $Config (revísela)."
}
if (-not $Model) {
    # 1) resultado del plan de 118 M (GGUF Q8_0 o, si no, la exportación TITAN), 2) models\quantized, 3) models\base\titan
    $planGguf = Join-Path $App "runs\pc_120m\gguf\titan-120m-Q8_0.gguf"
    $planExport = Join-Path $App "runs\pc_120m\export"
    $gguf = Get-ChildItem (Join-Path $Root "models\quantized") -Filter *.gguf -ErrorAction SilentlyContinue | Select-Object -First 1
    & $VPy -c "import llama_cpp" 2>$null
    $hasLlama = ($LASTEXITCODE -eq 0)
    if ((Test-Path $planGguf) -and $hasLlama) { $Model = $planGguf }
    elseif (Test-Path (Join-Path $planExport "model_config.json")) { $Model = $planExport }
    elseif ($gguf) { $Model = $gguf.FullName }
    else { $Model = Join-Path $Root "models\base\titan" }
}
Write-Host "Modelo: $Model"
Start-Job -ScriptBlock { param($p) Start-Sleep 6; Start-Process "http://127.0.0.1:$p" } -ArgumentList $Port | Out-Null
$ErrorActionPreference = "Continue"   # los avisos de Python en stderr no deben abortar el servidor
& $VPy -m titan.cli serve --model $Model --port $Port --agent-config $Config
