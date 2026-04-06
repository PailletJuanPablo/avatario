[CmdletBinding()]
param(
    [string]$RuntimeRoot = "E:\FasterLivePortrait-windows-20241228\FasterLivePortrait-windows",
    [string]$ApiPython = "",
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8010,
    [string]$ApiToken = "dev-token",
    [switch]$DisableWarmup,
    [switch]$SkipTrtEngineBuild,
    [switch]$ForceTrtEngineRebuild,
    [string]$LogFile = "",
    [switch]$NoRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ApiPython)) {
    $ApiPython = Join-Path $ProjectRoot ".venv-liveportrait\Scripts\python.exe"
}
if ([string]::IsNullOrWhiteSpace($LogFile)) {
    $LogFile = Join-Path $ProjectRoot "output_fasterliveportrait\local_e2e_api.log"
}

$RunnerPython = Join-Path $RuntimeRoot "venv\python.exe"
$RunnerRepo = Join-Path $ProjectRoot "third_party\FasterLivePortrait"
$WorkerLogFile = Join-Path $ProjectRoot "output_fasterliveportrait\worker_queue\worker.log"
$GridSamplePluginPath = Join-Path $RuntimeRoot "checkpoints\liveportrait_onnx\grid_sample_3d_plugin.dll"

if (-not (Test-Path $RunnerPython -PathType Leaf)) {
    throw "Runner python not found: $RunnerPython"
}
if (-not (Test-Path $ApiPython -PathType Leaf)) {
    throw "API python not found: $ApiPython"
}
if (-not (Test-Path $RunnerRepo -PathType Container)) {
    throw "Runner repo not found: $RunnerRepo"
}
if (-not (Test-Path $GridSamplePluginPath -PathType Leaf)) {
    throw "TensorRT GridSample plugin not found: $GridSamplePluginPath"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WorkerLogFile) | Out-Null

$env:ANIMATION_RUNNER_PYTHON = $RunnerPython
$env:ANIMATION_BACKEND = "trt"
$env:ANIMATION_TRT_RUNTIME = "local"
$env:ANIMATION_SKIP_TRT_ENGINE_BUILD = $(if ($SkipTrtEngineBuild) { "1" } else { "0" })
$env:ANIMATION_FORCE_TRT_ENGINE_REBUILD = $(if ($ForceTrtEngineRebuild) { "1" } else { "0" })
$env:ANIMATION_TRT_GRID_SAMPLE_PLUGIN_PATH = $GridSamplePluginPath
$env:ANIMATION_API_TOKEN = $ApiToken
$env:ANIMATION_API_HOST = $ApiHost
$env:ANIMATION_API_PORT = "$Port"

if ($DisableWarmup) {
    $env:ANIMATION_WARMUP_ENABLED = "0"
}

Write-Host "[local-e2e] project root: $ProjectRoot"
Write-Host "[local-e2e] api python:   $ApiPython"
Write-Host "[local-e2e] runner python:$RunnerPython"
Write-Host "[local-e2e] runner repo:  $RunnerRepo"
Write-Host "[local-e2e] base url:     http://$ApiHost`:$Port"
Write-Host "[local-e2e] auth token:   $ApiToken"
Write-Host "[local-e2e] check trt:    $(-not $SkipTrtEngineBuild)"
Write-Host "[local-e2e] force rebuild: $ForceTrtEngineRebuild"
Write-Host "[local-e2e] trt plugin:   $GridSamplePluginPath"
Write-Host "[local-e2e] api log file: $LogFile"
Write-Host "[local-e2e] worker log:   $WorkerLogFile"

if ($NoRun) {
    return
}

if (Test-Path $WorkerLogFile -PathType Leaf) {
    Remove-Item -LiteralPath $WorkerLogFile -Force
}

Push-Location $ProjectRoot
try {
    $previousErrorActionPreference = $ErrorActionPreference
    $apiExitCode = 0
    try {
        $ErrorActionPreference = "Continue"
        & $ApiPython ".\realtime_stream_api.py" --host $ApiHost --port $Port 2>&1 | Tee-Object -FilePath $LogFile
        $apiExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($apiExitCode -ne 0) {
        Write-Error "API process exited with code $apiExitCode. Review $LogFile and output_fasterliveportrait\\worker_queue\\worker.log"
        exit $apiExitCode
    }
}
finally {
    Pop-Location
}
