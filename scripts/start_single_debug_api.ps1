param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8010,
    [string]$ApiToken = "dev-token",
    [string]$RunnerPython = "E:\FasterLivePortrait-windows-20241228\FasterLivePortrait-windows\venv\python.exe",
    [string]$GridSamplePluginPath = "E:\FasterLivePortrait-windows-20241228\FasterLivePortrait-windows\checkpoints\liveportrait_onnx\grid_sample_3d_plugin.dll",
    [switch]$NoRun
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$apiPython = Join-Path $projectRoot ".venv-liveportrait\Scripts\python.exe"
$stdoutLog = Join-Path $projectRoot "output_fasterliveportrait\single_api.stdout.log"
$stderrLog = Join-Path $projectRoot "output_fasterliveportrait\single_api.stderr.log"

Write-Host "[single-debug-api] project root: $projectRoot"
Write-Host "[single-debug-api] api python:   $apiPython"
Write-Host "[single-debug-api] runner:       $RunnerPython"
Write-Host "[single-debug-api] host:         $ApiHost"
Write-Host "[single-debug-api] port:         $Port"
Write-Host "[single-debug-api] stdout log:   $stdoutLog"
Write-Host "[single-debug-api] stderr log:   $stderrLog"

if ($NoRun) {
    return
}

$launcherScript = @"
`$env:ANIMATION_RUNNER_PYTHON = '$RunnerPython'
`$env:ANIMATION_BACKEND = 'trt'
`$env:ANIMATION_TRT_RUNTIME = 'local'
`$env:ANIMATION_SKIP_TRT_ENGINE_BUILD = '1'
`$env:ANIMATION_FORCE_TRT_ENGINE_REBUILD = '0'
`$env:ANIMATION_TRT_GRID_SAMPLE_PLUGIN_PATH = '$GridSamplePluginPath'
`$env:ANIMATION_API_TOKEN = '$ApiToken'
`$env:ANIMATION_WARMUP_ENABLED = '0'
`$env:ANIMATION_AVATAR_STREAM_DEBUG = '1'
Set-Location '$projectRoot'
& '$apiPython' '.\realtime_stream_api.py' --host $ApiHost --port $Port
"@

if (Test-Path $stdoutLog) {
    Remove-Item $stdoutLog -Force
}
if (Test-Path $stderrLog) {
    Remove-Item $stderrLog -Force
}

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-Command", $launcherScript) `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host "[single-debug-api] started pid=$($process.Id)"
