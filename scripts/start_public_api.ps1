param(
  [string]$HostAddress = "0.0.0.0",
  [int]$Port = 8010,
  [string]$Backend = "trt",
  [string]$TrtRuntime = "local",
  [string]$TrtPrecision = "fp16",
  [string]$DockerGpuDevice = "",
  [switch]$NoWarmup
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExecutable {
  $candidatePaths = @(
    (Join-Path $PSScriptRoot "..\\.venv-liveportrait\\Scripts\\python.exe"),
    (Join-Path $env:USERPROFILE ".pyenv\\pyenv-win\\versions\\3.10.5\\python.exe")
  )

  foreach ($candidatePath in $candidatePaths) {
    $resolvedPath = [System.IO.Path]::GetFullPath($candidatePath)
    if (Test-Path $resolvedPath) {
      return $resolvedPath
    }
  }

  $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCommand) {
    return $pythonCommand.Source
  }

  throw "Python executable was not found."
}

function Stop-ExistingApiProcess {
  $existingProcesses = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like '*realtime_stream_api.py*' }

  foreach ($existingProcess in $existingProcesses) {
    try {
      Stop-Process -Id $existingProcess.ProcessId -Force -ErrorAction Stop
    } catch {
      Write-Warning "Failed to stop process $($existingProcess.ProcessId): $($_.Exception.Message)"
    }
  }
}

function Ensure-FirewallRule {
  param(
    [int]$RulePort
  )

  $ruleName = "Animation API $RulePort"
  $existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
  if ($existingRule) {
    return
  }

  try {
    New-NetFirewallRule `
      -DisplayName $ruleName `
      -Direction Inbound `
      -Action Allow `
      -Protocol TCP `
      -LocalPort $RulePort `
      -Profile Any | Out-Null
  } catch {
    Write-Warning "Firewall rule '$ruleName' was not created automatically: $($_.Exception.Message)"
  }
}

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pythonExecutable = Resolve-PythonExecutable
$stdoutLogPath = Join-Path $projectRoot "output_fasterliveportrait\\host_api_stdout.log"
$stderrLogPath = Join-Path $projectRoot "output_fasterliveportrait\\host_api_stderr.log"
$previousDockerGpuDevice = [Environment]::GetEnvironmentVariable("ANIMATION_DOCKER_GPU_DEVICE", "Process")

if ([string]::IsNullOrWhiteSpace($DockerGpuDevice)) {
  Remove-Item Env:ANIMATION_DOCKER_GPU_DEVICE -ErrorAction SilentlyContinue
} else {
  $env:ANIMATION_DOCKER_GPU_DEVICE = $DockerGpuDevice
}

Ensure-FirewallRule -RulePort $Port
Stop-ExistingApiProcess

$argumentList = @(
  "realtime_stream_api.py",
  "--host", $HostAddress,
  "--port", "$Port",
  "--backend", $Backend,
  "--trt-runtime", $TrtRuntime,
  "--trt-precision", $TrtPrecision
)

if ($NoWarmup) {
  $argumentList += "--no-warmup"
}

Start-Process `
  -FilePath $pythonExecutable `
  -ArgumentList $argumentList `
  -WorkingDirectory $projectRoot `
  -RedirectStandardOutput $stdoutLogPath `
  -RedirectStandardError $stderrLogPath

if ([string]::IsNullOrWhiteSpace($previousDockerGpuDevice)) {
  Remove-Item Env:ANIMATION_DOCKER_GPU_DEVICE -ErrorAction SilentlyContinue
} else {
  $env:ANIMATION_DOCKER_GPU_DEVICE = $previousDockerGpuDevice
}

Write-Output "API started on ${HostAddress}:$Port"
