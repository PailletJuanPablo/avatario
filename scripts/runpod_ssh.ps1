param(
  [string]$Target = "02cg563ucc0piq-6441116d@ssh.runpod.io",
  [string]$IdentityFile = "~/.ssh/id_ed25519",
  [string]$RemoteCommand = "",
  [switch]$Interactive
)

$ErrorActionPreference = "Stop"

function Resolve-IdentityPath {
  param([string]$PathValue)

  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    throw "IdentityFile cannot be empty."
  }

  if ($PathValue.StartsWith("~/")) {
    return Join-Path $HOME $PathValue.Substring(2)
  }

  return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
}

$resolvedIdentityFile = Resolve-IdentityPath -PathValue $IdentityFile

if (-not (Test-Path $resolvedIdentityFile)) {
  throw "SSH identity file not found: $resolvedIdentityFile"
}

$sshArgs = @(
  "-tt",
  "-i", $resolvedIdentityFile,
  "-o", "StrictHostKeyChecking=accept-new",
  $Target
)

if (-not $Interactive -and -not [string]::IsNullOrWhiteSpace($RemoteCommand)) {
  $commandBytes = [System.Text.Encoding]::UTF8.GetBytes($RemoteCommand)
  $encodedCommand = [System.Convert]::ToBase64String($commandBytes)
  $sshArgs += "bash -lc ""printf '%s' '$encodedCommand' | base64 -d | bash"""
}

& ssh @sshArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
  exit $exitCode
}
