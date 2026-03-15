param(
  [string]$ImageName = "pailletjp/avatario-runpod",
  [string]$Tag = "thin",
  [switch]$NoPush,
  [switch]$SkipLatestTag
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Description,
    [scriptblock]$Action
  )

  Write-Host "[info] $Description"
  & $Action
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Description"
  }
}

$fullTag = "${ImageName}:${Tag}"
$latestTag = "${ImageName}:latest"

Invoke-Step "Building Runpod image ${fullTag}" {
  docker build -f Dockerfile.runpod -t $fullTag .
}

if (-not $SkipLatestTag) {
  Invoke-Step "Tagging ${fullTag} as ${latestTag}" {
    docker tag $fullTag $latestTag
  }
}

if (-not $NoPush) {
  Invoke-Step "Pushing ${fullTag}" {
    docker push $fullTag
  }
  if (-not $SkipLatestTag) {
    Invoke-Step "Pushing ${latestTag}" {
      docker push $latestTag
    }
  }
}

Write-Host "[ok] Runpod image ready: $fullTag"
if (-not $SkipLatestTag) {
  Write-Host "[ok] Latest tag ready: $latestTag"
}
