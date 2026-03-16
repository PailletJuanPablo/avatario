[CmdletBinding()]
param(
    [string]$ImageName = "pailletjp/avatario-musetalk-runpod",
    [string]$Tag = "latest",
    [switch]$Push
)

$ErrorActionPreference = "Stop"

$imageRef = "{0}:{1}" -f $ImageName, $Tag

Write-Host ("[info] Building MuseTalk RunPod image: {0}" -f $imageRef)
docker build -f Dockerfile.runpod.musetalk -t $imageRef .

if ($Push) {
    Write-Host ("[info] Pushing MuseTalk RunPod image: {0}" -f $imageRef)
    docker push $imageRef
}

Write-Host ("[ok] MuseTalk RunPod image ready: {0}" -f $imageRef)
