$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
docker compose up --build -d
Write-Host "[ok] API started at http://127.0.0.1:8010/"
