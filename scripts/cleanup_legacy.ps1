param(
  [switch]$IncludeLegacyCode
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$legacyDirectories = @(
  ".venv-liveportrait",
  "node_modules",
  "models",
  "output_b_step1",
  "output_b_step2",
  "output_b_step3",
  "output_debug_logic",
  "output_liveportrait",
  "output_mediapipe_v2",
  "output_new_contract",
  "__pycache__"
)

$legacyLogPatterns = @(
  "api_*.log",
  "api_*.err",
  "api_run.log",
  "api_run.err",
  "serve.log",
  "temp_job_log.json"
)

$legacyCodeFiles = @(
  "background.png",
  "requirements.txt",
  "package-lock.json",
  "newtest.html",
  "viewer.html",
  "viewer.js",
  "viewer_recon.html",
  "viewer_render.html",
  "viewer_render.js",
  "preprocess.py",
  "preprocess_face_controls.py",
  "build_mouth_controls.py",
  "liveportrait_runner.py",
  "rig-config.json",
  "debug_check.py",
  "debug_landmarks.py",
  "debug_mesh.py",
  "debug_uv_format.py",
  "fix_uvs.py",
  "fix_current_output_uvs.py",
  "test_textures.html",
  "verify_output.py",
  "bootstrap_liveportrait_env.ps1",
  "input.mp4",
  "input_new.mp4"
)

foreach ($relativePath in $legacyDirectories) {
  $target = Join-Path $projectRoot $relativePath
  if (Test-Path $target) {
    Remove-Item -Path $target -Recurse -Force
    Write-Host "[removed-dir] $relativePath"
  }
}

foreach ($pattern in $legacyLogPatterns) {
  $matches = Get-ChildItem -Path $projectRoot -Filter $pattern -File -ErrorAction SilentlyContinue
  foreach ($item in $matches) {
    Remove-Item -Path $item.FullName -Force
    Write-Host "[removed-log] $($item.Name)"
  }
}

if ($IncludeLegacyCode) {
  foreach ($relativePath in $legacyCodeFiles) {
    $target = Join-Path $projectRoot $relativePath
    if (Test-Path $target) {
      Remove-Item -Path $target -Force
      Write-Host "[removed-code] $relativePath"
    }
  }
} else {
  Write-Host "[info] Legacy code files were preserved. Re-run with -IncludeLegacyCode to delete them."
}

Write-Host "[ok] cleanup complete"
