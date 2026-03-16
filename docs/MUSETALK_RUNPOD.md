# MuseTalk RunPod Scaffold

## Scope

This scaffold adds one isolated deployment path for `MuseTalk` on `RunPod Pods` with an initial `Gradio` user interface.

It does not replace the current `FasterLivePortrait` flow. The goal is to let the project run a first MuseTalk proof of concept on remote GPU hardware without depending on local inference.

## What Was Researched

Research date: `2026-03-16`.

Official MuseTalk sources used for this scaffold:

- Repository: https://github.com/TMElyralab/MuseTalk
- Installation guidance: https://raw.githubusercontent.com/TMElyralab/MuseTalk/main/README.md

RunPod-related references found during research:

- Community worker scaffold: https://github.com/runpod-workers/worker-musetalk
- Community serverless API wrapper: https://github.com/PY7H0N/musetalk-runpod-api

Observation:

- No official RunPod Pod template or official RunPod MuseTalk template page was found in the sources reviewed for this task.
- The existing public examples are community-maintained and focused on worker or serverless patterns.
- This repository therefore ships a `RunPod Pod + Gradio` scaffold instead of trying to reuse a worker-specific template.

## Files Added

- [Dockerfile.runpod.musetalk](/e:/animation/Dockerfile.runpod.musetalk)
- [scripts/runpod_musetalk_start.sh](/e:/animation/scripts/runpod_musetalk_start.sh)
- [scripts/build_runpod_musetalk_image.ps1](/e:/animation/scripts/build_runpod_musetalk_image.ps1)

## Design Choices

1. The scaffold is isolated from the current realtime avatar API to avoid colliding with the active FasterLivePortrait deployment.
2. The image is built around `CUDA 11.8` because the official MuseTalk README documents installation with `torch 2.0.1` and `torchvision 0.15.2` on `cu118`.
3. The container installs the MuseTalk Python stack at image build time so the Pod does not need to compile the environment on each boot.
4. Model weights are downloaded at Pod startup into `/workspace/musetalk/models` so they can persist on the RunPod volume.
5. The official `app.py` is launched through environment-driven `Gradio` host and port settings.

## Runtime Layout

At runtime the start script maps the official MuseTalk repository directories to persistent workspace paths:

- `/opt/MuseTalk/models` -> `/workspace/musetalk/models`
- `/opt/MuseTalk/results` -> `/workspace/musetalk/results`

This keeps downloaded weights and generated outputs outside the image layer without replacing internal repository assets that MuseTalk may expect inside the original source tree.

## Build The Image

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_runpod_musetalk_image.ps1 -ImageName pailletjp/avatario-musetalk-runpod -Tag gradio -Push
```

If you only want to build locally and skip the push:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_runpod_musetalk_image.ps1 -ImageName pailletjp/avatario-musetalk-runpod -Tag gradio
```

## Recommended RunPod Pod Settings

These values are a practical recommendation for the first pod, not a claim from an official MuseTalk deployment guide:

| Setting | Value |
| --- | --- |
| Container image | `pailletjp/avatario-musetalk-runpod:gradio` |
| GPU | `RTX 4090` or another `24 GB` class GPU |
| Container disk | `25 GB` minimum |
| Volume disk | `30 GB` minimum |
| Volume mount path | `/workspace` |
| HTTP ports | `7860` |
| TCP ports | `22` optional |
| Start command | leave empty |

## Environment Variables

Optional environment variables for the Pod:

```text
MUSETALK_GRADIO_HOST=0.0.0.0
MUSETALK_GRADIO_PORT=7860
MUSETALK_WORKSPACE_ROOT=/workspace/musetalk
MUSETALK_SKIP_WEIGHTS_DOWNLOAD=0
```

## First Start Behavior

On the first Pod boot, the container will:

1. Mount persistent workspace folders through symlinks.
2. Download MuseTalk weights with the official `download_weights.sh`.
3. Launch the official `Gradio` app.

The first boot can take a while because model downloads are large.

## Open The UI

After the Pod is running and the logs show the Gradio app is listening, open:

```text
https://<runpod-http-endpoint>:7860
```

If RunPod exposes a proxy URL for the HTTP port, use that URL instead.

## Notes

- This scaffold is intentionally minimal and aimed at remote validation first.
- It is a Pod-oriented UI path, not a serverless worker path.
- If the official MuseTalk project changes its dependency graph or `app.py` contract, rebuild the image from the latest repo state.
