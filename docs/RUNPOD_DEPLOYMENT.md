# Runpod TRT Deployment

## Scope

This runbook deploys the realtime avatar API on Runpod Pods with `TensorRT` as the only supported inference backend for this flow.

The deployment assets are:

- `Dockerfile.runpod`
- `requirements-runpod.txt`
- `scripts/runpod_bootstrap.sh`
- `scripts/runpod_pytorch_quickstart.sh`
- `scripts/runpod_image_start.sh`

`ONNX` artifacts are still required because FasterLivePortrait builds local TensorRT engines from `checkpoints/liveportrait_onnx/*.onnx`. This runbook does not use `ONNX Runtime` as the active inference backend.
The bootstrap downloads three checkpoint sources:

- `warmshao/FasterLivePortrait`
- `jdh-algo/JoyVASA`
- `TencentGameMate/chinese-hubert-base`

## Verified Repository Facts

| Fact | Source |
| --- | --- |
| The API defaults to the `trt` backend when `ANIMATION_BACKEND` is unset. | [realtime_stream_api.py](/e:/animation/realtime_stream_api.py#L225) |
| The local TRT flow builds engines from `liveportrait_onnx/*.onnx` files. | [faster_liveportrait_runner.py](/e:/animation/faster_liveportrait_runner.py#L1369) |
| The TensorRT build script imports `onnx`, `pycuda`, and `tensorrt`. | [third_party/FasterLivePortrait/scripts/onnx2trt.py](/e:/animation/third_party/FasterLivePortrait/scripts/onnx2trt.py#L15) |
| The FasterLivePortrait runtime imports `cv2`, `ffmpeg`, `PIL`, `tqdm`, `insightface`, `mediapipe`, `onnxruntime`, and `torchgeometry` in the TRT execution path. | [third_party/FasterLivePortrait/run.py](/e:/animation/third_party/FasterLivePortrait/run.py#L27), [third_party/FasterLivePortrait/src/models/__init__.py](/e:/animation/third_party/FasterLivePortrait/src/models/__init__.py#L7), [third_party/FasterLivePortrait/src/models/predictor.py](/e:/animation/third_party/FasterLivePortrait/src/models/predictor.py#L7), [third_party/FasterLivePortrait/src/utils/crop.py](/e:/animation/third_party/FasterLivePortrait/src/utils/crop.py#L14) |

## Recommended Path

Use a custom Runpod image built from `Dockerfile.runpod`.

This avoids:

- reinstalling TensorRT Python packages on every Pod start
- discovering missing runtime modules one by one
- paying Pod time for large dependency builds before the API starts

## Build the Custom Image

Build and push the image from a machine with Docker access:

```bash
docker build -f Dockerfile.runpod -t <dockerhub-user>/avatario-runpod:latest .
docker push <dockerhub-user>/avatario-runpod:latest
```

## Pod Settings

Use these Pod settings in Runpod:

| Setting | Value |
| --- | --- |
| Container image | `<dockerhub-user>/avatario-runpod:latest` |
| GPU | `RTX 4000 Ada` or another NVIDIA GPU with sufficient VRAM |
| Container disk | `20 GB` minimum |
| Volume disk | `0 GB` for disposable tests, persistent volume when you want cached checkpoints and TRT engines |
| Volume mount path | `/workspace` |
| HTTP ports | optional `8888` only if you want Jupyter |
| TCP ports | `22`, `8010` |

Do not configure `8010` as both HTTP and TCP in the same template. Runpod rejects duplicate port declarations.

## Start Command

Preferred start command for the custom image:

```bash
bash /app/scripts/runpod_image_start.sh
```

This start script launches the base Runpod image services through `/start.sh`, copies the image contents into `/workspace/animation`, and then runs the bootstrap from there. That keeps the image aligned with the base Runpod startup path while still allowing checkpoint and TensorRT engine reuse when a persistent Runpod volume is attached.

Fallback start command for the official Runpod PyTorch image:

```bash
bash -lc 'command -v curl >/dev/null 2>&1 || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates); curl -fsSL https://raw.githubusercontent.com/PailletJuanPablo/avatario/main/scripts/runpod_pytorch_quickstart.sh | bash'
```

The fallback path is slower because it installs Python runtime dependencies and TensorRT packages inside the Pod.

## Environment Variables

Recommended variables:

| Variable | Value |
| --- | --- |
| `ANIMATION_API_PORT` | `8010` |
| `ANIMATION_BACKEND` | `trt` |
| `ANIMATION_TRT_RUNTIME` | `local` |
| `ANIMATION_TRT_PRECISION` | `fp16` |
| `ANIMATION_VIDEO_ENCODER` | `cpu` |
| `ANIMATION_API_TOKEN` | optional fixed token |

`scripts/runpod_bootstrap.sh` exits immediately when `ANIMATION_BACKEND` is not `trt`.

## Validation Procedure

1. Confirm CUDA access before any large install step:

```bash
python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_0 =", torch.cuda.get_device_name(0))
PY
```

2. Start the bootstrap.

3. Wait until the script prints:

- `Local health URL`
- `Token`
- optional `Direct TCP UI URL`

4. Validate the API locally:

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8010/api/health
```

5. Open the direct TCP URL when `8010/tcp` is exposed:

```text
http://<public-ip>:<mapped-port>/?token=<token>
```

Direct TCP is preferred for long-lived websocket sessions.

## Failure Policy

Terminate the Pod immediately when either of these checks fails:

- `torch.cuda.is_available()` returns `False`
- `curl http://127.0.0.1:8010/api/health` does not become healthy after bootstrap

Use the application log for triage:

```bash
tail -n 120 /app/output_fasterliveportrait/runpod_api.log
```

## References

- Runpod Pod templates: https://docs.runpod.io/pods/templates/create-custom-template
- Runpod template management: https://docs.runpod.io/pods/templates/manage-templates
- Runpod SSH for Pods: https://docs.runpod.io/pods/configuration/use-ssh
- Runpod exposed ports: https://docs.runpod.io/pods/configuration/expose-ports
