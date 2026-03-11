# Runpod TRT Deployment

## Scope

This runbook deploys the realtime avatar API on a Runpod Pod without Docker-in-Docker.

It uses:

- Runpod Pod networking and storage
- `scripts/runpod_pytorch_quickstart.sh`
- `scripts/runpod_bootstrap.sh`
- `realtime_stream_api.py`
- `faster_liveportrait_runner.py`

The runtime path is `TRT` with `ANIMATION_TRT_RUNTIME=local`.
The supported Pod base is an official Runpod PyTorch image with CUDA enabled.

## Preconditions

1. Create a Runpod Pod with an NVIDIA GPU that has enough VRAM for the project workload.
2. Use an official Runpod PyTorch image with CUDA enabled, for example `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
3. Mount persistent storage at `/workspace`.
4. Expose port `8010/http` for browser access.
5. Expose port `8010/tcp` when you want a direct URL for long-lived websocket sessions.
6. Expose port `22/tcp` only when you want full SSH or SCP access.

## Pod Settings

| Setting | Value |
| --- | --- |
| Container image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Volume mount path | `/workspace` |
| Required HTTP port | `8010/http` |
| Optional TCP port | `8010/tcp` |
| Optional SSH port | `22/tcp` |
| Recommended env | `ANIMATION_API_PORT=8010` |
| Recommended env | `ANIMATION_BACKEND=trt` |
| Recommended env | `ANIMATION_TRT_RUNTIME=local` |
| Recommended env | `ANIMATION_TRT_PRECISION=fp16` |
| Optional env | `ANIMATION_API_TOKEN=<your token>` |
| Optional env | `RUNPOD_ENABLE_SSH=1` |
| Optional env | `PUBLIC_KEY=<your ssh public key>` |
| Optional env | `RUNPOD_GIT_REPO=https://github.com/PailletJuanPablo/avatario.git` |
| Optional env | `RUNPOD_GIT_REF=main` |

## Start Command

Use this as the Pod start command:

```bash
bash -lc 'command -v curl >/dev/null 2>&1 || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates); curl -fsSL https://raw.githubusercontent.com/PailletJuanPablo/avatario/main/scripts/runpod_pytorch_quickstart.sh | bash'
```

If you deploy from a fork or a non-`main` branch, replace the raw GitHub URL in the command so it points to the matching `scripts/runpod_pytorch_quickstart.sh`.

This command:

1. Ensures `curl` is available.
2. Downloads `scripts/runpod_pytorch_quickstart.sh` from GitHub.
3. Clones the repository into `/workspace/animation` when it is missing.
4. Installs missing base packages when needed.
5. Installs TensorRT Python packages when they are not present in the PyTorch image.
6. Bootstraps `third_party/FasterLivePortrait`.
7. Downloads the required checkpoints into `/workspace/animation/third_party/FasterLivePortrait/checkpoints`.
8. Starts `realtime_stream_api.py` with the local TRT runtime.

## Validation

1. Open the Pod logs and wait for the bootstrap to print `Local health URL` and `Token`.
2. Validate the local health endpoint from SSH or the web terminal:

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8010/api/health
```

3. Open the UI through the HTTP proxy when only `8010/http` is exposed:

```text
https://<pod-id>-8010.proxy.runpod.net/?token=<token>
```

4. Prefer the direct TCP URL when `8010/tcp` is also exposed:

```text
http://<public-ip>:<mapped-tcp-port>/?token=<token>
```

The bootstrap script prints both URLs when Runpod provides the required environment variables.

## SSH

Runpod supports a basic proxied SSH flow for Pods. Full SSH and SCP require a public IP and a TCP port mapping for `22/tcp`.

When `RUNPOD_ENABLE_SSH=1` and `PUBLIC_KEY` are set, `scripts/runpod_bootstrap.sh` starts `sshd` inside the container and writes the authorized key for the `root` user.

## Data Persistence

The bootstrap stores runtime state in the persistent volume under `/workspace/animation`:

- repository clone
- generated TRT engines under `third_party/FasterLivePortrait/checkpoints`
- output artifacts under `output/` and `output_fasterliveportrait/`
- generated API token under `.runpod/api_token`

## Stop and Restart

1. Stop the Pod from Runpod when you want to release the GPU.
2. Start the same Pod again to reuse the persistent volume.
3. If `RUNPOD_GIT_AUTO_UPDATE=1` is set, the entrypoint updates the repository checkout on startup.

## Unknowns

- The repository does not include a measured minimum volume size for checkpoints, generated TRT engines, and output artifacts.
  Verification action: allocate persistent storage with headroom for model downloads and generated outputs before the first run.
- The repository does not include a benchmark of long-lived avatar websocket streams through the Runpod HTTP proxy.
  Verification action: expose `8010/tcp` and validate one end-to-end streaming session if the HTTP proxy shows websocket instability.

## References

- Runpod Pod overview: https://docs.runpod.io/pods/overview
- Runpod custom container arguments: https://docs.runpod.io/pods/configuration/launch-custom-image
- Runpod exposed ports: https://docs.runpod.io/pods/configuration/expose-ports
- Runpod SSH access: https://docs.runpod.io/pods/configuration/use-ssh
