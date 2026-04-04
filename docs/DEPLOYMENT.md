# Deployment Runbook

## Scope

This repository exposes a Dockerized API and static UI with the following runtime entry points:

- `docker-compose.yml`
- `Dockerfile`
- `realtime_stream_api.py`
- `index.html`

The health endpoint is `GET /api/health`.

## Git Baseline

Generated or machine-specific files are excluded from version control:

- `.env`
- `logs/`
- `output/`
- `output_fasterliveportrait/`
- `output_liveportrait_dataset_poc/`
- `output_liveportrait_test/`
- `.tmp_*`
- `_tmp_*`
- `Microsoft/`
- `liveportrait_dataset_*.json`
- `person.png`
- `sin nombre.mp3`

The repository keeps `inputs/idlevid.mp4` and `inputs/idlevid_breath.mp4` as versioned base assets.

## VM Preconditions

1. Clone the repository onto a Linux VM.
2. Install Docker with the Compose plugin.
3. If `ANIMATION_BACKEND=trt`, provide an NVIDIA GPU runtime compatible with Docker because the Compose service requests `gpus: all`.
4. Ensure outbound network access on the first run so `scripts/bootstrap_faster_liveportrait.sh` can clone `third_party/FasterLivePortrait` when it is missing.
5. Ensure the required model files exist under `third_party/FasterLivePortrait/checkpoints`.

## VM Deployment

1. Review `.env.example`.
2. Create `.env` if it does not exist.
3. Set at least:

```dotenv
ANIMATION_API_PORT=8010
ANIMATION_API_TOKEN=change-me
ANIMATION_BACKEND=trt
ANIMATION_TRT_RUNTIME=local
ANIMATION_TRT_PRECISION=fp16
ANIMATION_IDLE_VIDEO=inputs/idlevid_breath.mp4
```

4. Run the deployment script:

```bash
bash scripts/deploy_vm.sh
```

5. Validate the API:

```bash
curl -H "Authorization: Bearer change-me" \
  http://127.0.0.1:8010/api/health
```

6. Open the UI:

```text
http://<vm-ip>:8010/?token=change-me
```

## Rollback

1. Stop the deployment:

```bash
docker compose down
```

2. Remove generated runtime data only when you want a clean local state:

```bash
rm -rf output output_fasterliveportrait logs
```

## Google Colab Notebook

Use `notebooks/google_colab_animation.ipynb`.

The notebook:

1. Prompts for the Git repository URL at runtime.
2. Clones this repository.
3. Bootstraps `third_party/FasterLivePortrait` when it is missing.
4. Installs the Python dependencies needed by `realtime_stream_api.py`.
5. Downloads the required checkpoints.
6. Starts the API with the ONNX backend for Colab execution.
7. Exposes the UI through the Colab port proxy.

Use a Colab runtime with GPU enabled.

## Unknowns

- The repository does not include OS-specific installation steps for Docker, Docker Compose, or the NVIDIA container runtime.
  Verification action: install those prerequisites according to the target Linux distribution before running `bash scripts/deploy_vm.sh`.
- The repository does not include checkpoint download automation for the VM path.
  Verification action: populate `third_party/FasterLivePortrait/checkpoints` before the first deployment.
