# Animation Realtime TRT (Stable)

Versión consolidada y lista para uso directo en Docker.

## Qué queda como implementación activa

- UI: `index.html`
- API: `realtime_stream_api.py`
- Runner: `faster_liveportrait_runner.py`
- Audio → template: `faster_liveportrait_audio_to_pkl.py`
- Motor base: `third_party/FasterLivePortrait`
- Entrada de frame neutro: `output/frames/frame_00061.png` (editable desde la UI)

## Arranque 1-click (Docker)

### 1) Levantar

```powershell
docker compose up --build -d
```

### 2) Abrir

```text
http://127.0.0.1:8010/
```

### 3) Ver logs

```powershell
docker compose logs -f animation-api
```

### 4) Bajar

```powershell
docker compose down
```

## Contrato runtime

- Backend por defecto: TensorRT
- Runtime TRT por defecto: `local` (dentro del contenedor, sin Docker-in-Docker)
- Precisión por defecto: `fp16`
- Warmup habilitado por defecto
- Worker persistente habilitado por defecto (reutiliza modelos entre jobs para reducir latencia)

Variables útiles (en `docker-compose.yml`):

- `ANIMATION_BACKEND` = `trt|onnx`
- `ANIMATION_TRT_RUNTIME` = `local|docker`
- `ANIMATION_TRT_PRECISION` = `fp32|fp16|int8`
- `ANIMATION_WARMUP_ENABLED` = `0|1`
- `ANIMATION_AUDIO_MOTION_STRIDE` = `1..6`

## Limpieza de legacy/output

Limpieza de outputs y logs legacy:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_legacy.ps1
```

Limpieza completa (incluye archivos legacy de código/UI antiguos):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_legacy.ps1 -IncludeLegacyCode
```

## Notas

- Persistencia de trabajo:
  - `output/`
  - `output_fasterliveportrait/`
  - `third_party/FasterLivePortrait/checkpoints/`
  - `third_party/FasterLivePortrait/results/`
- Si no hay GPU NVIDIA disponible en Docker, TensorRT no va a iniciar.
