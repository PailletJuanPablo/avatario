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

## Build offline de visemas (TTS)

Genera un set base de 14 visemas (`sil`, `AA`, `E`, `I`, `O`, `U`, `MBP`, `FV`, `L`, `TH`, `CH`, `SS`, `RR`, `DD`)
con audios WAV normalizados y manifest JSON:

```powershell
python .\scripts\generate_viseme_tts.py `
  --base-image output/frames/frame_00095.png `
  --overwrite
```

Salidas:

- `output_fasterliveportrait/viseme_library/audio/*.wav`
- `output_fasterliveportrait/viseme_library/viseme_audio_manifest.json`

Generar templates `.pkl` por visema (usa Docker `animation_api` por defecto):

```powershell
python .\scripts\build_viseme_pkls.py --overwrite
```

Modo suavidad (mas frames por visema):

```powershell
python .\scripts\build_viseme_pkls.py --overwrite --motion-upsample-factor 2
```

Transiciones visema->visema (offline, todos los pares):

```powershell
python .\scripts\generate_viseme_transition_audio.py --overwrite

python .\scripts\build_viseme_pkls.py `
  --audio-manifest output_fasterliveportrait/viseme_library/viseme_transition_audio_manifest.json `
  --output-dir output_fasterliveportrait/viseme_library/pkl_transitions `
  --output-manifest output_fasterliveportrait/viseme_library/viseme_transition_motion_manifest.json `
  --motion-upsample-factor 2 `
  --overwrite
```

Con 14 visemas se generan 196 transiciones (`A_to_B`), cada una con su `.pkl`.

Si quieres usar esas transiciones en Pixi (tambien como atlas):

```powershell
python .\scripts\build_viseme_clips.py `
  --motion-manifest output_fasterliveportrait/viseme_library/viseme_transition_motion_manifest.json `
  --output-dir output_fasterliveportrait/viseme_library/clips_transitions `
  --output-manifest output_fasterliveportrait/viseme_library/viseme_transition_clip_manifest.json `
  --overwrite

python .\scripts\extract_viseme_frames.py `
  --clip-manifest output_fasterliveportrait/viseme_library/viseme_transition_clip_manifest.json `
  --output-dir output_fasterliveportrait/viseme_library/frames_transitions `
  --output-manifest output_fasterliveportrait/viseme_library/viseme_transition_frames_manifest.json `
  --target-fps 60 `
  --interpolation-mode minterpolate `
  --sharpen-amount 0.28 `
  --overwrite

python .\scripts\build_viseme_atlas.py `
  --frames-manifest output_fasterliveportrait/viseme_library/viseme_transition_frames_manifest.json `
  --motion-manifest output_fasterliveportrait/viseme_library/viseme_transition_motion_manifest.json `
  --output-dir output_fasterliveportrait/viseme_library/atlas_transitions `
  --output-manifest output_fasterliveportrait/viseme_library/pixi_transition_library_manifest.json `
  --overwrite
```

Salidas:

- `output_fasterliveportrait/viseme_library/pkl/*.pkl`
- `output_fasterliveportrait/viseme_library/viseme_motion_manifest.json`

Render de clips por visema (`.pkl -> .mp4`) con base `frame_00095.png`:

```powershell
python .\scripts\build_viseme_clips.py --overwrite
```

Salidas:

- `output_fasterliveportrait/viseme_library/clips/<viseme>/result_org.mp4`
- `output_fasterliveportrait/viseme_library/clips/<viseme>/result_crop.mp4`
- `output_fasterliveportrait/viseme_library/viseme_clip_manifest.json`

Extracción de frames PNG para runtime (con ventanas `attack/hold/release`):

```powershell
python .\scripts\extract_viseme_frames.py --overwrite
```

Modo calidad/suavidad (interpolacion temporal + sharpen leve):

```powershell
python .\scripts\extract_viseme_frames.py --overwrite `
  --target-fps 60 `
  --interpolation-mode minterpolate `
  --sharpen-amount 0.28
```

Salidas:

- `output_fasterliveportrait/viseme_library/frames/<viseme>/frame_*.png`
- `output_fasterliveportrait/viseme_library/viseme_frames_manifest.json`

Generar atlas/spritesheet JSON para Pixi + manifest runtime final:

```powershell
python .\scripts\build_viseme_atlas.py --overwrite
```

Salidas:

- `output_fasterliveportrait/viseme_library/atlas/<viseme>.png`
- `output_fasterliveportrait/viseme_library/atlas/<viseme>.json`
- `output_fasterliveportrait/viseme_library/pixi_library_manifest.json`

## Preview Pixi (runtime sin IA)

Interfaz de validación runtime:

- `pixi_preview.html`

Usa:

- `vendor/pixi/pixi.min.js` (local, sin CDN)
- `output_fasterliveportrait/viseme_library/pixi_library_manifest.json`

Ejecutar servidor estático rápido:

```powershell
python -m http.server 3010 --bind 127.0.0.1
```

Abrir:

```text
http://127.0.0.1:3010/pixi_preview.html
```
