# PKL Video Mini Framework

Flujo offline, centrado en `pkl`, para generar video en 3 pasos:

1. `audio -> pkl`
2. `pkl -> pkl tuned` (ojos)
3. `pkl tuned -> mp4`

Todo queda en `scripts/pkl_video_framework.py`.

## 1) Build PKL from audio

```powershell
python scripts/pkl_video_framework.py build-pkl `
  --audio output_fasterliveportrait/quick_check/aa_long_single/audio/AA.wav `
  --output-pkl output_fasterliveportrait/quick_check/aa_long_single/pkl/AA_raw.pkl
```

## 2) Tune eyes in PKL (recommended in Docker mode)

```powershell
python scripts/pkl_video_framework.py tune-pkl `
  --input-pkl output_fasterliveportrait/quick_check/aa_long_single/pkl/AA_raw.pkl `
  --output-pkl output_fasterliveportrait/quick_check/aa_long_single/pkl/AA_eye_tamed.pkl `
  --runtime docker
```

Default tuning profile:
- Soft upper-face damping: `0.45`
- Hard eye channels damping: `0.18`
- Hard vertical clamp: `[-0.0045, 0.0035]`

## 3) Render from PKL

```powershell
python scripts/pkl_video_framework.py render-pkl `
  --pkl output_fasterliveportrait/quick_check/aa_long_single/pkl/AA_eye_tamed.pkl `
  --output-dir output_fasterliveportrait/quick_check/aa_long_single/render_eye_tamed `
  --source-image output/frames/frame_00095.png
```

## End-to-end helper

```powershell
python scripts/pkl_video_framework.py build-tune-render `
  --audio output_fasterliveportrait/quick_check/aa_long_single/audio/AA.wav `
  --work-dir output_fasterliveportrait/quick_check/aa_framework_run `
  --source-image output/frames/frame_00095.png
```

Outputs:
- `motion_raw.pkl`
- `motion_eye_tamed.pkl`
- `render/*.mp4`

