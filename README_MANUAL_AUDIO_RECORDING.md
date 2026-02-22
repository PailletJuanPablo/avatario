# Manual Audio Recording Contract (Visemes)

This guide defines the exact file names and phonetic targets to record audios manually without breaking the offline pipeline.

## 1) Output folders and file names

- Base viseme audios (14 files):
  - `output_fasterliveportrait/viseme_library/audio/<VISEME>.wav`
- Transition audios (all pair combinations):
  - `output_fasterliveportrait/viseme_library/audio_transitions/<FROM>_to_<TO>.wav`

## 2) Required WAV format

All recorded files must be:

- WAV PCM 16-bit (`pcm_s16le`)
- Mono (`1` channel)
- `16000` Hz

Recommended:

- Base clip duration: `~1.05s`
- Transition clip duration: `~0.78s`
- Stable volume, no clipping, no reverb, no background noise.

## 3) Base viseme list (record each one)

Record one file per viseme with the exact filename below.

| Viseme | Required filename | IPA target (approx) | Recording cue |
|---|---|---|---|
| `sil` | `sil.wav` | silence | No voice. Keep low room noise. |
| `AA` | `AA.wav` | /a/ | Sustained open "aaaaaaa". |
| `E` | `E.wav` | /e/ | Sustained "eeeeeee". |
| `I` | `I.wav` | /i/ | Sustained "iiiiiii". |
| `O` | `O.wav` | /o/ | Sustained rounded "ooooooo". |
| `U` | `U.wav` | /u/ | Sustained rounded "uuuuuuu". |
| `MBP` | `MBP.wav` | /m b p/ | Bilabial closure. Use "mmmmmmm" or soft "mbp" loop. |
| `FV` | `FV.wav` | /f v/ | Labiodental friction. Use "fffffff" (optionally mix with "vvvv"). |
| `L` | `L.wav` | /l/ | Repeated "lalalalalala". |
| `TH` | `TH.wav` | /th/ dental | Repeated "thathathatha" (or similar dental tongue placement). |
| `CH` | `CH.wav` | /t-sh/ | Repeated "chachachacha". |
| `SS` | `SS.wav` | /s/ | Sustained "ssssssss". |
| `RR` | `RR.wav` | /r/ | Sustained or repeated trill "rrrrrrrr". |
| `DD` | `DD.wav` | /d/ | Repeated "dadadadada". |

## 4) Transition audios naming rule

Transitions must follow:

- `<FROM>_to_<TO>.wav`
- Example: `AA_to_O.wav`, `sil_to_MBP.wav`, `RR_to_DD.wav`

Valid viseme keys:

- `sil`, `AA`, `E`, `I`, `O`, `U`, `MBP`, `FV`, `L`, `TH`, `CH`, `SS`, `RR`, `DD`

Total transition files if fully covered:

- `14 x 14 = 196` files.

## 5) Quick quality checklist before pipeline

- Mouth shape is stable during each base viseme.
- No hard attack click at start/end of the file.
- Similar loudness across all files.
- No music, compression pumping, or strong denoise artifacts.

## 6) Next offline steps after recording

Once files are in place, run the existing build pipeline:

1. `audio -> pkl`
2. `pkl -> clips`
3. `clips -> frames`
4. `transition frame interpolation (optical flow)`
5. `atlas + manifest`

Use the commands already documented in `README.md`.
