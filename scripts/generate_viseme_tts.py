"""
Generate offline viseme TTS audio clips and a manifest for library builds.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_IMAGE = "output/frames/frame_00095.png"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/audio"
DEFAULT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_audio_manifest.json"
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_DURATION_SEC = 1.05
DEFAULT_TTS_RATE = -2
DEFAULT_TTS_VOLUME = 100


@dataclass(frozen=True)
class VisemeSpec:
    """
    Viseme definition used for TTS generation.
    """

    key: str
    phrase: str


DEFAULT_VISEME_SPECS: tuple[VisemeSpec, ...] = (
    VisemeSpec("sil", ""),
    VisemeSpec("AA", "aaaa aaaa aaaa"),
    VisemeSpec("E", "eeee eeee eeee"),
    VisemeSpec("I", "iiii iiii iiii"),
    VisemeSpec("O", "oooo oooo oooo"),
    VisemeSpec("U", "uuuu uuuu uuuu"),
    VisemeSpec("MBP", "mmm mmm mmm"),
    VisemeSpec("FV", "fafa fefe fifi"),
    VisemeSpec("L", "lala lala lala"),
    VisemeSpec("TH", "thaa thee thoo"),
    VisemeSpec("CH", "chaa chee choo"),
    VisemeSpec("SS", "ssss ssss ssss"),
    VisemeSpec("RR", "rrrr rrrr rrrr"),
    VisemeSpec("DD", "dada dede didi"),
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(description="Generate viseme TTS audio set for offline avatar library.")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--tts-rate", type=int, default=DEFAULT_TTS_RATE, help="Windows SAPI speech rate [-10..10].")
    parser.add_argument("--tts-volume", type=int, default=DEFAULT_TTS_VOLUME, help="Windows SAPI volume [0..100].")
    parser.add_argument("--voice-name", default="", help="Optional Windows SAPI voice name.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve to absolute path using project root for relative values.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """
    Convert absolute path to project-relative POSIX path when possible.
    """
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def shell_escape_ps_literal(value: str) -> str:
    """
    Escape single quotes for PowerShell single-quoted literals.
    """
    return value.replace("'", "''")


def run_powershell_script(script: str) -> None:
    """
    Execute PowerShell script using UTF-16LE encoded command.
    """
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(f"PowerShell TTS failed. stdout={stdout} stderr={stderr}")


def synthesize_with_windows_tts(
    output_path: Path,
    phrase: str,
    voice_name: str,
    tts_rate: int,
    tts_volume: int,
) -> None:
    """
    Synthesize a WAV file with Windows SAPI TTS.
    """
    output_literal = shell_escape_ps_literal(str(output_path))
    phrase_literal = shell_escape_ps_literal(phrase)
    voice_stmt = ""
    if voice_name.strip():
        voice_literal = shell_escape_ps_literal(voice_name.strip())
        voice_stmt = f"$synth.SelectVoice('{voice_literal}');"
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"{voice_stmt}"
        f"$synth.Rate = {int(tts_rate)};"
        f"$synth.Volume = {int(tts_volume)};"
        f"$synth.SetOutputToWaveFile('{output_literal}');"
        f"$synth.Speak('{phrase_literal}');"
        "$synth.Dispose();"
    )
    run_powershell_script(script)


def normalize_audio_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    sample_rate_hz: int,
    duration_sec: float,
) -> None:
    """
    Normalize to mono PCM16 WAV with fixed target duration.
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is required but not found in PATH.")
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate_hz)),
        "-af",
        "apad",
        "-t",
        f"{float(duration_sec):.3f}",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed: {(result.stderr or '').strip()}")


def generate_silence_wav(output_path: Path, sample_rate_hz: int, duration_sec: float) -> None:
    """
    Generate silent mono PCM16 WAV file.
    """
    frame_count = max(1, int(round(float(sample_rate_hz) * float(duration_sec))))
    silence = b"\x00\x00" * frame_count
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate_hz))
        handle.writeframes(silence)


def main() -> None:
    """
    Generate viseme audio set and manifest.
    """
    args = parse_args()
    base_image_path = resolve_path(args.base_image)
    output_dir = resolve_path(args.output_dir)
    manifest_path = resolve_path(args.manifest_path)
    sample_rate_hz = max(8000, int(args.sample_rate_hz))
    duration_sec = max(0.3, float(args.duration_sec))
    tts_rate = max(-10, min(10, int(args.tts_rate)))
    tts_volume = max(0, min(100, int(args.tts_volume)))

    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    viseme_entries: list[dict[str, object]] = []
    for spec in DEFAULT_VISEME_SPECS:
        output_wav = output_dir / f"{spec.key}.wav"
        if output_wav.exists() and not args.overwrite:
            print(f"[skip] {spec.key} already exists: {output_wav}")
        else:
            if spec.key == "sil":
                generate_silence_wav(output_wav, sample_rate_hz, duration_sec)
            else:
                raw_wav = tmp_dir / f"{spec.key}_raw.wav"
                synthesize_with_windows_tts(
                    output_path=raw_wav,
                    phrase=spec.phrase,
                    voice_name=args.voice_name,
                    tts_rate=tts_rate,
                    tts_volume=tts_volume,
                )
                normalize_audio_with_ffmpeg(
                    input_path=raw_wav,
                    output_path=output_wav,
                    sample_rate_hz=sample_rate_hz,
                    duration_sec=duration_sec,
                )
            print(f"[ok] {spec.key} -> {output_wav}")

        viseme_entries.append(
            {
                "viseme": spec.key,
                "phrase": spec.phrase,
                "audio": to_project_relative(output_wav),
            }
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "baseImage": to_project_relative(base_image_path),
        "sampleRateHz": sample_rate_hz,
        "targetDurationSec": duration_sec,
        "tts": {
            "engine": "windows-sapi",
            "voiceName": args.voice_name.strip(),
            "rate": tts_rate,
            "volume": tts_volume,
        },
        "visemeCount": len(viseme_entries),
        "visemes": viseme_entries,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[ok] manifest -> {manifest_path}")
    print(f"[ok] viseme set size -> {len(viseme_entries)}")


if __name__ == "__main__":
    main()

