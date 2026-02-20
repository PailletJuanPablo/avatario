#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate base viseme audios and transition audios for the project contract.

Outputs:
1) output_fasterliveportrait/viseme_library/audio/{viseme}.wav
2) output_fasterliveportrait/viseme_library/audio_transitions/{FROM}_to_{TO}.wav
3) output_fasterliveportrait/viseme_library/viseme_audio_manifest.json
4) output_fasterliveportrait/viseme_library/viseme_transition_audio_manifest.json

Final format:
- WAV PCM 16-bit (pcm_s16le)
- mono
- 16 kHz
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_IMAGE = "output/frames/frame_00095.png"
DEFAULT_BASE_DIR = "output_fasterliveportrait/viseme_library"
DEFAULT_AUDIO_DIR = "output_fasterliveportrait/viseme_library/audio"
DEFAULT_TRANSITION_DIR = "output_fasterliveportrait/viseme_library/audio_transitions"
DEFAULT_BASE_MANIFEST = "output_fasterliveportrait/viseme_library/viseme_audio_manifest.json"
DEFAULT_TRANSITION_MANIFEST = "output_fasterliveportrait/viseme_library/viseme_transition_audio_manifest.json"
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_BASE_DURATION_SEC = 1.05
DEFAULT_TRANSITION_DURATION_SEC = 0.78
DEFAULT_SLEEP_SEC = 0.25
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_API_BASE = "https://api.elevenlabs.io"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_BACKOFF_SEC = 1.5

# -------------------------
# ElevenLabs config (requested hardcoded defaults)
# -------------------------
ELEVEN_API_KEY = "sk_2efdd567c2f1c8b95d69af96ac4887fb0bcaea163219a20b"
ELEVEN_VOICE_ID = "TumdjBNWanlT3ysvclWh"
ELEVEN_MODEL_ID = "eleven_multilingual_v2"

VOICE_SETTINGS = {
    "stability": 0.85,
    "similarity_boost": 0.6,
    "style": 0.0,
    "use_speaker_boost": True,
}


@dataclass(frozen=True)
class VisemeSpec:
    """Viseme definition and generation prompt."""

    key: str
    prompt: str


VISEME_SPECS: tuple[VisemeSpec, ...] = (
    VisemeSpec("sil", ""),
    VisemeSpec("AA", "aaaaaaa"),
    VisemeSpec("E", "eeeeeee"),
    VisemeSpec("I", "iiiiiii"),
    VisemeSpec("O", "ooooooo"),
    VisemeSpec("U", "uuuuuuu"),
    VisemeSpec("MBP", "mmmmmmm"),
    VisemeSpec("FV", "fffffff"),
    VisemeSpec("L", "lalalalalala"),
    VisemeSpec("TH", "tatatatata"),
    VisemeSpec("CH", "chachachacha"),
    VisemeSpec("SS", "ssssssss"),
    VisemeSpec("RR", "rrrrrrrr"),
    VisemeSpec("DD", "dadadadada"),
)

VISEME_KEYS: tuple[str, ...] = tuple(spec.key for spec in VISEME_SPECS)
VISEME_PROMPT_BY_KEY = {spec.key: spec.prompt for spec in VISEME_SPECS}


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="Generate ElevenLabs viseme and transition audio libraries.")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--transition-dir", default=DEFAULT_TRANSITION_DIR)
    parser.add_argument("--base-manifest", default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--transition-manifest", default=DEFAULT_TRANSITION_MANIFEST)
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--base-duration-sec", type=float, default=DEFAULT_BASE_DURATION_SEC)
    parser.add_argument("--transition-duration-sec", type=float, default=DEFAULT_TRANSITION_DURATION_SEC)
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff-sec", type=float, default=DEFAULT_RETRY_BACKOFF_SEC)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--api-key", default=os.environ.get("ELEVENLABS_API_KEY", ELEVEN_API_KEY))
    parser.add_argument("--voice-id", default=os.environ.get("ELEVENLABS_VOICE_ID", ELEVEN_VOICE_ID))
    parser.add_argument("--model-id", default=os.environ.get("ELEVENLABS_MODEL_ID", ELEVEN_MODEL_ID))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """Resolve path relative to project root when needed."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """Convert absolute path to project-relative POSIX path when possible."""
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def ensure_ffmpeg() -> None:
    """Validate ffmpeg availability."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required in PATH.")


def run_command(command: list[str]) -> None:
    """Run command and raise on error."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")


def normalize_to_wav_16k_mono(input_path: Path, output_path: Path, sample_rate_hz: int, duration_sec: float) -> None:
    """Normalize arbitrary audio to wav mono 16k pcm_s16le with fixed duration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
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
    run_command(command)


def create_silence_wav(output_path: Path, sample_rate_hz: int, duration_sec: float) -> None:
    """Generate silence wav directly."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={int(sample_rate_hz)}:cl=mono",
        "-t",
        f"{float(duration_sec):.3f}",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    run_command(command)


def build_transition_prompt(from_viseme: str, to_viseme: str) -> str:
    """Build prompt for viseme transition."""
    if from_viseme == "sil" and to_viseme == "sil":
        return ""
    if from_viseme == "sil":
        return VISEME_PROMPT_BY_KEY[to_viseme]
    if to_viseme == "sil":
        return VISEME_PROMPT_BY_KEY[from_viseme]
    if from_viseme == to_viseme:
        return VISEME_PROMPT_BY_KEY[from_viseme]
    return f"{VISEME_PROMPT_BY_KEY[from_viseme]}{VISEME_PROMPT_BY_KEY[to_viseme]}"


def request_elevenlabs_audio(
    api_base: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
    text: str,
) -> bytes:
    """Request raw audio bytes from ElevenLabs."""
    if not api_key.strip():
        raise RuntimeError("ELEVENLABS_API_KEY is required.")
    if not voice_id.strip():
        raise RuntimeError("ELEVENLABS_VOICE_ID is required.")

    url = f"{api_base.rstrip('/')}/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Accept": "application/octet-stream",
        "Content-Type": "application/json",
    }
    params = {"output_format": output_format}
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": VOICE_SETTINGS,
    }
    safe_max_retries = max(0, int(max_retries))
    safe_backoff = max(0.1, float(retry_backoff_sec))
    attempt = 0
    while True:
        attempt += 1
        response = requests.post(url, headers=headers, params=params, json=payload, timeout=int(timeout_sec))
        if response.status_code < 400:
            return response.content
        retryable = response.status_code in {429, 500, 502, 503, 504}
        if retryable and attempt <= safe_max_retries:
            wait_seconds = safe_backoff * (2 ** (attempt - 1))
            print(
                f"[retry] elevenlabs status={response.status_code} "
                f"attempt={attempt}/{safe_max_retries} wait={wait_seconds:.2f}s"
            )
            time.sleep(wait_seconds)
            continue
        error_payload = (response.text or "")[:500]
        raise RuntimeError(f"ElevenLabs request failed ({response.status_code}): {error_payload}")


def write_bytes_atomic(path_value: Path, payload: bytes) -> None:
    """Write bytes atomically."""
    path_value.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path_value.with_suffix(path_value.suffix + ".tmp")
    temp_path.write_bytes(payload)
    os.replace(temp_path, path_value)


def generate_audio_file(
    output_wav: Path,
    prompt: str,
    duration_sec: float,
    sample_rate_hz: int,
    overwrite: bool,
    api_base: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
    temp_dir: Path,
) -> bool:
    """Generate one viseme wav and return True when file was created/overwritten."""
    if output_wav.exists() and output_wav.stat().st_size > 1024 and not overwrite:
        return False

    if not prompt.strip():
        create_silence_wav(output_wav, sample_rate_hz, duration_sec)
        return True

    raw_path = temp_dir / f"{output_wav.stem}.rawaudio"
    audio_bytes = request_elevenlabs_audio(
        api_base=api_base,
        api_key=api_key,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
        text=prompt,
    )
    write_bytes_atomic(raw_path, audio_bytes)
    normalize_to_wav_16k_mono(raw_path, output_wav, sample_rate_hz, duration_sec)
    return True


def main() -> None:
    """Run base + transition audio generation and write manifests."""
    args = parse_args()
    ensure_ffmpeg()

    base_image_path = resolve_path(args.base_image)
    base_dir = resolve_path(args.base_dir)
    audio_dir = resolve_path(args.audio_dir)
    transition_dir = resolve_path(args.transition_dir)
    base_manifest_path = resolve_path(args.base_manifest)
    transition_manifest_path = resolve_path(args.transition_manifest)
    sample_rate_hz = max(8000, int(args.sample_rate_hz))
    base_duration_sec = max(0.3, float(args.base_duration_sec))
    transition_duration_sec = max(0.3, float(args.transition_duration_sec))
    sleep_sec = max(0.0, float(args.sleep_sec))
    timeout_sec = max(10, int(args.timeout_sec))
    api_key = str(args.api_key)
    voice_id = str(args.voice_id)
    model_id = str(args.model_id)
    api_base = str(args.api_base)
    output_format = str(args.output_format)
    max_retries = max(0, int(args.max_retries))
    retry_backoff_sec = max(0.1, float(args.retry_backoff_sec))

    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    if not api_key.strip():
        raise RuntimeError("ELEVENLABS_API_KEY is empty.")
    if not voice_id.strip():
        raise RuntimeError("ELEVENLABS_VOICE_ID is empty.")

    audio_dir.mkdir(parents=True, exist_ok=True)
    transition_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / "_tmp_eleven_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)

    base_entries: list[dict[str, Any]] = []
    for index, spec in enumerate(VISEME_SPECS, start=1):
        output_wav = audio_dir / f"{spec.key}.wav"
        print(f"[base] {index}/{len(VISEME_SPECS)} {output_wav.name}")
        generated = generate_audio_file(
            output_wav=output_wav,
            prompt=spec.prompt,
            duration_sec=base_duration_sec,
            sample_rate_hz=sample_rate_hz,
            overwrite=bool(args.overwrite),
            api_base=api_base,
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            temp_dir=temp_dir,
        )
        if generated and sleep_sec > 0:
            time.sleep(sleep_sec)
        base_entries.append(
            {
                "viseme": spec.key,
                "phrase": spec.prompt,
                "audio": to_project_relative(output_wav),
            }
        )

    transition_entries: list[dict[str, Any]] = []
    total_transitions = len(VISEME_KEYS) * len(VISEME_KEYS)
    counter = 0
    for from_viseme in VISEME_KEYS:
        for to_viseme in VISEME_KEYS:
            counter += 1
            viseme_key = f"{from_viseme}_to_{to_viseme}"
            output_wav = transition_dir / f"{viseme_key}.wav"
            print(f"[trn ] {counter}/{total_transitions} {output_wav.name}")
            generated = generate_audio_file(
                output_wav=output_wav,
                prompt=build_transition_prompt(from_viseme, to_viseme),
                duration_sec=transition_duration_sec,
                sample_rate_hz=sample_rate_hz,
                overwrite=bool(args.overwrite),
                api_base=api_base,
                api_key=api_key,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
                temp_dir=temp_dir,
            )
            if generated and sleep_sec > 0:
                time.sleep(sleep_sec)
            transition_entries.append(
                {
                    "viseme": viseme_key,
                    "fromViseme": from_viseme,
                    "toViseme": to_viseme,
                    "audio": to_project_relative(output_wav),
                }
            )

    shutil.rmtree(temp_dir, ignore_errors=True)

    base_manifest_payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "baseImage": to_project_relative(base_image_path),
        "sampleRateHz": sample_rate_hz,
        "targetDurationSec": base_duration_sec,
        "tts": {
            "engine": "elevenlabs",
            "voiceId": voice_id,
            "modelId": model_id,
            "outputFormat": output_format,
            "voiceSettings": {
                "stability": VOICE_SETTINGS["stability"],
                "similarityBoost": VOICE_SETTINGS["similarity_boost"],
                "style": VOICE_SETTINGS["style"],
                "useSpeakerBoost": VOICE_SETTINGS["use_speaker_boost"],
            },
        },
        "visemeCount": len(base_entries),
        "visemes": base_entries,
    }
    base_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with base_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(base_manifest_payload, handle, indent=2)

    transition_manifest_payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceVisemeAudioManifest": to_project_relative(base_manifest_path),
        "baseImage": to_project_relative(base_image_path),
        "sampleRateHz": sample_rate_hz,
        "targetDurationSec": transition_duration_sec,
        "includeSelfTransitions": True,
        "transitionCount": len(transition_entries),
        "visemeCount": len(transition_entries),
        "outputDir": to_project_relative(transition_dir),
        "visemes": transition_entries,
        "transitions": transition_entries,
    }
    transition_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with transition_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(transition_manifest_payload, handle, indent=2)

    print(f"[ok] base manifest -> {base_manifest_path}")
    print(f"[ok] transition manifest -> {transition_manifest_path}")
    print(f"[ok] base audios -> {len(base_entries)}")
    print(f"[ok] transition audios -> {len(transition_entries)}")


if __name__ == "__main__":
    main()
