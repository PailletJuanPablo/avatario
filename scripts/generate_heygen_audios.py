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
DEFAULT_BASE_DURATION_SEC = 1.35
DEFAULT_TRANSITION_DURATION_SEC = 0.78
DEFAULT_SLEEP_SEC = 0.25
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_API_BASE = "https://api.elevenlabs.io"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_BACKOFF_SEC = 1.5
DEFAULT_LANGUAGE = "es"
DEFAULT_SIL_BASE_DURATION_SEC = 20.0
DEFAULT_TRANSITION_PAIR_REPEATS = 6
DEFAULT_PROMPT_AUDIT_PATH = "output_fasterliveportrait/viseme_library/viseme_prompt_audit.json"

# -------------------------
# ElevenLabs config defaults
# -------------------------
ELEVEN_API_KEY = ""
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


VISEME_SPECS_EN: tuple[VisemeSpec, ...] = (
    VisemeSpec("sil", ""),
    VisemeSpec("AA", "ah ah ah ah ah"),
    VisemeSpec("E", "eh eh eh eh eh"),
    VisemeSpec("I", "ee ee ee ee ee"),
    VisemeSpec("O", "oh oh oh oh oh"),
    VisemeSpec("U", "oo oo oo oo oo"),
    VisemeSpec("MBP", "mama baba papa mama baba papa"),
    VisemeSpec("FV", "fife five fife five"),
    VisemeSpec("L", "la la la la la la"),
    VisemeSpec("TH", "think this think this"),
    VisemeSpec("CH", "cheese chair cheese chair"),
    VisemeSpec("SS", "see soup see soup"),
    VisemeSpec("RR", "red road red road"),
    VisemeSpec("DD", "did do did do"),
)

VISEME_SPECS_ES: tuple[VisemeSpec, ...] = (
    VisemeSpec("sil", ""),
    VisemeSpec("AA", "aaaaaa aaaaaa aaaaaa"),
    VisemeSpec("E", "eeeeee eeeeee eeeeee"),
    VisemeSpec("I", "iiiiii iiiiii iiiiii"),
    VisemeSpec("O", "oooooo oooooo oooooo"),
    VisemeSpec("U", "uuuuuu uuuuuu uuuuuu"),
    VisemeSpec("MBP", "ma ba pa ma ba pa"),
    VisemeSpec("FV", "fa fe fi fo fu"),
    VisemeSpec("L", "la le li lo lu"),
    VisemeSpec("TH", "ta da ta da ta da"),
    VisemeSpec("CH", "cha che chi cho chu"),
    VisemeSpec("SS", "sa se si so su"),
    VisemeSpec("RR", "rra rre rri rro rru"),
    VisemeSpec("DD", "da de di do du"),
)

VISEME_SPECS_BY_LANGUAGE: dict[str, tuple[VisemeSpec, ...]] = {
    "en": VISEME_SPECS_EN,
    "es": VISEME_SPECS_ES,
}

VOWEL_VISEME_KEYS: tuple[str, ...] = ("AA", "E", "I", "O", "U")

VOWEL_TOKEN_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "en": {
        "AA": "ah",
        "E": "eh",
        "I": "ee",
        "O": "oh",
        "U": "oo",
    },
    "es": {
        "AA": "a",
        "E": "e",
        "I": "i",
        "O": "o",
        "U": "u",
    },
}

CONSONANT_TOKEN_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "en": {
        "MBP": "m",
        "FV": "f",
        "L": "l",
        "TH": "th",
        "CH": "ch",
        "SS": "s",
        "RR": "r",
        "DD": "d",
    },
    "es": {
        "MBP": "m",
        "FV": "f",
        "L": "l",
        "TH": "t",
        "CH": "ch",
        "SS": "s",
        "RR": "r",
        "DD": "d",
    },
}

ALLOWED_CONSONANT_CLUSTERS_BY_LANGUAGE: dict[str, set[str]] = {
    "en": {
        "fl",
        "fr",
        "tr",
        "dr",
        "thr",
    },
    "es": {
        "fl",
        "fr",
        "tr",
        "dr",
    },
}

SPECIAL_TRANSITION_PROMPTS_BY_LANGUAGE: dict[str, dict[tuple[str, str], str]] = {
    "en": {
        ("DD", "RR"): "dra dre dri dro dru",
    },
    "es": {
        ("DD", "RR"): "dra dre dri dro dru",
    },
}


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
    parser.add_argument("--sil-base-duration-sec", type=float, default=DEFAULT_SIL_BASE_DURATION_SEC)
    parser.add_argument("--transition-duration-sec", type=float, default=DEFAULT_TRANSITION_DURATION_SEC)
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff-sec", type=float, default=DEFAULT_RETRY_BACKOFF_SEC)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--language", choices=["es", "en"], default=DEFAULT_LANGUAGE)
    parser.add_argument("--transition-pair-repeats", type=int, default=DEFAULT_TRANSITION_PAIR_REPEATS)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--prompt-audit-path", default=DEFAULT_PROMPT_AUDIT_PATH)
    parser.add_argument("--generate-base-only", action="store_true")
    parser.add_argument("--generate-transitions-only", action="store_true")
    parser.add_argument(
        "--base-viseme-keys",
        default="",
        help="Comma-separated viseme keys for base generation. Empty means all.",
    )
    parser.add_argument(
        "--transition-from-keys",
        default="",
        help="Comma-separated source viseme keys for transition generation. Empty means all.",
    )
    parser.add_argument(
        "--transition-to-keys",
        default="",
        help="Comma-separated target viseme keys for transition generation. Empty means all.",
    )
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


def parse_key_filter(raw_value: str, allowed_keys: tuple[str, ...], field_name: str) -> set[str]:
    """Parse a comma-separated key filter and validate every key."""
    token_list = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not token_list:
        return set(allowed_keys)
    allowed_key_set = set(allowed_keys)
    invalid = [token for token in token_list if token not in allowed_key_set]
    if invalid:
        raise ValueError(f"Invalid {field_name}: {','.join(invalid)}")
    return set(token_list)


def merge_base_entries_with_existing(
    manifest_path: Path,
    selected_key_set: set[str],
    all_key_order: tuple[str, ...],
    new_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge partial base updates with existing manifest entries."""
    if len(selected_key_set) == len(all_key_order):
        return new_entries
    existing_entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("visemes"), list):
            existing_entries = [item for item in payload["visemes"] if isinstance(item, dict)]
    merged_by_key: dict[str, dict[str, Any]] = {}
    for item in existing_entries:
        viseme_key = str(item.get("viseme", "")).strip()
        if viseme_key and viseme_key not in selected_key_set:
            merged_by_key[viseme_key] = item
    for item in new_entries:
        viseme_key = str(item.get("viseme", "")).strip()
        if viseme_key:
            merged_by_key[viseme_key] = item
    ordered_entries: list[dict[str, Any]] = []
    for viseme_key in all_key_order:
        if viseme_key in merged_by_key:
            ordered_entries.append(merged_by_key[viseme_key])
    return ordered_entries


def merge_transition_entries_with_existing(
    manifest_path: Path,
    selected_from_set: set[str],
    selected_to_set: set[str],
    all_key_order: tuple[str, ...],
    new_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge partial transition updates with existing manifest entries."""
    is_full_update = len(selected_from_set) == len(all_key_order) and len(selected_to_set) == len(all_key_order)
    if is_full_update:
        return new_entries
    existing_entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("visemes"), list):
            existing_entries = [item for item in payload["visemes"] if isinstance(item, dict)]
    merged_by_key: dict[str, dict[str, Any]] = {}
    for item in existing_entries:
        from_viseme = str(item.get("fromViseme", "")).strip()
        to_viseme = str(item.get("toViseme", "")).strip()
        viseme_key = str(item.get("viseme", "")).strip()
        if not viseme_key:
            continue
        if from_viseme in selected_from_set and to_viseme in selected_to_set:
            continue
        merged_by_key[viseme_key] = item
    for item in new_entries:
        viseme_key = str(item.get("viseme", "")).strip()
        if viseme_key:
            merged_by_key[viseme_key] = item
    ordered_entries: list[dict[str, Any]] = []
    for from_viseme in all_key_order:
        for to_viseme in all_key_order:
            viseme_key = f"{from_viseme}_to_{to_viseme}"
            if viseme_key in merged_by_key:
                ordered_entries.append(merged_by_key[viseme_key])
    return ordered_entries


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


def build_transition_prompt(
    from_viseme: str,
    to_viseme: str,
    viseme_prompt_by_key: dict[str, str],
    language: str,
    transition_pair_repeats: int,
) -> str:
    """Build prompt for viseme transition."""
    if from_viseme == "sil" and to_viseme == "sil":
        return ""
    if from_viseme == "sil":
        return viseme_prompt_by_key[to_viseme]
    if to_viseme == "sil":
        return viseme_prompt_by_key[from_viseme]
    if from_viseme == to_viseme:
        return viseme_prompt_by_key[from_viseme]
    special_prompt = SPECIAL_TRANSITION_PROMPTS_BY_LANGUAGE.get(language, {}).get((from_viseme, to_viseme), "")
    if special_prompt:
        return special_prompt

    vowel_tokens = VOWEL_TOKEN_BY_LANGUAGE[language]
    consonant_tokens = CONSONANT_TOKEN_BY_LANGUAGE[language]
    from_is_vowel = from_viseme in VOWEL_VISEME_KEYS
    to_is_vowel = to_viseme in VOWEL_VISEME_KEYS
    repeats = max(2, int(transition_pair_repeats))

    if from_is_vowel and to_is_vowel:
        from_vowel = vowel_tokens[from_viseme]
        to_vowel = vowel_tokens[to_viseme]
        pair = f"{from_vowel}{to_vowel}"
        return " ".join(pair for _ in range(repeats))

    if from_is_vowel and not to_is_vowel:
        from_vowel = vowel_tokens[from_viseme]
        to_consonant = consonant_tokens[to_viseme]
        pair = f"{from_vowel}{to_consonant}"
        return " ".join(pair for _ in range(repeats))

    if (not from_is_vowel) and to_is_vowel:
        from_consonant = consonant_tokens[from_viseme]
        to_vowel = vowel_tokens[to_viseme]
        pair = f"{from_consonant}{to_vowel}"
        return " ".join(pair for _ in range(repeats))

    from_consonant = consonant_tokens[from_viseme]
    to_consonant = consonant_tokens[to_viseme]
    cluster = f"{from_consonant}{to_consonant}"
    vowels = tuple(vowel_tokens[key] for key in VOWEL_VISEME_KEYS)
    allowed_clusters = ALLOWED_CONSONANT_CLUSTERS_BY_LANGUAGE.get(language, set())

    if cluster in allowed_clusters:
        return " ".join(f"{cluster}{vowel}" for vowel in vowels)

    # For non-natural clusters, keep both articulations in one syllabic unit.
    # Example: d->s => "dasa dese disi doso dusu"
    return " ".join(f"{from_consonant}{vowel}{to_consonant}{vowel}" for vowel in vowels)


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
    if bool(args.generate_base_only) and bool(args.generate_transitions_only):
        raise ValueError("Use only one of --generate-base-only or --generate-transitions-only.")
    if not bool(args.audit_only):
        ensure_ffmpeg()

    base_image_path = resolve_path(args.base_image)
    base_dir = resolve_path(args.base_dir)
    audio_dir = resolve_path(args.audio_dir)
    transition_dir = resolve_path(args.transition_dir)
    base_manifest_path = resolve_path(args.base_manifest)
    transition_manifest_path = resolve_path(args.transition_manifest)
    sample_rate_hz = max(8000, int(args.sample_rate_hz))
    base_duration_sec = max(0.3, float(args.base_duration_sec))
    sil_base_duration_sec = max(1.0, float(args.sil_base_duration_sec))
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
    language = str(args.language).strip().lower()
    transition_pair_repeats = max(2, int(args.transition_pair_repeats))
    audit_path = resolve_path(str(args.prompt_audit_path))
    generate_base = not bool(args.generate_transitions_only)
    generate_transitions = not bool(args.generate_base_only)
    viseme_specs = VISEME_SPECS_BY_LANGUAGE[language]
    viseme_keys: tuple[str, ...] = tuple(spec.key for spec in viseme_specs)
    selected_base_key_set = parse_key_filter(args.base_viseme_keys, viseme_keys, "base viseme key")
    selected_transition_from_set = parse_key_filter(args.transition_from_keys, viseme_keys, "transition from key")
    selected_transition_to_set = parse_key_filter(args.transition_to_keys, viseme_keys, "transition to key")
    viseme_prompt_by_key = {spec.key: spec.prompt for spec in viseme_specs}

    if bool(args.audit_only):
        selected_base_specs = [spec for spec in viseme_specs if spec.key in selected_base_key_set]
        base_audit_entries = [
            {
                "viseme": spec.key,
                "phrase": spec.prompt,
                "language": language,
                "durationSec": sil_base_duration_sec if spec.key == "sil" else base_duration_sec,
            }
            for spec in selected_base_specs
        ]
        transition_audit_entries: list[dict[str, Any]] = []
        for from_viseme in viseme_keys:
            for to_viseme in viseme_keys:
                if from_viseme not in selected_transition_from_set or to_viseme not in selected_transition_to_set:
                    continue
                transition_audit_entries.append(
                    {
                        "viseme": f"{from_viseme}_to_{to_viseme}",
                        "fromViseme": from_viseme,
                        "toViseme": to_viseme,
                        "language": language,
                        "phrase": build_transition_prompt(
                            from_viseme,
                            to_viseme,
                            viseme_prompt_by_key,
                            language,
                            transition_pair_repeats,
                        ),
                    }
                )

        audit_payload = {
            "version": 1,
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "baseVisemeCount": len(base_audit_entries),
            "transitionCount": len(transition_audit_entries),
            "transitionPairRepeats": transition_pair_repeats,
            "baseVisemes": base_audit_entries,
            "transitions": transition_audit_entries,
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as handle:
            json.dump(audit_payload, handle, indent=2)
        print(f"[ok] prompt audit -> {audit_path}")
        print(f"[ok] base visemes -> {len(base_audit_entries)}")
        print(f"[ok] transitions -> {len(transition_audit_entries)}")
        return

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
    selected_base_specs = [spec for spec in viseme_specs if spec.key in selected_base_key_set]
    for index, spec in enumerate(selected_base_specs, start=1):
        output_wav = audio_dir / f"{spec.key}.wav"
        print(f"[base] {index}/{len(selected_base_specs)} {output_wav.name}")
        generated = False
        if generate_base:
            generated = generate_audio_file(
                output_wav=output_wav,
                prompt=spec.prompt,
                duration_sec=sil_base_duration_sec if spec.key == "sil" else base_duration_sec,
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
                "language": language,
                "durationSec": sil_base_duration_sec if spec.key == "sil" else base_duration_sec,
                "audio": to_project_relative(output_wav),
            }
        )

    transition_entries: list[dict[str, Any]] = []
    selected_transition_pairs = [
        (from_viseme, to_viseme)
        for from_viseme in viseme_keys
        for to_viseme in viseme_keys
        if from_viseme in selected_transition_from_set and to_viseme in selected_transition_to_set
    ]
    total_transitions = len(selected_transition_pairs)
    counter = 0
    for from_viseme, to_viseme in selected_transition_pairs:
        counter += 1
        viseme_key = f"{from_viseme}_to_{to_viseme}"
        output_wav = transition_dir / f"{viseme_key}.wav"
        print(f"[trn ] {counter}/{total_transitions} {output_wav.name}")
        generated = False
        transition_phrase = build_transition_prompt(
            from_viseme,
            to_viseme,
            viseme_prompt_by_key,
            language,
            transition_pair_repeats,
        )
        if generate_transitions:
            generated = generate_audio_file(
                output_wav=output_wav,
                prompt=transition_phrase,
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
                "language": language,
                "phrase": transition_phrase,
                "audio": to_project_relative(output_wav),
            }
        )

    shutil.rmtree(temp_dir, ignore_errors=True)

    base_entries = merge_base_entries_with_existing(
        manifest_path=base_manifest_path,
        selected_key_set=selected_base_key_set,
        all_key_order=viseme_keys,
        new_entries=base_entries,
    )
    transition_entries = merge_transition_entries_with_existing(
        manifest_path=transition_manifest_path,
        selected_from_set=selected_transition_from_set,
        selected_to_set=selected_transition_to_set,
        all_key_order=viseme_keys,
        new_entries=transition_entries,
    )

    base_manifest_payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "baseImage": to_project_relative(base_image_path),
        "sampleRateHz": sample_rate_hz,
        "targetDurationSec": base_duration_sec,
        "silDurationSec": sil_base_duration_sec,
        "tts": {
            "engine": "elevenlabs",
            "language": language,
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
