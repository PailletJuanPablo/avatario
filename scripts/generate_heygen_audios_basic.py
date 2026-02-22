#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "output_fasterliveportrait/viseme_library/heygen_basic_audio.mp3"
DEFAULT_API_BASE = "https://api.heygen.com"
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_TEXT_TYPE = "text"
DEFAULT_SPEED = "1"
DEFAULT_LANGUAGE = "es"
DEFAULT_LOCALE = "es-ES"
DEFAULT_COMPAT_API_KEY = ""
DEFAULT_COMPAT_VOICE_ID = "TumdjBNWanlT3ysvclWh"

VISEMES = ["sil", "AA", "E", "I", "O", "U", "MBP", "FV", "L", "TH", "CH", "SS", "RR", "DD"]

# “Sonidos” base (en español neutro, sin palabras reales)
PROMPT = {
    "sil": "mmm",         # “sil” real no existe en TTS; usamos mmm como reposo/cierre
    "AA": "aaaaaaa",
    "E": "eeeeeee",
    "I": "iiiiiii",
    "O": "ooooooo",
    "U": "uuuuuuu",
    "MBP": "mmmmmmm",
    "FV": "fffffff",
    "L": "lalalalalala",
    "TH": "tatatatata",   # dental/alveolar (t/d) para esa familia
    "CH": "chachachacha",
    "SS": "ssssssss",
    "RR": "rrrrrrrr",
    "DD": "dadadadada",
}

DELIM = " | "  # separador entre segmentos (útil para cortar luego)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera audio básico de fonemas usando API de HeyGen.")
    parser.add_argument(
        "--voice-id",
        default=os.environ.get("HEYGEN_VOICE_ID", os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_COMPAT_VOICE_ID)),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HEYGEN_API_KEY", os.environ.get("ELEVENLABS_API_KEY", DEFAULT_COMPAT_API_KEY)),
    )
    parser.add_argument("--api-base", default=os.environ.get("HEYGEN_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--text-type", choices=["text", "ssml"], default=DEFAULT_TEXT_TYPE)
    parser.add_argument("--speed", default=DEFAULT_SPEED)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--locale", default=DEFAULT_LOCALE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--print-text-only", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def transition(frm: str, to: str) -> str:
    a = PROMPT[frm]
    b = PROMPT[to]

    # Para que la transición “tenga” ambos extremos:
    if frm == "sil" and to == "sil":
        return a
    if frm == "sil":
        return a + b            # mmm + visema
    if to == "sil":
        return a + PROMPT["sil"]  # visema + mmm
    if frm == to:
        return a
    return a + b


def build_text() -> str:
    parts: list[str] = []

    # 1) Bases (en el mismo orden del contrato)
    for viseme in VISEMES:
        parts.append(PROMPT[viseme])

    # 2) Transiciones 14x14 en orden determinista
    for frm in VISEMES:
        for to in VISEMES:
            parts.append(transition(frm, to))

    return DELIM.join(parts)


def request_heygen_tts(
    api_base: str,
    api_key: str,
    voice_id: str,
    text: str,
    text_type: str,
    speed: str,
    language: str,
    locale: str,
    timeout_sec: int,
) -> bytes:
    if not api_key.strip():
        raise RuntimeError("API key vacía. Define HEYGEN_API_KEY o ELEVENLABS_API_KEY.")
    if not voice_id.strip():
        raise RuntimeError("Voice ID vacío. Define HEYGEN_VOICE_ID o ELEVENLABS_VOICE_ID.")

    url = f"{api_base.rstrip('/')}/v1/audio/text_to_speech"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    payload = {
        "text": text,
        "voice_id": voice_id,
        "input_type": text_type,
        "speed": str(speed),
        "language": language,
        "locale": locale,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=max(10, int(timeout_sec)))
    if response.status_code >= 400:
        detail = (response.text or "")[:800]
        raise RuntimeError(f"HeyGen TTS falló ({response.status_code}): {detail}")

    data = response.json()
    audio_url = (
        data.get("audio_url")
        or data.get("url")
        or (data.get("data") or {}).get("audio_url")
        or (data.get("data") or {}).get("url")
        or (data.get("result") or {}).get("audio_url")
        or (data.get("result") or {}).get("url")
    )
    if not audio_url:
        raise RuntimeError(f"No se encontró audio_url en la respuesta de HeyGen: {data}")

    audio_response = requests.get(audio_url, timeout=max(10, int(timeout_sec)))
    if audio_response.status_code >= 400:
        detail = (audio_response.text or "")[:500]
        raise RuntimeError(f"Descarga de audio falló ({audio_response.status_code}): {detail}")
    return audio_response.content


def infer_output_path(base_output: Path, source_url: str | None) -> Path:
    if base_output.suffix:
        return base_output
    if source_url:
        parsed = urlparse(source_url)
        suffix = Path(parsed.path).suffix
        if suffix:
            return base_output.with_suffix(suffix)
    return base_output.with_suffix(".mp3")


def main() -> None:
    args = parse_args()
    final_text = build_text()

    if bool(args.print_text_only):
        print(final_text)
        return

    payload_output = resolve_path(str(args.output))
    payload_output = infer_output_path(payload_output, None)
    payload_output.parent.mkdir(parents=True, exist_ok=True)

    audio_bytes = request_heygen_tts(
        api_base=str(args.api_base),
        api_key=str(args.api_key),
        voice_id=str(args.voice_id),
        text=final_text,
        text_type=str(args.text_type),
        speed=str(args.speed),
        language=str(args.language),
        locale=str(args.locale),
        timeout_sec=int(args.timeout_sec),
    )

    payload_output.write_bytes(audio_bytes)
    print(f"[ok] audio generado: {payload_output}")
    print(f"[ok] chars: {len(final_text)}")


if __name__ == "__main__":
    main()
