"""
Generate offline transition-audio clips for viseme pair transitions (from->to).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISEME_AUDIO_MANIFEST = "output_fasterliveportrait/viseme_library/viseme_audio_manifest.json"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/audio_transitions"
DEFAULT_OUTPUT_MANIFEST = "output_fasterliveportrait/viseme_library/viseme_transition_audio_manifest.json"
DEFAULT_SEGMENT_SEC = 0.48
DEFAULT_CROSSFADE_SEC = 0.18
DEFAULT_TARGET_DURATION_SEC = 0.0
DEFAULT_INCLUDE_SELF = True
MIN_SEGMENT_SEC = 0.08
MIN_CROSSFADE_SEC = 0.03


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(description="Generate transition audio clips for viseme pair blending.")
    parser.add_argument("--viseme-audio-manifest", default=DEFAULT_VISEME_AUDIO_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--segment-sec", type=float, default=DEFAULT_SEGMENT_SEC)
    parser.add_argument("--crossfade-sec", type=float, default=DEFAULT_CROSSFADE_SEC)
    parser.add_argument(
        "--target-duration-sec",
        type=float,
        default=DEFAULT_TARGET_DURATION_SEC,
        help="Optional fixed target duration for each transition. 0 keeps natural duration.",
    )
    parser.add_argument("--include-self", action="store_true", default=DEFAULT_INCLUDE_SELF)
    parser.add_argument("--exclude-self", dest="include_self", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve path relative to project root when needed.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """
    Convert absolute path to project-relative POSIX format when possible.
    """
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path_value: Path) -> dict[str, Any]:
    """
    Read JSON object from disk.
    """
    with path_value.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def run_command(command: list[str]) -> None:
    """
    Execute command and raise on failure.
    """
    print(f"[cmd] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")


def run_command_capture(command: list[str]) -> str:
    """
    Execute command and return stdout.
    """
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")
    return (result.stdout or "").strip()


def get_audio_duration_sec(audio_path: Path) -> float:
    """
    Resolve media duration using ffprobe.
    """
    stdout = run_command_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
    )
    return float(stdout or 0.0)


def normalize_viseme_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Load viseme audio entries from base manifest.
    """
    visemes = payload.get("visemes")
    if not isinstance(visemes, list) or not visemes:
        raise ValueError("Viseme audio manifest has no entries.")

    normalized: list[dict[str, Any]] = []
    for item in visemes:
        if not isinstance(item, dict):
            raise ValueError("Invalid viseme entry in audio manifest.")
        viseme = str(item.get("viseme", "")).strip()
        audio_rel = str(item.get("audio", "")).strip()
        if not viseme or not audio_rel:
            raise ValueError("Viseme entry requires 'viseme' and 'audio'.")
        audio_path = resolve_path(audio_rel)
        if not audio_path.exists():
            raise FileNotFoundError(f"Viseme audio file not found: {audio_path}")
        normalized.append(
            {
                "viseme": viseme,
                "audioPath": audio_path,
                "audioRel": to_project_relative(audio_path),
                "phrase": str(item.get("phrase", "")).strip(),
            }
        )
    return normalized


def build_transition_name(from_viseme: str, to_viseme: str) -> str:
    """
    Build transition key used across manifests and filenames.
    """
    return f"{from_viseme}_to_{to_viseme}"


def build_transition_audio(
    from_audio_path: Path,
    to_audio_path: Path,
    output_audio_path: Path,
    requested_segment_sec: float,
    requested_crossfade_sec: float,
    requested_target_duration_sec: float,
    sample_rate_hz: int,
    overwrite: bool,
) -> tuple[float, float, float]:
    """
    Create one transition audio file from source->target viseme.
    """
    if output_audio_path.exists() and not overwrite:
        from_duration = get_audio_duration_sec(from_audio_path)
        to_duration = get_audio_duration_sec(to_audio_path)
        safe_segment = max(
            MIN_SEGMENT_SEC,
            min(float(requested_segment_sec), from_duration, to_duration),
        )
        safe_crossfade = max(
            MIN_CROSSFADE_SEC,
            min(float(requested_crossfade_sec), safe_segment * 0.9),
        )
        return safe_segment, safe_crossfade, max(0.0, float(requested_target_duration_sec))

    from_duration = get_audio_duration_sec(from_audio_path)
    to_duration = get_audio_duration_sec(to_audio_path)
    safe_segment = max(
        MIN_SEGMENT_SEC,
        min(float(requested_segment_sec), from_duration, to_duration),
    )
    safe_crossfade = max(
        MIN_CROSSFADE_SEC,
        min(float(requested_crossfade_sec), safe_segment * 0.9),
    )
    from_start = max(0.0, from_duration - safe_segment)
    natural_duration = (safe_segment * 2.0) - safe_crossfade
    target_duration = (
        max(natural_duration, float(requested_target_duration_sec))
        if float(requested_target_duration_sec) > 0
        else natural_duration
    )

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    filter_complex = (
        f"[0:a]atrim=start={from_start:.3f}:duration={safe_segment:.3f},asetpts=PTS-STARTPTS,"
        f"aformat=sample_fmts=fltp:sample_rates={int(sample_rate_hz)}:channel_layouts=mono[a0];"
        f"[1:a]atrim=start=0:duration={safe_segment:.3f},asetpts=PTS-STARTPTS,"
        f"aformat=sample_fmts=fltp:sample_rates={int(sample_rate_hz)}:channel_layouts=mono[a1];"
        f"[a0][a1]acrossfade=d={safe_crossfade:.3f}:c1=tri:c2=tri,apad,atrim=0:{target_duration:.3f}[aout]"
    )

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(from_audio_path),
        "-i",
        str(to_audio_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate_hz)),
        "-c:a",
        "pcm_s16le",
        str(output_audio_path),
    ]
    run_command(command)
    return safe_segment, safe_crossfade, target_duration


def main() -> None:
    """
    Generate transition audio set and corresponding manifest.
    """
    args = parse_args()
    base_manifest_path = resolve_path(args.viseme_audio_manifest)
    output_dir = resolve_path(args.output_dir)
    output_manifest_path = resolve_path(args.output_manifest)
    safe_segment_sec = max(MIN_SEGMENT_SEC, float(args.segment_sec))
    safe_crossfade_sec = max(MIN_CROSSFADE_SEC, float(args.crossfade_sec))
    safe_target_duration_sec = max(0.0, float(args.target_duration_sec))

    if not base_manifest_path.exists():
        raise FileNotFoundError(f"Viseme audio manifest not found: {base_manifest_path}")

    payload = read_json(base_manifest_path)
    base_image = str(payload.get("baseImage", "")).strip()
    sample_rate_hz = int(payload.get("sampleRateHz", 16000) or 16000)
    viseme_entries = normalize_viseme_entries(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    transitions: list[dict[str, Any]] = []
    for from_entry in viseme_entries:
        for to_entry in viseme_entries:
            if not bool(args.include_self) and from_entry["viseme"] == to_entry["viseme"]:
                continue
            from_viseme = str(from_entry["viseme"])
            to_viseme = str(to_entry["viseme"])
            transition_key = build_transition_name(from_viseme, to_viseme)
            output_audio_path = output_dir / f"{transition_key}.wav"
            segment_sec, crossfade_sec, target_duration_sec = build_transition_audio(
                from_audio_path=Path(from_entry["audioPath"]),
                to_audio_path=Path(to_entry["audioPath"]),
                output_audio_path=output_audio_path,
                requested_segment_sec=safe_segment_sec,
                requested_crossfade_sec=safe_crossfade_sec,
                requested_target_duration_sec=safe_target_duration_sec,
                sample_rate_hz=sample_rate_hz,
                overwrite=bool(args.overwrite),
            )
            transitions.append(
                {
                    "viseme": transition_key,
                    "fromViseme": from_viseme,
                    "toViseme": to_viseme,
                    "audio": to_project_relative(output_audio_path),
                    "segmentSec": round(segment_sec, 3),
                    "crossfadeSec": round(crossfade_sec, 3),
                    "targetDurationSec": round(target_duration_sec, 3),
                }
            )
            print(f"[ok] {transition_key} -> {output_audio_path}")

    output_payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceVisemeAudioManifest": to_project_relative(base_manifest_path),
        "baseImage": base_image,
        "sampleRateHz": sample_rate_hz,
        "segmentSec": safe_segment_sec,
        "crossfadeSec": safe_crossfade_sec,
        "targetDurationSec": safe_target_duration_sec,
        "includeSelfTransitions": bool(args.include_self),
        "transitionCount": len(transitions),
        "visemeCount": len(transitions),
        "outputDir": to_project_relative(output_dir),
        "visemes": transitions,
        "transitions": transitions,
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    print(f"[ok] transition audio manifest -> {output_manifest_path}")
    print(f"[ok] transition audio clips -> {len(transitions)}")


if __name__ == "__main__":
    main()
