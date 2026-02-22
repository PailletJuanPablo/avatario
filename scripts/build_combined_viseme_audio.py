"""
Build one long WAV that concatenates all base visemes and viseme transitions.
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_AUDIO_DIR = "output_fasterliveportrait/viseme_library/audio"
DEFAULT_TRANSITION_AUDIO_DIR = "output_fasterliveportrait/viseme_library/audio_transitions"
DEFAULT_OUTPUT_AUDIO = "output_fasterliveportrait/viseme_library/viseme_all_segments.wav"
DEFAULT_OUTPUT_MANIFEST = "output_fasterliveportrait/viseme_library/viseme_all_segments_manifest.json"
DEFAULT_GAP_SEC = 0.20
VISEME_ORDER: tuple[str, ...] = (
    "sil",
    "AA",
    "E",
    "I",
    "O",
    "U",
    "MBP",
    "FV",
    "L",
    "TH",
    "CH",
    "SS",
    "RR",
    "DD",
)


@dataclass(frozen=True)
class AudioEntry:
    """
    One input audio segment to append.
    """

    index: int
    kind: str
    viseme: str
    source_path: Path


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(
        description="Concatenate base viseme + transition WAV files into one long WAV with silence gaps."
    )
    parser.add_argument("--base-audio-dir", default=DEFAULT_BASE_AUDIO_DIR)
    parser.add_argument("--transition-audio-dir", default=DEFAULT_TRANSITION_AUDIO_DIR)
    parser.add_argument("--output-audio", default=DEFAULT_OUTPUT_AUDIO)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument(
        "--gap-sec",
        type=float,
        default=DEFAULT_GAP_SEC,
        help="Silence duration inserted between adjacent segments.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve relative path from project root.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """
    Convert path to project-relative POSIX representation when possible.
    """
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def collect_audio_entries(base_audio_dir: Path, transition_audio_dir: Path) -> list[AudioEntry]:
    """
    Build deterministic append order: all base visemes, then all pair transitions.
    """
    entries: list[AudioEntry] = []
    next_index = 0
    for viseme in VISEME_ORDER:
        entries.append(
            AudioEntry(
                index=next_index,
                kind="base",
                viseme=viseme,
                source_path=base_audio_dir / f"{viseme}.wav",
            )
        )
        next_index += 1
    for from_viseme in VISEME_ORDER:
        for to_viseme in VISEME_ORDER:
            transition_key = f"{from_viseme}_to_{to_viseme}"
            entries.append(
                AudioEntry(
                    index=next_index,
                    kind="transition",
                    viseme=transition_key,
                    source_path=transition_audio_dir / f"{transition_key}.wav",
                )
            )
            next_index += 1
    return entries


def validate_input_files(entries: list[AudioEntry]) -> None:
    """
    Fail fast with all missing files listed.
    """
    missing_paths = [item.source_path for item in entries if not item.source_path.exists()]
    if missing_paths:
        preview = "\n".join(str(path) for path in missing_paths[:20])
        extra_count = max(0, len(missing_paths) - 20)
        extra_label = f"\n... (+{extra_count} more)" if extra_count else ""
        raise FileNotFoundError(f"Missing input WAV files:\n{preview}{extra_label}")


def read_wave_data(audio_path: Path) -> tuple[int, int, int, bytes]:
    """
    Read PCM WAV and return (channels, sample_width, sample_rate, frames_bytes).
    """
    with wave.open(str(audio_path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frame_count = int(handle.getnframes())
        compression_type = str(handle.getcomptype())
        if compression_type != "NONE":
            raise ValueError(f"Unsupported WAV compression for {audio_path}: {compression_type}")
        frames_bytes = handle.readframes(frame_count)
    return channels, sample_width, sample_rate, frames_bytes


def sec_from_frames(frame_index: int, sample_rate_hz: int) -> float:
    """
    Convert absolute frame index to seconds.
    """
    if sample_rate_hz <= 0:
        return 0.0
    return float(frame_index) / float(sample_rate_hz)


def build_long_audio(
    entries: list[AudioEntry],
    output_audio_path: Path,
    output_manifest_path: Path,
    gap_sec: float,
) -> None:
    """
    Concatenate all WAV inputs into one output WAV and write per-segment timing manifest.
    """
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width: int | None = None
    gap_sample_count: int | None = None
    written_frame_count = 0
    segment_manifest_entries: list[dict[str, Any]] = []

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_audio_path), "wb") as writer:
        for entry in entries:
            entry_channels, entry_sample_width, entry_sample_rate, entry_frames = read_wave_data(entry.source_path)
            bytes_per_frame = int(entry_channels * entry_sample_width)
            entry_frame_count = int(len(entry_frames) // bytes_per_frame) if bytes_per_frame > 0 else 0

            if sample_rate_hz is None:
                sample_rate_hz = entry_sample_rate
                channels = entry_channels
                sample_width = entry_sample_width
                safe_gap_sec = max(0.0, float(gap_sec))
                gap_sample_count = int(round(safe_gap_sec * float(sample_rate_hz)))
                writer.setnchannels(channels)
                writer.setsampwidth(sample_width)
                writer.setframerate(sample_rate_hz)
            else:
                if entry_sample_rate != sample_rate_hz:
                    raise ValueError(
                        f"Sample-rate mismatch in {entry.source_path}: {entry_sample_rate} != {sample_rate_hz}"
                    )
                if entry_channels != channels:
                    raise ValueError(f"Channel mismatch in {entry.source_path}: {entry_channels} != {channels}")
                if entry_sample_width != sample_width:
                    raise ValueError(
                        f"Sample-width mismatch in {entry.source_path}: {entry_sample_width} != {sample_width}"
                    )

            if entry.index > 0 and int(gap_sample_count or 0) > 0:
                gap_bytes = b"\x00" * (int(gap_sample_count) * int(channels or 1) * int(sample_width or 2))
                writer.writeframes(gap_bytes)
                written_frame_count += int(gap_sample_count)

            segment_start_frame = int(written_frame_count)
            writer.writeframes(entry_frames)
            written_frame_count += int(entry_frame_count)
            segment_end_frame = int(written_frame_count)

            segment_manifest_entries.append(
                {
                    "index": int(entry.index),
                    "kind": entry.kind,
                    "viseme": entry.viseme,
                    "audio": to_project_relative(entry.source_path),
                    "startSample": segment_start_frame,
                    "endSample": segment_end_frame,
                    "durationSamples": int(segment_end_frame - segment_start_frame),
                    "startSec": round(sec_from_frames(segment_start_frame, int(sample_rate_hz or 1)), 6),
                    "endSec": round(sec_from_frames(segment_end_frame, int(sample_rate_hz or 1)), 6),
                    "durationSec": round(sec_from_frames(segment_end_frame - segment_start_frame, int(sample_rate_hz or 1)), 6),
                }
            )

    manifest_payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "outputAudio": to_project_relative(output_audio_path),
        "sampleRateHz": int(sample_rate_hz or 0),
        "channels": int(channels or 0),
        "sampleWidthBytes": int(sample_width or 0),
        "gapSec": float(max(0.0, gap_sec)),
        "baseCount": len(VISEME_ORDER),
        "transitionCount": len(VISEME_ORDER) * len(VISEME_ORDER),
        "segmentCount": len(segment_manifest_entries),
        "totalDurationSec": round(sec_from_frames(written_frame_count, int(sample_rate_hz or 1)), 6),
        "segments": segment_manifest_entries,
    }
    output_manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")


def main() -> None:
    """
    Program entry point.
    """
    args = parse_args()
    base_audio_dir = resolve_path(str(args.base_audio_dir))
    transition_audio_dir = resolve_path(str(args.transition_audio_dir))
    output_audio_path = resolve_path(str(args.output_audio))
    output_manifest_path = resolve_path(str(args.output_manifest))
    safe_gap_sec = max(0.0, float(args.gap_sec))

    entries = collect_audio_entries(base_audio_dir=base_audio_dir, transition_audio_dir=transition_audio_dir)
    validate_input_files(entries)

    if output_audio_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output audio already exists. Use --overwrite: {output_audio_path}")
    if output_manifest_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output manifest already exists. Use --overwrite: {output_manifest_path}")

    build_long_audio(
        entries=entries,
        output_audio_path=output_audio_path,
        output_manifest_path=output_manifest_path,
        gap_sec=safe_gap_sec,
    )
    print(f"[ok] long audio -> {output_audio_path}")
    print(f"[ok] segment manifest -> {output_manifest_path}")
    print(f"[ok] segments total -> {len(entries)}")


if __name__ == "__main__":
    main()
