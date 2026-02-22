"""
POC: extract AA from combined viseme audio and build repeated AA test audios.
"""

from __future__ import annotations

import argparse
import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED_AUDIO_PATH = "output_fasterliveportrait/viseme_library/viseme_all_segments.wav"
DEFAULT_SEGMENT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_all_segments_manifest.json"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/poc_aa_repeat"
DEFAULT_BASE_IMAGE_PATH = "output/frames/frame_00095.png"
DEFAULT_REPEAT_NO_PAUSE = 4
DEFAULT_REPEAT_WITH_PAUSE = 4
DEFAULT_PAUSE_SEC = 0.20
TARGET_VISEME_KEY = "AA"
DEFAULT_MIXED_VISEME_KEY = "AA_poc_repeat_mixed"
DEFAULT_AUDIO_MANIFEST_NAME = "aa_repeat_audio_manifest.json"


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(
        description="Build AA repeated POC audio from combined viseme audio and emit single-entry audio manifest."
    )
    parser.add_argument("--combined-audio", default=DEFAULT_COMBINED_AUDIO_PATH)
    parser.add_argument("--segment-manifest", default=DEFAULT_SEGMENT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE_PATH)
    parser.add_argument("--repeat-no-pause", type=int, default=DEFAULT_REPEAT_NO_PAUSE)
    parser.add_argument("--repeat-with-pause", type=int, default=DEFAULT_REPEAT_WITH_PAUSE)
    parser.add_argument("--pause-sec", type=float, default=DEFAULT_PAUSE_SEC)
    parser.add_argument("--mixed-viseme-key", default=DEFAULT_MIXED_VISEME_KEY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve path relative to project root.
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


def read_json(path_value: Path) -> dict[str, Any]:
    """
    Read and validate JSON object from disk.
    """
    payload = json.loads(path_value.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def find_aa_segment(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Find AA base segment in combined-audio segment manifest.
    """
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Segment manifest missing 'segments' list.")
    matches = [
        item
        for item in raw_segments
        if isinstance(item, dict) and str(item.get("kind", "")).strip() == "base" and str(item.get("viseme", "")).strip() == TARGET_VISEME_KEY
    ]
    if not matches:
        raise ValueError("AA base segment was not found in segment manifest.")
    return matches[0]


def read_combined_audio(combined_audio_path: Path) -> tuple[int, int, int, int, bytes]:
    """
    Read full combined WAV audio.
    """
    with wave.open(str(combined_audio_path), "rb") as reader:
        channels = int(reader.getnchannels())
        sample_width = int(reader.getsampwidth())
        sample_rate_hz = int(reader.getframerate())
        frame_count = int(reader.getnframes())
        compression_type = str(reader.getcomptype())
        if compression_type != "NONE":
            raise ValueError(f"Combined audio must be PCM WAV. Found compression={compression_type}")
        audio_bytes = reader.readframes(frame_count)
    return channels, sample_width, sample_rate_hz, frame_count, audio_bytes


def slice_frames(audio_bytes: bytes, channels: int, sample_width: int, start_sample: int, end_sample: int) -> bytes:
    """
    Slice frame bytes in sample-index space.
    """
    bytes_per_frame = int(channels * sample_width)
    if bytes_per_frame <= 0:
        raise ValueError("Invalid bytes per frame.")
    byte_start = int(start_sample) * bytes_per_frame
    byte_end = int(end_sample) * bytes_per_frame
    if byte_start < 0 or byte_end <= byte_start or byte_end > len(audio_bytes):
        raise ValueError("Invalid AA sample range for combined audio.")
    return audio_bytes[byte_start:byte_end]


def write_wav(path_value: Path, channels: int, sample_width: int, sample_rate_hz: int, frames_bytes: bytes, overwrite: bool) -> None:
    """
    Write PCM WAV file.
    """
    if path_value.exists() and not overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite: {path_value}")
    path_value.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path_value), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames_bytes)


def repeat_frames(segment_frames: bytes, repeat_count: int) -> bytes:
    """
    Repeat segment bytes N times.
    """
    safe_repeat = max(1, int(repeat_count))
    return segment_frames * safe_repeat


def build_frames_with_pause(
    segment_frames: bytes,
    repeat_count: int,
    pause_sec: float,
    channels: int,
    sample_width: int,
    sample_rate_hz: int,
) -> bytes:
    """
    Build repeated segment bytes with silence gaps between repeats.
    """
    safe_repeat = max(1, int(repeat_count))
    safe_pause_sec = max(0.0, float(pause_sec))
    pause_frames = int(round(float(sample_rate_hz) * safe_pause_sec))
    pause_bytes = b"\x00" * (pause_frames * channels * sample_width)
    chunks: list[bytes] = []
    for index in range(safe_repeat):
        chunks.append(segment_frames)
        if index < safe_repeat - 1 and pause_bytes:
            chunks.append(pause_bytes)
    return b"".join(chunks)


def build_audio_manifest(
    output_dir: Path,
    base_image_path: Path,
    mixed_audio_path: Path,
    sample_rate_hz: int,
    target_duration_sec: float,
    mixed_viseme_key: str,
    repeat_no_pause: int,
    repeat_with_pause: int,
    pause_sec: float,
    overwrite: bool,
) -> Path:
    """
    Write single-entry audio manifest compatible with build_viseme_pkls.py.
    """
    manifest_path = output_dir / DEFAULT_AUDIO_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Audio manifest already exists. Use --overwrite: {manifest_path}")
    payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "baseImage": to_project_relative(base_image_path),
        "sampleRateHz": int(sample_rate_hz),
        "targetDurationSec": float(round(target_duration_sec, 6)),
        "sourceType": "poc-aa-repeat-from-combined",
        "repeatNoPause": int(repeat_no_pause),
        "repeatWithPause": int(repeat_with_pause),
        "pauseSec": float(round(max(0.0, pause_sec), 6)),
        "visemeCount": 1,
        "visemes": [
            {
                "viseme": str(mixed_viseme_key).strip(),
                "audio": to_project_relative(mixed_audio_path),
                "fromViseme": TARGET_VISEME_KEY,
                "toViseme": TARGET_VISEME_KEY,
                "kind": "poc",
                "phrase": "AA repeated mixed (no-pause + with-pause)",
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    """
    Program entry point.
    """
    args = parse_args()
    combined_audio_path = resolve_path(str(args.combined_audio))
    segment_manifest_path = resolve_path(str(args.segment_manifest))
    output_dir = resolve_path(str(args.output_dir))
    base_image_path = resolve_path(str(args.base_image))
    mixed_viseme_key = str(args.mixed_viseme_key).strip() or DEFAULT_MIXED_VISEME_KEY
    repeat_no_pause = max(1, int(args.repeat_no_pause))
    repeat_with_pause = max(1, int(args.repeat_with_pause))
    pause_sec = max(0.0, float(args.pause_sec))
    overwrite = bool(args.overwrite)

    if not combined_audio_path.exists():
        raise FileNotFoundError(f"Combined audio not found: {combined_audio_path}")
    if not segment_manifest_path.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest_path}")
    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")

    segment_manifest_payload = read_json(segment_manifest_path)
    aa_segment = find_aa_segment(segment_manifest_payload)
    aa_start_sample = int(aa_segment.get("startSample", -1))
    aa_end_sample = int(aa_segment.get("endSample", -1))
    if aa_start_sample < 0 or aa_end_sample <= aa_start_sample:
        raise ValueError("Invalid AA segment bounds in segment manifest.")

    channels, sample_width, sample_rate_hz, _, audio_bytes = read_combined_audio(combined_audio_path)
    aa_frames = slice_frames(
        audio_bytes=audio_bytes,
        channels=channels,
        sample_width=sample_width,
        start_sample=aa_start_sample,
        end_sample=aa_end_sample,
    )
    no_pause_frames = repeat_frames(aa_frames, repeat_no_pause)
    with_pause_frames = build_frames_with_pause(
        segment_frames=aa_frames,
        repeat_count=repeat_with_pause,
        pause_sec=pause_sec,
        channels=channels,
        sample_width=sample_width,
        sample_rate_hz=sample_rate_hz,
    )
    mixed_frames = b"".join([no_pause_frames, with_pause_frames])

    audio_output_dir = output_dir / "audio"
    aa_from_long_path = audio_output_dir / "AA_from_long.wav"
    no_pause_path = audio_output_dir / "AA_repeat_no_pause.wav"
    with_pause_path = audio_output_dir / "AA_repeat_with_pause.wav"
    mixed_path = audio_output_dir / f"{mixed_viseme_key}.wav"

    write_wav(
        path_value=aa_from_long_path,
        channels=channels,
        sample_width=sample_width,
        sample_rate_hz=sample_rate_hz,
        frames_bytes=aa_frames,
        overwrite=overwrite,
    )
    write_wav(
        path_value=no_pause_path,
        channels=channels,
        sample_width=sample_width,
        sample_rate_hz=sample_rate_hz,
        frames_bytes=no_pause_frames,
        overwrite=overwrite,
    )
    write_wav(
        path_value=with_pause_path,
        channels=channels,
        sample_width=sample_width,
        sample_rate_hz=sample_rate_hz,
        frames_bytes=with_pause_frames,
        overwrite=overwrite,
    )
    write_wav(
        path_value=mixed_path,
        channels=channels,
        sample_width=sample_width,
        sample_rate_hz=sample_rate_hz,
        frames_bytes=mixed_frames,
        overwrite=overwrite,
    )

    frame_denominator = max(1, channels * sample_width)
    mixed_duration_sec = float(len(mixed_frames) / frame_denominator) / float(sample_rate_hz)
    manifest_path = build_audio_manifest(
        output_dir=output_dir,
        base_image_path=base_image_path,
        mixed_audio_path=mixed_path,
        sample_rate_hz=sample_rate_hz,
        target_duration_sec=mixed_duration_sec,
        mixed_viseme_key=mixed_viseme_key,
        repeat_no_pause=repeat_no_pause,
        repeat_with_pause=repeat_with_pause,
        pause_sec=pause_sec,
        overwrite=overwrite,
    )

    print(f"[ok] AA extracted from combined -> {aa_from_long_path}")
    print(f"[ok] AA repeated (no pause) -> {no_pause_path}")
    print(f"[ok] AA repeated (with pause) -> {with_pause_path}")
    print(f"[ok] AA repeated (mixed) -> {mixed_path}")
    print(f"[ok] audio manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
