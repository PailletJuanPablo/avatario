"""
Prepare viseme audio manifests from one combined viseme WAV and its segment manifest.
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
DEFAULT_COMBINED_AUDIO_PATH = "output_fasterliveportrait/viseme_library/viseme_all_segments.wav"
DEFAULT_SEGMENTS_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_all_segments_manifest.json"
DEFAULT_OUTPUT_AUDIO_DIR = "output_fasterliveportrait/viseme_library/audio_from_combined"
DEFAULT_OUTPUT_ALL_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_all_audio_from_combined_manifest.json"
DEFAULT_OUTPUT_BASE_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_audio_from_combined_manifest.json"
DEFAULT_OUTPUT_TRANSITION_MANIFEST_PATH = (
    "output_fasterliveportrait/viseme_library/viseme_transition_audio_from_combined_manifest.json"
)
DEFAULT_BASE_IMAGE = "output/frames/frame_00095.png"


@dataclass(frozen=True)
class SegmentSpec:
    """
    One segment entry from the combined-audio manifest.
    """

    index: int
    kind: str
    viseme: str
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(
        description="Extract viseme/transition WAV clips from one combined audio file and build compatible manifests."
    )
    parser.add_argument("--combined-audio", default=DEFAULT_COMBINED_AUDIO_PATH)
    parser.add_argument("--segments-manifest", default=DEFAULT_SEGMENTS_MANIFEST_PATH)
    parser.add_argument("--output-audio-dir", default=DEFAULT_OUTPUT_AUDIO_DIR)
    parser.add_argument("--output-all-manifest", default=DEFAULT_OUTPUT_ALL_MANIFEST_PATH)
    parser.add_argument("--output-base-manifest", default=DEFAULT_OUTPUT_BASE_MANIFEST_PATH)
    parser.add_argument("--output-transition-manifest", default=DEFAULT_OUTPUT_TRANSITION_MANIFEST_PATH)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve relative paths from the project root.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """
    Convert an absolute path into project-relative POSIX path when possible.
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


def parse_segment_specs(segments_manifest_path: Path) -> tuple[int, float, list[SegmentSpec]]:
    """
    Parse and validate segment list from combined-audio manifest.
    """
    payload = read_json(segments_manifest_path)
    sample_rate_hz = int(payload.get("sampleRateHz", 0) or 0)
    gap_sec = float(payload.get("gapSec", 0.0) or 0.0)
    raw_segments = payload.get("segments")
    if sample_rate_hz <= 0:
        raise ValueError(f"Invalid sampleRateHz in {segments_manifest_path}")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"No segments found in {segments_manifest_path}")

    parsed: list[SegmentSpec] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise ValueError("Invalid segment entry type.")
        index = int(item.get("index", -1))
        kind = str(item.get("kind", "")).strip()
        viseme = str(item.get("viseme", "")).strip()
        start_sample = int(item.get("startSample", -1))
        end_sample = int(item.get("endSample", -1))
        start_sec = float(item.get("startSec", 0.0) or 0.0)
        end_sec = float(item.get("endSec", 0.0) or 0.0)
        duration_sec = float(item.get("durationSec", 0.0) or 0.0)
        if index < 0:
            raise ValueError("Segment index must be >= 0.")
        if kind not in {"base", "transition"}:
            raise ValueError(f"Invalid segment kind for index {index}: {kind}")
        if not viseme:
            raise ValueError(f"Missing viseme for segment index {index}")
        if start_sample < 0 or end_sample <= start_sample:
            raise ValueError(f"Invalid sample range for segment {viseme} ({start_sample}, {end_sample})")
        parsed.append(
            SegmentSpec(
                index=index,
                kind=kind,
                viseme=viseme,
                start_sample=start_sample,
                end_sample=end_sample,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration_sec,
            )
        )

    parsed.sort(key=lambda item: item.index)
    expected_indices = list(range(len(parsed)))
    actual_indices = [item.index for item in parsed]
    if actual_indices != expected_indices:
        raise ValueError("Segment indices are not contiguous from 0.")
    return sample_rate_hz, max(0.0, gap_sec), parsed


def split_transition_viseme(viseme_key: str) -> tuple[str, str]:
    """
    Parse transition key in FROM_to_TO format.
    """
    if "_to_" not in viseme_key:
        return "", ""
    from_viseme, to_viseme = viseme_key.split("_to_", 1)
    return from_viseme.strip(), to_viseme.strip()


def extract_segment_wav(
    reader: wave.Wave_read,
    channels: int,
    sample_width: int,
    sample_rate_hz: int,
    segment: SegmentSpec,
    output_path: Path,
    overwrite: bool,
) -> None:
    """
    Extract one WAV segment from the combined source.
    """
    if output_path.exists() and not overwrite:
        return
    frame_count = int(segment.end_sample - segment.start_sample)
    reader.setpos(int(segment.start_sample))
    frames = reader.readframes(frame_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames)


def build_manifest_entry(
    segment: SegmentSpec,
    output_audio_path: Path,
) -> dict[str, Any]:
    """
    Build one viseme manifest entry compatible with build_viseme_pkls.py.
    """
    from_viseme, to_viseme = split_transition_viseme(segment.viseme)
    if segment.kind == "base":
        from_viseme = segment.viseme
        to_viseme = segment.viseme
    return {
        "viseme": segment.viseme,
        "audio": to_project_relative(output_audio_path),
        "fromViseme": from_viseme,
        "toViseme": to_viseme,
        "kind": segment.kind,
        "startSec": round(float(segment.start_sec), 6),
        "endSec": round(float(segment.end_sec), 6),
        "durationSec": round(float(segment.duration_sec), 6),
        "startSample": int(segment.start_sample),
        "endSample": int(segment.end_sample),
    }


def write_manifest(path_value: Path, payload: dict[str, Any], overwrite: bool) -> None:
    """
    Write JSON manifest with overwrite guard.
    """
    if path_value.exists() and not overwrite:
        raise FileExistsError(f"Output manifest already exists. Use --overwrite: {path_value}")
    path_value.parent.mkdir(parents=True, exist_ok=True)
    path_value.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """
    Program entry point.
    """
    args = parse_args()
    combined_audio_path = resolve_path(str(args.combined_audio))
    segments_manifest_path = resolve_path(str(args.segments_manifest))
    output_audio_dir = resolve_path(str(args.output_audio_dir))
    output_all_manifest = resolve_path(str(args.output_all_manifest))
    output_base_manifest = resolve_path(str(args.output_base_manifest))
    output_transition_manifest = resolve_path(str(args.output_transition_manifest))
    base_image_path = resolve_path(str(args.base_image))
    overwrite = bool(args.overwrite)

    if not combined_audio_path.exists():
        raise FileNotFoundError(f"Combined audio not found: {combined_audio_path}")
    if not segments_manifest_path.exists():
        raise FileNotFoundError(f"Segments manifest not found: {segments_manifest_path}")
    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")

    sample_rate_hz, gap_sec, segment_specs = parse_segment_specs(segments_manifest_path)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict[str, Any]] = []
    with wave.open(str(combined_audio_path), "rb") as reader:
        channels = int(reader.getnchannels())
        sample_width = int(reader.getsampwidth())
        reader_sample_rate = int(reader.getframerate())
        compression_type = str(reader.getcomptype())
        total_frames = int(reader.getnframes())
        if compression_type != "NONE":
            raise ValueError(f"Combined WAV must be uncompressed PCM. Found: {compression_type}")
        if reader_sample_rate != sample_rate_hz:
            raise ValueError(
                f"Sample-rate mismatch: manifest={sample_rate_hz} wav={reader_sample_rate} ({combined_audio_path})"
            )
        for segment in segment_specs:
            if segment.end_sample > total_frames:
                raise ValueError(
                    f"Segment {segment.viseme} exceeds source audio length ({segment.end_sample} > {total_frames})"
                )
            output_audio_path = output_audio_dir / f"{segment.viseme}.wav"
            extract_segment_wav(
                reader=reader,
                channels=channels,
                sample_width=sample_width,
                sample_rate_hz=sample_rate_hz,
                segment=segment,
                output_path=output_audio_path,
                overwrite=overwrite,
            )
            all_entries.append(build_manifest_entry(segment=segment, output_audio_path=output_audio_path))

    base_entries = [item for item in all_entries if str(item.get("kind")) == "base"]
    transition_entries = [item for item in all_entries if str(item.get("kind")) == "transition"]

    now_iso = datetime.now(timezone.utc).isoformat()
    common_payload = {
        "version": 1,
        "generatedAtUtc": now_iso,
        "sourceCombinedAudio": to_project_relative(combined_audio_path),
        "sourceSegmentManifest": to_project_relative(segments_manifest_path),
        "baseImage": to_project_relative(base_image_path),
        "sampleRateHz": sample_rate_hz,
        "segmentGapSec": gap_sec,
        "outputDir": to_project_relative(output_audio_dir),
    }
    all_payload = dict(common_payload)
    all_payload["visemeCount"] = len(all_entries)
    all_payload["transitionCount"] = len(transition_entries)
    all_payload["visemes"] = all_entries

    base_payload = dict(common_payload)
    base_payload["visemeCount"] = len(base_entries)
    base_payload["visemes"] = base_entries

    transition_payload = dict(common_payload)
    transition_payload["visemeCount"] = len(transition_entries)
    transition_payload["transitionCount"] = len(transition_entries)
    transition_payload["includeSelfTransitions"] = True
    transition_payload["visemes"] = transition_entries
    transition_payload["transitions"] = transition_entries

    write_manifest(path_value=output_all_manifest, payload=all_payload, overwrite=overwrite)
    write_manifest(path_value=output_base_manifest, payload=base_payload, overwrite=overwrite)
    write_manifest(path_value=output_transition_manifest, payload=transition_payload, overwrite=overwrite)

    print(f"[ok] extracted segment audios -> {output_audio_dir}")
    print(f"[ok] all manifest -> {output_all_manifest}")
    print(f"[ok] base manifest -> {output_base_manifest}")
    print(f"[ok] transition manifest -> {output_transition_manifest}")
    print(f"[ok] base={len(base_entries)} transitions={len(transition_entries)} total={len(all_entries)}")


if __name__ == "__main__":
    main()
