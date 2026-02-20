"""
Build Pixi-compatible spritesheets and runtime manifest from viseme frames.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_frames_manifest.json"
DEFAULT_MOTION_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_motion_manifest.json"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/atlas"
DEFAULT_OUTPUT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/pixi_library_manifest.json"
DEFAULT_FPS = 25.0
DEFAULT_MAX_COLUMNS = 8


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(description="Build Pixi spritesheet atlases for viseme frame libraries.")
    parser.add_argument("--frames-manifest", default=DEFAULT_FRAMES_MANIFEST_PATH)
    parser.add_argument("--motion-manifest", default=DEFAULT_MOTION_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST_PATH)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-columns", type=int, default=DEFAULT_MAX_COLUMNS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve path using project root for relative values.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """
    Convert path to project-relative POSIX string when possible.
    """
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path_value: Path) -> dict[str, Any]:
    """
    Read a JSON object from disk.
    """
    with path_value.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def resolve_library_fps(
    frames_manifest_payload: dict[str, Any],
    motion_manifest_path: Path,
    fallback_fps: float,
) -> float:
    """
    Resolve library fps from frames manifest first, then motion manifest.
    """
    target_fps = frames_manifest_payload.get("targetFps")
    if isinstance(target_fps, (int, float)) and float(target_fps) > 0:
        return float(target_fps)

    if not motion_manifest_path.exists():
        return float(fallback_fps)
    payload = read_json(motion_manifest_path)
    visemes = payload.get("visemes")
    if not isinstance(visemes, list):
        return float(fallback_fps)
    for item in visemes:
        if not isinstance(item, dict):
            continue
        stats = item.get("stats")
        if not isinstance(stats, dict):
            continue
        fps_value = stats.get("fps")
        if isinstance(fps_value, (int, float)) and float(fps_value) > 0:
            return float(fps_value)
    return float(fallback_fps)


def build_motion_entry_index(motion_manifest_path: Path) -> dict[str, dict[str, Any]]:
    """
    Build lookup table by viseme key from motion manifest.
    """
    if not motion_manifest_path.exists():
        return {}

    payload = read_json(motion_manifest_path)
    visemes = payload.get("visemes")
    if not isinstance(visemes, list):
        return {}

    motion_index: dict[str, dict[str, Any]] = {}
    for item in visemes:
        if not isinstance(item, dict):
            continue
        viseme = str(item.get("viseme", "")).strip()
        if not viseme:
            continue
        motion_index[viseme] = item
    return motion_index


def build_grid(frame_count: int, max_columns: int) -> tuple[int, int]:
    """
    Compute atlas grid columns/rows.
    """
    safe_count = max(1, int(frame_count))
    auto_columns = int(math.ceil(math.sqrt(safe_count)))
    columns = max(1, min(int(max_columns), auto_columns))
    rows = int(math.ceil(safe_count / float(columns)))
    return columns, rows


def build_spritesheet_for_viseme(
    viseme: str,
    frame_paths: list[Path],
    output_dir: Path,
    fps: float,
    max_columns: int,
    overwrite: bool,
) -> tuple[Path, Path, int, int, int, int]:
    """
    Create one atlas image + spritesheet json for a viseme.
    """
    if not frame_paths:
        raise ValueError(f"Viseme {viseme} has no frame paths.")

    first_frame = Image.open(frame_paths[0]).convert("RGBA")
    frame_width, frame_height = first_frame.size
    frame_count = len(frame_paths)
    columns, rows = build_grid(frame_count, max_columns)
    sheet_width = frame_width * columns
    sheet_height = frame_height * rows

    atlas_image_path = output_dir / f"{viseme}.png"
    spritesheet_json_path = output_dir / f"{viseme}.json"
    if atlas_image_path.exists() and spritesheet_json_path.exists() and not overwrite:
        return (
            atlas_image_path,
            spritesheet_json_path,
            frame_count,
            frame_width,
            frame_height,
            int(round(1000.0 / max(0.001, fps))),
        )

    sheet_image = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
    frame_entries: dict[str, Any] = {}
    animation_frames: list[str] = []
    frame_duration_ms = int(round(1000.0 / max(0.001, fps)))

    for index, frame_path in enumerate(frame_paths):
        frame_name = f"{viseme}_{index:03d}.png"
        column = index % columns
        row = index // columns
        x = column * frame_width
        y = row * frame_height

        frame_image = Image.open(frame_path).convert("RGBA")
        if frame_image.size != (frame_width, frame_height):
            raise ValueError(
                f"Viseme {viseme} frame size mismatch at {frame_path}: "
                f"expected {(frame_width, frame_height)} got {frame_image.size}"
            )
        sheet_image.paste(frame_image, (x, y))
        frame_entries[frame_name] = {
            "frame": {"x": x, "y": y, "w": frame_width, "h": frame_height},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": frame_width, "h": frame_height},
            "sourceSize": {"w": frame_width, "h": frame_height},
            "duration": frame_duration_ms,
        }
        animation_frames.append(frame_name)

    sheet_image.save(atlas_image_path, format="PNG")
    spritesheet_payload = {
        "frames": frame_entries,
        "animations": {viseme: animation_frames},
        "meta": {
            "app": "offline-viseme-builder",
            "version": "1.0",
            "image": atlas_image_path.name,
            "format": "RGBA8888",
            "size": {"w": sheet_width, "h": sheet_height},
            "scale": "1",
        },
    }
    with spritesheet_json_path.open("w", encoding="utf-8") as handle:
        json.dump(spritesheet_payload, handle, indent=2)
    return (
        atlas_image_path,
        spritesheet_json_path,
        frame_count,
        frame_width,
        frame_height,
        frame_duration_ms,
    )


def main() -> None:
    """
    Build all viseme atlases and write runtime manifest.
    """
    args = parse_args()
    frames_manifest_path = resolve_path(args.frames_manifest)
    motion_manifest_path = resolve_path(args.motion_manifest)
    output_dir = resolve_path(args.output_dir)
    output_manifest_path = resolve_path(args.output_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not frames_manifest_path.exists():
        raise FileNotFoundError(f"Frames manifest not found: {frames_manifest_path}")

    frames_payload = read_json(frames_manifest_path)
    visemes = frames_payload.get("visemes")
    if not isinstance(visemes, list) or not visemes:
        raise ValueError(f"No viseme entries in {frames_manifest_path}")

    resolved_fps = resolve_library_fps(frames_payload, motion_manifest_path, float(args.fps))
    safe_max_columns = max(1, int(args.max_columns))
    motion_entry_index = build_motion_entry_index(motion_manifest_path)
    requires_motion_alignment = motion_manifest_path.exists()

    runtime_visemes: list[dict[str, Any]] = []
    for item in visemes:
        if not isinstance(item, dict):
            raise ValueError("Invalid viseme entry in frames manifest.")
        viseme = str(item.get("viseme", "")).strip()
        frame_rel_paths = item.get("frames")
        if not viseme:
            raise ValueError("Viseme entry missing 'viseme'.")
        if not isinstance(frame_rel_paths, list) or not frame_rel_paths:
            raise ValueError(f"Viseme {viseme} has no frames list.")

        frame_paths = [resolve_path(str(frame_rel)) for frame_rel in frame_rel_paths]
        for frame_path in frame_paths:
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame missing for {viseme}: {frame_path}")

        (
            atlas_image_path,
            spritesheet_json_path,
            frame_count,
            frame_width,
            frame_height,
            frame_duration_ms,
        ) = build_spritesheet_for_viseme(
            viseme=viseme,
            frame_paths=frame_paths,
            output_dir=output_dir,
            fps=resolved_fps,
            max_columns=safe_max_columns,
            overwrite=bool(args.overwrite),
        )

        if requires_motion_alignment and viseme not in motion_entry_index:
            raise ValueError(f"Viseme {viseme} does not exist in motion manifest {motion_manifest_path}.")

        runtime_entry: dict[str, Any] = {
            "viseme": viseme,
            "animationKey": viseme,
            "sheetImage": to_project_relative(atlas_image_path),
            "sheetJson": to_project_relative(spritesheet_json_path),
            "frameCount": frame_count,
            "frameSize": {"w": frame_width, "h": frame_height},
            "frameDurationMs": frame_duration_ms,
            "neutralIndex": int(item.get("neutralIndex", max(0, frame_count // 2))),
            "attackIndices": item.get("attackIndices", []),
            "holdIndices": item.get("holdIndices", []),
            "releaseIndices": item.get("releaseIndices", []),
        }

        motion_entry = motion_entry_index.get(viseme)
        if motion_entry:
            from_viseme = str(motion_entry.get("fromViseme", "")).strip()
            to_viseme = str(motion_entry.get("toViseme", "")).strip()
            if from_viseme or to_viseme:
                if not from_viseme or not to_viseme:
                    raise ValueError(
                        f"Viseme {viseme} has incomplete transition mapping in motion manifest: "
                        f"fromViseme='{from_viseme}' toViseme='{to_viseme}'."
                    )
                runtime_entry["fromViseme"] = from_viseme
                runtime_entry["toViseme"] = to_viseme

        runtime_visemes.append(runtime_entry)
        print(f"[ok] atlas {viseme} -> {atlas_image_path.name} + {spritesheet_json_path.name}")

    runtime_manifest = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceFramesManifest": to_project_relative(frames_manifest_path),
        "sourceMotionManifest": to_project_relative(motion_manifest_path)
        if motion_manifest_path.exists()
        else "",
        "baseImage": str(frames_payload.get("baseImage", "")),
        "fps": resolved_fps,
        "visemeCount": len(runtime_visemes),
        "atlasDir": to_project_relative(output_dir),
        "rendering": {
            "defaultCrossfadeMs": 80,
            "idleViseme": "sil",
            "maxMicroJitterPx": 0.8,
        },
        "visemes": runtime_visemes,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(runtime_manifest, handle, indent=2)

    print(f"[ok] pixi library manifest -> {output_manifest_path}")
    print(f"[ok] atlases generated -> {len(runtime_visemes)}")


if __name__ == "__main__":
    main()
