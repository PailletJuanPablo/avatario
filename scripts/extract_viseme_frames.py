"""
Extract PNG frames from rendered viseme clips and build frame manifest.
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
DEFAULT_CLIP_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_clip_manifest.json"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/frames"
DEFAULT_OUTPUT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_frames_manifest.json"
DEFAULT_CLIP_VARIANT = "org"
DEFAULT_TARGET_FPS = 0.0
DEFAULT_INTERPOLATION_MODE = "none"
DEFAULT_SHARPEN_AMOUNT = 0.0
FRAME_FILE_PATTERN = "frame_{:03d}.png"
INTERPOLATION_MODE_NONE = "none"
INTERPOLATION_MODE_DUPLICATE = "duplicate"
INTERPOLATION_MODE_MINTERPOLATE = "minterpolate"


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(description="Extract viseme frame PNGs from clip manifest.")
    parser.add_argument("--clip-manifest", default=DEFAULT_CLIP_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST_PATH)
    parser.add_argument(
        "--clip-variant",
        choices=["org", "crop"],
        default=DEFAULT_CLIP_VARIANT,
        help="Select source clip variant from manifest.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Target extraction fps. 0 keeps source fps.",
    )
    parser.add_argument(
        "--interpolation-mode",
        choices=[INTERPOLATION_MODE_NONE, INTERPOLATION_MODE_DUPLICATE, INTERPOLATION_MODE_MINTERPOLATE],
        default=DEFAULT_INTERPOLATION_MODE,
        help="Frame interpolation strategy when target fps is enabled.",
    )
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=DEFAULT_SHARPEN_AMOUNT,
        help="Optional unsharp intensity (0 disables sharpening).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve to absolute path using project root for relative inputs.
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


def run_command(command: list[str]) -> None:
    """
    Execute command and raise if it fails.
    """
    print(f"[cmd] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")


def read_json(path_value: Path) -> dict[str, Any]:
    """
    Read JSON object from file.
    """
    with path_value.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def build_frame_windows(frame_count: int) -> dict[str, list[int] | int]:
    """
    Build default attack/hold/release frame windows.
    """
    if frame_count <= 0:
        return {
            "neutralIndex": 0,
            "attackIndices": [],
            "holdIndices": [],
            "releaseIndices": [],
        }

    neutral_index = frame_count // 2
    attack_start = max(0, neutral_index - 3)
    attack_end = max(attack_start, neutral_index - 1)
    release_start = min(frame_count - 1, neutral_index + 1)
    release_end = min(frame_count - 1, neutral_index + 3)

    attack = list(range(attack_start, attack_end + 1))
    hold = list(range(max(0, neutral_index - 1), min(frame_count - 1, neutral_index + 1) + 1))
    release = list(range(release_start, release_end + 1))
    return {
        "neutralIndex": neutral_index,
        "attackIndices": attack,
        "holdIndices": hold,
        "releaseIndices": release,
    }


def build_video_filter_chain(target_fps: float, interpolation_mode: str, sharpen_amount: float) -> str:
    """
    Build ffmpeg filter graph for temporal interpolation and optional sharpening.
    """
    filters: list[str] = []
    safe_target_fps = float(target_fps)
    if safe_target_fps > 0:
        if interpolation_mode == INTERPOLATION_MODE_MINTERPOLATE:
            filters.append(
                "minterpolate="
                f"fps={safe_target_fps:.3f}:"
                "mi_mode=mci:mc_mode=aobmc:vsbmc=1:me_mode=bidir:me=epzs"
            )
        elif interpolation_mode == INTERPOLATION_MODE_DUPLICATE:
            filters.append(f"fps={safe_target_fps:.3f}")

    safe_sharpen = max(0.0, float(sharpen_amount))
    if safe_sharpen > 0:
        filters.append(f"unsharp=5:5:{safe_sharpen:.3f}:5:5:0.000")

    return ",".join(filters)


def extract_frames(
    clip_path: Path,
    target_dir: Path,
    overwrite: bool,
    target_fps: float,
    interpolation_mode: str,
    sharpen_amount: float,
) -> list[Path]:
    """
    Extract all clip frames as PNG sequence.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_file in target_dir.glob("*.png"):
            old_file.unlink()

    existing = sorted(target_dir.glob("*.png"))
    if existing and not overwrite:
        return existing

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    frame_pattern = str(target_dir / "frame_%03d.png")
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(clip_path),
        "-vsync",
        "0",
    ]
    filter_chain = build_video_filter_chain(target_fps, interpolation_mode, sharpen_amount)
    if filter_chain:
        command.extend(["-vf", filter_chain])
    command.extend([
        frame_pattern,
    ])
    run_command(command)
    extracted = sorted(target_dir.glob("frame_*.png"))
    return extracted


def main() -> None:
    """
    Extract viseme frames and build manifest.
    """
    args = parse_args()
    clip_manifest_path = resolve_path(args.clip_manifest)
    output_dir = resolve_path(args.output_dir)
    output_manifest_path = resolve_path(args.output_manifest)

    if not clip_manifest_path.exists():
        raise FileNotFoundError(f"Clip manifest not found: {clip_manifest_path}")

    payload = read_json(clip_manifest_path)
    visemes = payload.get("visemes")
    if not isinstance(visemes, list) or not visemes:
        raise ValueError(f"No visemes in clip manifest: {clip_manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_visemes: list[dict[str, Any]] = []
    for item in visemes:
        if not isinstance(item, dict):
            raise ValueError("Invalid viseme entry in clip manifest.")
        viseme = str(item.get("viseme", "")).strip()
        if not viseme:
            raise ValueError("Viseme entry missing 'viseme'.")

        clip_field = "clipOrg" if args.clip_variant == "org" else "clipCrop"
        clip_rel = str(item.get(clip_field, "")).strip()
        if not clip_rel:
            raise ValueError(f"Viseme {viseme} missing {clip_field}.")
        clip_path = resolve_path(clip_rel)
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found for {viseme}: {clip_path}")

        viseme_output_dir = output_dir / viseme
        frame_paths = extract_frames(
            clip_path=clip_path,
            target_dir=viseme_output_dir,
            overwrite=bool(args.overwrite),
            target_fps=float(args.target_fps),
            interpolation_mode=str(args.interpolation_mode),
            sharpen_amount=float(args.sharpen_amount),
        )
        frame_paths = sorted(frame_paths)
        frame_count = len(frame_paths)
        frame_files = [to_project_relative(frame_path) for frame_path in frame_paths]
        windows = build_frame_windows(frame_count)

        result_visemes.append(
            {
                "viseme": viseme,
                "clip": to_project_relative(clip_path),
                "frameCount": frame_count,
                "frames": frame_files,
                "neutralIndex": windows["neutralIndex"],
                "attackIndices": windows["attackIndices"],
                "holdIndices": windows["holdIndices"],
                "releaseIndices": windows["releaseIndices"],
            }
        )
        print(f"[ok] {viseme} frames -> {frame_count}")

    output_manifest = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceClipManifest": to_project_relative(clip_manifest_path),
        "baseImage": payload.get("baseImage", ""),
        "clipVariant": args.clip_variant,
        "targetFps": float(args.target_fps),
        "interpolationMode": str(args.interpolation_mode),
        "sharpenAmount": float(args.sharpen_amount),
        "outputDir": to_project_relative(output_dir),
        "visemeCount": len(result_visemes),
        "visemes": result_visemes,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2)

    print(f"[ok] frame manifest -> {output_manifest_path}")
    print(f"[ok] extracted viseme frame sets -> {len(result_visemes)}")


if __name__ == "__main__":
    main()
