"""
Generate denser transition frame sequences using pairwise optical flow interpolation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_transition_frames_manifest.json"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/frames_transitions_flow"
DEFAULT_OUTPUT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_transition_frames_flow_manifest.json"
DEFAULT_SUBFRAMES_PER_PAIR = 2
DEFAULT_PARALLEL_JOBS = 4
DEFAULT_CONSISTENCY_THRESHOLD = 1.5
DEFAULT_CONSISTENCY_SCALE = 0.05
DEFAULT_MASK_BLUR_KERNEL = 5
FRAME_FILE_PATTERN = "frame_{:03d}.png"


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Apply optical-flow pair interpolation to transition frame sets."
    )
    parser.add_argument("--frames-manifest", default=DEFAULT_INPUT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST_PATH)
    parser.add_argument(
        "--subframes-per-pair",
        type=int,
        default=DEFAULT_SUBFRAMES_PER_PAIR,
        help="How many synthetic in-between frames to insert for each pair.",
    )
    parser.add_argument("--jobs", type=int, default=DEFAULT_PARALLEL_JOBS)
    parser.add_argument("--max-visemes", type=int, default=0, help="Optional cap for quick tests.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--flow-pyr-scale", type=float, default=0.5)
    parser.add_argument("--flow-levels", type=int, default=3)
    parser.add_argument("--flow-winsize", type=int, default=21)
    parser.add_argument("--flow-iterations", type=int, default=5)
    parser.add_argument("--flow-poly-n", type=int, default=7)
    parser.add_argument("--flow-poly-sigma", type=float, default=1.5)

    parser.add_argument("--consistency-threshold", type=float, default=DEFAULT_CONSISTENCY_THRESHOLD)
    parser.add_argument("--consistency-scale", type=float, default=DEFAULT_CONSISTENCY_SCALE)
    parser.add_argument(
        "--mask-blur-kernel",
        type=int,
        default=DEFAULT_MASK_BLUR_KERNEL,
        help="Odd kernel size for occlusion-mask blur. <=1 disables blur.",
    )
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
    Convert absolute path to project-relative POSIX path when possible.
    """
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path_value: Path) -> dict[str, Any]:
    """
    Read JSON object from path.
    """
    with path_value.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def load_frame(path_value: Path) -> np.ndarray:
    """
    Load frame as BGR uint8.
    """
    frame = cv2.imread(str(path_value), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Cannot decode frame image: {path_value}")
    return frame


def build_frame_windows(frame_count: int) -> dict[str, Any]:
    """
    Build default attack/hold/release windows for runtime playback.
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
    return {
        "neutralIndex": neutral_index,
        "attackIndices": list(range(attack_start, attack_end + 1)),
        "holdIndices": list(range(max(0, neutral_index - 1), min(frame_count - 1, neutral_index + 1) + 1)),
        "releaseIndices": list(range(release_start, release_end + 1)),
    }


def build_pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build pixel-coordinate grids.
    """
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return x, y


def compute_optical_flow(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    Compute dense optical flow from gray_a to gray_b.
    """
    return cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=float(args.flow_pyr_scale),
        levels=int(args.flow_levels),
        winsize=int(args.flow_winsize),
        iterations=int(args.flow_iterations),
        poly_n=int(args.flow_poly_n),
        poly_sigma=float(args.flow_poly_sigma),
        flags=0,
    )


def warp_with_flow(
    image: np.ndarray,
    flow: np.ndarray,
    factor: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """
    Warp an image using scaled flow displacement.
    """
    map_x = grid_x - (flow[..., 0] * factor)
    map_y = grid_y - (flow[..., 1] * factor)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def flow_consistency_mask(
    flow_forward: np.ndarray,
    flow_backward: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    threshold: float,
    scale: float,
    blur_kernel: int,
) -> np.ndarray:
    """
    Estimate valid-flow mask with forward/backward consistency.
    """
    map_x = grid_x + flow_forward[..., 0]
    map_y = grid_y + flow_forward[..., 1]
    sampled_backward = cv2.remap(
        flow_backward,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    residual = np.linalg.norm(flow_forward + sampled_backward, axis=2)
    magnitude = np.linalg.norm(flow_forward, axis=2) + np.linalg.norm(sampled_backward, axis=2)
    valid = (residual <= (float(threshold) + (float(scale) * magnitude))).astype(np.float32)

    kernel = int(blur_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        valid = cv2.GaussianBlur(valid, (kernel, kernel), 0)
    return np.clip(valid, 0.0, 1.0)


def synthesize_intermediate_frame(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    flow_ab: np.ndarray,
    flow_ba: np.ndarray,
    mask_ab: np.ndarray,
    mask_ba: np.ndarray,
    alpha: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """
    Synthesize one intermediate frame at alpha in [0, 1].
    """
    warped_a = warp_with_flow(frame_a, flow_ab, alpha, grid_x, grid_y).astype(np.float32)
    warped_b = warp_with_flow(frame_b, flow_ba, 1.0 - alpha, grid_x, grid_y).astype(np.float32)

    warped_mask_a = warp_with_flow(mask_ab, flow_ab, alpha, grid_x, grid_y).astype(np.float32)
    warped_mask_b = warp_with_flow(mask_ba, flow_ba, 1.0 - alpha, grid_x, grid_y).astype(np.float32)

    weight_a = ((1.0 - alpha) * warped_mask_a)[..., None]
    weight_b = (alpha * warped_mask_b)[..., None]
    weight_sum = np.maximum(weight_a + weight_b, 1e-6)
    blended = ((warped_a * weight_a) + (warped_b * weight_b)) / weight_sum
    return np.clip(blended, 0, 255).astype(np.uint8)


def clear_png_files(directory: Path) -> None:
    """
    Delete png files from directory.
    """
    for file_path in directory.glob("*.png"):
        file_path.unlink()


def process_viseme_entry(
    index: int,
    entry: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    """
    Interpolate one viseme frame sequence and return manifest entry.
    """
    viseme = str(entry.get("viseme", "")).strip()
    if not viseme:
        raise ValueError("Viseme entry missing 'viseme'.")

    frame_paths_raw = entry.get("frames")
    if not isinstance(frame_paths_raw, list) or not frame_paths_raw:
        raise ValueError(f"Viseme {viseme} has empty frames list.")

    input_frames = [resolve_path(str(frame_rel)) for frame_rel in frame_paths_raw]
    for input_path in input_frames:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input frame for {viseme}: {input_path}")

    viseme_output_dir = output_dir / viseme
    viseme_output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.overwrite):
        clear_png_files(viseme_output_dir)

    existing_output_frames = sorted(viseme_output_dir.glob("frame_*.png"))
    if existing_output_frames and not bool(args.overwrite):
        existing_rel = [to_project_relative(path) for path in existing_output_frames]
        windows = build_frame_windows(len(existing_rel))
        cached_entry = {
            "viseme": viseme,
            "clip": str(entry.get("clip", "")),
            "frameCount": len(existing_rel),
            "frames": existing_rel,
            "neutralIndex": windows["neutralIndex"],
            "attackIndices": windows["attackIndices"],
            "holdIndices": windows["holdIndices"],
            "releaseIndices": windows["releaseIndices"],
            "sourceFrameCount": len(input_frames),
            "subframesPerPair": int(args.subframes_per_pair),
        }
        if "fromViseme" in entry:
            cached_entry["fromViseme"] = str(entry.get("fromViseme", ""))
        if "toViseme" in entry:
            cached_entry["toViseme"] = str(entry.get("toViseme", ""))
        return index, cached_entry

    subframes = max(0, int(args.subframes_per_pair))
    input_arrays = [load_frame(path) for path in input_frames]
    height, width = input_arrays[0].shape[:2]
    grid_x, grid_y = build_pixel_grid(width, height)

    output_arrays: list[np.ndarray] = []
    for pair_index in range(len(input_arrays) - 1):
        frame_a = input_arrays[pair_index]
        frame_b = input_arrays[pair_index + 1]
        output_arrays.append(frame_a)

        if subframes <= 0:
            continue

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        flow_ab = compute_optical_flow(gray_a, gray_b, args)
        flow_ba = compute_optical_flow(gray_b, gray_a, args)
        mask_ab = flow_consistency_mask(
            flow_forward=flow_ab,
            flow_backward=flow_ba,
            grid_x=grid_x,
            grid_y=grid_y,
            threshold=float(args.consistency_threshold),
            scale=float(args.consistency_scale),
            blur_kernel=int(args.mask_blur_kernel),
        )
        mask_ba = flow_consistency_mask(
            flow_forward=flow_ba,
            flow_backward=flow_ab,
            grid_x=grid_x,
            grid_y=grid_y,
            threshold=float(args.consistency_threshold),
            scale=float(args.consistency_scale),
            blur_kernel=int(args.mask_blur_kernel),
        )

        for subframe_index in range(1, subframes + 1):
            alpha = float(subframe_index) / float(subframes + 1)
            synthetic_frame = synthesize_intermediate_frame(
                frame_a=frame_a,
                frame_b=frame_b,
                flow_ab=flow_ab,
                flow_ba=flow_ba,
                mask_ab=mask_ab,
                mask_ba=mask_ba,
                alpha=alpha,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            output_arrays.append(synthetic_frame)

    output_arrays.append(input_arrays[-1])

    frame_rel_paths: list[str] = []
    for output_index, frame_array in enumerate(output_arrays, start=1):
        filename = FRAME_FILE_PATTERN.format(output_index)
        output_path = viseme_output_dir / filename
        if not cv2.imwrite(str(output_path), frame_array):
            raise RuntimeError(f"Failed writing output frame: {output_path}")
        frame_rel_paths.append(to_project_relative(output_path))

    windows = build_frame_windows(len(frame_rel_paths))
    output_entry: dict[str, Any] = {
        "viseme": viseme,
        "clip": str(entry.get("clip", "")),
        "frameCount": len(frame_rel_paths),
        "frames": frame_rel_paths,
        "neutralIndex": windows["neutralIndex"],
        "attackIndices": windows["attackIndices"],
        "holdIndices": windows["holdIndices"],
        "releaseIndices": windows["releaseIndices"],
        "sourceFrameCount": len(input_frames),
        "subframesPerPair": subframes,
    }
    if "fromViseme" in entry:
        output_entry["fromViseme"] = str(entry.get("fromViseme", ""))
    if "toViseme" in entry:
        output_entry["toViseme"] = str(entry.get("toViseme", ""))
    return index, output_entry


def run() -> None:
    """
    Interpolate transition frames and write a new manifest.
    """
    args = parse_args()
    input_manifest_path = resolve_path(args.frames_manifest)
    output_dir = resolve_path(args.output_dir)
    output_manifest_path = resolve_path(args.output_manifest)

    if not input_manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest_path}")

    payload = read_json(input_manifest_path)
    visemes = payload.get("visemes")
    if not isinstance(visemes, list) or not visemes:
        raise ValueError(f"No viseme entries in manifest: {input_manifest_path}")

    selected_visemes = visemes
    max_visemes = int(args.max_visemes)
    if max_visemes > 0:
        selected_visemes = visemes[:max_visemes]

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))
    indexed_results: list[tuple[int, dict[str, Any]]] = []

    print(
        f"[info] visemes={len(selected_visemes)} subframesPerPair={int(args.subframes_per_pair)} "
        f"jobs={jobs}"
    )

    if jobs == 1:
        for idx, entry in enumerate(selected_visemes):
            result = process_viseme_entry(idx, entry, output_dir, args)
            indexed_results.append(result)
            print(f"[ok] {result[1]['viseme']} frames={result[1]['frameCount']}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures: dict[concurrent.futures.Future[tuple[int, dict[str, Any]]], str] = {}
            for idx, entry in enumerate(selected_visemes):
                viseme = str(entry.get("viseme", "")).strip()
                future = executor.submit(process_viseme_entry, idx, entry, output_dir, args)
                futures[future] = viseme
            for future in concurrent.futures.as_completed(futures):
                viseme = futures[future]
                result = future.result()
                indexed_results.append(result)
                print(f"[ok] {viseme} frames={result[1]['frameCount']}")

    indexed_results.sort(key=lambda pair: pair[0])
    output_visemes = [pair[1] for pair in indexed_results]

    source_target_fps = float(payload.get("targetFps", 0.0) or 0.0)
    upsample_multiplier = max(1, int(args.subframes_per_pair) + 1)
    output_target_fps = source_target_fps * upsample_multiplier if source_target_fps > 0 else 0.0

    output_manifest = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceFramesManifest": to_project_relative(input_manifest_path),
        "baseImage": str(payload.get("baseImage", "")),
        "clipVariant": str(payload.get("clipVariant", "org")),
        "targetFps": float(output_target_fps),
        "interpolationMode": "optical_flow_pairwise",
        "subframesPerPair": int(args.subframes_per_pair),
        "flowModel": "opencv_farneback",
        "flowParams": {
            "pyrScale": float(args.flow_pyr_scale),
            "levels": int(args.flow_levels),
            "winSize": int(args.flow_winsize),
            "iterations": int(args.flow_iterations),
            "polyN": int(args.flow_poly_n),
            "polySigma": float(args.flow_poly_sigma),
            "consistencyThreshold": float(args.consistency_threshold),
            "consistencyScale": float(args.consistency_scale),
            "maskBlurKernel": int(args.mask_blur_kernel),
        },
        "outputDir": to_project_relative(output_dir),
        "visemeCount": len(output_visemes),
        "visemes": output_visemes,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2)

    print(f"[ok] optical-flow manifest -> {output_manifest_path}")
    print(f"[ok] interpolated viseme sets -> {len(output_visemes)}")


if __name__ == "__main__":
    run()
