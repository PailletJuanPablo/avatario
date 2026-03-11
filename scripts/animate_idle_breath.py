from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT_VIDEO = Path("inputs/idlevid.mp4")
DEFAULT_OUTPUT_VIDEO = Path("inputs/idlevid_breath.mp4")
DEFAULT_VIDEO_CODEC = "mp4v"
DEFAULT_BREATH_CYCLE_SEC = 4.9
DEFAULT_BREATH_STRENGTH = 1.15
DEFAULT_SECONDARY_WAVE_SCALE = 0.18
DEFAULT_SECONDARY_WAVE_PHASE_RAD = 0.58
DEFAULT_TRANSCODE_CRF = 18
DEFAULT_TRANSCODE_PRESET = "medium"
DEFAULT_MASK_BLUR_SIGMA = 19.0
DEFAULT_SUBJECT_MASK_BLUR_SIGMA = 2.5
DEFAULT_SUBJECT_MASK_ERODE_PX = 3
SUBJECT_GRABCUT_ITERATIONS = 5
PROGRESS_FRAME_INTERVAL = 60

TORSO_ZONE_NAME = "torso"
LEFT_SHOULDER_ZONE_NAME = "left_shoulder"
RIGHT_SHOULDER_ZONE_NAME = "right_shoulder"

TORSO_CENTER_X_RATIO = 0.50
TORSO_CENTER_Y_RATIO = 0.77
TORSO_RADIUS_X_RATIO = 0.18
TORSO_RADIUS_Y_RATIO = 0.27
TORSO_TOP_FADE_START_RATIO = 0.43
TORSO_TOP_FADE_END_RATIO = 0.58
TORSO_BOTTOM_FADE_START_RATIO = 0.91
TORSO_BOTTOM_FADE_END_RATIO = 0.995
TORSO_ANCHOR_X_RATIO = 0.50
TORSO_ANCHOR_Y_RATIO = 0.68
TORSO_SCALE_X_AMPLITUDE = 0.0042
TORSO_SCALE_Y_AMPLITUDE = 0.0082
TORSO_SHIFT_X_RATIO = 0.0
TORSO_SHIFT_Y_RATIO = -0.0022
TORSO_PHASE_OFFSET_RAD = 0.0

SHOULDER_CENTER_Y_RATIO = 0.60
SHOULDER_RADIUS_X_RATIO = 0.115
SHOULDER_RADIUS_Y_RATIO = 0.11
SHOULDER_TOP_FADE_START_RATIO = 0.41
SHOULDER_TOP_FADE_END_RATIO = 0.49
SHOULDER_BOTTOM_FADE_START_RATIO = 0.68
SHOULDER_BOTTOM_FADE_END_RATIO = 0.78
SHOULDER_ANCHOR_Y_RATIO = 0.58
SHOULDER_SCALE_X_AMPLITUDE = 0.0021
SHOULDER_SCALE_Y_AMPLITUDE = 0.0045
SHOULDER_SHIFT_Y_RATIO = -0.0027
SHOULDER_OUTWARD_SHIFT_RATIO = 0.00095
SHOULDER_PHASE_OFFSET_RAD = 0.16
SHOULDER_INHALE_BIAS = 0.22


@dataclass(frozen=True)
class MotionZoneConfig:
    """One masked motion zone used to fake idle breathing in a fixed shot."""

    name: str
    center_x_ratio: float
    center_y_ratio: float
    radius_x_ratio: float
    radius_y_ratio: float
    top_fade_start_ratio: float
    top_fade_end_ratio: float
    bottom_fade_start_ratio: float
    bottom_fade_end_ratio: float
    anchor_x_ratio: float
    anchor_y_ratio: float
    scale_x_amplitude: float
    scale_y_amplitude: float
    shift_x_ratio: float
    shift_y_ratio: float
    phase_offset_rad: float
    inhale_bias: float
    blur_sigma: float


@dataclass(frozen=True)
class BreathAnimationConfig:
    """Aggregate configuration for the full idle breathing render."""

    breath_cycle_sec: float
    secondary_wave_scale: float
    secondary_wave_phase_rad: float
    motion_zones: tuple[MotionZoneConfig, ...]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for idle breathing rendering."""

    parser = argparse.ArgumentParser(
        description="Render one subtle torso and shoulder breathing loop for a fixed idle MP4.",
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help="Source idle video path.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=DEFAULT_OUTPUT_VIDEO,
        help="Output video path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output video when it already exists.",
    )
    parser.add_argument(
        "--breath-cycle-sec",
        type=float,
        default=DEFAULT_BREATH_CYCLE_SEC,
        help="Duration of one inhale-exhale cycle in seconds.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=DEFAULT_BREATH_STRENGTH,
        help="Global scalar for the breathing deformation strength.",
    )
    return parser.parse_args()


def create_motion_zone(
    name: str,
    center_x_ratio: float,
    center_y_ratio: float,
    radius_x_ratio: float,
    radius_y_ratio: float,
    top_fade_start_ratio: float,
    top_fade_end_ratio: float,
    bottom_fade_start_ratio: float,
    bottom_fade_end_ratio: float,
    anchor_x_ratio: float,
    anchor_y_ratio: float,
    scale_x_amplitude: float,
    scale_y_amplitude: float,
    shift_x_ratio: float,
    shift_y_ratio: float,
    phase_offset_rad: float,
    inhale_bias: float,
    blur_sigma: float,
    strength: float,
) -> MotionZoneConfig:
    """Create one motion zone with the global strength applied."""

    safe_strength = max(0.0, strength)
    return MotionZoneConfig(
        name=name,
        center_x_ratio=center_x_ratio,
        center_y_ratio=center_y_ratio,
        radius_x_ratio=radius_x_ratio,
        radius_y_ratio=radius_y_ratio,
        top_fade_start_ratio=top_fade_start_ratio,
        top_fade_end_ratio=top_fade_end_ratio,
        bottom_fade_start_ratio=bottom_fade_start_ratio,
        bottom_fade_end_ratio=bottom_fade_end_ratio,
        anchor_x_ratio=anchor_x_ratio,
        anchor_y_ratio=anchor_y_ratio,
        scale_x_amplitude=scale_x_amplitude * safe_strength,
        scale_y_amplitude=scale_y_amplitude * safe_strength,
        shift_x_ratio=shift_x_ratio * safe_strength,
        shift_y_ratio=shift_y_ratio * safe_strength,
        phase_offset_rad=phase_offset_rad,
        inhale_bias=inhale_bias,
        blur_sigma=blur_sigma,
    )


def build_breath_config(args: argparse.Namespace) -> BreathAnimationConfig:
    """Create one normalized breathing configuration from parsed arguments."""

    safe_strength = max(0.0, float(args.strength or 0.0))
    torso_zone = create_motion_zone(
        name=TORSO_ZONE_NAME,
        center_x_ratio=TORSO_CENTER_X_RATIO,
        center_y_ratio=TORSO_CENTER_Y_RATIO,
        radius_x_ratio=TORSO_RADIUS_X_RATIO,
        radius_y_ratio=TORSO_RADIUS_Y_RATIO,
        top_fade_start_ratio=TORSO_TOP_FADE_START_RATIO,
        top_fade_end_ratio=TORSO_TOP_FADE_END_RATIO,
        bottom_fade_start_ratio=TORSO_BOTTOM_FADE_START_RATIO,
        bottom_fade_end_ratio=TORSO_BOTTOM_FADE_END_RATIO,
        anchor_x_ratio=TORSO_ANCHOR_X_RATIO,
        anchor_y_ratio=TORSO_ANCHOR_Y_RATIO,
        scale_x_amplitude=TORSO_SCALE_X_AMPLITUDE,
        scale_y_amplitude=TORSO_SCALE_Y_AMPLITUDE,
        shift_x_ratio=TORSO_SHIFT_X_RATIO,
        shift_y_ratio=TORSO_SHIFT_Y_RATIO,
        phase_offset_rad=TORSO_PHASE_OFFSET_RAD,
        inhale_bias=0.0,
        blur_sigma=DEFAULT_MASK_BLUR_SIGMA,
        strength=safe_strength,
    )
    left_shoulder_zone = create_motion_zone(
        name=LEFT_SHOULDER_ZONE_NAME,
        center_x_ratio=0.385,
        center_y_ratio=SHOULDER_CENTER_Y_RATIO,
        radius_x_ratio=SHOULDER_RADIUS_X_RATIO,
        radius_y_ratio=SHOULDER_RADIUS_Y_RATIO,
        top_fade_start_ratio=SHOULDER_TOP_FADE_START_RATIO,
        top_fade_end_ratio=SHOULDER_TOP_FADE_END_RATIO,
        bottom_fade_start_ratio=SHOULDER_BOTTOM_FADE_START_RATIO,
        bottom_fade_end_ratio=SHOULDER_BOTTOM_FADE_END_RATIO,
        anchor_x_ratio=0.39,
        anchor_y_ratio=SHOULDER_ANCHOR_Y_RATIO,
        scale_x_amplitude=SHOULDER_SCALE_X_AMPLITUDE,
        scale_y_amplitude=SHOULDER_SCALE_Y_AMPLITUDE,
        shift_x_ratio=-SHOULDER_OUTWARD_SHIFT_RATIO,
        shift_y_ratio=SHOULDER_SHIFT_Y_RATIO,
        phase_offset_rad=SHOULDER_PHASE_OFFSET_RAD,
        inhale_bias=SHOULDER_INHALE_BIAS,
        blur_sigma=DEFAULT_MASK_BLUR_SIGMA,
        strength=safe_strength,
    )
    right_shoulder_zone = create_motion_zone(
        name=RIGHT_SHOULDER_ZONE_NAME,
        center_x_ratio=0.615,
        center_y_ratio=SHOULDER_CENTER_Y_RATIO,
        radius_x_ratio=SHOULDER_RADIUS_X_RATIO,
        radius_y_ratio=SHOULDER_RADIUS_Y_RATIO,
        top_fade_start_ratio=SHOULDER_TOP_FADE_START_RATIO,
        top_fade_end_ratio=SHOULDER_TOP_FADE_END_RATIO,
        bottom_fade_start_ratio=SHOULDER_BOTTOM_FADE_START_RATIO,
        bottom_fade_end_ratio=SHOULDER_BOTTOM_FADE_END_RATIO,
        anchor_x_ratio=0.61,
        anchor_y_ratio=SHOULDER_ANCHOR_Y_RATIO,
        scale_x_amplitude=SHOULDER_SCALE_X_AMPLITUDE,
        scale_y_amplitude=SHOULDER_SCALE_Y_AMPLITUDE,
        shift_x_ratio=SHOULDER_OUTWARD_SHIFT_RATIO,
        shift_y_ratio=SHOULDER_SHIFT_Y_RATIO,
        phase_offset_rad=SHOULDER_PHASE_OFFSET_RAD,
        inhale_bias=SHOULDER_INHALE_BIAS,
        blur_sigma=DEFAULT_MASK_BLUR_SIGMA,
        strength=safe_strength,
    )
    return BreathAnimationConfig(
        breath_cycle_sec=max(1.0, float(args.breath_cycle_sec or DEFAULT_BREATH_CYCLE_SEC)),
        secondary_wave_scale=DEFAULT_SECONDARY_WAVE_SCALE,
        secondary_wave_phase_rad=DEFAULT_SECONDARY_WAVE_PHASE_RAD,
        motion_zones=(torso_zone, left_shoulder_zone, right_shoulder_zone),
    )


def resolve_video_writer(
    output_path: Path,
    frames_per_second: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Create one OpenCV MP4 writer or raise a clear error when it cannot open."""

    fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, frames_per_second, frame_size)
    if writer.isOpened():
        return writer
    raise RuntimeError(f"Unable to open video writer for {output_path}")


def fill_polygon(mask_image: np.ndarray, polygon_points: list[tuple[int, int]], value: int) -> None:
    """Fill one convex polygon in-place."""

    polygon_array = np.asarray(polygon_points, dtype=np.int32)
    cv2.fillConvexPoly(mask_image, polygon_array, int(value))


def build_subject_seed_mask(frame_width: int, frame_height: int) -> np.ndarray:
    """Create one guided GrabCut seed mask tuned for the fixed idle framing."""

    safe_width = max(1, int(frame_width))
    safe_height = max(1, int(frame_height))
    seed_mask = np.full((safe_height, safe_width), cv2.GC_PR_BGD, dtype=np.uint8)
    seed_mask[: int(safe_height * 0.12), :] = cv2.GC_BGD
    seed_mask[:, : int(safe_width * 0.16)] = cv2.GC_BGD
    seed_mask[:, int(safe_width * 0.84) :] = cv2.GC_BGD
    seed_mask[int(safe_height * 0.92) :, :] = cv2.GC_BGD

    fill_polygon(
        seed_mask,
        [
            (int(safe_width * 0.34), int(safe_height * 0.20)),
            (int(safe_width * 0.44), int(safe_height * 0.16)),
            (int(safe_width * 0.57), int(safe_height * 0.16)),
            (int(safe_width * 0.66), int(safe_height * 0.22)),
            (int(safe_width * 0.71), int(safe_height * 0.38)),
            (int(safe_width * 0.73), int(safe_height * 0.62)),
            (int(safe_width * 0.71), int(safe_height * 0.88)),
            (int(safe_width * 0.29), int(safe_height * 0.88)),
            (int(safe_width * 0.27), int(safe_height * 0.62)),
            (int(safe_width * 0.29), int(safe_height * 0.38)),
        ],
        cv2.GC_PR_FGD,
    )
    fill_polygon(
        seed_mask,
        [
            (int(safe_width * 0.40), int(safe_height * 0.18)),
            (int(safe_width * 0.50), int(safe_height * 0.12)),
            (int(safe_width * 0.60), int(safe_height * 0.18)),
            (int(safe_width * 0.60), int(safe_height * 0.42)),
            (int(safe_width * 0.40), int(safe_height * 0.42)),
        ],
        cv2.GC_FGD,
    )
    fill_polygon(
        seed_mask,
        [
            (0, 0),
            (int(safe_width * 0.37), 0),
            (int(safe_width * 0.37), int(safe_height * 0.44)),
            (int(safe_width * 0.30), int(safe_height * 0.55)),
            (0, int(safe_height * 0.55)),
        ],
        cv2.GC_BGD,
    )
    return seed_mask


def build_subject_mask(reference_frame: np.ndarray) -> np.ndarray:
    """Build one soft subject matte so the background stays completely static."""

    frame_height, frame_width = reference_frame.shape[:2]
    seed_mask = build_subject_seed_mask(frame_width, frame_height)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        reference_frame,
        seed_mask,
        None,
        background_model,
        foreground_model,
        SUBJECT_GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_MASK,
    )
    subject_mask = np.where(
        (seed_mask == cv2.GC_FGD) | (seed_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    erode_size = max(1, int(DEFAULT_SUBJECT_MASK_ERODE_PX))
    erode_kernel = np.ones((erode_size, erode_size), dtype=np.uint8)
    subject_mask = cv2.erode(subject_mask, erode_kernel, iterations=1)
    subject_mask = cv2.morphologyEx(subject_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    subject_mask = cv2.GaussianBlur(subject_mask, (0, 0), DEFAULT_SUBJECT_MASK_BLUR_SIGMA)
    normalized_mask = np.clip(subject_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.repeat(normalized_mask[:, :, None], 3, axis=2)


def build_motion_zone_mask(frame_width: int, frame_height: int, zone: MotionZoneConfig) -> np.ndarray:
    """Build one soft alpha mask for a specific motion zone."""

    safe_width = max(1, int(frame_width))
    safe_height = max(1, int(frame_height))
    grid_x, grid_y = np.meshgrid(
        np.arange(safe_width, dtype=np.float32),
        np.arange(safe_height, dtype=np.float32),
    )
    center_x = float(safe_width) * zone.center_x_ratio
    center_y = float(safe_height) * zone.center_y_ratio
    radius_x = max(1.0, float(safe_width) * zone.radius_x_ratio)
    radius_y = max(1.0, float(safe_height) * zone.radius_y_ratio)
    ellipse_distance = (
        ((grid_x - center_x) / radius_x) ** 2
        + ((grid_y - center_y) / radius_y) ** 2
    )
    ellipse_mask = np.clip(1.0 - ellipse_distance, 0.0, 1.0) ** 1.5

    top_start = float(safe_height) * zone.top_fade_start_ratio
    top_end = max(top_start + 1.0, float(safe_height) * zone.top_fade_end_ratio)
    top_fade = np.clip((grid_y - top_start) / (top_end - top_start), 0.0, 1.0)

    bottom_start = float(safe_height) * zone.bottom_fade_start_ratio
    bottom_end = max(bottom_start + 1.0, float(safe_height) * zone.bottom_fade_end_ratio)
    bottom_fade = 1.0 - np.clip((grid_y - bottom_start) / (bottom_end - bottom_start), 0.0, 1.0)

    mask = ellipse_mask * top_fade * bottom_fade
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), zone.blur_sigma)
    normalized_mask = np.clip(blurred_mask, 0.0, 1.0)
    return np.repeat(normalized_mask[:, :, None], 3, axis=2)


def build_zone_wave(
    frame_index: int,
    frames_per_second: float,
    animation_config: BreathAnimationConfig,
    zone: MotionZoneConfig,
) -> float:
    """Compute one smooth breathing waveform for a specific motion zone."""

    safe_fps = max(1.0, float(frames_per_second or 0.0))
    elapsed_sec = float(frame_index) / safe_fps
    phase = ((2.0 * math.pi * elapsed_sec) / animation_config.breath_cycle_sec) + zone.phase_offset_rad
    primary_wave = math.sin(phase)
    secondary_wave = math.sin((phase * 2.0) + animation_config.secondary_wave_phase_rad)
    weighted_wave = (
        primary_wave + (animation_config.secondary_wave_scale * secondary_wave)
    ) / (1.0 + animation_config.secondary_wave_scale)
    inhale_component = max(0.0, weighted_wave)
    biased_wave = weighted_wave + (zone.inhale_bias * inhale_component)
    return max(-1.0, min(1.0, biased_wave))


def build_affine_matrix(
    frame_width: int,
    frame_height: int,
    zone_wave: float,
    zone: MotionZoneConfig,
) -> np.ndarray:
    """Create one subtle affine transform for a motion zone."""

    anchor_x = float(frame_width) * zone.anchor_x_ratio
    anchor_y = float(frame_height) * zone.anchor_y_ratio
    scale_x = 1.0 + (zone.scale_x_amplitude * zone_wave)
    scale_y = 1.0 + (zone.scale_y_amplitude * zone_wave)
    shift_x = float(frame_width) * zone.shift_x_ratio * zone_wave
    shift_y = float(frame_height) * zone.shift_y_ratio * zone_wave
    return np.asarray(
        [
            [scale_x, 0.0, anchor_x - (scale_x * anchor_x) + shift_x],
            [0.0, scale_y, anchor_y - (scale_y * anchor_y) + shift_y],
        ],
        dtype=np.float32,
    )


def apply_motion_zone(
    frame_image: np.ndarray,
    zone_mask: np.ndarray,
    zone_wave: float,
    zone: MotionZoneConfig,
) -> np.ndarray:
    """Blend one masked affine motion zone into the current frame."""

    frame_height, frame_width = frame_image.shape[:2]
    affine_matrix = build_affine_matrix(frame_width, frame_height, zone_wave, zone)
    warped_frame = cv2.warpAffine(
        frame_image,
        affine_matrix,
        (frame_width, frame_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    base_frame = frame_image.astype(np.float32)
    warped_float = warped_frame.astype(np.float32)
    blended = (zone_mask * warped_float) + ((1.0 - zone_mask) * base_frame)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def transcode_with_ffmpeg(source_path: Path, output_path: Path) -> bool:
    """Transcode the temporary MP4 into browser-friendly H.264 when FFmpeg is available."""

    ffmpeg_binary = shutil.which("ffmpeg")
    if not ffmpeg_binary:
        return False
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        DEFAULT_TRANSCODE_PRESET,
        "-crf",
        str(DEFAULT_TRANSCODE_CRF),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        source_path.unlink(missing_ok=True)
        return True
    print(completed.stdout)
    print(completed.stderr)
    return False


def render_idle_breath_video(
    input_video: Path,
    output_video: Path,
    animation_config: BreathAnimationConfig,
) -> None:
    """Render one breathing-enhanced MP4 from a fixed idle source video."""

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_video}")

    frames_per_second = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    safe_fps = frames_per_second if frames_per_second > 0 else 29.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid input frame size: {frame_width}x{frame_height}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_video.with_name(f"{output_video.stem}.tmp_render.mp4")
    ok, reference_frame = capture.read()
    if not ok or reference_frame is None:
        capture.release()
        raise RuntimeError(f"Unable to read reference frame from {input_video}")
    subject_mask = build_subject_mask(reference_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    zone_masks = {
        zone.name: np.clip(build_motion_zone_mask(frame_width, frame_height, zone) * subject_mask, 0.0, 1.0)
        for zone in animation_config.motion_zones
    }
    writer = resolve_video_writer(temporary_output, safe_fps, (frame_width, frame_height))

    frame_index = 0
    try:
        while True:
            ok, frame_image = capture.read()
            if not ok or frame_image is None:
                break
            animated_frame = frame_image
            for zone in animation_config.motion_zones:
                zone_wave = build_zone_wave(frame_index, safe_fps, animation_config, zone)
                animated_frame = apply_motion_zone(animated_frame, zone_masks[zone.name], zone_wave, zone)
            writer.write(animated_frame)
            frame_index += 1
            if frame_index % PROGRESS_FRAME_INTERVAL == 0:
                total_label = str(frame_total) if frame_total > 0 else "?"
                print(f"[render] frame={frame_index}/{total_label}")
    finally:
        writer.release()
        capture.release()

    if transcode_with_ffmpeg(temporary_output, output_video):
        print(f"[ok] rendered {output_video} with ffmpeg")
        return

    if output_video.exists():
        output_video.unlink()
    temporary_output.replace(output_video)
    print(f"[ok] rendered {output_video} without ffmpeg transcode")


def main() -> None:
    """Program entry point."""

    args = parse_args()
    input_video = Path(args.input_video).resolve()
    output_video = Path(args.output_video).resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if output_video.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output video already exists: {output_video}")
    animation_config = build_breath_config(args)
    render_idle_breath_video(input_video, output_video, animation_config)


if __name__ == "__main__":
    main()
