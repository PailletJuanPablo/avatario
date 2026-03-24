"""
Build one subtle idle motion PKL by repeating a neutral source frame and applying
the repository idle enhancement profile.
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import build_dataset_liveportrait_clips as dataset_clips


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PKL = "output_fasterliveportrait/driving_audio_preview.pkl"
DEFAULT_OUTPUT_PKL = "output_fasterliveportrait/idle/idle_subtle_30s.pkl"
DEFAULT_DURATION_SEC = 30.0
DEFAULT_OUTPUT_FPS = 25.0
DEFAULT_FRAME_INDEX = -1
DEFAULT_PROFILE = "subtle_idle_v2"

DEFAULT_IDLE_EDGE_FADE_SEC = 1.35
DEFAULT_IDLE_PRIMARY_FREQ_HZ = 0.075
DEFAULT_IDLE_SECONDARY_FREQ_SCALE = 1.70
DEFAULT_IDLE_YAW_AMPLITUDE_DEG = 0.48
DEFAULT_IDLE_PITCH_AMPLITUDE_DEG = 0.30
DEFAULT_IDLE_ROLL_AMPLITUDE_DEG = 0.22
DEFAULT_IDLE_TRANSLATION_AMPLITUDE = 0.00105
DEFAULT_IDLE_BLINK_MIN_INTERVAL_SEC = 3.1
DEFAULT_IDLE_BLINK_MAX_INTERVAL_SEC = 5.9
DEFAULT_IDLE_BLINK_DURATION_SEC = 0.070
DEFAULT_IDLE_BLINK_AMPLITUDE = 0.0190
DEFAULT_IDLE_BLINK_SIGN = 1.0
DEFAULT_IDLE_BLINK_CLOSE_SEC = 0.050
DEFAULT_IDLE_BLINK_HOLD_SEC = 0.052
DEFAULT_IDLE_BLINK_DOUBLE_PROBABILITY = 0.18
DEFAULT_IDLE_BLINK_IRREGULARITY = 0.32
DEFAULT_IDLE_BLINK_CLOSE_TARGET_SCALE = 1.65
DEFAULT_IDLE_BLINK_SOFT_UPPER_SCALE = 0.0
DEFAULT_IDLE_BLINK_SOFT_LOWER_SCALE = 0.0
DEFAULT_IDLE_BLINK_CENTER_SCALE = 0.050
DEFAULT_IDLE_BLINK_CENTER_FORCE_SCALE = 0.12
DEFAULT_IDLE_MOUTH_NEUTRAL_STRENGTH = 1.0
DEFAULT_IDLE_MOUTH_TARGET_QUANTILE = 0.50
DEFAULT_IDLE_MOUTH_FLOOR_SIGMA = 0.20
DEFAULT_IDLE_MOUTH_OUTLIER_Z = 2.35
DEFAULT_IDLE_MOUTH_DELTA_Z = 1.95
DEFAULT_IDLE_MOUTH_EMA_ALPHA = 0.18
DEFAULT_IDLE_MOUTH_BLINK_LOCK_STRENGTH = 1.0
DEFAULT_IDLE_MOUTH_BLINK_LOCK_THRESHOLD = 0.08
DEFAULT_IDLE_MOUTH_BLINK_HARD_LOCK_THRESHOLD = 0.38
DEFAULT_IDLE_MOUTH_BLINK_REFERENCE_EMA_ALPHA = 0.01
DEFAULT_IDLE_MOUTH_BLINK_FREEZE_THRESHOLD = 0.05
DEFAULT_IDLE_MOUTH_BLINK_FREEZE_PADDING_FRAMES = 2

DEFAULT_EXTRA_HEAD_YAW_DEG = 0.22
DEFAULT_EXTRA_HEAD_PITCH_DEG = 0.14
DEFAULT_EXTRA_HEAD_ROLL_DEG = 0.10
DEFAULT_EXTRA_SCALE_AMPLITUDE = 0.0023
DEFAULT_EXTRA_TX_AMPLITUDE = 0.00042
DEFAULT_EXTRA_TY_AMPLITUDE = 0.00058
DEFAULT_EXTRA_TZ_AMPLITUDE = 0.00024
DEFAULT_FACE_BROW_Y_AMPLITUDE = 0.0017
DEFAULT_FACE_BROW_Z_AMPLITUDE = 0.0008
DEFAULT_FACE_LOWER_EYE_Y_AMPLITUDE = 0.0008
DEFAULT_FACE_CENTER_EYE_Y_AMPLITUDE = 0.0010
DEFAULT_FACE_SOFT_SQUINT_AMPLITUDE = 0.00075
DEFAULT_FACE_FREQ_SCALE = 1.42
DEFAULT_FACE_SECONDARY_FREQ_SCALE = 1.95


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one subtle 30s idle PKL with blinks and minimal head drift."
    )
    parser.add_argument("--source-pkl", default=DEFAULT_SOURCE_PKL)
    parser.add_argument("--output-pkl", default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="0 keeps the source fps or falls back to 25.",
    )
    parser.add_argument(
        "--base-frame-index",
        type=int,
        default=DEFAULT_FRAME_INDEX,
        help="-1 selects the most neutral frame automatically.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE)

    parser.add_argument("--eye-hard-dy-min", type=float, default=dataset_clips.DEFAULT_EYE_HARD_DY_MIN)
    parser.add_argument("--eye-hard-dy-max", type=float, default=dataset_clips.DEFAULT_EYE_HARD_DY_MAX)
    parser.add_argument("--idle-motion-enhancement", action="store_true", default=True)
    parser.add_argument("--no-idle-motion-enhancement", dest="idle_motion_enhancement", action="store_false")
    parser.add_argument("--force-idle-motion-enhancement", action="store_true", default=True)
    parser.add_argument("--idle-edge-fade-sec", type=float, default=DEFAULT_IDLE_EDGE_FADE_SEC)
    parser.add_argument("--idle-primary-freq-hz", type=float, default=DEFAULT_IDLE_PRIMARY_FREQ_HZ)
    parser.add_argument("--idle-secondary-freq-scale", type=float, default=DEFAULT_IDLE_SECONDARY_FREQ_SCALE)
    parser.add_argument("--idle-yaw-amplitude-deg", type=float, default=DEFAULT_IDLE_YAW_AMPLITUDE_DEG)
    parser.add_argument("--idle-pitch-amplitude-deg", type=float, default=DEFAULT_IDLE_PITCH_AMPLITUDE_DEG)
    parser.add_argument("--idle-roll-amplitude-deg", type=float, default=DEFAULT_IDLE_ROLL_AMPLITUDE_DEG)
    parser.add_argument("--idle-translation-amplitude", type=float, default=DEFAULT_IDLE_TRANSLATION_AMPLITUDE)
    parser.add_argument("--idle-blink-min-interval-sec", type=float, default=DEFAULT_IDLE_BLINK_MIN_INTERVAL_SEC)
    parser.add_argument("--idle-blink-max-interval-sec", type=float, default=DEFAULT_IDLE_BLINK_MAX_INTERVAL_SEC)
    parser.add_argument("--idle-blink-duration-sec", type=float, default=DEFAULT_IDLE_BLINK_DURATION_SEC)
    parser.add_argument("--idle-blink-amplitude", type=float, default=DEFAULT_IDLE_BLINK_AMPLITUDE)
    parser.add_argument("--idle-blink-sign", type=float, default=DEFAULT_IDLE_BLINK_SIGN)
    parser.add_argument("--idle-blink-close-sec", type=float, default=DEFAULT_IDLE_BLINK_CLOSE_SEC)
    parser.add_argument("--idle-blink-hold-sec", type=float, default=DEFAULT_IDLE_BLINK_HOLD_SEC)
    parser.add_argument("--idle-blink-double-probability", type=float, default=DEFAULT_IDLE_BLINK_DOUBLE_PROBABILITY)
    parser.add_argument("--idle-blink-irregularity", type=float, default=DEFAULT_IDLE_BLINK_IRREGULARITY)
    parser.add_argument("--idle-blink-close-target-scale", type=float, default=DEFAULT_IDLE_BLINK_CLOSE_TARGET_SCALE)
    parser.add_argument("--idle-blink-soft-upper-scale", type=float, default=DEFAULT_IDLE_BLINK_SOFT_UPPER_SCALE)
    parser.add_argument("--idle-blink-soft-lower-scale", type=float, default=DEFAULT_IDLE_BLINK_SOFT_LOWER_SCALE)
    parser.add_argument("--idle-blink-center-scale", type=float, default=DEFAULT_IDLE_BLINK_CENTER_SCALE)
    parser.add_argument(
        "--idle-blink-center-force-scale",
        type=float,
        default=DEFAULT_IDLE_BLINK_CENTER_FORCE_SCALE,
    )
    parser.add_argument("--idle-mouth-neutral-strength", type=float, default=DEFAULT_IDLE_MOUTH_NEUTRAL_STRENGTH)
    parser.add_argument("--idle-mouth-target-quantile", type=float, default=DEFAULT_IDLE_MOUTH_TARGET_QUANTILE)
    parser.add_argument("--idle-mouth-floor-sigma", type=float, default=DEFAULT_IDLE_MOUTH_FLOOR_SIGMA)
    parser.add_argument("--idle-mouth-outlier-z", type=float, default=DEFAULT_IDLE_MOUTH_OUTLIER_Z)
    parser.add_argument("--idle-mouth-delta-z", type=float, default=DEFAULT_IDLE_MOUTH_DELTA_Z)
    parser.add_argument("--idle-mouth-ema-alpha", type=float, default=DEFAULT_IDLE_MOUTH_EMA_ALPHA)
    parser.add_argument("--idle-mouth-blink-lock-strength", type=float, default=DEFAULT_IDLE_MOUTH_BLINK_LOCK_STRENGTH)
    parser.add_argument("--idle-mouth-blink-lock-threshold", type=float, default=DEFAULT_IDLE_MOUTH_BLINK_LOCK_THRESHOLD)
    parser.add_argument(
        "--idle-mouth-blink-hard-lock-threshold",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_HARD_LOCK_THRESHOLD,
    )
    parser.add_argument(
        "--idle-mouth-blink-reference-ema-alpha",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_REFERENCE_EMA_ALPHA,
    )
    parser.add_argument(
        "--idle-mouth-blink-freeze-threshold",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_FREEZE_THRESHOLD,
    )
    parser.add_argument(
        "--idle-mouth-blink-freeze-padding-frames",
        type=int,
        default=DEFAULT_IDLE_MOUTH_BLINK_FREEZE_PADDING_FRAMES,
    )
    parser.add_argument("--extra-head-yaw-deg", type=float, default=DEFAULT_EXTRA_HEAD_YAW_DEG)
    parser.add_argument("--extra-head-pitch-deg", type=float, default=DEFAULT_EXTRA_HEAD_PITCH_DEG)
    parser.add_argument("--extra-head-roll-deg", type=float, default=DEFAULT_EXTRA_HEAD_ROLL_DEG)
    parser.add_argument("--extra-scale-amplitude", type=float, default=DEFAULT_EXTRA_SCALE_AMPLITUDE)
    parser.add_argument("--extra-tx-amplitude", type=float, default=DEFAULT_EXTRA_TX_AMPLITUDE)
    parser.add_argument("--extra-ty-amplitude", type=float, default=DEFAULT_EXTRA_TY_AMPLITUDE)
    parser.add_argument("--extra-tz-amplitude", type=float, default=DEFAULT_EXTRA_TZ_AMPLITUDE)
    parser.add_argument("--face-brow-y-amplitude", type=float, default=DEFAULT_FACE_BROW_Y_AMPLITUDE)
    parser.add_argument("--face-brow-z-amplitude", type=float, default=DEFAULT_FACE_BROW_Z_AMPLITUDE)
    parser.add_argument("--face-lower-eye-y-amplitude", type=float, default=DEFAULT_FACE_LOWER_EYE_Y_AMPLITUDE)
    parser.add_argument("--face-center-eye-y-amplitude", type=float, default=DEFAULT_FACE_CENTER_EYE_Y_AMPLITUDE)
    parser.add_argument("--face-soft-squint-amplitude", type=float, default=DEFAULT_FACE_SOFT_SQUINT_AMPLITUDE)
    parser.add_argument("--face-freq-scale", type=float, default=DEFAULT_FACE_FREQ_SCALE)
    parser.add_argument("--face-secondary-freq-scale", type=float, default=DEFAULT_FACE_SECONDARY_FREQ_SCALE)
    return parser.parse_args()


def ensure_numpy_pickle_compatibility() -> None:
    numpy_core_numeric = sys.modules.get("numpy.core.numeric")
    if numpy_core_numeric is not None and "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = numpy_core_numeric


def load_pkl(path: Path) -> dict[str, Any]:
    ensure_numpy_pickle_compatibility()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported PKL payload type: {type(payload)}")
    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        raise ValueError(f"Invalid PKL motion payload: {path}")
    return payload


def extract_scalar(frame: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    return float(dataset_clips.extract_scalar_from_frame(frame, key, fallback))


def get_lip_energy_from_frame(frame: dict[str, Any]) -> float:
    exp_array = np.asarray(frame.get("exp"), dtype=np.float32)
    if exp_array.ndim == 3:
        frame_exp = exp_array[0]
    else:
        frame_exp = exp_array
    frame_exp = np.asarray(frame_exp, dtype=np.float32)
    positive = lambda value: float(max(0.0, value))
    return (
        positive(frame_exp[19, 1])
        + positive(frame_exp[20, 1])
        + 0.45 * positive(frame_exp[14, 1])
        + 0.20 * positive(frame_exp[17, 1])
    )


def robust_zscores(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, np.newaxis]
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = np.maximum(mad * 1.4826, 1e-6)
    zscores = np.abs((values - center) / scale)
    return np.sum(zscores, axis=1).astype(np.float32, copy=False)


def select_neutral_frame_index(motion: list[dict[str, Any]]) -> int:
    lip_energy = np.asarray([get_lip_energy_from_frame(frame) for frame in motion], dtype=np.float32)
    pose_vectors = np.asarray(
        [
            [
                extract_scalar(frame, "pitch", 0.0),
                extract_scalar(frame, "yaw", 0.0),
                extract_scalar(frame, "roll", 0.0),
            ]
            for frame in motion
        ],
        dtype=np.float32,
    )
    translation_vectors = np.asarray(
        [
            np.asarray(frame.get("t", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(-1)[:3]
            for frame in motion
        ],
        dtype=np.float32,
    )
    if translation_vectors.shape[1] < 3:
        padding = np.zeros((translation_vectors.shape[0], 3 - translation_vectors.shape[1]), dtype=np.float32)
        translation_vectors = np.concatenate((translation_vectors, padding), axis=1)
    eye_vectors = np.asarray(
        [
            np.asarray(frame.get("exp"), dtype=np.float32).reshape(21, 3)[[11, 13, 15], 1]
            for frame in motion
        ],
        dtype=np.float32,
    )

    lip_score = lip_energy / max(float(np.percentile(lip_energy, 90)) if lip_energy.size else 0.0, 1e-6)
    pose_score = robust_zscores(pose_vectors)
    translation_score = robust_zscores(translation_vectors)
    eye_score = robust_zscores(eye_vectors)
    total_score = (lip_score * 2.4) + pose_score + (translation_score * 0.45) + (eye_score * 0.55)
    return int(np.argmin(total_score))


def clone_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(frame)


def build_static_idle_payload(
    source_payload: dict[str, Any],
    base_frame_index: int,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    motion = source_payload.get("motion")
    if not isinstance(motion, list) or not motion:
        raise ValueError("Source PKL has no motion frames.")
    safe_index = max(0, min(int(base_frame_index), len(motion) - 1))
    base_frame = motion[safe_index]
    if not isinstance(base_frame, dict):
        raise ValueError(f"Base frame is not a dict: index={safe_index}")

    processed = dict(source_payload)
    processed_motion = [clone_frame(base_frame) for _ in range(max(1, int(frame_count)))]
    processed["motion"] = processed_motion
    processed["n_frames"] = len(processed_motion)
    processed["output_fps"] = int(round(float(fps)))
    processed["c_eyes_lst"] = []
    processed["c_lip_lst"] = []
    return processed


def annotate_output_metadata(
    payload: dict[str, Any],
    source_pkl_path: Path,
    base_frame_index: int,
    duration_sec: float,
    fps: float,
    args: argparse.Namespace,
) -> None:
    metadata = dict(payload.get("idlePklBuild") or {})
    metadata.update(
        {
            "profile": str(args.profile_name),
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "sourcePkl": dataset_clips.to_project_relative(source_pkl_path),
            "baseFrameIndex": int(base_frame_index),
            "durationSecRequested": float(duration_sec),
            "durationSecApprox": float(len(payload.get("motion", [])) / max(float(fps), 1e-6)),
            "fps": float(fps),
            "headMotionIntent": "subtle_alive",
            "mouthMotionIntent": "frozen_neutral",
            "notes": "Static neutral base frame with synthetic idle blink, head drift, and facial micro-motion.",
        }
    )
    payload["idlePklBuild"] = metadata


def build_idle_entry(duration_sec: float) -> dataset_clips.DatasetEntrySpec:
    safe_duration = max(0.1, float(duration_sec))
    return dataset_clips.DatasetEntrySpec(
        dataset_index=0,
        dataset_id="idle_subtle_30s",
        phrase="IDLE_SUBTLE_30S",
        duration_sec=safe_duration,
        audio_base64="",
        visemes=[
            dataset_clips.VisemeSpec(
                index=0,
                char="_",
                viseme="IDLE",
                start_sec=0.0,
                end_sec=safe_duration,
                duration_sec=safe_duration,
            )
        ],
        transitions=[],
    )


def build_wave(
    time_sec: np.ndarray,
    primary_hz: float,
    secondary_scale: float,
    phase_primary: float,
    phase_secondary: float,
) -> np.ndarray:
    return dataset_clips.build_idle_dual_wave(
        time_sec=time_sec,
        primary_hz=primary_hz,
        secondary_scale=secondary_scale,
        phase_primary=phase_primary,
        phase_secondary=phase_secondary,
    )


def apply_extra_idle_liveliness(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    motion = payload.get("motion")
    if not isinstance(motion, list) or len(motion) < 3:
        return payload

    frame_count = len(motion)
    fps = max(1.0, float(payload.get("output_fps", DEFAULT_OUTPUT_FPS) or DEFAULT_OUTPUT_FPS))
    duration_sec = float(frame_count - 1) / fps
    time_sec = np.linspace(0.0, duration_sec, frame_count, dtype=np.float32)
    envelope = dataset_clips.build_temporal_edge_envelope(
        frame_count=frame_count,
        fps=fps,
        fade_sec=float(args.idle_edge_fade_sec),
    ).astype(np.float32, copy=False)

    rng = np.random.default_rng(int(args.seed) + 41003)
    primary_hz = max(0.01, float(args.idle_primary_freq_hz))
    face_primary_hz = primary_hz * max(1.05, float(args.face_freq_scale))
    face_secondary_scale = max(1.1, float(args.face_secondary_freq_scale))

    yaw_wave = build_wave(
        time_sec=time_sec,
        primary_hz=primary_hz * float(rng.uniform(0.95, 1.12)),
        secondary_scale=max(1.05, float(args.idle_secondary_freq_scale)),
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    pitch_wave = build_wave(
        time_sec=time_sec,
        primary_hz=primary_hz * float(rng.uniform(0.92, 1.08)),
        secondary_scale=max(1.05, float(args.idle_secondary_freq_scale)),
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    roll_wave = build_wave(
        time_sec=time_sec,
        primary_hz=primary_hz * float(rng.uniform(0.98, 1.16)),
        secondary_scale=max(1.05, float(args.idle_secondary_freq_scale)),
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    scale_wave = build_wave(
        time_sec=time_sec,
        primary_hz=primary_hz * float(rng.uniform(0.72, 0.90)),
        secondary_scale=1.55,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    tx_wave = build_wave(
        time_sec=time_sec,
        primary_hz=primary_hz * float(rng.uniform(0.85, 1.05)),
        secondary_scale=1.7,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    ty_wave = build_wave(
        time_sec=time_sec,
        primary_hz=primary_hz * float(rng.uniform(0.74, 0.96)),
        secondary_scale=1.6,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    tz_wave = build_wave(
        time_sec=time_sec,
        primary_hz=primary_hz * float(rng.uniform(0.88, 1.10)),
        secondary_scale=1.65,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    brow_wave = build_wave(
        time_sec=time_sec,
        primary_hz=face_primary_hz * float(rng.uniform(0.94, 1.08)),
        secondary_scale=face_secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    lower_eye_wave = build_wave(
        time_sec=time_sec,
        primary_hz=face_primary_hz * float(rng.uniform(1.00, 1.14)),
        secondary_scale=face_secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    center_eye_wave = build_wave(
        time_sec=time_sec,
        primary_hz=face_primary_hz * float(rng.uniform(1.08, 1.24)),
        secondary_scale=face_secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    soft_squint_wave = np.clip(
        build_wave(
            time_sec=time_sec,
            primary_hz=face_primary_hz * float(rng.uniform(1.16, 1.32)),
            secondary_scale=1.7,
            phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
            phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
        ),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    for frame_index, frame in enumerate(motion):
        if not isinstance(frame, dict):
            continue
        exp_array = np.asarray(frame.get("exp"), dtype=np.float32).reshape(21, 3).copy()
        pitch_value = extract_scalar(frame, "pitch", 0.0)
        yaw_value = extract_scalar(frame, "yaw", 0.0)
        roll_value = extract_scalar(frame, "roll", 0.0)
        scale_value = extract_scalar(frame, "scale", 1.0)
        translation_value = np.asarray(frame.get("t", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if translation_value.size < 3:
            translation_value = np.pad(translation_value, (0, max(0, 3 - translation_value.size)), mode="constant")

        envelope_value = float(envelope[frame_index])

        yaw_value += envelope_value * float(args.extra_head_yaw_deg) * float(yaw_wave[frame_index])
        pitch_value += envelope_value * float(args.extra_head_pitch_deg) * float(pitch_wave[frame_index])
        roll_value += envelope_value * float(args.extra_head_roll_deg) * float(roll_wave[frame_index])
        scale_value += envelope_value * float(args.extra_scale_amplitude) * float(np.clip(scale_wave[frame_index], -0.8, 1.0))
        translation_value[0] += envelope_value * float(args.extra_tx_amplitude) * float(tx_wave[frame_index])
        translation_value[1] += envelope_value * float(args.extra_ty_amplitude) * float(ty_wave[frame_index])
        translation_value[2] += envelope_value * float(args.extra_tz_amplitude) * float(tz_wave[frame_index])

        upper_delta_y = envelope_value * float(args.face_brow_y_amplitude) * float(brow_wave[frame_index])
        upper_delta_z = envelope_value * float(args.face_brow_z_amplitude) * float(brow_wave[frame_index])
        lower_delta_y = envelope_value * float(args.face_lower_eye_y_amplitude) * float(lower_eye_wave[frame_index])
        center_delta_y = envelope_value * float(args.face_center_eye_y_amplitude) * float(center_eye_wave[frame_index])
        soft_squint_delta = envelope_value * float(args.face_soft_squint_amplitude) * float(soft_squint_wave[frame_index])

        for index, scale_factor in ((1, 0.92), (2, 1.08), (4, 1.05), (5, 0.95)):
            exp_array[index, 1] += upper_delta_y * scale_factor
            exp_array[index, 2] += upper_delta_z * scale_factor
        for index, scale_factor in ((0, 0.90), (3, 1.05), (7, 1.00), (10, 0.94)):
            exp_array[index, 1] += lower_delta_y * scale_factor
        exp_array[13, 1] += center_delta_y
        exp_array[11, 1] += soft_squint_delta * 0.85
        exp_array[15, 1] += soft_squint_delta * 0.85

        frame["exp"] = exp_array.reshape(1, 21, 3).astype(np.float32)
        frame["pitch"] = np.asarray([[pitch_value]], dtype=np.float32)
        frame["yaw"] = np.asarray([[yaw_value]], dtype=np.float32)
        frame["roll"] = np.asarray([[roll_value]], dtype=np.float32)
        frame["scale"] = np.asarray([[scale_value]], dtype=np.float32)
        frame["t"] = translation_value[:3].reshape(1, 3).astype(np.float32)
        frame["R"] = dataset_clips.build_rotation_matrix_from_degrees(
            pitch_deg=float(pitch_value),
            yaw_deg=float(yaw_value),
            roll_deg=float(roll_value),
        )

    metadata = dict(payload.get("idlePklBuild") or {})
    metadata.update(
        {
            "extraHeadYawDeg": float(args.extra_head_yaw_deg),
            "extraHeadPitchDeg": float(args.extra_head_pitch_deg),
            "extraHeadRollDeg": float(args.extra_head_roll_deg),
            "extraScaleAmplitude": float(args.extra_scale_amplitude),
            "extraTxAmplitude": float(args.extra_tx_amplitude),
            "extraTyAmplitude": float(args.extra_ty_amplitude),
            "extraTzAmplitude": float(args.extra_tz_amplitude),
            "faceBrowYAmplitude": float(args.face_brow_y_amplitude),
            "faceBrowZAmplitude": float(args.face_brow_z_amplitude),
            "faceLowerEyeYAmplitude": float(args.face_lower_eye_y_amplitude),
            "faceCenterEyeYAmplitude": float(args.face_center_eye_y_amplitude),
            "faceSoftSquintAmplitude": float(args.face_soft_squint_amplitude),
        }
    )
    payload["idlePklBuild"] = metadata
    return payload


def main() -> None:
    args = parse_args()
    source_pkl_path = resolve_path(str(args.source_pkl))
    output_pkl_path = resolve_path(str(args.output_pkl))

    if not source_pkl_path.exists():
        raise FileNotFoundError(f"Source PKL not found: {source_pkl_path}")

    source_payload = load_pkl(source_pkl_path)
    source_motion = source_payload["motion"]
    source_fps = float(source_payload.get("output_fps", 0) or 0)
    output_fps = float(args.output_fps) if float(args.output_fps) > 0.0 else source_fps
    if output_fps <= 0.0:
        output_fps = DEFAULT_OUTPUT_FPS
    duration_sec = max(0.5, float(args.duration_sec))
    frame_count = max(3, int(round(duration_sec * output_fps)))

    if int(args.base_frame_index) >= 0:
        base_frame_index = max(0, min(int(args.base_frame_index), len(source_motion) - 1))
    else:
        base_frame_index = select_neutral_frame_index(source_motion)

    static_payload = build_static_idle_payload(
        source_payload=source_payload,
        base_frame_index=base_frame_index,
        frame_count=frame_count,
        fps=output_fps,
    )

    entry = build_idle_entry(duration_sec=duration_sec)
    processed_payload = dataset_clips.apply_idle_motion_enhancement_to_payload(
        payload=static_payload,
        args=args,
        entry=entry,
        visemes=entry.visemes,
        transitions=entry.transitions,
    )
    processed_payload = apply_extra_idle_liveliness(
        payload=processed_payload,
        args=args,
    )
    annotate_output_metadata(
        payload=processed_payload,
        source_pkl_path=source_pkl_path,
        base_frame_index=base_frame_index,
        duration_sec=duration_sec,
        fps=output_fps,
        args=args,
    )

    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl_path.open("wb") as handle:
        pickle.dump(processed_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ok] idle pkl -> {output_pkl_path}")
    print(f"[info] source={source_pkl_path}")
    print(f"[info] base_frame_index={base_frame_index}")
    print(f"[info] frames={len(processed_payload.get('motion', []))} fps={output_fps:.3f}")


if __name__ == "__main__":
    main()
