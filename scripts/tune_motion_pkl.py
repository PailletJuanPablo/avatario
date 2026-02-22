"""
Tune FasterLivePortrait motion PKL files without regenerating audio.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRESET_EYE_TAMED_V1 = "eye_tamed_v1"
PRESET_CHOICES = (PRESET_EYE_TAMED_V1,)
DEFAULT_SOFT_INDICES = (0, 1, 2, 3, 4, 5, 7, 10, 13)
DEFAULT_HARD_INDICES = (11, 15)
DEFAULT_SOFT_FACTOR = 0.45
DEFAULT_HARD_FACTOR = 0.18
DEFAULT_HARD_DY_MIN = -0.0045
DEFAULT_HARD_DY_MAX = 0.0035


def resolve_path(path_value: str) -> Path:
    """
    Resolve path relative to project root.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def parse_indices(text_value: str, fallback: Iterable[int]) -> tuple[int, ...]:
    """
    Parse comma-separated index list.
    """
    raw = str(text_value or "").strip()
    if not raw:
        return tuple(int(item) for item in fallback)
    values: list[int] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(int(token))
    return tuple(values)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Tune expression channels in motion PKL.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default=PRESET_EYE_TAMED_V1)
    parser.add_argument("--soft-indices", default="")
    parser.add_argument("--hard-indices", default="")
    parser.add_argument("--soft-factor", type=float, default=DEFAULT_SOFT_FACTOR)
    parser.add_argument("--hard-factor", type=float, default=DEFAULT_HARD_FACTOR)
    parser.add_argument("--hard-dy-min", type=float, default=DEFAULT_HARD_DY_MIN)
    parser.add_argument("--hard-dy-max", type=float, default=DEFAULT_HARD_DY_MAX)
    return parser.parse_args()


def apply_eye_tamed_preset(
    payload: dict,
    soft_indices: tuple[int, ...],
    hard_indices: tuple[int, ...],
    soft_factor: float,
    hard_factor: float,
    hard_dy_min: float,
    hard_dy_max: float,
) -> dict:
    """
    Apply upper-face damping relative to frame-0 baseline.
    """
    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        raise ValueError("Invalid motion payload: missing non-empty motion list.")

    first_frame = motion[0]
    if not isinstance(first_frame, dict) or "exp" not in first_frame:
        raise ValueError("Invalid motion payload: first frame has no exp.")

    base_exp = np.asarray(first_frame["exp"], dtype=np.float32).reshape(21, 3).copy()
    safe_soft = float(np.clip(soft_factor, 0.0, 1.0))
    safe_hard = float(np.clip(hard_factor, 0.0, 1.0))
    safe_min = float(min(hard_dy_min, hard_dy_max))
    safe_max = float(max(hard_dy_min, hard_dy_max))

    processed = dict(payload)
    processed_motion: list[dict] = []
    for frame in motion:
        if not isinstance(frame, dict) or "exp" not in frame:
            processed_motion.append(frame)
            continue
        exp_array = np.asarray(frame["exp"], dtype=np.float32).reshape(21, 3).copy()
        for index in soft_indices:
            exp_array[index, :] = base_exp[index, :] + (exp_array[index, :] - base_exp[index, :]) * safe_soft
        for index in hard_indices:
            exp_array[index, :] = base_exp[index, :] + (exp_array[index, :] - base_exp[index, :]) * safe_hard
            delta_y = exp_array[index, 1] - base_exp[index, 1]
            exp_array[index, 1] = base_exp[index, 1] + float(np.clip(delta_y, safe_min, safe_max))
        next_frame = dict(frame)
        next_frame["exp"] = exp_array.reshape(1, 21, 3)
        processed_motion.append(next_frame)

    processed["motion"] = processed_motion
    return processed


def main() -> None:
    """
    Entry point.
    """
    args = parse_args()
    input_pkl = resolve_path(args.input_pkl)
    output_pkl = resolve_path(args.output_pkl)

    if not input_pkl.exists():
        raise FileNotFoundError(f"Input pkl not found: {input_pkl}")
    with input_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported pkl payload type: {type(payload)}")

    soft_indices = parse_indices(args.soft_indices, DEFAULT_SOFT_INDICES)
    hard_indices = parse_indices(args.hard_indices, DEFAULT_HARD_INDICES)
    if args.preset == PRESET_EYE_TAMED_V1:
        processed = apply_eye_tamed_preset(
            payload=payload,
            soft_indices=soft_indices,
            hard_indices=hard_indices,
            soft_factor=float(args.soft_factor),
            hard_factor=float(args.hard_factor),
            hard_dy_min=float(args.hard_dy_min),
            hard_dy_max=float(args.hard_dy_max),
        )
    else:
        processed = payload

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[ok] tuned pkl -> {output_pkl}")


if __name__ == "__main__":
    main()
