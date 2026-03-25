"""
Generate FasterLivePortrait driving motion pickle from an audio file using JoyVASA.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


DEFAULT_MOTION_STRIDE = 2
DEFAULT_ENABLE_EYE_TAMED_PRESET = True
DEFAULT_EYE_TAMED_SOFT_INDICES = (0, 1, 2, 3, 4, 5, 7, 10, 13)
DEFAULT_EYE_TAMED_HARD_INDICES = (11, 15)
DEFAULT_EYE_SOFT_FACTOR = 0.45
DEFAULT_EYE_HARD_FACTOR = 0.18
DEFAULT_EYE_HARD_DY_MIN = -0.0045
DEFAULT_EYE_HARD_DY_MAX = 0.0035


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Build JoyVASA motion template (.pkl) from driving audio.")
    parser.add_argument("--faster-repo-dir", required=True, help="Path to third_party/FasterLivePortrait")
    parser.add_argument("--cfg", required=True, help="Path to FasterLivePortrait cfg yaml")
    parser.add_argument("--driving-audio", required=True, help="Path to driving audio (.wav/.mp3/...)")
    parser.add_argument("--output-pkl", required=True, help="Output motion template .pkl")
    parser.add_argument("--seed", type=int, default=1234, help="Global seed for deterministic motion generation.")
    parser.add_argument(
        "--motion-stride",
        type=int,
        default=DEFAULT_MOTION_STRIDE,
        help="Audio motion decimation factor used while building the template.",
    )
    parser.add_argument(
        "--generation-frame-count",
        type=int,
        default=0,
        help="Optional exact number of frames to generate. Zero keeps the automatic duration plan.",
    )
    parser.add_argument(
        "--enable-eye-tamed-preset",
        dest="enable_eye_tamed_preset",
        action="store_true",
        help="Apply conservative eye/upper-face damping to reduce unnatural eye opening.",
    )
    parser.add_argument(
        "--disable-eye-tamed-preset",
        dest="enable_eye_tamed_preset",
        action="store_false",
        help="Disable conservative eye/upper-face damping.",
    )
    parser.add_argument(
        "--eye-soft-factor",
        type=float,
        default=DEFAULT_EYE_SOFT_FACTOR,
        help="Damping factor [0..1] for soft upper-face eye-adjacent indices.",
    )
    parser.add_argument(
        "--eye-hard-factor",
        type=float,
        default=DEFAULT_EYE_HARD_FACTOR,
        help="Damping factor [0..1] for hard eyelid indices.",
    )
    parser.add_argument(
        "--eye-hard-dy-min",
        type=float,
        default=DEFAULT_EYE_HARD_DY_MIN,
        help="Minimum vertical eyelid delta allowed relative to frame zero.",
    )
    parser.add_argument(
        "--eye-hard-dy-max",
        type=float,
        default=DEFAULT_EYE_HARD_DY_MAX,
        help="Maximum vertical eyelid delta allowed relative to frame zero.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.2,
        help="JoyVASA classifier-free guidance scale override [0..10].",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=15,
        help="JoyVASA diffusion inference step override [1..100].",
    )
    parser.set_defaults(enable_eye_tamed_preset=DEFAULT_ENABLE_EYE_TAMED_PRESET)
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve path value to absolute path.
    """
    return Path(path_value).expanduser().resolve()


def set_global_seed(seed_value: int) -> None:
    """Set deterministic seed across python, numpy and torch."""
    safe_seed = int(seed_value)
    random.seed(safe_seed)
    np.random.seed(safe_seed)
    torch.manual_seed(safe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(safe_seed)


def build_motion_sequence(
    converter: Any,
    driving_audio_path: Path,
    motion_stride: int,
    generation_frame_count: int | None,
) -> dict[str, Any]:
    """
    Build one motion payload while preserving the reduced-motion planning contract.
    """
    stream_meta, motion_iter = converter.prepare_audio_motion_stream(
        str(driving_audio_path),
        motion_stride=max(1, int(motion_stride)),
        generation_frame_count=generation_frame_count,
    )
    motion_list = [motion_info[0] for motion_info in motion_iter]
    return {
        "n_frames": len(motion_list),
        "output_fps": int(stream_meta.get("output_fps", 0) or 0),
        "motion": motion_list,
        "c_eyes_lst": [],
        "c_lip_lst": [],
    }


def apply_eye_tamed_preset_to_motion_data(
    motion_data: dict[str, Any],
    soft_factor: float,
    hard_factor: float,
    hard_dy_min: float,
    hard_dy_max: float,
) -> dict[str, Any]:
    """
    Dampen eye-sensitive channels relative to frame zero and clamp excessive eye opening.
    """
    motion = motion_data.get("motion")
    if not isinstance(motion, list) or not motion:
        return motion_data

    first_frame = motion[0]
    if not isinstance(first_frame, dict) or "exp" not in first_frame:
        return motion_data

    base_exp = np.asarray(first_frame["exp"], dtype=np.float32).reshape(21, 3).copy()
    safe_soft = float(np.clip(float(soft_factor), 0.0, 1.0))
    safe_hard = float(np.clip(float(hard_factor), 0.0, 1.0))
    safe_min = float(min(hard_dy_min, hard_dy_max))
    safe_max = float(max(hard_dy_min, hard_dy_max))

    processed_motion: list[dict[str, Any]] = []
    for frame in motion:
        if not isinstance(frame, dict) or "exp" not in frame:
            processed_motion.append(frame)
            continue
        exp_array = np.asarray(frame["exp"], dtype=np.float32).reshape(21, 3).copy()
        for index in DEFAULT_EYE_TAMED_SOFT_INDICES:
            exp_array[index, :] = base_exp[index, :] + (exp_array[index, :] - base_exp[index, :]) * safe_soft
        for index in DEFAULT_EYE_TAMED_HARD_INDICES:
            exp_array[index, :] = base_exp[index, :] + (exp_array[index, :] - base_exp[index, :]) * safe_hard
            delta_y = exp_array[index, 1] - base_exp[index, 1]
            exp_array[index, 1] = base_exp[index, 1] + float(np.clip(delta_y, safe_min, safe_max))
        next_frame = dict(frame)
        next_frame["exp"] = exp_array.reshape(1, 21, 3)
        processed_motion.append(next_frame)

    processed_motion_data = dict(motion_data)
    processed_motion_data["motion"] = processed_motion
    processed_motion_data["n_frames"] = len(processed_motion)
    return processed_motion_data


def main() -> None:
    """
    Run JoyVASA audio-to-motion conversion and export pkl.
    """
    args = parse_args()
    faster_repo_dir = resolve_path(args.faster_repo_dir)
    cfg_path = resolve_path(args.cfg)
    driving_audio_path = resolve_path(args.driving_audio)
    output_pkl_path = resolve_path(args.output_pkl)
    set_global_seed(args.seed)

    if not faster_repo_dir.exists():
        raise FileNotFoundError(f"FasterLivePortrait repo not found: {faster_repo_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg file not found: {cfg_path}")
    if not driving_audio_path.exists():
        raise FileNotFoundError(f"driving audio not found: {driving_audio_path}")

    if str(faster_repo_dir) not in sys.path:
        sys.path.insert(0, str(faster_repo_dir))

    from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

    cfg = OmegaConf.load(str(cfg_path))
    joyvasa_motion_model_path = (faster_repo_dir / cfg.joyvasa_models.motion_model_path).resolve()
    joyvasa_audio_model_path = (faster_repo_dir / cfg.joyvasa_models.audio_model_path).resolve()
    joyvasa_template_path = (faster_repo_dir / cfg.joyvasa_models.motion_template_path).resolve()
    if not joyvasa_motion_model_path.exists():
        raise FileNotFoundError(
            "JoyVASA motion model not found: "
            f"{joyvasa_motion_model_path}. "
            "Download it into third_party/FasterLivePortrait/checkpoints/JoyVASA."
        )
    if not joyvasa_template_path.exists():
        raise FileNotFoundError(
            "JoyVASA motion template not found: "
            f"{joyvasa_template_path}. "
            "Download it into third_party/FasterLivePortrait/checkpoints/JoyVASA."
        )
    if not joyvasa_audio_model_path.exists():
        raise FileNotFoundError(
            "HuBERT model path not found: "
            f"{joyvasa_audio_model_path}. "
            "Download TencentGameMate/chinese-hubert-base into "
            "third_party/FasterLivePortrait/checkpoints/chinese-hubert-base."
        )

    converter = JoyVASAAudio2MotionPipeline(
        motion_model_path=str(joyvasa_motion_model_path),
        audio_model_path=str(joyvasa_audio_model_path),
        motion_template_path=str(joyvasa_template_path),
        cfg_mode=str(cfg.infer_params.cfg_mode),
        cfg_scale=float(np.clip(float(args.cfg_scale), 0.0, 10.0)),
        inference_steps=int(np.clip(int(args.inference_steps), 1, 100)),
    )
    generation_frame_count = max(0, int(args.generation_frame_count or 0))
    motion_data = build_motion_sequence(
        converter=converter,
        driving_audio_path=driving_audio_path,
        motion_stride=max(1, int(args.motion_stride)),
        generation_frame_count=generation_frame_count or None,
    )
    if bool(args.enable_eye_tamed_preset):
        motion_data = apply_eye_tamed_preset_to_motion_data(
            motion_data=motion_data,
            soft_factor=float(args.eye_soft_factor),
            hard_factor=float(args.eye_hard_factor),
            hard_dy_min=float(args.eye_hard_dy_min),
            hard_dy_max=float(args.eye_hard_dy_max),
        )

    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl_path.open("wb") as handle:
        pickle.dump(motion_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ok] joyvasa template -> {output_pkl_path}")
    print(f"[info] frames={motion_data.get('n_frames', 0)} fps={motion_data.get('output_fps', 0)}")


if __name__ == "__main__":
    main()
