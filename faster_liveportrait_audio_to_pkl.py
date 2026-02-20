"""
Generate FasterLivePortrait driving motion pickle from an audio file using JoyVASA.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Build JoyVASA motion template (.pkl) from driving audio.")
    parser.add_argument("--faster-repo-dir", required=True, help="Path to third_party/FasterLivePortrait")
    parser.add_argument("--cfg", required=True, help="Path to FasterLivePortrait cfg yaml")
    parser.add_argument("--driving-audio", required=True, help="Path to driving audio (.wav/.mp3/...)")
    parser.add_argument("--output-pkl", required=True, help="Output motion template .pkl")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve path value to absolute path.
    """
    return Path(path_value).expanduser().resolve()


def main() -> None:
    """
    Run JoyVASA audio-to-motion conversion and export pkl.
    """
    args = parse_args()
    faster_repo_dir = resolve_path(args.faster_repo_dir)
    cfg_path = resolve_path(args.cfg)
    driving_audio_path = resolve_path(args.driving_audio)
    output_pkl_path = resolve_path(args.output_pkl)

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
        cfg_scale=float(cfg.infer_params.cfg_scale),
    )
    motion_data = converter.gen_motion_sequence(str(driving_audio_path))

    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl_path.open("wb") as handle:
        pickle.dump(motion_data, handle)

    print(f"[ok] joyvasa template -> {output_pkl_path}")
    print(f"[info] frames={motion_data.get('n_frames', 0)} fps={motion_data.get('output_fps', 0)}")


if __name__ == "__main__":
    main()
