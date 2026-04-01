"""
Run FasterLivePortrait from local frames and export browser-ready result videos.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import hashlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from runtime_env import apply_runtime_library_environment, build_process_env

PROJECT_ROOT = Path(__file__).resolve().parent
apply_runtime_library_environment(PROJECT_ROOT)

import cv2
import numpy as np


def read_env_bool(env_key: str, default_value: bool) -> bool:
    """
    Read one boolean environment override with a safe fallback.
    """
    raw_value = os.getenv(env_key, "1" if default_value else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return bool(default_value)


def read_env_float(
    env_key: str,
    default_value: float,
    minimum_value: float | None = None,
    maximum_value: float | None = None,
) -> float:
    """
    Read one float environment override and clamp it when bounds are provided.
    """
    raw_value = os.getenv(env_key, str(default_value)).strip()
    try:
        parsed_value = float(raw_value or str(default_value))
    except ValueError:
        parsed_value = float(default_value)
    if minimum_value is not None:
        parsed_value = max(float(minimum_value), parsed_value)
    if maximum_value is not None:
        parsed_value = min(float(maximum_value), parsed_value)
    return float(parsed_value)


def read_env_int(
    env_key: str,
    default_value: int,
    minimum_value: int | None = None,
    maximum_value: int | None = None,
) -> int:
    """
    Read one integer environment override and clamp it when bounds are provided.
    """
    raw_value = os.getenv(env_key, str(default_value)).strip()
    try:
        parsed_value = int(raw_value or str(default_value))
    except ValueError:
        parsed_value = int(default_value)
    if minimum_value is not None:
        parsed_value = max(int(minimum_value), parsed_value)
    if maximum_value is not None:
        parsed_value = min(int(maximum_value), parsed_value)
    return int(parsed_value)


MODE_PREVIEW = "preview"
MODE_FULL = "full"
BACKEND_ONNX = "onnx"
BACKEND_TRT = "trt"
TRT_RUNTIME_LOCAL = "local"
TRT_RUNTIME_DOCKER = "docker"
TRT_PRECISION_FP32 = "fp32"
TRT_PRECISION_FP16 = "fp16"
TRT_PRECISION_INT8 = "int8"
TRT_PRECISION_CHOICES = (TRT_PRECISION_FP32, TRT_PRECISION_FP16, TRT_PRECISION_INT8)
DEFAULT_TRT_DOCKER_IMAGE = "animation/faster_liveportrait:v3-runtime"
DEFAULT_TRT_DOCKER_PYTHON = "/root/miniconda3/bin/python"
DEFAULT_TRT_DOCKER_LD_PATH = "/opt/TensorRT-8.6.1.6/lib"
DEFAULT_TRT_DOCKER_GPU_DEVICE = "auto"
DEFAULT_TRT_DOCKER_CONTAINER_NAME = "animation_faster_liveportrait_runtime"
DEFAULT_TRT_DOCKER_REUSE_CONTAINER = True
DEFAULT_TRT_PRECISION = TRT_PRECISION_FP16
DOCKER_IPC_MODE_ENV_KEY = "ANIMATION_DOCKER_IPC_MODE"
DOCKER_PARENT_CONTAINER_ENV_KEY = "ANIMATION_DOCKER_PARENT_CONTAINER"

FRAME_PATTERN = "frame_%05d.png"
DEFAULT_FPS = 30.0
RAW_RESULTS_DIR_NAME = "raw_results"
RUN_REPORT_NAME = "run_report.json"
DRIVING_PUBLIC_NAME = "driving.mp4"
RESULT_PUBLIC_NAME = "result.mp4"
RESULT_CONCAT_PUBLIC_NAME = "result_concat.mp4"
PREVIEW_COMPOSITION_META_NAME = "preview_composition.json"
PREVIEW_COMPOSITION_MASK_NAME = "preview_composition_mask.png"
AUDIO_TO_PKL_SCRIPT_NAME = "faster_liveportrait_audio_to_pkl.py"
DRIVING_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")
AUDIO_TEMPLATE_META_NAME = "audio_template_meta.json"
DEFAULT_AUDIO_TEMPLATE_CACHE_DIR = "output_fasterliveportrait/audio_template_cache"
DEFAULT_SOURCE_CACHE_DIR = "output_fasterliveportrait/source_preprocess_cache"
DEFAULT_PERSISTENT_WORKER_QUEUE_DIR = "output_fasterliveportrait/worker_queue"
FIXED_AUDIO_MOTION_STRIDE = 2
GENERATION_FRAME_COUNT_MIN = 1
GENERATION_FRAME_COUNT_MAX = 1200
FIXED_AUDIO_MOTION_TARGET_FPS_ENV_KEY = "ANIMATION_AUDIO_MOTION_TARGET_FPS"
FIXED_AUDIO_MOTION_TARGET_FPS_DEFAULT = 22.0
try:
    FIXED_AUDIO_MOTION_TARGET_FPS = max(
        1.0,
        float(
            os.getenv(
                FIXED_AUDIO_MOTION_TARGET_FPS_ENV_KEY,
                str(FIXED_AUDIO_MOTION_TARGET_FPS_DEFAULT),
            ).strip()
            or str(FIXED_AUDIO_MOTION_TARGET_FPS_DEFAULT)
        ),
    )
except ValueError:
    FIXED_AUDIO_MOTION_TARGET_FPS = FIXED_AUDIO_MOTION_TARGET_FPS_DEFAULT
DEFAULT_AUDIO_EYE_TAMED_PRESET_ENV_KEY = "ANIMATION_AUDIO_EYE_TAMED_PRESET"
DEFAULT_AUDIO_EYE_SOFT_FACTOR_ENV_KEY = "ANIMATION_AUDIO_EYE_SOFT_FACTOR"
DEFAULT_AUDIO_EYE_HARD_FACTOR_ENV_KEY = "ANIMATION_AUDIO_EYE_HARD_FACTOR"
DEFAULT_AUDIO_EYE_HARD_DY_MIN_ENV_KEY = "ANIMATION_AUDIO_EYE_HARD_DY_MIN"
DEFAULT_AUDIO_EYE_HARD_DY_MAX_ENV_KEY = "ANIMATION_AUDIO_EYE_HARD_DY_MAX"
DEFAULT_AUDIO_MOTION_TUNING_ENABLED_ENV_KEY = "ANIMATION_AUDIO_MOTION_TUNING_ENABLED"
DEFAULT_AUDIO_REANCHOR_FIRST_N_ENV_KEY = "ANIMATION_AUDIO_REANCHOR_FIRST_N"
DEFAULT_AUDIO_MOUTH_OPEN_FACTOR_ENV_KEY = "ANIMATION_AUDIO_MOUTH_OPEN_FACTOR"
DEFAULT_AUDIO_POSE_SMOOTH_WINDOW_ENV_KEY = "ANIMATION_AUDIO_POSE_SMOOTH_WINDOW"
DEFAULT_AUDIO_EXP_SMOOTH_WINDOW_ENV_KEY = "ANIMATION_AUDIO_EXP_SMOOTH_WINDOW"
DEFAULT_AUDIO_POSE_JUMP_THRESHOLD_ENV_KEY = "ANIMATION_AUDIO_POSE_JUMP_THRESHOLD"
DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD_ENV_KEY = "ANIMATION_AUDIO_TRANSLATION_JUMP_THRESHOLD"
DEFAULT_AUDIO_LIP_SYNC_ASSIST_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_ASSIST"
DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_MIN_RATIO"
DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_MAX_RATIO"
DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_SMOOTH_WINDOW"
DEFAULT_AUDIO_LIP_SYNC_STRENGTH_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_STRENGTH"
DEFAULT_AUDIO_LIP_SYNC_POWER_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_POWER"
DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_OFFSET_MS"
DEFAULT_AUDIO_EYE_TAMED_PRESET = read_env_bool(DEFAULT_AUDIO_EYE_TAMED_PRESET_ENV_KEY, True)
DEFAULT_AUDIO_EYE_SOFT_FACTOR = read_env_float(DEFAULT_AUDIO_EYE_SOFT_FACTOR_ENV_KEY, 0.45, 0.0, 1.0)
DEFAULT_AUDIO_EYE_HARD_FACTOR = read_env_float(DEFAULT_AUDIO_EYE_HARD_FACTOR_ENV_KEY, 0.18, 0.0, 1.0)
DEFAULT_AUDIO_EYE_HARD_DY_MIN = read_env_float(DEFAULT_AUDIO_EYE_HARD_DY_MIN_ENV_KEY, -0.0045)
DEFAULT_AUDIO_EYE_HARD_DY_MAX = read_env_float(DEFAULT_AUDIO_EYE_HARD_DY_MAX_ENV_KEY, 0.0035)
DEFAULT_AUDIO_MOTION_TUNING_ENABLED = read_env_bool(DEFAULT_AUDIO_MOTION_TUNING_ENABLED_ENV_KEY, True)
DEFAULT_AUDIO_REANCHOR_FIRST_N = read_env_int(DEFAULT_AUDIO_REANCHOR_FIRST_N_ENV_KEY, 5, 1, 15)
DEFAULT_AUDIO_MOUTH_OPEN_FACTOR = read_env_float(DEFAULT_AUDIO_MOUTH_OPEN_FACTOR_ENV_KEY, 1.18, 0.0, 3.0)
DEFAULT_AUDIO_POSE_SMOOTH_WINDOW = read_env_int(DEFAULT_AUDIO_POSE_SMOOTH_WINDOW_ENV_KEY, 5, 0, 21)
DEFAULT_AUDIO_EXP_SMOOTH_WINDOW = read_env_int(DEFAULT_AUDIO_EXP_SMOOTH_WINDOW_ENV_KEY, 3, 0, 21)
DEFAULT_AUDIO_POSE_JUMP_THRESHOLD = read_env_float(DEFAULT_AUDIO_POSE_JUMP_THRESHOLD_ENV_KEY, 8.0, 0.0, 60.0)
DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD = read_env_float(
    DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD_ENV_KEY,
    0.03,
    0.0,
    1.0,
)
DEFAULT_AUDIO_LIP_SYNC_ASSIST = read_env_bool(DEFAULT_AUDIO_LIP_SYNC_ASSIST_ENV_KEY, False)
DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO = read_env_float(DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO_ENV_KEY, 0.03, 0.0, 1.0)
DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO = read_env_float(DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO_ENV_KEY, 0.32, 0.0, 1.0)
DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW = read_env_int(DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW_ENV_KEY, 5, 0, 21)
DEFAULT_AUDIO_LIP_SYNC_STRENGTH = read_env_float(DEFAULT_AUDIO_LIP_SYNC_STRENGTH_ENV_KEY, 1.15, 0.0, 4.0)
DEFAULT_AUDIO_LIP_SYNC_POWER = read_env_float(DEFAULT_AUDIO_LIP_SYNC_POWER_ENV_KEY, 0.85, 0.001, 4.0)
DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS = read_env_int(DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS_ENV_KEY, 0, -1000, 1000)
DEFAULT_DRIVING_MULTIPLIER = read_env_float("ANIMATION_DRIVING_MULTIPLIER", 1.0, 0.0, 2.0)
DEFAULT_CFG_SCALE = read_env_float("ANIMATION_CFG_SCALE", 1.2, 0.0, 10.0)
DEFAULT_JOYVASA_INFERENCE_STEPS = read_env_int("ANIMATION_JOYVASA_INFERENCE_STEPS", 15, 1, 100)
ENGINE_PRECISION_MARKER_SUFFIX = ".precision.txt"
ENGINE_BATCH_MARKER_SUFFIX = ".batch.txt"
TRT_INT8_CALIBRATION_BATCHES = 12
TRT_INT8_CALIBRATION_CACHE_SUFFIX = ".int8.cache"
PERSISTENT_WORKER_HEARTBEAT_FILE_NAME = "worker_heartbeat.json"
PERSISTENT_WORKER_REQUEST_SUFFIX = ".request.json"
PERSISTENT_WORKER_RESPONSE_SUFFIX = ".response.json"
PERSISTENT_WORKER_HEARTBEAT_STALE_SEC = 8.0
PERSISTENT_WORKER_STARTUP_TIMEOUT_SEC = 45.0
PERSISTENT_WORKER_RESPONSE_TIMEOUT_SEC = 1200.0
PERSISTENT_WORKER_POLL_SLEEP_SEC = 0.08
ANIMATION_REGION_CHOICES = ("all", "exp", "lip", "eyes", "pose")
VIDEO_ENCODER_AUTO = "auto"
VIDEO_ENCODER_NVENC = "nvenc"
VIDEO_ENCODER_CPU = "cpu"
VIDEO_ENCODER_CHOICES = (VIDEO_ENCODER_AUTO, VIDEO_ENCODER_NVENC, VIDEO_ENCODER_CPU)
FFMPEG_H264_NVENC = "h264_nvenc"
FFMPEG_LIBX264 = "libx264"
DEFAULT_RENDER_BATCH_SIZE = max(
    1,
    int(os.getenv("ANIMATION_RENDER_BATCH_SIZE", "4").strip() or "4"),
)
DEFAULT_TRT_ENGINE_BATCH_SIZE = max(
    DEFAULT_RENDER_BATCH_SIZE,
    int(
        os.getenv(
            "ANIMATION_TRT_ENGINE_BATCH_SIZE",
            str(DEFAULT_RENDER_BATCH_SIZE),
        ).strip()
        or str(DEFAULT_RENDER_BATCH_SIZE)
    ),
)
DEFAULT_VIDEO_ENCODER = os.getenv("ANIMATION_VIDEO_ENCODER", VIDEO_ENCODER_AUTO).strip().lower() or VIDEO_ENCODER_AUTO
if DEFAULT_VIDEO_ENCODER not in VIDEO_ENCODER_CHOICES:
    DEFAULT_VIDEO_ENCODER = VIDEO_ENCODER_AUTO
FFMPEG_ENCODER_SUPPORT_CACHE: dict[str, bool] = {}
SOURCE_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv"}


@dataclass
class RunnerConfig:
    project_root: Path
    source_frame: Path
    frames_dir: Path
    meta_path: Path
    output_dir: Path
    faster_repo_dir: Path
    python_executable: Path
    mode: str
    backend: str
    trt_runtime: str
    trt_precision: str
    docker_image: str
    docker_gpu_device: str
    docker_container_name: str
    docker_reuse_container: bool
    audio_template_cache_dir: Path
    source_cache_dir: Path
    persistent_worker_queue_dir: Path
    use_persistent_trt_worker: bool
    driving_audio: Path | None
    audio_motion_stride: int
    generation_frame_count: int | None
    audio_eye_tamed_preset: bool
    audio_eye_soft_factor: float
    audio_eye_hard_factor: float
    audio_eye_hard_dy_min: float
    audio_eye_hard_dy_max: float
    audio_motion_tuning_enabled: bool
    audio_reanchor_first_n: int
    audio_mouth_open_factor: float
    audio_pose_smooth_window: int
    audio_exp_smooth_window: int
    audio_pose_jump_threshold: float
    audio_translation_jump_threshold: float
    audio_lip_sync_assist: bool
    audio_lip_sync_min_ratio: float
    audio_lip_sync_max_ratio: float
    audio_lip_sync_smooth_window: int
    audio_lip_sync_strength: float
    audio_lip_sync_power: float
    audio_lip_sync_offset_ms: int
    driving_multiplier: float
    cfg_scale: float
    joyvasa_inference_steps: int
    render_batch_size: int
    trt_engine_batch_size: int
    stream_dir: Path
    stream_shm_prefix: str
    stream_enabled: bool
    frame_step: int
    skip_driving_video_build: bool
    rebuild_driving_template: bool
    skip_trt_engine_build: bool
    video_encoder: str
    paste_back: bool
    defer_paste_back: bool
    animation_region: str
    stitching_enabled: bool
    relative_motion_enabled: bool


def parse_args() -> RunnerConfig:
    """
    Parse CLI args and normalize runtime config.
    """
    parser = argparse.ArgumentParser(description="Run FasterLivePortrait with local project assets.")
    parser.add_argument("--source-frame", default="output/frames/frame_00096.png")
    parser.add_argument("--frames-dir", default="output/frames")
    parser.add_argument("--meta-path", default="output/meta.json")
    parser.add_argument("--output-dir", default="output_fasterliveportrait")
    parser.add_argument("--faster-repo-dir", default="third_party/FasterLivePortrait")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--driving-audio", default="")
    parser.add_argument(
        "--audio-motion-stride",
        type=int,
        default=FIXED_AUDIO_MOTION_STRIDE,
        help="Audio motion frame decimation factor. Default 2 uses the reduced-motion profile.",
    )
    parser.add_argument(
        "--generation-frame-count",
        type=int,
        default=0,
        help="Optional exact number of motion frames to generate from audio. Zero keeps automatic planning.",
    )
    parser.add_argument(
        "--audio-eye-tamed-preset",
        dest="audio_eye_tamed_preset",
        action="store_true",
        help="Apply conservative eye-opening damping when building audio motion templates.",
    )
    parser.add_argument(
        "--no-audio-eye-tamed-preset",
        dest="audio_eye_tamed_preset",
        action="store_false",
        help="Disable conservative eye-opening damping when building audio motion templates.",
    )
    parser.set_defaults(audio_eye_tamed_preset=DEFAULT_AUDIO_EYE_TAMED_PRESET)
    parser.add_argument(
        "--audio-eye-soft-factor",
        type=float,
        default=DEFAULT_AUDIO_EYE_SOFT_FACTOR,
        help="Soft upper-face damping factor [0..1] used for audio template eye taming.",
    )
    parser.add_argument(
        "--audio-eye-hard-factor",
        type=float,
        default=DEFAULT_AUDIO_EYE_HARD_FACTOR,
        help="Hard eyelid damping factor [0..1] used for audio template eye taming.",
    )
    parser.add_argument(
        "--audio-eye-hard-dy-min",
        type=float,
        default=DEFAULT_AUDIO_EYE_HARD_DY_MIN,
        help="Minimum vertical eyelid delta allowed while building audio motion templates.",
    )
    parser.add_argument(
        "--audio-eye-hard-dy-max",
        type=float,
        default=DEFAULT_AUDIO_EYE_HARD_DY_MAX,
        help="Maximum vertical eyelid delta allowed while building audio motion templates.",
    )
    parser.add_argument(
        "--audio-motion-tuning-enabled",
        dest="audio_motion_tuning_enabled",
        action="store_true",
        help="Apply deterministic PKL cleanup for audio-generated motion templates.",
    )
    parser.add_argument(
        "--no-audio-motion-tuning",
        dest="audio_motion_tuning_enabled",
        action="store_false",
        help="Disable deterministic PKL cleanup for audio-generated motion templates.",
    )
    parser.set_defaults(audio_motion_tuning_enabled=DEFAULT_AUDIO_MOTION_TUNING_ENABLED)
    parser.add_argument(
        "--audio-reanchor-first-n",
        type=int,
        default=DEFAULT_AUDIO_REANCHOR_FIRST_N,
        help="Initial median anchor span used to stabilize audio-generated motion templates.",
    )
    parser.add_argument(
        "--audio-mouth-open-factor",
        type=float,
        default=DEFAULT_AUDIO_MOUTH_OPEN_FACTOR,
        help="Relative mouth-opening gain applied to audio-generated motion templates.",
    )
    parser.add_argument(
        "--audio-pose-smooth-window",
        type=int,
        default=DEFAULT_AUDIO_POSE_SMOOTH_WINDOW,
        help="Median smoothing window applied to pitch/yaw/roll in audio-generated motion templates.",
    )
    parser.add_argument(
        "--audio-exp-smooth-window",
        type=int,
        default=DEFAULT_AUDIO_EXP_SMOOTH_WINDOW,
        help="Median smoothing window applied to non-mouth expression channels.",
    )
    parser.add_argument(
        "--audio-pose-jump-threshold",
        type=float,
        default=DEFAULT_AUDIO_POSE_JUMP_THRESHOLD,
        help="Maximum per-frame pose delta in degrees before clamping.",
    )
    parser.add_argument(
        "--audio-translation-jump-threshold",
        type=float,
        default=DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD,
        help="Maximum per-frame translation delta before clamping.",
    )
    parser.add_argument(
        "--audio-lip-sync-assist",
        dest="audio_lip_sync_assist",
        action="store_true",
        help="Reinforce mouth motion using the audio envelope.",
    )
    parser.add_argument(
        "--no-audio-lip-sync-assist",
        dest="audio_lip_sync_assist",
        action="store_false",
        help="Disable mouth reinforcement from the audio envelope.",
    )
    parser.set_defaults(audio_lip_sync_assist=DEFAULT_AUDIO_LIP_SYNC_ASSIST)
    parser.add_argument(
        "--audio-lip-sync-min-ratio",
        type=float,
        default=DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO,
        help="Minimum lip ratio derived from the audio envelope.",
    )
    parser.add_argument(
        "--audio-lip-sync-max-ratio",
        type=float,
        default=DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO,
        help="Maximum lip ratio derived from the audio envelope.",
    )
    parser.add_argument(
        "--audio-lip-sync-smooth-window",
        type=int,
        default=DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW,
        help="Moving-average window used on the audio envelope before lip sync.",
    )
    parser.add_argument(
        "--audio-lip-sync-strength",
        type=float,
        default=DEFAULT_AUDIO_LIP_SYNC_STRENGTH,
        help="Gain applied to the audio-driven lip-sync assist.",
    )
    parser.add_argument(
        "--audio-lip-sync-power",
        type=float,
        default=DEFAULT_AUDIO_LIP_SYNC_POWER,
        help="Envelope exponent shaping low- versus high-energy syllables.",
    )
    parser.add_argument(
        "--audio-lip-sync-offset-ms",
        type=int,
        default=DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS,
        help="Time offset in milliseconds applied when sampling the audio envelope.",
    )
    parser.add_argument(
        "--driving-multiplier",
        type=float,
        default=DEFAULT_DRIVING_MULTIPLIER,
        help="Global motion amplitude multiplier applied during render [0..2].",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=DEFAULT_CFG_SCALE,
        help="JoyVASA classifier-free guidance scale used for audio motion generation [0..10].",
    )
    parser.add_argument(
        "--joyvasa-inference-steps",
        type=int,
        default=DEFAULT_JOYVASA_INFERENCE_STEPS,
        help="JoyVASA diffusion inference steps [1..100].",
    )
    parser.add_argument(
        "--render-batch-size",
        type=int,
        default=DEFAULT_RENDER_BATCH_SIZE,
        help="Mini-batch size used when rendering precomputed .pkl motion.",
    )
    parser.add_argument(
        "--trt-engine-batch-size",
        type=int,
        default=DEFAULT_TRT_ENGINE_BATCH_SIZE,
        help="Maximum dynamic batch size for TensorRT engines used by batched render paths.",
    )
    parser.add_argument("--stream-dir", default="output_fasterliveportrait/stream")
    parser.add_argument("--stream-shm-prefix", default="")
    parser.add_argument("--disable-stream", action="store_true")
    parser.add_argument("--mode", choices=[MODE_PREVIEW, MODE_FULL], default=MODE_PREVIEW)
    parser.add_argument("--backend", choices=[BACKEND_ONNX, BACKEND_TRT], default=BACKEND_ONNX)
    parser.add_argument(
        "--trt-precision",
        choices=TRT_PRECISION_CHOICES,
        default=DEFAULT_TRT_PRECISION,
        help="TensorRT precision target for generated engines.",
    )
    parser.add_argument(
        "--trt-runtime",
        choices=[TRT_RUNTIME_LOCAL, TRT_RUNTIME_DOCKER],
        default=TRT_RUNTIME_DOCKER,
    )
    parser.add_argument("--docker-image", default=DEFAULT_TRT_DOCKER_IMAGE)
    parser.add_argument(
        "--docker-gpu-device",
        default=DEFAULT_TRT_DOCKER_GPU_DEVICE,
        help="Docker GPU selector. Use 'auto', 'all', or a device index list such as '1' or '0,1'.",
    )
    parser.add_argument(
        "--docker-container-name",
        default=DEFAULT_TRT_DOCKER_CONTAINER_NAME,
        help="Persistent container name used when docker reuse is enabled.",
    )
    parser.add_argument(
        "--docker-reuse-container",
        dest="docker_reuse_container",
        action="store_true",
        help="Reuse a persistent Docker container to avoid per-job cold starts.",
    )
    parser.add_argument(
        "--no-docker-reuse-container",
        dest="docker_reuse_container",
        action="store_false",
        help="Disable persistent Docker container reuse.",
    )
    parser.set_defaults(docker_reuse_container=DEFAULT_TRT_DOCKER_REUSE_CONTAINER)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument(
        "--audio-template-cache-dir",
        default=DEFAULT_AUDIO_TEMPLATE_CACHE_DIR,
        help="Global cache directory for audio-to-motion templates (.pkl).",
    )
    parser.add_argument(
        "--source-cache-dir",
        default=DEFAULT_SOURCE_CACHE_DIR,
        help="Global cache directory for source preprocess artifacts.",
    )
    parser.add_argument(
        "--persistent-worker-queue-dir",
        default=DEFAULT_PERSISTENT_WORKER_QUEUE_DIR,
        help="Queue directory used by persistent TRT worker.",
    )
    parser.add_argument(
        "--disable-persistent-trt-worker",
        action="store_true",
        help="Disable persistent Docker TRT worker and use per-job run.py execution.",
    )
    parser.add_argument("--skip-driving-video-build", action="store_true")
    parser.add_argument("--rebuild-driving-template", action="store_true")
    parser.add_argument("--skip-trt-engine-build", action="store_true")
    parser.add_argument("--video-encoder", choices=VIDEO_ENCODER_CHOICES, default=DEFAULT_VIDEO_ENCODER)
    parser.add_argument("--no-paste-back", action="store_true")
    parser.add_argument("--defer-paste-back", action="store_true")
    parser.add_argument("--animation-region", choices=ANIMATION_REGION_CHOICES, default="all")
    parser.add_argument("--stitching-enabled", dest="stitching_enabled", action="store_true")
    parser.add_argument("--no-stitching", dest="stitching_enabled", action="store_false")
    parser.add_argument("--relative-motion-enabled", dest="relative_motion_enabled", action="store_true")
    parser.add_argument("--no-relative-motion", dest="relative_motion_enabled", action="store_false")
    parser.set_defaults(stitching_enabled=True, relative_motion_enabled=True)
    args = parser.parse_args()

    frame_step = max(1, args.frame_step)
    if args.mode == MODE_FULL:
        frame_step = 1
    render_batch_size = max(1, int(args.render_batch_size))
    trt_engine_batch_size = max(render_batch_size, int(args.trt_engine_batch_size))
    generation_frame_count = int(args.generation_frame_count or 0)
    if generation_frame_count < 0:
        parser.error("--generation-frame-count must be zero or a positive integer")
    if 0 < generation_frame_count < GENERATION_FRAME_COUNT_MIN:
        parser.error(
            f"--generation-frame-count must be >= {GENERATION_FRAME_COUNT_MIN} when provided"
        )
    if generation_frame_count > GENERATION_FRAME_COUNT_MAX:
        parser.error(
            f"--generation-frame-count must be <= {GENERATION_FRAME_COUNT_MAX}"
        )
    audio_eye_soft_factor = float(np.clip(float(args.audio_eye_soft_factor), 0.0, 1.0))
    audio_eye_hard_factor = float(np.clip(float(args.audio_eye_hard_factor), 0.0, 1.0))
    audio_eye_hard_dy_min = float(min(args.audio_eye_hard_dy_min, args.audio_eye_hard_dy_max))
    audio_eye_hard_dy_max = float(max(args.audio_eye_hard_dy_min, args.audio_eye_hard_dy_max))
    audio_reanchor_first_n = int(np.clip(int(args.audio_reanchor_first_n), 1, 15))
    audio_mouth_open_factor = float(np.clip(float(args.audio_mouth_open_factor), 0.0, 3.0))
    audio_pose_smooth_window = max(0, int(args.audio_pose_smooth_window))
    audio_exp_smooth_window = max(0, int(args.audio_exp_smooth_window))
    audio_pose_jump_threshold = float(np.clip(float(args.audio_pose_jump_threshold), 0.0, 60.0))
    audio_translation_jump_threshold = float(np.clip(float(args.audio_translation_jump_threshold), 0.0, 1.0))
    audio_lip_sync_min_ratio = float(np.clip(float(args.audio_lip_sync_min_ratio), 0.0, 1.0))
    audio_lip_sync_max_ratio = float(np.clip(float(args.audio_lip_sync_max_ratio), audio_lip_sync_min_ratio, 1.0))
    audio_lip_sync_smooth_window = max(0, int(args.audio_lip_sync_smooth_window))
    audio_lip_sync_strength = float(np.clip(float(args.audio_lip_sync_strength), 0.0, 4.0))
    audio_lip_sync_power = float(np.clip(float(args.audio_lip_sync_power), 0.001, 4.0))
    audio_lip_sync_offset_ms = int(np.clip(int(args.audio_lip_sync_offset_ms), -1000, 1000))
    driving_multiplier = float(np.clip(float(args.driving_multiplier), 0.0, 2.0))
    cfg_scale = float(np.clip(float(args.cfg_scale), 0.0, 10.0))
    joyvasa_inference_steps = int(np.clip(int(args.joyvasa_inference_steps), 1, 100))

    project_root = Path(__file__).resolve().parent
    driving_audio = (project_root / args.driving_audio).resolve() if args.driving_audio else None
    docker_gpu_device = str(args.docker_gpu_device).strip()
    if docker_gpu_device.lower() == "auto":
        docker_gpu_device = detect_preferred_gpu_device()
        print(f"[info] selected docker gpu device: {docker_gpu_device}")
    return RunnerConfig(
        project_root=project_root,
        source_frame=(project_root / args.source_frame).resolve(),
        frames_dir=(project_root / args.frames_dir).resolve(),
        meta_path=(project_root / args.meta_path).resolve(),
        output_dir=(project_root / args.output_dir).resolve(),
        faster_repo_dir=(project_root / args.faster_repo_dir).resolve(),
        python_executable=(project_root / args.python_executable).resolve(),
        mode=args.mode,
        backend=args.backend,
        trt_runtime=args.trt_runtime,
        trt_precision=args.trt_precision,
        docker_image=args.docker_image,
        docker_gpu_device=docker_gpu_device,
        docker_container_name=str(args.docker_container_name).strip(),
        docker_reuse_container=bool(args.docker_reuse_container),
        audio_template_cache_dir=(project_root / args.audio_template_cache_dir).resolve(),
        source_cache_dir=(project_root / args.source_cache_dir).resolve(),
        persistent_worker_queue_dir=(project_root / args.persistent_worker_queue_dir).resolve(),
        use_persistent_trt_worker=not args.disable_persistent_trt_worker,
        driving_audio=driving_audio,
        audio_motion_stride=max(1, int(args.audio_motion_stride)),
        generation_frame_count=generation_frame_count or None,
        audio_eye_tamed_preset=bool(args.audio_eye_tamed_preset),
        audio_eye_soft_factor=audio_eye_soft_factor,
        audio_eye_hard_factor=audio_eye_hard_factor,
        audio_eye_hard_dy_min=audio_eye_hard_dy_min,
        audio_eye_hard_dy_max=audio_eye_hard_dy_max,
        audio_motion_tuning_enabled=bool(args.audio_motion_tuning_enabled),
        audio_reanchor_first_n=audio_reanchor_first_n,
        audio_mouth_open_factor=audio_mouth_open_factor,
        audio_pose_smooth_window=audio_pose_smooth_window,
        audio_exp_smooth_window=audio_exp_smooth_window,
        audio_pose_jump_threshold=audio_pose_jump_threshold,
        audio_translation_jump_threshold=audio_translation_jump_threshold,
        audio_lip_sync_assist=bool(args.audio_lip_sync_assist),
        audio_lip_sync_min_ratio=audio_lip_sync_min_ratio,
        audio_lip_sync_max_ratio=audio_lip_sync_max_ratio,
        audio_lip_sync_smooth_window=audio_lip_sync_smooth_window,
        audio_lip_sync_strength=audio_lip_sync_strength,
        audio_lip_sync_power=audio_lip_sync_power,
        audio_lip_sync_offset_ms=audio_lip_sync_offset_ms,
        driving_multiplier=driving_multiplier,
        cfg_scale=cfg_scale,
        joyvasa_inference_steps=joyvasa_inference_steps,
        render_batch_size=render_batch_size,
        trt_engine_batch_size=trt_engine_batch_size,
        stream_dir=(project_root / args.stream_dir).resolve(),
        stream_shm_prefix=str(args.stream_shm_prefix or "").strip(),
        stream_enabled=not args.disable_stream,
        frame_step=frame_step,
        skip_driving_video_build=args.skip_driving_video_build,
        rebuild_driving_template=args.rebuild_driving_template,
        skip_trt_engine_build=args.skip_trt_engine_build,
        video_encoder=str(args.video_encoder).strip().lower(),
        paste_back=not args.no_paste_back,
        defer_paste_back=bool(args.defer_paste_back),
        animation_region=str(args.animation_region).strip().lower(),
        stitching_enabled=bool(args.stitching_enabled),
        relative_motion_enabled=bool(args.relative_motion_enabled),
    )


def assert_path_exists(path: Path, label: str) -> None:
    """
    Ensure required filesystem path exists.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def is_deferred_paste_back_enabled(config: RunnerConfig) -> bool:
    """
    Enable deferred paste-back only when the final contract still requires paste-back output.
    """
    if config.source_frame.suffix.lower() in SOURCE_VIDEO_EXTENSIONS:
        return False
    return bool(config.defer_paste_back and config.paste_back and config.stitching_enabled)


def should_render_paste_back(config: RunnerConfig) -> bool:
    """
    Resolve whether the heavy core pipeline should execute paste-back internally.
    """
    return bool(config.paste_back and not is_deferred_paste_back_enabled(config))


def should_export_preview_composition(config: RunnerConfig) -> bool:
    """
    Resolve whether the core pipeline must emit lightweight composition metadata.
    """
    return is_deferred_paste_back_enabled(config)


def read_fps(meta_path: Path) -> float:
    """
    Read source FPS from meta file with fallback default.
    """
    if not meta_path.exists():
        return DEFAULT_FPS

    with meta_path.open("r", encoding="utf-8") as handle:
        meta_data = json.load(handle)

    fps_value = meta_data.get("fps", DEFAULT_FPS)
    if not isinstance(fps_value, (int, float)):
        return DEFAULT_FPS
    if fps_value <= 0:
        return DEFAULT_FPS
    return float(fps_value)


def read_source_video_fps(source_path: Path) -> float:
    """
    Read FPS from a source video with a safe default.
    """
    capture = cv2.VideoCapture(str(source_path))
    try:
        fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if fps_value <= 0 or not np.isfinite(fps_value):
        return DEFAULT_FPS
    return float(fps_value)


def resolve_source_fps(config: RunnerConfig) -> float:
    """
    Resolve effective source FPS from source video when available, else from metadata.
    """
    if config.source_frame.suffix.lower() in SOURCE_VIDEO_EXTENSIONS:
        return read_source_video_fps(config.source_frame)
    return read_fps(config.meta_path)


def read_template_fps(template_path: Path, fallback_fps: float) -> float:
    """
    Read FPS from motion template pickle with fallback.
    """
    if not template_path.exists():
        return fallback_fps
    with template_path.open("rb") as handle:
        template_data = pickle.load(handle)
    fps_value = template_data.get("output_fps", fallback_fps)
    if not isinstance(fps_value, (int, float)):
        return fallback_fps
    if fps_value <= 0:
        return fallback_fps
    return float(fps_value)


def onnx2trt_supports_max_batch_size(script_path: Path) -> bool:
    """
    Detect whether the current FasterLivePortrait onnx2trt.py accepts --max-batch-size.
    """
    if not script_path.exists():
        return False
    try:
        script_text = script_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "--max-batch-size" in script_text


def build_motion_sample_indices(source_count: int, target_count: int) -> list[int]:
    """
    Build monotonically increasing sample indices for motion downsampling.
    """
    if source_count <= 0 or target_count <= 0:
        return []
    if target_count >= source_count:
        return list(range(source_count))
    if target_count == 1:
        return [0]
    indices: list[int] = []
    denominator = float(target_count - 1)
    scale = float(source_count - 1)
    for target_idx in range(target_count):
        source_position = scale * (float(target_idx) / denominator)
        source_idx = int(round(source_position))
        if source_idx < 0:
            source_idx = 0
        if source_idx >= source_count:
            source_idx = source_count - 1
        indices.append(source_idx)
    return indices


def resolve_motion_stride_target_fps(source_fps: float, motion_stride: int) -> int:
    """
    Resolve one deterministic target FPS for reduced-motion generation.
    """
    safe_source_fps = max(1.0, float(source_fps))
    safe_motion_stride = max(1, int(motion_stride))
    if safe_motion_stride <= 1:
        return max(1, int(round(safe_source_fps)))
    if safe_motion_stride == FIXED_AUDIO_MOTION_STRIDE:
        return max(1, int(round(min(safe_source_fps, FIXED_AUDIO_MOTION_TARGET_FPS))))
    return max(1, int(round(safe_source_fps / float(safe_motion_stride))))


def build_strided_audio_template(
    source_template_path: Path,
    output_template_path: Path,
    motion_stride: int,
) -> None:
    """
    Build a reduced-motion template to speed up inference while preserving approximate duration.
    """
    if motion_stride <= 1:
        shutil.copy2(source_template_path, output_template_path)
        return

    with source_template_path.open("rb") as handle:
        source_template = pickle.load(handle)

    if not isinstance(source_template, dict):
        shutil.copy2(source_template_path, output_template_path)
        return

    source_motion = source_template.get("motion")
    if not isinstance(source_motion, list) or not source_motion:
        shutil.copy2(source_template_path, output_template_path)
        return

    source_frame_count = len(source_motion)
    source_fps_raw = source_template.get("output_fps", DEFAULT_FPS)
    source_fps = int(source_fps_raw) if isinstance(source_fps_raw, (int, float)) else int(DEFAULT_FPS)
    if source_fps <= 0:
        source_fps = int(DEFAULT_FPS)

    source_duration_sec = float(source_frame_count) / float(source_fps)
    target_fps = resolve_motion_stride_target_fps(source_fps, motion_stride)
    target_frame_count = max(1, int(round(source_duration_sec * float(target_fps))))
    sample_indices = build_motion_sample_indices(source_frame_count, target_frame_count)
    if not sample_indices:
        shutil.copy2(source_template_path, output_template_path)
        return

    output_template: dict = dict(source_template)
    output_template["motion"] = [source_motion[idx] for idx in sample_indices]
    output_template["n_frames"] = len(output_template["motion"])
    output_template["output_fps"] = target_fps

    for channel_key in ("c_eyes_lst", "c_lip_lst"):
        channel_value = source_template.get(channel_key)
        if isinstance(channel_value, list) and len(channel_value) == source_frame_count:
            output_template[channel_key] = [channel_value[idx] for idx in sample_indices]

    output_template_path.parent.mkdir(parents=True, exist_ok=True)
    with output_template_path.open("wb") as handle:
        pickle.dump(output_template, handle, protocol=pickle.HIGHEST_PROTOCOL)


def is_audio_file(path: Path) -> bool:
    """
    Determine whether path extension is treated as driving audio.
    """
    return path.suffix.lower() in DRIVING_AUDIO_EXTENSIONS


def resolve_driving_public_name(driving_media: Path) -> str:
    """
    Build stable public alias for driving media file.
    """
    if is_audio_file(driving_media):
        return f"driving{driving_media.suffix.lower()}"
    return DRIVING_PUBLIC_NAME


def prepare_stream_dir(config: RunnerConfig) -> None:
    """
    Ensure stream artifact directory exists for auxiliary preview assets.
    """
    if not config.stream_enabled:
        return
    config.stream_dir.mkdir(parents=True, exist_ok=True)


def build_runtime_env() -> dict[str, str]:
    """
    Build runtime environment for subprocess execution.
    """
    return build_process_env(PROJECT_ROOT, os.environ)


def run_command(command: Sequence[str], cwd: Path | None = None) -> None:
    """
    Execute subprocess command and fail fast on errors.
    """
    printable = " ".join(command)
    print(f"\n[cmd] {printable}")
    runtime_env = build_runtime_env()
    started_at = time.perf_counter()
    try:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True, env=runtime_env)
    except subprocess.CalledProcessError:
        duration_sec = time.perf_counter() - started_at
        print(f"[error] command failed duration_sec={duration_sec:.3f}: {printable}")
        raise
    duration_sec = time.perf_counter() - started_at
    print(f"[info] command completed duration_sec={duration_sec:.3f}: {printable}")


def start_detached_process(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    """
    Start detached background process and append logs to the provided file.
    """
    runtime_env = build_runtime_env()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_handle:
        if os.name == "nt":
            detached_process = 0x00000008
            new_process_group = 0x00000200
            subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=runtime_env,
                creationflags=detached_process | new_process_group,
            )
            return
        subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=runtime_env,
            start_new_session=True,
        )


def detect_preferred_gpu_device() -> str:
    """
    Detect the most capable GPU index using nvidia-smi.
    Returns "all" when no reliable detection is available.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "all"

    candidates: list[tuple[int, int, int, str]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        index_text, name_text, memory_text = parts
        if not index_text.isdigit():
            continue
        index_value = int(index_text)
        memory_value = int(memory_text) if memory_text.isdigit() else 0
        model_numbers = [int(token) for token in re.findall(r"\b\d{3,4}\b", name_text)]
        model_score = max(model_numbers) if model_numbers else 0
        candidates.append((model_score, memory_value, -index_value, index_text))

    if not candidates:
        return "all"
    candidates.sort(reverse=True)
    return candidates[0][3]


def resolve_docker_gpus_argument(config: RunnerConfig) -> str:
    """
    Convert configured GPU selector into Docker --gpus argument value.
    """
    value = str(config.docker_gpu_device).strip()
    if not value or value.lower() == "all":
        return "all"
    return f"device={value}"


def resolve_docker_visible_devices(config: RunnerConfig) -> str:
    """
    Resolve CUDA_VISIBLE_DEVICES value for containerized TRT runs.
    """
    value = str(config.docker_gpu_device).strip()
    if not value or value.lower() == "all":
        return ""
    return value


def inspect_container_running(container_name: str) -> bool | None:
    """
    Inspect container running state.
    Returns:
    - True if container exists and is running
    - False if container exists but is stopped
    - None if container does not exist
    """
    command = ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    state_text = completed.stdout.strip().lower()
    return state_text == "true"


def inspect_container_ipc_mode(container_name: str) -> str | None:
    """
    Inspect container IPC mode when the container exists.
    """
    command = ["docker", "inspect", "-f", "{{.HostConfig.IpcMode}}", container_name]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def running_inside_container() -> bool:
    """
    Detect whether the current API process itself runs inside a container.
    """
    if Path("/.dockerenv").exists():
        return True
    for cgroup_path in (Path("/proc/self/cgroup"), Path("/proc/1/cgroup")):
        try:
            cgroup_text = cgroup_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = cgroup_text.lower()
        if "docker" in lowered or "containerd" in lowered or "kubepods" in lowered:
            return True
    return False


def resolve_current_container_reference() -> str:
    """
    Resolve the current outer container identifier for docker IPC sharing.
    """
    explicit_reference = str(os.getenv(DOCKER_PARENT_CONTAINER_ENV_KEY, "")).strip()
    if explicit_reference:
        return explicit_reference
    hostname_reference = str(os.getenv("HOSTNAME", "")).strip()
    if hostname_reference and inspect_container_running(hostname_reference) is not None:
        return hostname_reference
    for cgroup_path in (Path("/proc/self/cgroup"), Path("/proc/1/cgroup")):
        try:
            cgroup_text = cgroup_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for candidate in re.findall(r"[0-9a-f]{12,64}", cgroup_text.lower()):
            if inspect_container_running(candidate) is not None:
                return candidate
    raise RuntimeError(
        "Unable to resolve current container reference for docker IPC sharing. "
        f"Set {DOCKER_PARENT_CONTAINER_ENV_KEY} explicitly."
    )


def resolve_docker_ipc_mode() -> str:
    """
    Resolve the IPC mode required so the docker runtime can share stream memory with the API process.
    """
    explicit_mode = str(os.getenv(DOCKER_IPC_MODE_ENV_KEY, "")).strip()
    if explicit_mode:
        return explicit_mode
    if running_inside_container():
        return f"container:{resolve_current_container_reference()}"
    return "host"


def ensure_runtime_container(config: RunnerConfig) -> None:
    """
    Ensure persistent runtime container exists and is running.
    """
    if not config.docker_reuse_container:
        return

    desired_ipc_mode = resolve_docker_ipc_mode()
    running_state = inspect_container_running(config.docker_container_name)
    existing_ipc_mode = inspect_container_ipc_mode(config.docker_container_name)
    if running_state is not None and existing_ipc_mode != desired_ipc_mode:
        remove_command = ["docker", "rm"]
        if running_state:
            remove_command.append("-f")
        remove_command.append(config.docker_container_name)
        run_command(remove_command)
        running_state = None
    if running_state is True:
        return
    if running_state is False:
        run_command(["docker", "start", config.docker_container_name])
        return

    command = [
        "docker",
        "run",
        "-d",
        "--gpus",
        resolve_docker_gpus_argument(config),
        "--name",
        config.docker_container_name,
        "--ipc",
        desired_ipc_mode,
        "-v",
        f"{to_docker_host_path(config.project_root)}:/workspace",
        "-w",
        "/workspace",
    ]
    visible_devices = resolve_docker_visible_devices(config)
    if visible_devices:
        command.extend(["-e", f"CUDA_VISIBLE_DEVICES={visible_devices}"])
    command.extend(
        [
            config.docker_image,
            "bash",
            "-lc",
            "tail -f /dev/null",
        ]
    )
    run_command(command)


def run_docker_shell(config: RunnerConfig, workdir: str, script: str) -> None:
    """
    Execute shell script inside Docker runtime using ephemeral or persistent mode.
    """
    visible_devices = resolve_docker_visible_devices(config)
    shell_prefix = ""
    if visible_devices:
        shell_prefix = f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)}; "
    if config.docker_reuse_container:
        ensure_runtime_container(config)
        command = [
            "docker",
            "exec",
            config.docker_container_name,
            "bash",
            "-lc",
            f"{shell_prefix}cd {shlex.quote(workdir)} && {script}",
        ]
        run_command(command)
        return

    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        resolve_docker_gpus_argument(config),
        "--ipc",
        resolve_docker_ipc_mode(),
        "-v",
        f"{to_docker_host_path(config.project_root)}:/workspace",
        "-w",
    ]
    if visible_devices:
        command.extend(["-e", f"CUDA_VISIBLE_DEVICES={visible_devices}"])
    command.extend(
        [
            workdir,
            config.docker_image,
            "bash",
            "-lc",
            f"{shell_prefix}{script}",
        ]
    )
    run_command(command)


def resolve_driving_paths(config: RunnerConfig) -> tuple[Path, Path, Path]:
    """
    Build mode-specific driving paths plus public alias.
    """
    mode_suffix = MODE_PREVIEW if config.mode == MODE_PREVIEW else MODE_FULL
    driving_video = config.output_dir / f"driving_{mode_suffix}.mp4"
    driving_template = config.output_dir / f"driving_{mode_suffix}.pkl"
    driving_public = config.output_dir / DRIVING_PUBLIC_NAME
    return driving_video, driving_template, driving_public


def resolve_audio_template_path(config: RunnerConfig) -> Path:
    """
    Build mode-specific JoyVASA motion template path for audio driving.
    """
    mode_suffix = MODE_PREVIEW if config.mode == MODE_PREVIEW else MODE_FULL
    return config.output_dir / f"driving_audio_{mode_suffix}.pkl"


def resolve_audio_template_meta_path(config: RunnerConfig) -> Path:
    """
    Build mode-specific metadata path for audio template cache validation.
    """
    mode_suffix = MODE_PREVIEW if config.mode == MODE_PREVIEW else MODE_FULL
    return config.output_dir / f"driving_audio_{mode_suffix}.{AUDIO_TEMPLATE_META_NAME}"


def resolve_audio_template_input_wav_path(config: RunnerConfig) -> Path:
    """
    Build mode-specific normalized WAV path for JoyVASA ingestion.
    """
    mode_suffix = MODE_PREVIEW if config.mode == MODE_PREVIEW else MODE_FULL
    return config.output_dir / f"driving_audio_{mode_suffix}.wav"


def compute_file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA1 hash for cache signature.
    """
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def ffmpeg_supports_encoder(encoder_name: str) -> bool:
    """
    Check whether the local FFmpeg runtime can actually initialize one specific video encoder.
    """
    safe_encoder_name = str(encoder_name or "").strip().lower()
    if not safe_encoder_name:
        return False
    cached_value = FFMPEG_ENCODER_SUPPORT_CACHE.get(safe_encoder_name)
    if cached_value is not None:
        return cached_value
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=1",
                "-frames:v",
                "1",
                "-c:v",
                safe_encoder_name,
                "-pix_fmt",
                "yuv420p",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=build_runtime_env(),
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        FFMPEG_ENCODER_SUPPORT_CACHE[safe_encoder_name] = False
        return False
    support_detected = completed.returncode == 0
    FFMPEG_ENCODER_SUPPORT_CACHE[safe_encoder_name] = support_detected
    return support_detected


def resolve_ffmpeg_video_encoder_name(video_encoder: str) -> str:
    """
    Resolve the concrete FFmpeg video codec name with NVENC auto fallback.
    """
    safe_video_encoder = str(video_encoder or VIDEO_ENCODER_AUTO).strip().lower()
    if safe_video_encoder not in VIDEO_ENCODER_CHOICES:
        safe_video_encoder = VIDEO_ENCODER_AUTO
    if safe_video_encoder == VIDEO_ENCODER_CPU:
        return FFMPEG_LIBX264
    if ffmpeg_supports_encoder(FFMPEG_H264_NVENC):
        return FFMPEG_H264_NVENC
    if safe_video_encoder == VIDEO_ENCODER_NVENC:
        print("[warn] requested NVENC video encoder is unavailable; falling back to libx264")
    return FFMPEG_LIBX264


def build_ffmpeg_video_encode_args(video_encoder: str, quality_value: int | str) -> list[str]:
    """
    Build FFmpeg codec arguments for browser-compatible H.264 output.
    """
    codec_name = resolve_ffmpeg_video_encoder_name(video_encoder)
    quality_text = str(quality_value)
    if codec_name == FFMPEG_H264_NVENC:
        return [
            "-c:v",
            codec_name,
            "-preset",
            "fast",
            "-rc",
            "vbr",
            "-cq",
            quality_text,
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        codec_name,
        "-preset",
        "veryfast",
        "-crf",
        quality_text,
        "-pix_fmt",
        "yuv420p",
    ]


def normalize_audio_for_joyvasa(input_audio_path: Path, output_wav_path: Path) -> None:
    """
    Convert audio to 16k mono PCM WAV for robust JoyVASA loading.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_wav_path),
    ]
    run_command(command)


def build_audio_signature(audio_path: Path) -> dict[str, str | int]:
    """
    Build deterministic signature for driving audio cache.
    """
    audio_stat = audio_path.stat()
    return {
        "size": int(audio_stat.st_size),
        "suffix": audio_path.suffix.lower(),
        "sha1": compute_file_sha1(audio_path),
    }


def is_audio_template_cache_hit(
    template_path: Path,
    meta_path: Path,
    expected_signature: dict[str, str | int],
    expected_generation_profile: dict[str, str | int | float | bool],
) -> bool:
    """
    Validate template path and metadata signature for cache hit checks.
    """
    if not template_path.exists():
        return False
    meta_payload = load_audio_template_meta(meta_path)
    if meta_payload is None:
        return False
    return (
        meta_payload.get("audioSignature") == expected_signature
        and meta_payload.get("generationProfile") == expected_generation_profile
    )


def build_audio_template_generation_profile(
    config: RunnerConfig,
) -> dict[str, str | int | float | bool]:
    """
    Build one stable description of how audio motion templates are generated.
    """
    cfg_path = resolve_cfg_path(config)
    return {
        "backend": config.backend,
        "cfg": cfg_path.name,
        "motionStride": int(config.audio_motion_stride),
        "generationFrameCount": int(config.generation_frame_count or 0),
        "eyeTamedPreset": bool(config.audio_eye_tamed_preset),
        "eyeSoftFactor": round(float(config.audio_eye_soft_factor), 6),
        "eyeHardFactor": round(float(config.audio_eye_hard_factor), 6),
        "eyeHardDyMin": round(float(config.audio_eye_hard_dy_min), 6),
        "eyeHardDyMax": round(float(config.audio_eye_hard_dy_max), 6),
        "audioMotionTuningEnabled": bool(config.audio_motion_tuning_enabled),
        "audioReanchorFirstN": int(config.audio_reanchor_first_n),
        "audioMouthOpenFactor": round(float(config.audio_mouth_open_factor), 6),
        "audioPoseSmoothWindow": int(config.audio_pose_smooth_window),
        "audioExpSmoothWindow": int(config.audio_exp_smooth_window),
        "audioPoseJumpThreshold": round(float(config.audio_pose_jump_threshold), 6),
        "audioTranslationJumpThreshold": round(float(config.audio_translation_jump_threshold), 6),
        "audioLipSyncAssist": bool(config.audio_lip_sync_assist),
        "audioLipSyncMinRatio": round(float(config.audio_lip_sync_min_ratio), 6),
        "audioLipSyncMaxRatio": round(float(config.audio_lip_sync_max_ratio), 6),
        "audioLipSyncSmoothWindow": int(config.audio_lip_sync_smooth_window),
        "audioLipSyncStrength": round(float(config.audio_lip_sync_strength), 6),
        "audioLipSyncPower": round(float(config.audio_lip_sync_power), 6),
        "audioLipSyncOffsetMs": int(config.audio_lip_sync_offset_ms),
        "cfgScale": round(float(config.cfg_scale), 6),
        "joyvasaInferenceSteps": int(config.joyvasa_inference_steps),
    }


def should_prebuild_audio_template(config: RunnerConfig) -> bool:
    """
    Route audio requests through PKL generation whenever one template-only control is active.
    """
    return bool(
        config.generation_frame_count is not None
        or config.audio_eye_tamed_preset
        or config.audio_motion_tuning_enabled
    )


def build_audio_template_cache_key(
    audio_signature: dict[str, str | int],
    generation_profile: dict[str, str | int | float | bool],
) -> str:
    """
    Build stable key for cross-job audio template cache.
    """
    payload = {
        "audio": audio_signature,
        "generationProfile": generation_profile,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()


def resolve_global_audio_cache_paths(
    config: RunnerConfig,
    audio_signature: dict[str, str | int],
) -> tuple[Path, Path]:
    """
    Resolve global shared cache paths for audio templates.
    """
    generation_profile = build_audio_template_generation_profile(config)
    cache_key = build_audio_template_cache_key(audio_signature, generation_profile)
    cache_root = config.audio_template_cache_dir / config.backend
    template_path = cache_root / f"{cache_key}.pkl"
    meta_path = cache_root / f"{cache_key}.{AUDIO_TEMPLATE_META_NAME}"
    return template_path, meta_path


def build_audio_template_meta_payload(
    audio_signature: dict[str, str | int],
    template_path: Path,
    config: RunnerConfig,
    normalized_audio_path: Path,
) -> dict:
    """
    Build metadata payload for local/global audio template cache.
    """
    generation_profile = build_audio_template_generation_profile(config)
    return {
        "audioSignature": audio_signature,
        "generationProfile": generation_profile,
        "audioTemplateInputWav": to_viewer_path(normalized_audio_path, config.project_root),
        "templatePath": to_viewer_path(template_path, config.project_root),
        "backend": generation_profile["backend"],
        "cfg": generation_profile["cfg"],
        "updatedAtMs": int(time.time() * 1000),
    }


def load_audio_template_meta(meta_path: Path) -> dict | None:
    """
    Read cached audio template metadata if present.
    """
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def write_audio_template_meta(meta_path: Path, payload: dict) -> None:
    """
    Persist audio template metadata.
    """
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_driving_video(config: RunnerConfig, driving_video: Path, source_fps: float) -> float:
    """
    Build deterministic driving MP4 from frame sequence.
    """
    if not any(config.frames_dir.glob("frame_*.png")):
        raise RuntimeError(f"No frame_*.png files found in {config.frames_dir}")

    target_fps = float(resolve_motion_stride_target_fps(source_fps, config.frame_step))

    if config.frame_step > 1:
        video_filter = f"fps={target_fps:.6f},scale=720:420:flags=lanczos,format=yuv420p"
    else:
        video_filter = "scale=720:420:flags=lanczos,format=yuv420p"

    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{source_fps:.6f}",
        "-start_number",
        "1",
        "-i",
        str(config.frames_dir / FRAME_PATTERN),
        "-vf",
        video_filter,
        *build_ffmpeg_video_encode_args(config.video_encoder, 16),
        "-movflags",
        "+faststart",
        "-r",
        f"{target_fps:.6f}",
        str(driving_video),
    ]
    run_command(command)
    return target_fps


def resolve_cfg_path(config: RunnerConfig) -> Path:
    """
    Resolve FasterLivePortrait config path based on selected backend.
    """
    cfg_name = "onnx_infer.yaml" if config.backend == BACKEND_ONNX else "trt_infer.yaml"
    return config.faster_repo_dir / "configs" / cfg_name


def build_driving_template_from_audio_local(
    config: RunnerConfig,
    driving_audio: Path,
    output_template: Path,
) -> None:
    """
    Build JoyVASA motion template from driving audio using local Python runtime.
    """
    script_path = config.project_root / AUDIO_TO_PKL_SCRIPT_NAME
    assert_path_exists(script_path, "Audio-to-PKL script")
    cfg_path = resolve_cfg_path(config)
    assert_path_exists(cfg_path, "FasterLivePortrait config")
    command = [
        str(config.python_executable),
        str(script_path),
        "--faster-repo-dir",
        str(config.faster_repo_dir),
        "--cfg",
        str(cfg_path),
        "--driving-audio",
        str(driving_audio),
        "--output-pkl",
        str(output_template),
    ]
    command.extend(build_audio_to_pkl_extra_args(config))
    run_command(command)


def build_audio_to_pkl_extra_args(config: RunnerConfig) -> list[str]:
    """
    Build extra audio-to-PKL tuning arguments shared by local and docker invocations.
    """
    command = [
        "--motion-stride",
        str(int(config.audio_motion_stride)),
        "--eye-soft-factor",
        f"{float(config.audio_eye_soft_factor):.6f}",
        "--eye-hard-factor",
        f"{float(config.audio_eye_hard_factor):.6f}",
        "--eye-hard-dy-min",
        f"{float(config.audio_eye_hard_dy_min):.6f}",
        "--eye-hard-dy-max",
        f"{float(config.audio_eye_hard_dy_max):.6f}",
        "--cfg-scale",
        f"{float(config.cfg_scale):.6f}",
        "--inference-steps",
        str(int(config.joyvasa_inference_steps)),
        "--reanchor-first-n",
        str(int(config.audio_reanchor_first_n)),
        "--mouth-open-factor",
        f"{float(config.audio_mouth_open_factor):.6f}",
        "--pose-smooth-window",
        str(int(config.audio_pose_smooth_window)),
        "--exp-smooth-window",
        str(int(config.audio_exp_smooth_window)),
        "--pose-jump-threshold",
        f"{float(config.audio_pose_jump_threshold):.6f}",
        "--translation-jump-threshold",
        f"{float(config.audio_translation_jump_threshold):.6f}",
        "--lip-sync-min-ratio",
        f"{float(config.audio_lip_sync_min_ratio):.6f}",
        "--lip-sync-max-ratio",
        f"{float(config.audio_lip_sync_max_ratio):.6f}",
        "--lip-sync-smooth-window",
        str(int(config.audio_lip_sync_smooth_window)),
        "--lip-sync-strength",
        f"{float(config.audio_lip_sync_strength):.6f}",
        "--lip-sync-power",
        f"{float(config.audio_lip_sync_power):.6f}",
        "--lip-sync-offset-ms",
        str(int(config.audio_lip_sync_offset_ms)),
    ]
    if config.generation_frame_count is not None:
        command.extend(
            [
                "--generation-frame-count",
                str(int(config.generation_frame_count)),
            ]
        )
    command.append("--enable-eye-tamed-preset" if config.audio_eye_tamed_preset else "--disable-eye-tamed-preset")
    command.append(
        "--enable-audio-motion-tuning" if config.audio_motion_tuning_enabled else "--disable-audio-motion-tuning"
    )
    command.append(
        "--enable-audio-lip-sync-assist" if config.audio_lip_sync_assist else "--disable-audio-lip-sync-assist"
    )
    return command


def build_audio_to_pkl_shell_args(config: RunnerConfig) -> str:
    """
    Build shell-quoted extra audio-to-PKL tuning arguments.
    """
    return " ".join(shlex.quote(token) for token in build_audio_to_pkl_extra_args(config))


def build_driving_template_from_audio_docker(
    config: RunnerConfig,
    driving_audio: Path,
    output_template: Path,
) -> None:
    """
    Build JoyVASA motion template from driving audio inside TRT Docker runtime.
    """
    script_path = config.project_root / AUDIO_TO_PKL_SCRIPT_NAME
    assert_path_exists(script_path, "Audio-to-PKL script")
    cfg_path = resolve_cfg_path(config)
    assert_path_exists(cfg_path, "FasterLivePortrait config")

    project_root = config.project_root
    script_rel = to_project_relative(script_path, project_root, "Audio-to-PKL script")
    faster_repo_rel = to_project_relative(config.faster_repo_dir, project_root, "FasterLivePortrait repo")
    cfg_rel = to_project_relative(cfg_path, project_root, "FasterLivePortrait cfg")
    audio_rel = to_project_relative(driving_audio, project_root, "Driving audio")
    template_rel = to_project_relative(output_template, project_root, "Output template")
    extra_args = build_audio_to_pkl_shell_args(config)

    script = (
        f"export LD_LIBRARY_PATH={DEFAULT_TRT_DOCKER_LD_PATH}:$LD_LIBRARY_PATH; "
        f"{DEFAULT_TRT_DOCKER_PYTHON} -c \"import transformers as t; import sys; "
        f"v=str(t.__version__); "
        f"print('[check] transformers=' + v); "
        f"sys.exit(0 if v.startswith('4.40.') else 2)\"; "
        f"{DEFAULT_TRT_DOCKER_PYTHON} /workspace/{script_rel} "
        f"--faster-repo-dir /workspace/{faster_repo_rel} "
        f"--cfg /workspace/{cfg_rel} "
        f"--driving-audio /workspace/{audio_rel} "
        f"--output-pkl /workspace/{template_rel}"
    )
    if extra_args:
        script += f" {extra_args}"
    run_docker_shell(config, "/workspace", script)


def build_driving_template_from_audio(
    config: RunnerConfig,
    driving_audio: Path,
    output_template: Path,
) -> None:
    """
    Build JoyVASA motion template using runtime aligned with selected backend.
    """
    if config.backend == BACKEND_TRT and config.trt_runtime == TRT_RUNTIME_DOCKER:
        build_driving_template_from_audio_docker(config, driving_audio, output_template)
        return
    build_driving_template_from_audio_local(config, driving_audio, output_template)


def mux_audio_into_video(video_path: Path, audio_path: Path, video_encoder: str) -> Path:
    """
    Add audio track and normalize to browser-compatible H.264/AAC.
    """
    output_path = video_path.with_name(f"{video_path.stem}-audio.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        *build_ffmpeg_video_encode_args(video_encoder, 20),
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(output_path),
    ]
    run_command(command)
    return output_path


def read_primary_video_codec(video_path: Path) -> str:
    """
    Read primary video codec name with ffprobe.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip().lower()


def ensure_browser_compatible_video(video_path: Path, video_encoder: str) -> Path:
    """
    Ensure output video uses browser-compatible H.264 video.
    """
    codec_name = read_primary_video_codec(video_path)
    if codec_name == "h264":
        return video_path

    output_path = video_path.with_name(f"{video_path.stem}-web.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        *build_ffmpeg_video_encode_args(video_encoder, 20),
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    run_command(command)
    return output_path


def read_preview_composition_meta(run_output_dir: Path) -> dict[str, object]:
    """
    Read deferred paste-back metadata emitted by the core render path.
    """
    meta_path = run_output_dir / PREVIEW_COMPOSITION_META_NAME
    assert_path_exists(meta_path, "Preview composition metadata")
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid preview composition metadata: {meta_path}")
    return payload


def load_preview_composition_mask(run_output_dir: Path, mask_file_name: str) -> np.ndarray:
    """
    Load one persisted alpha mask as a float blend tensor.
    """
    mask_path = run_output_dir / str(mask_file_name or PREVIEW_COMPOSITION_MASK_NAME)
    assert_path_exists(mask_path, "Preview composition mask")
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise RuntimeError(f"Unable to read preview composition mask: {mask_path}")
    if mask_image.ndim != 3 or mask_image.shape[2] < 4:
        raise RuntimeError(f"Invalid preview composition mask format: {mask_path}")
    alpha_channel = mask_image[..., 3].astype(np.float32) / 255.0
    return np.repeat(alpha_channel[..., None], 3, axis=2)


def compose_deferred_paste_back_video(
    config: RunnerConfig,
    run_output_dir: Path,
    crop_video_path: Path,
) -> Path:
    """
    Compose one full-frame result from crop-only output using exported preview metadata.
    """
    metadata = read_preview_composition_meta(run_output_dir)
    matrix_value = metadata.get("matrix")
    if not isinstance(matrix_value, list):
        raise RuntimeError("Preview composition matrix is missing.")
    transform_matrix = np.asarray(matrix_value, dtype=np.float32)
    if transform_matrix.shape != (2, 3):
        raise RuntimeError("Preview composition matrix has an invalid shape.")

    source_frame_image = cv2.imread(str(config.source_frame), cv2.IMREAD_COLOR)
    if source_frame_image is None:
        raise RuntimeError(f"Unable to read source frame for deferred paste-back: {config.source_frame}")
    source_height, source_width = source_frame_image.shape[:2]
    mask_float = load_preview_composition_mask(
        run_output_dir,
        str(metadata.get("maskFile") or PREVIEW_COMPOSITION_MASK_NAME),
    )
    if mask_float.shape[:2] != (source_height, source_width):
        raise RuntimeError("Preview composition mask dimensions do not match the source frame.")

    capture = cv2.VideoCapture(str(crop_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open crop video for deferred paste-back: {crop_video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = DEFAULT_FPS

    output_path = crop_video_path.with_name(f"{crop_video_path.stem}-deferred-org.mp4")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (source_width, source_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create deferred paste-back video: {output_path}")

    try:
        while True:
            ok, crop_frame_image = capture.read()
            if not ok:
                break
            warped_frame_image = cv2.warpAffine(
                crop_frame_image,
                transform_matrix,
                (source_width, source_height),
            )
            composed_frame_image = np.clip(
                (mask_float * warped_frame_image) + ((1.0 - mask_float) * source_frame_image),
                0.0,
                255.0,
            ).astype(np.uint8)
            writer.write(composed_frame_image)
    finally:
        capture.release()
        writer.release()

    assert_path_exists(output_path, "Deferred paste-back video")
    return output_path


def to_docker_host_path(path: Path) -> str:
    """
    Convert host filesystem path to Docker-friendly path for bind mount.
    """
    return str(path.resolve()).replace("\\", "/")


def to_project_relative(path: Path, project_root: Path, label: str) -> str:
    """
    Convert absolute project path to workspace-relative posix path.
    """
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"{label} must be inside project root for Docker mode: {path}"
        ) from exc


def to_viewer_path(path: Path, project_root: Path) -> str:
    """
    Convert absolute path to web-friendly project-relative path.
    """
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def list_result_dirs(results_root: Path) -> set[Path]:
    """
    Return existing result directory set.
    """
    if not results_root.exists():
        return set()
    return {path for path in results_root.iterdir() if path.is_dir()}


def choose_latest_path(paths: Iterable[Path], token: str) -> Path:
    """
    Select latest file path containing token in file name.
    """
    candidates = [path for path in paths if token in path.name and path.suffix.lower() == ".mp4"]
    if not candidates:
        raise RuntimeError(f"Cannot find output file containing token '{token}'.")
    return max(candidates, key=lambda value: value.stat().st_mtime)


def find_run_output_dir(results_root: Path, before_dirs: set[Path]) -> Path:
    """
    Find fresh run directory. Fallback to latest results directory.
    """
    after_dirs = list_result_dirs(results_root)
    new_dirs = [path for path in after_dirs if path not in before_dirs]
    if new_dirs:
        return max(new_dirs, key=lambda value: value.stat().st_mtime)
    if not after_dirs:
        raise RuntimeError(f"No results directory found in {results_root}")
    return max(after_dirs, key=lambda value: value.stat().st_mtime)


def resolve_engine_precision_marker_path(engine_path: Path) -> Path:
    """
    Resolve marker file path that tracks engine precision mode.
    """
    return engine_path.with_suffix(engine_path.suffix + ENGINE_PRECISION_MARKER_SUFFIX)


def resolve_engine_batch_marker_path(engine_path: Path) -> Path:
    """
    Resolve marker file path that tracks engine batch capacity.
    """
    return engine_path.with_suffix(engine_path.suffix + ENGINE_BATCH_MARKER_SUFFIX)


def read_engine_precision_marker(engine_path: Path) -> str:
    """
    Read precision marker for an existing TRT engine.
    """
    marker_path = resolve_engine_precision_marker_path(engine_path)
    if not marker_path.exists():
        return ""
    try:
        return marker_path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""


def read_engine_batch_marker(engine_path: Path) -> int:
    """
    Read batch marker for an existing TRT engine.
    """
    marker_path = resolve_engine_batch_marker_path(engine_path)
    if not marker_path.exists():
        return 1
    try:
        marker_text = marker_path.read_text(encoding="utf-8").strip()
    except OSError:
        return 1
    try:
        return max(1, int(marker_text))
    except ValueError:
        return 1


def write_engine_precision_marker(engine_path: Path, precision: str) -> None:
    """
    Persist precision marker for TRT engine.
    """
    marker_path = resolve_engine_precision_marker_path(engine_path)
    marker_path.write_text(precision, encoding="utf-8")


def write_engine_batch_marker(engine_path: Path, batch_size: int) -> None:
    """
    Persist batch marker for TRT engine.
    """
    marker_path = resolve_engine_batch_marker_path(engine_path)
    marker_path.write_text(str(max(1, int(batch_size))), encoding="utf-8")


def ensure_trt_engines(config: RunnerConfig) -> None:
    """
    Ensure required TensorRT engines exist for trt_infer config.
    """
    engine_jobs: tuple[tuple[str, str | None, int], ...] = (
        # warping_spade includes custom plugin path and is kept on FP16 for stability.
        ("warping_spade-fix.onnx", TRT_PRECISION_FP16, config.trt_engine_batch_size),
        ("landmark.onnx", None, 1),
        ("motion_extractor.onnx", TRT_PRECISION_FP32, 1),
        ("retinaface_det_static.onnx", None, 1),
        ("face_2dpose_106_static.onnx", None, 1),
        ("appearance_feature_extractor.onnx", None, 1),
        ("stitching.onnx", None, config.trt_engine_batch_size),
        ("stitching_eye.onnx", None, config.trt_engine_batch_size),
        ("stitching_lip.onnx", None, config.trt_engine_batch_size),
    )
    checkpoints_dir = config.faster_repo_dir / "checkpoints" / "liveportrait_onnx"
    build_jobs: list[dict[str, str]] = []
    for onnx_name, precision_override, target_batch_size in engine_jobs:
        target_precision = precision_override or config.trt_precision
        engine_name = onnx_name.replace(".onnx", ".trt")
        engine_path = checkpoints_dir / engine_name
        precision_marker = read_engine_precision_marker(engine_path)
        batch_marker = read_engine_batch_marker(engine_path)
        rebuild_reasons: list[str] = []
        requires_rebuild = not engine_path.exists()
        if requires_rebuild:
            rebuild_reasons.append("missing_engine")
        if not requires_rebuild:
            if target_precision == TRT_PRECISION_FP16 and precision_marker in {"", TRT_PRECISION_FP16}:
                requires_rebuild = False
            else:
                requires_rebuild = precision_marker != target_precision
                if requires_rebuild:
                    rebuild_reasons.append(
                        f"precision_marker={precision_marker or 'missing'} expected={target_precision}"
                    )
        if not requires_rebuild:
            requires_rebuild = batch_marker != max(1, int(target_batch_size))
            if requires_rebuild:
                rebuild_reasons.append(
                    f"batch_marker={batch_marker} expected={max(1, int(target_batch_size))}"
                )

        if not requires_rebuild:
            continue

        source_onnx_path = checkpoints_dir / onnx_name
        assert_path_exists(source_onnx_path, "Source ONNX for TRT build")
        print(
            "[info] TRT engine rebuild queued engine={} source={} precision={} max_batch_size={} reasons={}".format(
                engine_name,
                onnx_name,
                target_precision,
                max(1, int(target_batch_size)),
                ",".join(rebuild_reasons),
            )
        )
        build_jobs.append(
            {
                "engine_name": engine_name,
                "onnx_name": onnx_name,
                "onnx_path": source_onnx_path.as_posix(),
                "engine_path": engine_path.as_posix(),
                "precision": target_precision,
                "max_batch_size": str(max(1, int(target_batch_size))),
            }
        )

    if not build_jobs:
        print("[info] all required TRT engines already exist")
        return

    print(f"[info] building TRT engines: {len(build_jobs)} (target={config.trt_precision})")
    if config.trt_runtime == TRT_RUNTIME_DOCKER:
        faster_repo_rel = to_project_relative(
            config.faster_repo_dir,
            config.project_root,
            "FasterLivePortrait repo",
        )
        commands: list[str] = []
        for build_job in build_jobs:
            onnx_rel = to_project_relative(Path(build_job["onnx_path"]), config.faster_repo_dir, "TRT source ONNX")
            engine_rel = to_project_relative(Path(build_job["engine_path"]), config.faster_repo_dir, "TRT engine output")
            convert_cmd = (
                f"{DEFAULT_TRT_DOCKER_PYTHON} scripts/onnx2trt.py "
                f"-o {shlex.quote(onnx_rel)} "
                f"-e {shlex.quote(engine_rel)} "
                f"-p {shlex.quote(build_job['precision'])} "
                f"--max-batch-size {shlex.quote(build_job['max_batch_size'])}"
            )
            if build_job["precision"] == TRT_PRECISION_INT8:
                calibration_cache_rel = f"{engine_rel}{TRT_INT8_CALIBRATION_CACHE_SUFFIX}"
                convert_cmd += (
                    f" --calibration-cache {shlex.quote(calibration_cache_rel)}"
                    f" --calibration-batches {TRT_INT8_CALIBRATION_BATCHES}"
                )
            commands.append(convert_cmd)

        script = (
            f"export LD_LIBRARY_PATH={DEFAULT_TRT_DOCKER_LD_PATH}:$LD_LIBRARY_PATH; "
            + " && ".join(commands)
        )
        run_docker_shell(config, f"/workspace/{faster_repo_rel}", script)
    else:
        onnx2trt_script = config.faster_repo_dir / "scripts" / "onnx2trt.py"
        assert_path_exists(onnx2trt_script, "TensorRT build script")
        supports_max_batch_size = onnx2trt_supports_max_batch_size(onnx2trt_script)
        if not supports_max_batch_size:
            print("[warn] onnx2trt.py does not support --max-batch-size; building fixed-batch TensorRT engines.")
        for build_job in build_jobs:
            onnx_rel = to_project_relative(Path(build_job["onnx_path"]), config.faster_repo_dir, "TRT source ONNX")
            engine_rel = to_project_relative(Path(build_job["engine_path"]), config.faster_repo_dir, "TRT engine output")
            print(
                "[info] TRT engine build start engine={} precision={} max_batch_size={}".format(
                    build_job["engine_name"],
                    build_job["precision"],
                    build_job["max_batch_size"],
                )
            )
            build_started_at = time.perf_counter()
            command = [
                str(config.python_executable),
                "scripts/onnx2trt.py",
                "-o",
                onnx_rel,
                "-e",
                engine_rel,
                "-p",
                build_job["precision"],
            ]
            if supports_max_batch_size:
                command.extend(
                    [
                        "--max-batch-size",
                        str(build_job["max_batch_size"]),
                    ]
                )
            if build_job["precision"] == TRT_PRECISION_INT8:
                calibration_cache_rel = f"{engine_rel}{TRT_INT8_CALIBRATION_CACHE_SUFFIX}"
                command.extend(
                    [
                        "--calibration-cache",
                        calibration_cache_rel,
                        "--calibration-batches",
                        str(TRT_INT8_CALIBRATION_BATCHES),
                    ]
                )
            run_command(command, cwd=config.faster_repo_dir)
            print(
                "[info] TRT engine build done engine={} duration_sec={:.3f}".format(
                    build_job["engine_name"],
                    time.perf_counter() - build_started_at,
                )
            )
    for build_job in build_jobs:
        engine_path = Path(build_job["engine_path"])
        write_engine_precision_marker(engine_path, build_job["precision"])
        write_engine_batch_marker(engine_path, int(build_job["max_batch_size"]))


def to_container_workspace_path(path: Path, project_root: Path, label: str) -> str:
    """
    Convert host project path to container /workspace path.
    """
    relative_path = to_project_relative(path, project_root, label)
    return f"/workspace/{relative_path}"


def from_container_workspace_path(path_text: str, project_root: Path) -> Path:
    """
    Convert container /workspace path back to host project path.
    """
    value = str(path_text or "").strip()
    workspace_prefix = "/workspace/"
    if value.startswith(workspace_prefix):
        relative_path = value[len(workspace_prefix):]
        return (project_root / relative_path).resolve()
    return Path(value).resolve()


def read_json_file(path: Path) -> dict | None:
    """
    Read JSON dictionary from file path.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_json_atomic(path: Path, payload: dict) -> None:
    """
    Write JSON atomically using temp file replacement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(str(tmp_path), str(path))


def resolve_worker_queue_paths(config: RunnerConfig) -> tuple[Path, Path, Path, Path, Path]:
    """
    Resolve host paths for persistent worker queue artifacts.
    """
    queue_root = config.persistent_worker_queue_dir
    requests_dir = queue_root / "requests"
    responses_dir = queue_root / "responses"
    heartbeat_path = queue_root / PERSISTENT_WORKER_HEARTBEAT_FILE_NAME
    worker_log_path = queue_root / "worker.log"
    return queue_root, requests_dir, responses_dir, heartbeat_path, worker_log_path


def process_pid_is_alive(pid: object) -> bool:
    """
    Check whether a numeric process id still exists on the current host.
    """
    try:
        safe_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if safe_pid <= 0:
        return False
    if os.name == "nt":
        try:
            completed = subprocess.run(
                [
                    "tasklist",
                    "/FI",
                    f"PID eq {safe_pid}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return False
        stdout_text = str(completed.stdout or "").lower()
        if "no tasks are running" in stdout_text:
            return False
        return str(safe_pid) in stdout_text
    try:
        os.kill(safe_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, SystemError, ValueError):
        return False
    return True


def heartbeat_is_fresh(heartbeat_path: Path, require_live_pid: bool = False) -> bool:
    """
    Check whether worker heartbeat is recent enough.
    """
    payload = read_json_file(heartbeat_path)
    if payload is None:
        return False
    updated_at_ms = payload.get("updatedAtMs")
    if not isinstance(updated_at_ms, int):
        return False
    age_ms = int(time.time() * 1000) - updated_at_ms
    if age_ms > int(PERSISTENT_WORKER_HEARTBEAT_STALE_SEC * 1000):
        return False
    if not require_live_pid:
        return True
    return process_pid_is_alive(payload.get("pid"))


def ensure_persistent_trt_worker_docker(config: RunnerConfig) -> None:
    """
    Ensure persistent TRT worker is alive inside Docker runtime container.
    """
    if not config.docker_reuse_container:
        raise RuntimeError("Persistent TRT worker requires docker_reuse_container enabled.")

    queue_root, requests_dir, responses_dir, heartbeat_path, _ = resolve_worker_queue_paths(config)
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    if heartbeat_is_fresh(heartbeat_path):
        return

    ensure_runtime_container(config)
    cfg_path = resolve_cfg_path(config)
    assert_path_exists(cfg_path, "FasterLivePortrait config")
    worker_script = config.faster_repo_dir / "run_persistent_worker.py"
    assert_path_exists(worker_script, "Persistent worker script")

    queue_rel = to_project_relative(queue_root, config.project_root, "Persistent worker queue")
    faster_repo_rel = to_project_relative(config.faster_repo_dir, config.project_root, "FasterLivePortrait repo")
    source_cache_rel = to_project_relative(config.source_cache_dir, config.project_root, "Source cache directory")
    source_frame_workspace_path = to_container_workspace_path(config.source_frame, config.project_root, "Source frame")
    cfg_rel = cfg_path.relative_to(config.faster_repo_dir).as_posix()

    worker_command = (
        f"mkdir -p /workspace/{queue_rel}/requests /workspace/{queue_rel}/responses; "
        f"cd /workspace/{faster_repo_rel}; "
        f"export LD_LIBRARY_PATH={DEFAULT_TRT_DOCKER_LD_PATH}:$LD_LIBRARY_PATH; "
        f"nohup {DEFAULT_TRT_DOCKER_PYTHON} run_persistent_worker.py "
        f"--cfg {shlex.quote(cfg_rel)} "
        f"--queue_dir {shlex.quote(f'/workspace/{queue_rel}')} "
        f"--source_cache_dir {shlex.quote(f'/workspace/{source_cache_rel}')} "
        f"--preload_source_image {shlex.quote(source_frame_workspace_path)} "
        f"--render_batch_size {int(config.render_batch_size)} "
        f"--animation_region {shlex.quote(config.animation_region)} "
        f"--driving_multiplier {float(config.driving_multiplier):.6f} "
        f"--cfg_scale {float(config.cfg_scale):.6f} "
        f"--joyvasa_inference_steps {int(config.joyvasa_inference_steps)} "
    )
    if should_render_paste_back(config):
        worker_command += "--paste_back "
    if should_export_preview_composition(config):
        worker_command += "--export_preview_composition "
    if not config.stitching_enabled:
        worker_command += "--no_stitching "
    if not config.relative_motion_enabled:
        worker_command += "--no_relative_motion "
    worker_command += f"> /workspace/{queue_rel}/worker.log 2>&1 < /dev/null &"
    run_docker_shell(config, "/workspace", worker_command)

    deadline = time.time() + PERSISTENT_WORKER_STARTUP_TIMEOUT_SEC
    while time.time() < deadline:
        if heartbeat_is_fresh(heartbeat_path):
            print(f"[info] persistent TRT worker ready: {heartbeat_path}")
            return
        time.sleep(PERSISTENT_WORKER_POLL_SLEEP_SEC)

    raise RuntimeError(
        "Persistent TRT worker startup timed out. Check worker log: "
        f"{(queue_root / 'worker.log')}"
    )


def ensure_persistent_trt_worker_local(config: RunnerConfig) -> None:
    """
    Ensure persistent TRT worker is alive in local runtime.
    """
    queue_root, requests_dir, responses_dir, heartbeat_path, worker_log_path = resolve_worker_queue_paths(config)
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    if heartbeat_is_fresh(heartbeat_path, require_live_pid=True):
        return

    cfg_path = resolve_cfg_path(config)
    assert_path_exists(cfg_path, "FasterLivePortrait config")
    worker_script = config.faster_repo_dir / "run_persistent_worker.py"
    assert_path_exists(worker_script, "Persistent worker script")
    assert_path_exists(config.python_executable, "Python executable")

    worker_command = [
        str(config.python_executable),
        str(worker_script),
        "--cfg",
        str(cfg_path),
        "--queue_dir",
        str(queue_root),
        "--source_cache_dir",
        str(config.source_cache_dir),
        "--preload_source_image",
        str(config.source_frame),
        "--render_batch_size",
        str(int(config.render_batch_size)),
        "--animation_region",
        str(config.animation_region),
        "--driving_multiplier",
        f"{float(config.driving_multiplier):.6f}",
        "--cfg_scale",
        f"{float(config.cfg_scale):.6f}",
        "--joyvasa_inference_steps",
        str(int(config.joyvasa_inference_steps)),
    ]
    if should_render_paste_back(config):
        worker_command.append("--paste_back")
    if should_export_preview_composition(config):
        worker_command.append("--export_preview_composition")
    if not config.stitching_enabled:
        worker_command.append("--no_stitching")
    if not config.relative_motion_enabled:
        worker_command.append("--no_relative_motion")

    printable = " ".join(worker_command)
    print(f"\n[cmd] {printable}")
    start_detached_process(worker_command, cwd=config.faster_repo_dir, log_path=worker_log_path)

    deadline = time.time() + PERSISTENT_WORKER_STARTUP_TIMEOUT_SEC
    while time.time() < deadline:
        if heartbeat_is_fresh(heartbeat_path, require_live_pid=True):
            print(f"[info] persistent TRT local worker ready: {heartbeat_path}")
            return
        time.sleep(PERSISTENT_WORKER_POLL_SLEEP_SEC)

    raise RuntimeError(
        "Persistent TRT local worker startup timed out. Check worker log: "
        f"{worker_log_path}"
    )


def wait_for_persistent_worker_result(
    request_id: str,
    request_path: Path,
    response_path: Path,
    request_payload: dict,
    worker_log_path: Path,
) -> dict:
    """
    Submit one worker request and wait for result payload.
    """
    if response_path.exists():
        response_path.unlink()
    write_json_atomic(request_path, request_payload)
    print(f"[info] submitted persistent worker request: {request_id}")

    deadline = time.time() + PERSISTENT_WORKER_RESPONSE_TIMEOUT_SEC
    response_payload: dict | None = None
    while time.time() < deadline:
        response_payload = read_json_file(response_path)
        if response_payload is not None:
            break
        time.sleep(PERSISTENT_WORKER_POLL_SLEEP_SEC)

    if response_payload is None:
        raise RuntimeError(
            "Persistent worker response timed out. "
            f"request={request_id} worker_log={worker_log_path}"
        )

    try:
        response_path.unlink(missing_ok=True)
    except OSError:
        pass

    if str(response_payload.get("status", "")).lower() != "ok":
        error_message = str(response_payload.get("error", "unknown worker error"))
        raise RuntimeError(
            f"Persistent worker failed: {error_message}. Worker log: {worker_log_path}"
        )

    result_payload = response_payload.get("result")
    if not isinstance(result_payload, dict):
        raise RuntimeError("Persistent worker response missing result payload.")
    return result_payload


def build_local_driving_args(config: RunnerConfig, driving_input: Path) -> list[str]:
    """
    Build local run.py driving arguments using one single audio/video contract.
    """
    if is_audio_file(driving_input):
        command = [
            "--driving_audio",
            str(driving_input),
            "--motion_stride",
            str(int(config.audio_motion_stride)),
            "--cfg_scale",
            f"{float(config.cfg_scale):.6f}",
            "--joyvasa_inference_steps",
            str(int(config.joyvasa_inference_steps)),
        ]
        if config.generation_frame_count is not None:
            command.extend(
                [
                    "--generation_frame_count",
                    str(int(config.generation_frame_count)),
                ]
            )
    else:
        command = [
        "--dri_video",
        str(driving_input),
        ]
    command.extend(
        [
            "--driving_multiplier",
            f"{float(config.driving_multiplier):.6f}",
        ]
    )
    return command


def build_docker_driving_args(config: RunnerConfig, driving_workspace_path: str) -> str:
    """
    Build docker run.py driving arguments using one single audio/video contract.
    """
    driving_suffix = Path(driving_workspace_path).suffix.lower()
    if driving_suffix in DRIVING_AUDIO_EXTENSIONS:
        command = (
            f"--driving_audio {shlex.quote(driving_workspace_path)} "
            f"--motion_stride {int(config.audio_motion_stride)} "
            f"--cfg_scale {float(config.cfg_scale):.6f} "
            f"--joyvasa_inference_steps {int(config.joyvasa_inference_steps)} "
        )
        if config.generation_frame_count is not None:
            command += f"--generation_frame_count {int(config.generation_frame_count)} "
    else:
        command = f"--dri_video {shlex.quote(driving_workspace_path)} "
    command += f"--driving_multiplier {float(config.driving_multiplier):.6f} "
    return command


def build_worker_driving_payload(
    config: RunnerConfig,
    driving_input: Path,
    containerized: bool,
) -> dict[str, str | int | float]:
    """
    Build persistent worker driving payload fields using one single audio/video contract.
    """
    if not is_audio_file(driving_input):
        driving_value = (
            to_container_workspace_path(driving_input, config.project_root, "Driving input")
            if containerized
            else str(driving_input)
        )
        return {
            "driVideo": driving_value,
            "drivingAudio": "",
            "motionStride": 1,
            "generationFrameCount": 0,
            "drivingMultiplier": float(config.driving_multiplier),
            "cfgScale": float(config.cfg_scale),
            "joyvasaInferenceSteps": int(config.joyvasa_inference_steps),
        }

    driving_value = (
        to_container_workspace_path(driving_input, config.project_root, "Driving audio")
        if containerized
        else str(driving_input)
    )
    return {
        "driVideo": "",
        "drivingAudio": driving_value,
        "motionStride": int(config.audio_motion_stride),
        "generationFrameCount": int(config.generation_frame_count or 0),
        "drivingMultiplier": float(config.driving_multiplier),
        "cfgScale": float(config.cfg_scale),
        "joyvasaInferenceSteps": int(config.joyvasa_inference_steps),
    }


def run_faster_pipeline_local(
    config: RunnerConfig,
    driving_input: Path,
    raw_results_root: Path,
) -> tuple[Path, Path, Path]:
    """
    Execute FasterLivePortrait and return resolved output paths.
    """
    cfg_path = resolve_cfg_path(config)
    assert_path_exists(cfg_path, "FasterLivePortrait config")

    before_dirs = list_result_dirs(raw_results_root)
    command = [
        str(config.python_executable),
        "run.py",
        "--src_image",
        str(config.source_frame),
        "--cfg",
        str(cfg_path.relative_to(config.faster_repo_dir)),
        "--source_cache_dir",
        str(config.source_cache_dir),
        "--render_batch_size",
        str(int(config.render_batch_size)),
        "--video_encoder",
        str(config.video_encoder),
        "--animation_region",
        str(config.animation_region),
    ]
    command.extend(build_local_driving_args(config, driving_input))
    if config.stream_enabled:
        command.extend(["--stream_dir", str(config.stream_dir)])
        if config.stream_shm_prefix:
            command.extend(["--stream_shm_prefix", str(config.stream_shm_prefix)])
    if should_render_paste_back(config):
        command.append("--paste_back")
    if should_export_preview_composition(config):
        command.append("--export_preview_composition")
    if not config.stitching_enabled:
        command.append("--no_stitching")
    if not config.relative_motion_enabled:
        command.append("--no_relative_motion")

    run_command(command, cwd=config.faster_repo_dir)
    output_dir = find_run_output_dir(raw_results_root, before_dirs)
    all_files = list(output_dir.glob("*.mp4"))
    result_org = choose_latest_path(all_files, "-org")
    result_crop = choose_latest_path(all_files, "-crop")
    return output_dir, result_org, result_crop


def run_faster_pipeline_local_trt_persistent_worker(
    config: RunnerConfig,
    driving_input: Path,
    raw_results_root: Path,
) -> tuple[Path, Path, Path]:
    """
    Execute TRT pipeline using persistent local worker with preloaded models.
    """
    ensure_persistent_trt_worker_local(config)

    _, requests_dir, responses_dir, _, worker_log_path = resolve_worker_queue_paths(config)
    request_id = uuid.uuid4().hex
    request_path = requests_dir / f"{request_id}{PERSISTENT_WORKER_REQUEST_SUFFIX}"
    response_path = responses_dir / f"{request_id}{PERSISTENT_WORKER_RESPONSE_SUFFIX}"
    run_output_dir = raw_results_root / f"worker_{request_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "requestId": request_id,
        "srcImage": str(config.source_frame),
        "streamDir": str(config.stream_dir) if config.stream_enabled else "",
        "streamShmPrefix": str(config.stream_shm_prefix) if config.stream_enabled else "",
        "saveDir": str(run_output_dir),
        "animal": False,
        "renderBatchSize": int(config.render_batch_size),
        "pasteBack": bool(should_render_paste_back(config)),
        "exportPreviewComposition": bool(should_export_preview_composition(config)),
        "animationRegion": str(config.animation_region),
        "stitchingEnabled": bool(config.stitching_enabled),
        "relativeMotionEnabled": bool(config.relative_motion_enabled),
        "sourceCacheDir": str(config.source_cache_dir),
    }
    request_payload.update(build_worker_driving_payload(config, driving_input, containerized=False))
    result_payload = wait_for_persistent_worker_result(
        request_id=request_id,
        request_path=request_path,
        response_path=response_path,
        request_payload=request_payload,
        worker_log_path=worker_log_path,
    )

    result_org = Path(str(result_payload.get("resultOrg", ""))).resolve()
    result_crop = Path(str(result_payload.get("resultCrop", ""))).resolve()
    resolved_output_dir = Path(str(result_payload.get("saveDir", ""))).resolve()
    assert_path_exists(resolved_output_dir, "Worker output directory")
    if not result_org.exists():
        result_org = result_crop
    assert_path_exists(result_org, "Worker result org video")
    assert_path_exists(result_crop, "Worker result crop video")
    return resolved_output_dir, result_org, result_crop


def run_faster_pipeline_docker_trt(
    config: RunnerConfig,
    driving_input: Path,
    raw_results_root: Path,
) -> tuple[Path, Path, Path]:
    """
    Execute FasterLivePortrait TRT pipeline in Docker and resolve output paths.
    """
    cfg_path = resolve_cfg_path(config)
    assert_path_exists(cfg_path, "FasterLivePortrait config")

    project_root = config.project_root
    faster_repo_rel = to_project_relative(config.faster_repo_dir, project_root, "FasterLivePortrait repo")
    src_rel = to_project_relative(config.source_frame, project_root, "Source frame")
    driving_rel = to_project_relative(driving_input, project_root, "Driving input")
    cfg_rel = cfg_path.relative_to(config.faster_repo_dir).as_posix()
    stream_rel = to_project_relative(config.stream_dir, project_root, "Stream directory")
    source_cache_rel = to_project_relative(config.source_cache_dir, project_root, "Source cache directory")

    script = (
        f"export LD_LIBRARY_PATH={DEFAULT_TRT_DOCKER_LD_PATH}:$LD_LIBRARY_PATH; "
        f"{DEFAULT_TRT_DOCKER_PYTHON} -c \"import colorama\"; "
        f"{DEFAULT_TRT_DOCKER_PYTHON} run.py "
        f"--src_image {shlex.quote(f'/workspace/{src_rel}')} "
        f"--cfg {shlex.quote(cfg_rel)} "
        f"--source_cache_dir {shlex.quote(f'/workspace/{source_cache_rel}')} "
        f"--render_batch_size {int(config.render_batch_size)} "
        f"--video_encoder {shlex.quote(config.video_encoder)} "
        f"--animation_region {shlex.quote(config.animation_region)}"
    )
    script += " " + build_docker_driving_args(config, f"/workspace/{driving_rel}").strip()
    if config.stream_enabled:
        script += f" --stream_dir {shlex.quote(f'/workspace/{stream_rel}')}"
        if config.stream_shm_prefix:
            script += f" --stream_shm_prefix {shlex.quote(str(config.stream_shm_prefix))}"
    if should_render_paste_back(config):
        script += " --paste_back"
    if should_export_preview_composition(config):
        script += " --export_preview_composition"
    if not config.stitching_enabled:
        script += " --no_stitching"
    if not config.relative_motion_enabled:
        script += " --no_relative_motion"

    before_dirs = list_result_dirs(raw_results_root)
    run_docker_shell(config, f"/workspace/{faster_repo_rel}", script)
    output_dir = find_run_output_dir(raw_results_root, before_dirs)
    all_files = list(output_dir.glob("*.mp4"))
    result_org = choose_latest_path(all_files, "-org")
    result_crop = choose_latest_path(all_files, "-crop")
    return output_dir, result_org, result_crop


def run_faster_pipeline_docker_trt_persistent_worker(
    config: RunnerConfig,
    driving_input: Path,
    raw_results_root: Path,
) -> tuple[Path, Path, Path]:
    """
    Execute TRT pipeline using persistent Docker worker with preloaded models.
    """
    ensure_persistent_trt_worker_docker(config)

    queue_root, requests_dir, responses_dir, _, worker_log_path = resolve_worker_queue_paths(config)
    request_id = uuid.uuid4().hex
    request_path = requests_dir / f"{request_id}{PERSISTENT_WORKER_REQUEST_SUFFIX}"
    response_path = responses_dir / f"{request_id}{PERSISTENT_WORKER_RESPONSE_SUFFIX}"
    run_output_dir = raw_results_root / f"worker_{request_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "requestId": request_id,
        "srcImage": to_container_workspace_path(config.source_frame, config.project_root, "Source frame"),
        "streamDir": (
            to_container_workspace_path(config.stream_dir, config.project_root, "Stream directory")
            if config.stream_enabled
            else ""
        ),
        "streamShmPrefix": str(config.stream_shm_prefix) if config.stream_enabled else "",
        "saveDir": to_container_workspace_path(run_output_dir, config.project_root, "Worker run output directory"),
        "animal": False,
        "renderBatchSize": int(config.render_batch_size),
        "pasteBack": bool(should_render_paste_back(config)),
        "exportPreviewComposition": bool(should_export_preview_composition(config)),
        "animationRegion": str(config.animation_region),
        "stitchingEnabled": bool(config.stitching_enabled),
        "relativeMotionEnabled": bool(config.relative_motion_enabled),
        "sourceCacheDir": to_container_workspace_path(
            config.source_cache_dir,
            config.project_root,
            "Source cache directory",
        ),
    }
    request_payload.update(build_worker_driving_payload(config, driving_input, containerized=True))
    result_payload = wait_for_persistent_worker_result(
        request_id=request_id,
        request_path=request_path,
        response_path=response_path,
        request_payload=request_payload,
        worker_log_path=worker_log_path,
    )

    result_org = from_container_workspace_path(str(result_payload.get("resultOrg", "")), config.project_root)
    result_crop = from_container_workspace_path(str(result_payload.get("resultCrop", "")), config.project_root)
    resolved_output_dir = from_container_workspace_path(str(result_payload.get("saveDir", "")), config.project_root)
    assert_path_exists(resolved_output_dir, "Worker output directory")
    if not result_org.exists():
        result_org = result_crop
    assert_path_exists(result_org, "Worker result org video")
    assert_path_exists(result_crop, "Worker result crop video")
    return resolved_output_dir, result_org, result_crop


def run_faster_pipeline(
    config: RunnerConfig,
    driving_input: Path,
    raw_results_root: Path,
) -> tuple[Path, Path, Path]:
    """
    Execute FasterLivePortrait and return resolved output paths.
    """
    if (
        config.backend == BACKEND_TRT
        and config.trt_runtime == TRT_RUNTIME_LOCAL
        and config.use_persistent_trt_worker
    ):
        return run_faster_pipeline_local_trt_persistent_worker(config, driving_input, raw_results_root)
    if config.backend == BACKEND_TRT and config.trt_runtime == TRT_RUNTIME_DOCKER:
        if config.use_persistent_trt_worker and config.docker_reuse_container:
            return run_faster_pipeline_docker_trt_persistent_worker(config, driving_input, raw_results_root)
        return run_faster_pipeline_docker_trt(config, driving_input, raw_results_root)
    return run_faster_pipeline_local(config, driving_input, raw_results_root)


def copy_public_outputs(
    output_dir: Path,
    driving_media: Path,
    result_org: Path,
    result_crop: Path,
) -> tuple[Path, Path, Path]:
    """
    Copy stable public files for browser playback.
    """
    driving_public = output_dir / resolve_driving_public_name(driving_media)
    result_public = output_dir / RESULT_PUBLIC_NAME
    result_concat_public = output_dir / RESULT_CONCAT_PUBLIC_NAME

    shutil.copy2(driving_media, driving_public)
    shutil.copy2(result_org, result_public)
    shutil.copy2(result_crop, result_concat_public)
    return driving_public, result_public, result_concat_public


def copy_template_if_present(
    run_output_dir: Path,
    output_template: Path,
    driving_input_name: str,
) -> bool:
    """
    Copy generated motion template from run output folder.
    """
    candidate = run_output_dir / f"{driving_input_name}.pkl"
    if not candidate.exists():
        return False
    shutil.copy2(candidate, output_template)
    return True


def write_report(
    config: RunnerConfig,
    source_fps: float,
    driving_fps: float,
    driving_input: Path,
    driving_public: Path,
    result_public: Path,
    result_concat_public: Path,
    run_output_dir: Path,
    template_used: bool,
    elapsed_seconds: float,
    phase_timings_seconds: dict[str, float],
) -> None:
    """
    Write execution report for browser viewer and debugging.
    """
    driving_public_viewer = to_viewer_path(driving_public, config.project_root)
    result_public_viewer = to_viewer_path(result_public, config.project_root)
    result_concat_public_viewer = to_viewer_path(result_concat_public, config.project_root)
    run_output_dir_viewer = to_viewer_path(run_output_dir, config.project_root)

    report = {
        "pipeline": "faster-liveportrait",
        "status": "ok",
        "backend": config.backend,
        "mode": config.mode,
        "templateUsed": template_used,
        "elapsedSeconds": round(elapsed_seconds, 3),
        "phaseTimingsSeconds": phase_timings_seconds,
        "sourceFps": source_fps,
        "drivingFps": driving_fps,
        "motionTargetFps": float(FIXED_AUDIO_MOTION_TARGET_FPS),
        "inputs": {
            "sourceFrame": str(config.source_frame),
            "sourceMediaType": "video" if config.source_frame.suffix.lower() in SOURCE_VIDEO_EXTENSIONS else "image",
            "framesDir": str(config.frames_dir),
            "metaPath": str(config.meta_path),
            "drivingInput": str(driving_input),
            "drivingMediaForViewer": driving_public_viewer,
            "drivingMediaType": "audio" if is_audio_file(driving_public) else "video",
            "drivingVideoForViewer": driving_public_viewer if not is_audio_file(driving_public) else "",
            "drivingAudioForViewer": driving_public_viewer if is_audio_file(driving_public) else "",
        },
        "outputs": {
            "resultVideo": result_public_viewer,
            "resultConcatVideo": result_concat_public_viewer,
            "rawResultDir": run_output_dir_viewer,
        },
        "runtimeConfig": asdict(config),
    }
    for key in (
        "project_root",
        "source_frame",
        "frames_dir",
        "meta_path",
        "output_dir",
        "faster_repo_dir",
        "python_executable",
        "audio_template_cache_dir",
        "source_cache_dir",
        "persistent_worker_queue_dir",
        "driving_audio",
        "stream_dir",
    ):
        value = report["runtimeConfig"][key]
        report["runtimeConfig"][key] = str(value) if value is not None else None

    report_path = config.output_dir / RUN_REPORT_NAME
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[ok] report -> {report_path}")


def main() -> None:
    """
    Execute full FasterLivePortrait runner.
    """
    started_at = time.time()
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_stream_dir(config)
    phase_started_at = time.time()

    assert_path_exists(config.source_frame, "Source frame")
    if config.driving_audio is None:
        assert_path_exists(config.frames_dir, "Frames directory")
    assert_path_exists(config.faster_repo_dir, "FasterLivePortrait repo")
    assert_path_exists(config.faster_repo_dir / "run.py", "FasterLivePortrait run.py")
    if not (config.backend == BACKEND_TRT and config.trt_runtime == TRT_RUNTIME_DOCKER):
        assert_path_exists(config.python_executable, "Python executable")

    if config.source_frame.suffix.lower() not in SOURCE_VIDEO_EXTENSIONS:
        assert_path_exists(config.meta_path, "Source metadata")
    source_fps = resolve_source_fps(config)
    driving_video, driving_template, _ = resolve_driving_paths(config)
    audio_template = resolve_audio_template_path(config)
    audio_template_meta = resolve_audio_template_meta_path(config)
    audio_template_input_wav = resolve_audio_template_input_wav_path(config)
    target_driving_fps = float(resolve_motion_stride_target_fps(source_fps, config.frame_step))
    print(
        "[info] backend={} trt_runtime={} trt_precision={} mode={} render_batch_size={} trt_engine_batch_size={} video_encoder={} source_fps={:.6f} target_driving_fps={:.6f}".format(
            config.backend,
            config.trt_runtime,
            config.trt_precision,
            config.mode,
            config.render_batch_size,
            config.trt_engine_batch_size,
            config.video_encoder,
            source_fps,
            target_driving_fps,
        )
    )
    print(
        "[info] motion-config audio_motion_stride={} fixed_audio_motion_stride={} effective_audio_motion_target_fps={:.6f}".format(
            config.audio_motion_stride,
            FIXED_AUDIO_MOTION_STRIDE,
            FIXED_AUDIO_MOTION_TARGET_FPS,
        )
    )

    driving_audio = config.driving_audio
    if driving_audio is not None:
        assert_path_exists(driving_audio, "Driving audio")
        print(f"[info] normalizing driving audio for progressive render: {driving_audio}")
        normalize_audio_for_joyvasa(driving_audio, audio_template_input_wav)
        assert_path_exists(audio_template_input_wav, "Normalized driving audio")
        if should_prebuild_audio_template(config):
            audio_signature = build_audio_signature(audio_template_input_wav)
            generation_profile = build_audio_template_generation_profile(config)
            template_used = (
                not config.rebuild_driving_template
                and is_audio_template_cache_hit(
                    audio_template,
                    audio_template_meta,
                    audio_signature,
                    generation_profile,
                )
            )
            if template_used:
                print(f"[info] using cached audio template: {audio_template}")
            else:
                if config.rebuild_driving_template and audio_template.exists():
                    print("[info] forcing audio template rebuild from normalized audio")
                build_driving_template_from_audio(config, audio_template_input_wav, audio_template)
                assert_path_exists(audio_template, "Audio motion template")
                write_audio_template_meta(
                    audio_template_meta,
                    build_audio_template_meta_payload(
                        audio_signature=audio_signature,
                        template_path=audio_template,
                        config=config,
                        normalized_audio_path=audio_template_input_wav,
                    ),
                )
                print(f"[ok] audio motion template -> {audio_template}")
            driving_input = audio_template
            driving_fps = read_template_fps(audio_template, source_fps)
        else:
            template_used = False
            driving_input = audio_template_input_wav
            driving_fps = float(resolve_motion_stride_target_fps(source_fps, config.audio_motion_stride))
        driving_media = driving_audio
    else:
        if config.skip_driving_video_build and driving_video.exists():
            print(f"[info] reusing driving video: {driving_video}")
            driving_fps = target_driving_fps
        elif config.skip_driving_video_build:
            raise RuntimeError(f"Driving video missing while skip flag is set: {driving_video}")
        else:
            driving_fps = build_driving_video(config, driving_video, source_fps)

        template_used = driving_template.exists() and not config.rebuild_driving_template
        if template_used:
            driving_input = driving_template
            print(f"[info] using cached template: {driving_input}")
        else:
            driving_input = driving_video
            if config.rebuild_driving_template and driving_template.exists():
                print("[info] forcing template rebuild from driving video")
        driving_media = driving_video
    phase_prepare_inputs_seconds = time.time() - phase_started_at

    if config.backend == BACKEND_TRT and not config.skip_trt_engine_build:
        ensure_trt_engines(config)

    raw_results_root = config.faster_repo_dir / "results"
    raw_results_root.mkdir(parents=True, exist_ok=True)
    phase_inference_started_at = time.time()
    run_output_dir, result_org, result_crop = run_faster_pipeline(config, driving_input, raw_results_root)
    phase_inference_seconds = time.time() - phase_inference_started_at

    phase_postprocess_started_at = time.time()
    if driving_audio is not None and driving_input.suffix.lower() != ".pkl":
        exported_audio_template = copy_template_if_present(run_output_dir, audio_template, driving_input.name)
        if exported_audio_template:
            driving_fps = read_template_fps(audio_template, driving_fps)
            print(f"[ok] exported audio motion template -> {audio_template}")
    if driving_input.suffix.lower() == ".mp4":
        template_written = copy_template_if_present(run_output_dir, driving_template, driving_input.name)
        if template_written:
            print(f"[ok] cached template -> {driving_template}")
    if is_deferred_paste_back_enabled(config):
        preview_composition_meta_path = run_output_dir / PREVIEW_COMPOSITION_META_NAME
        if preview_composition_meta_path.exists():
            result_org = compose_deferred_paste_back_video(config, run_output_dir, result_crop)
        else:
            print(
                f"[warn] deferred paste-back metadata missing; reusing original render: {preview_composition_meta_path}"
            )
    if driving_audio is not None:
        result_org = mux_audio_into_video(result_org, driving_audio, config.video_encoder)
        result_crop = mux_audio_into_video(result_crop, driving_audio, config.video_encoder)
    result_org = ensure_browser_compatible_video(result_org, config.video_encoder)
    result_crop = ensure_browser_compatible_video(result_crop, config.video_encoder)

    driving_public, result_public, result_concat_public = copy_public_outputs(
        output_dir=config.output_dir,
        driving_media=driving_media,
        result_org=result_org,
        result_crop=result_crop,
    )
    phase_postprocess_seconds = time.time() - phase_postprocess_started_at
    elapsed_seconds = time.time() - started_at
    phase_timings = {
        "prepareInputs": round(phase_prepare_inputs_seconds, 3),
        "inference": round(phase_inference_seconds, 3),
        "postprocess": round(phase_postprocess_seconds, 3),
    }
    write_report(
        config=config,
        source_fps=source_fps,
        driving_fps=driving_fps,
        driving_input=driving_input,
        driving_public=driving_public,
        result_public=result_public,
        result_concat_public=result_concat_public,
        run_output_dir=run_output_dir,
        template_used=template_used,
        elapsed_seconds=elapsed_seconds,
        phase_timings_seconds=phase_timings,
    )

    print(f"[ok] result -> {result_public}")
    print(f"[ok] concat -> {result_concat_public}")


if __name__ == "__main__":
    main()
