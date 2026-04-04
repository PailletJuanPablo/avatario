"""
API server for real-time FasterLivePortrait generation with WebSocket status and video streaming.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from collections import deque
import contextlib
from fractions import Fraction
import hmac
import json
import math
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime_env import apply_runtime_library_environment

import av
from av import AudioFrame, VideoFrame
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np

from job_stream_shared_memory import (
    JobStreamSharedMemoryReader,
    build_job_stream_shm_prefix,
)

PROJECT_ROOT = Path(__file__).resolve().parent
apply_runtime_library_environment(PROJECT_ROOT)
RUNNER_SCRIPT = PROJECT_ROOT / "faster_liveportrait_runner.py"
RUNNER_PYTHON = PROJECT_ROOT / ".venv-liveportrait" / "Scripts" / "python.exe"
if not RUNNER_PYTHON.exists():
    RUNNER_PYTHON = Path(sys.executable)


def resolve_media_tool_binary(tool_name: str) -> str:
    """
    Resolve the real media tool binary instead of the Chocolatey shim when possible.
    """
    binary_path_str = shutil.which(tool_name)
    if not binary_path_str:
        return tool_name
    binary_path = Path(binary_path_str)
    normalized_path = str(binary_path).replace("\\", "/").lower()
    if "/chocolatey/bin/" in normalized_path:
        chocolatey_root = binary_path.parent.parent
        real_binary_path = chocolatey_root / "lib" / "ffmpeg" / "tools" / "ffmpeg" / "bin" / binary_path.name
        if real_binary_path.exists():
            return str(real_binary_path.resolve())
    return str(binary_path.resolve())


def format_command_for_log(command: list[str]) -> str:
    """
    Render one subprocess command for human-readable logs.
    """
    return subprocess.list2cmdline([str(token) for token in command])


def emit_runtime_log(label: str, message: str) -> None:
    """
    Emit one flushed runtime log line.
    """
    print(f"[{label}] {message}", flush=True)


def pump_subprocess_output(
    stream: Any,
    log_handle: Any,
    label: str,
) -> None:
    """
    Mirror one subprocess combined stdout stream into file storage and terminal output.
    """
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
            text = str(line).rstrip("\r\n")
            if text:
                emit_runtime_log(label, text)
    finally:
        with contextlib.suppress(Exception):
            stream.close()


def read_text_file_tail(path_value: Path, maximum_chars: int = 2000) -> str:
    """
    Read a short tail slice from one text file for error reporting.
    """
    try:
        text_value = path_value.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if maximum_chars <= 0:
        return text_value
    return text_value[-maximum_chars:]


def terminate_process_tree(process: asyncio.subprocess.Process | None) -> None:
    """
    Terminate one subprocess and its child process tree when available.
    """
    if process is None or process.returncode is not None:
        return
    if os.name == "nt":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=5,
            )
            return
    with contextlib.suppress(Exception):
        process.kill()


FFMPEG_BINARY = resolve_media_tool_binary("ffmpeg")
FFPROBE_BINARY = resolve_media_tool_binary("ffprobe")


def ffmpeg_supports_encoder(encoder_name: str) -> bool:
    """
    Check whether the configured FFmpeg runtime can actually initialize one specific encoder.
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
                FFMPEG_BINARY,
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
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        FFMPEG_ENCODER_SUPPORT_CACHE[safe_encoder_name] = False
        return False
    support_detected = completed.returncode == 0
    FFMPEG_ENCODER_SUPPORT_CACHE[safe_encoder_name] = support_detected
    return support_detected


def read_env_int(
    env_key: str,
    default_value: int,
    minimum_value: int,
    maximum_value: int | None = None,
) -> int:
    """
    Read one integer environment override with bounded fallback.
    """
    raw_value = os.getenv(env_key, str(default_value)).strip()
    try:
        parsed_value = int(raw_value or str(default_value))
    except ValueError:
        parsed_value = int(default_value)
    bounded_value = max(int(minimum_value), parsed_value)
    if maximum_value is not None:
        bounded_value = min(int(maximum_value), bounded_value)
    return bounded_value


def read_env_float(
    env_key: str,
    default_value: float,
    minimum_value: float,
    maximum_value: float | None = None,
) -> float:
    """
    Read one float environment override with bounded fallback.
    """
    raw_value = os.getenv(env_key, str(default_value)).strip()
    try:
        parsed_value = float(raw_value or str(default_value))
    except ValueError:
        parsed_value = float(default_value)
    bounded_value = max(float(minimum_value), parsed_value)
    if maximum_value is not None:
        bounded_value = min(float(maximum_value), bounded_value)
    return bounded_value


def resolve_stream_video_encoder_name() -> str:
    """
    Resolve the concrete FFmpeg encoder name used by websocket/avatar stream encoders.
    """
    if DEFAULT_VIDEO_ENCODER == VIDEO_ENCODER_CPU:
        return FFMPEG_LIBX264
    if ffmpeg_supports_encoder(FFMPEG_H264_NVENC):
        return FFMPEG_H264_NVENC
    if DEFAULT_VIDEO_ENCODER == VIDEO_ENCODER_NVENC:
        print("[warn] requested NVENC stream encoder is unavailable; falling back to libx264")
    return FFMPEG_LIBX264


def build_stream_video_codec_args() -> list[str]:
    """
    Build encoder-specific FFmpeg arguments for fragmented MP4 stream outputs.
    """
    codec_name = resolve_stream_video_encoder_name()
    if codec_name == FFMPEG_H264_NVENC:
        return [
            "-c:v",
            codec_name,
            "-preset",
            "llhp",
            "-rc",
            "cbr",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-zerolatency",
            "1",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-g",
            VIDEO_STREAM_GOP,
            "-keyint_min",
            VIDEO_STREAM_KEYINT_MIN,
            "-b:v",
            VIDEO_STREAM_BITRATE,
            "-maxrate",
            VIDEO_STREAM_MAXRATE,
            "-bufsize",
            VIDEO_STREAM_BUFSIZE,
        ]
    return [
        "-c:v",
        codec_name,
        "-preset",
        VIDEO_STREAM_X264_PRESET,
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-bf",
        "0",
        "-refs",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-g",
        VIDEO_STREAM_GOP,
        "-keyint_min",
        VIDEO_STREAM_KEYINT_MIN,
        "-sc_threshold",
        "0",
        "-b:v",
        VIDEO_STREAM_BITRATE,
        "-maxrate",
        VIDEO_STREAM_MAXRATE,
        "-bufsize",
        VIDEO_STREAM_BUFSIZE,
    ]

DEFAULT_SOURCE_FRAME = "output/frames/frame_00061.png"
FIXED_SOURCE_FRAME_ENV_KEY = "ANIMATION_FIXED_SOURCE_FRAME"
FIXED_SOURCE_FRAME = os.getenv(FIXED_SOURCE_FRAME_ENV_KEY, "").strip()
DEFAULT_MODE = "preview"
BACKEND_TRT = "trt"
BACKEND_ONNX = "onnx"
TRT_RUNTIME_DOCKER = "docker"
TRT_RUNTIME_LOCAL = "local"
BACKEND_CHOICES = {BACKEND_TRT, BACKEND_ONNX}
TRT_RUNTIME_CHOICES = {TRT_RUNTIME_DOCKER, TRT_RUNTIME_LOCAL}
ANIMATION_REGION_CHOICES = {"all", "exp", "lip", "eyes", "pose"}

DEFAULT_BACKEND = os.getenv("ANIMATION_BACKEND", BACKEND_TRT).strip().lower() or BACKEND_TRT
if DEFAULT_BACKEND not in BACKEND_CHOICES:
    DEFAULT_BACKEND = BACKEND_TRT

DEFAULT_TRT_RUNTIME = os.getenv("ANIMATION_TRT_RUNTIME", TRT_RUNTIME_DOCKER).strip().lower() or TRT_RUNTIME_DOCKER
if DEFAULT_TRT_RUNTIME not in TRT_RUNTIME_CHOICES:
    DEFAULT_TRT_RUNTIME = TRT_RUNTIME_DOCKER

DEFAULT_TRT_PRECISION = os.getenv("ANIMATION_TRT_PRECISION", "fp16").strip().lower() or "fp16"
DEFAULT_DOCKER_GPU_DEVICE = os.getenv("ANIMATION_DOCKER_GPU_DEVICE", "").strip()
DEFAULT_SKIP_TRT_ENGINE_BUILD = (
    os.getenv("ANIMATION_SKIP_TRT_ENGINE_BUILD", "0").strip().lower() in {"1", "true", "yes"}
)
DEFAULT_AUDIO_MOTION_STRIDE_ENV_KEY = "ANIMATION_AUDIO_MOTION_STRIDE"
DEFAULT_AUDIO_MOTION_STRIDE_VALUE = 2
DEFAULT_AUDIO_MOTION_STRIDE = read_env_int(
    DEFAULT_AUDIO_MOTION_STRIDE_ENV_KEY,
    DEFAULT_AUDIO_MOTION_STRIDE_VALUE,
    1,
    6,
)
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
DEFAULT_AUDIO_LIP_SYNC_ATTACK_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_ATTACK"
DEFAULT_AUDIO_LIP_SYNC_RELEASE_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_RELEASE"
DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS_ENV_KEY = "ANIMATION_AUDIO_LIP_SYNC_OFFSET_MS"
DEFAULT_AUDIO_MOUTH_FLOOR_STRENGTH_ENV_KEY = "ANIMATION_AUDIO_MOUTH_FLOOR_STRENGTH"
DEFAULT_AUDIO_MOUTH_PEAK_CLAMP_ENV_KEY = "ANIMATION_AUDIO_MOUTH_PEAK_CLAMP"
DEFAULT_AUDIO_EYE_TAMED_PRESET = (
    os.getenv(DEFAULT_AUDIO_EYE_TAMED_PRESET_ENV_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
)
DEFAULT_AUDIO_EYE_SOFT_FACTOR = read_env_float(
    DEFAULT_AUDIO_EYE_SOFT_FACTOR_ENV_KEY,
    0.45,
    0.0,
    1.0,
)
DEFAULT_AUDIO_EYE_HARD_FACTOR = read_env_float(
    DEFAULT_AUDIO_EYE_HARD_FACTOR_ENV_KEY,
    0.18,
    0.0,
    1.0,
)
DEFAULT_AUDIO_EYE_HARD_DY_MIN = read_env_float(
    DEFAULT_AUDIO_EYE_HARD_DY_MIN_ENV_KEY,
    -0.0045,
    -1.0,
    1.0,
)
DEFAULT_AUDIO_EYE_HARD_DY_MAX = read_env_float(
    DEFAULT_AUDIO_EYE_HARD_DY_MAX_ENV_KEY,
    0.0035,
    -1.0,
    1.0,
)
DEFAULT_AUDIO_MOTION_TUNING_ENABLED = (
    os.getenv(DEFAULT_AUDIO_MOTION_TUNING_ENABLED_ENV_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
)
DEFAULT_AUDIO_REANCHOR_FIRST_N = read_env_int(DEFAULT_AUDIO_REANCHOR_FIRST_N_ENV_KEY, 5, 1, 15)
DEFAULT_AUDIO_MOUTH_OPEN_FACTOR = read_env_float(
    DEFAULT_AUDIO_MOUTH_OPEN_FACTOR_ENV_KEY,
    1.18,
    0.0,
    3.0,
)
DEFAULT_AUDIO_POSE_SMOOTH_WINDOW = read_env_int(DEFAULT_AUDIO_POSE_SMOOTH_WINDOW_ENV_KEY, 5, 0, 21)
DEFAULT_AUDIO_EXP_SMOOTH_WINDOW = read_env_int(DEFAULT_AUDIO_EXP_SMOOTH_WINDOW_ENV_KEY, 3, 0, 21)
DEFAULT_AUDIO_POSE_JUMP_THRESHOLD = read_env_float(
    DEFAULT_AUDIO_POSE_JUMP_THRESHOLD_ENV_KEY,
    8.0,
    0.0,
    60.0,
)
DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD = read_env_float(
    DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD_ENV_KEY,
    0.03,
    0.0,
    1.0,
)
DEFAULT_AUDIO_LIP_SYNC_ASSIST = (
    os.getenv(DEFAULT_AUDIO_LIP_SYNC_ASSIST_ENV_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
)
DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO = read_env_float(DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO_ENV_KEY, 0.03, 0.0, 1.0)
DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO = read_env_float(DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO_ENV_KEY, 0.32, 0.0, 1.0)
DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW = read_env_int(DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW_ENV_KEY, 5, 0, 21)
DEFAULT_AUDIO_LIP_SYNC_STRENGTH = read_env_float(DEFAULT_AUDIO_LIP_SYNC_STRENGTH_ENV_KEY, 1.15, 0.0, 4.0)
DEFAULT_AUDIO_LIP_SYNC_POWER = read_env_float(DEFAULT_AUDIO_LIP_SYNC_POWER_ENV_KEY, 0.85, 0.001, 4.0)
DEFAULT_AUDIO_LIP_SYNC_ATTACK = read_env_float(DEFAULT_AUDIO_LIP_SYNC_ATTACK_ENV_KEY, 1.0, 0.0, 1.0)
DEFAULT_AUDIO_LIP_SYNC_RELEASE = read_env_float(DEFAULT_AUDIO_LIP_SYNC_RELEASE_ENV_KEY, 1.0, 0.0, 1.0)
DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS = read_env_int(DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS_ENV_KEY, 0, -1000, 1000)
DEFAULT_AUDIO_MOUTH_FLOOR_STRENGTH = read_env_float(DEFAULT_AUDIO_MOUTH_FLOOR_STRENGTH_ENV_KEY, 0.26, 0.0, 4.0)
DEFAULT_AUDIO_MOUTH_PEAK_CLAMP = read_env_float(DEFAULT_AUDIO_MOUTH_PEAK_CLAMP_ENV_KEY, 0.0, 0.0, 4.0)
DEFAULT_DRIVING_MULTIPLIER = read_env_float(
    "ANIMATION_DRIVING_MULTIPLIER",
    1.0,
    0.0,
    2.0,
)
DEFAULT_CFG_SCALE = read_env_float(
    "ANIMATION_CFG_SCALE",
    1.2,
    0.0,
    10.0,
)
DEFAULT_JOYVASA_INFERENCE_STEPS = read_env_int(
    "ANIMATION_JOYVASA_INFERENCE_STEPS",
    15,
    1,
    100,
)
if DEFAULT_AUDIO_EYE_HARD_DY_MIN > DEFAULT_AUDIO_EYE_HARD_DY_MAX:
    (
        DEFAULT_AUDIO_EYE_HARD_DY_MIN,
        DEFAULT_AUDIO_EYE_HARD_DY_MAX,
    ) = (
        DEFAULT_AUDIO_EYE_HARD_DY_MAX,
        DEFAULT_AUDIO_EYE_HARD_DY_MIN,
    )
if DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO > DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO:
    (
        DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO,
        DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO,
    ) = (
        DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO,
        DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO,
    )
GENERATION_FRAME_COUNT_MIN = 1
GENERATION_FRAME_COUNT_MAX = 1200
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
VIDEO_ENCODER_AUTO = "auto"
VIDEO_ENCODER_NVENC = "nvenc"
VIDEO_ENCODER_CPU = "cpu"
VIDEO_ENCODER_CHOICES = {
    VIDEO_ENCODER_AUTO,
    VIDEO_ENCODER_NVENC,
    VIDEO_ENCODER_CPU,
}
FFMPEG_H264_NVENC = "h264_nvenc"
FFMPEG_LIBX264 = "libx264"
DEFAULT_VIDEO_ENCODER = os.getenv("ANIMATION_VIDEO_ENCODER", VIDEO_ENCODER_AUTO).strip().lower() or VIDEO_ENCODER_AUTO
if DEFAULT_VIDEO_ENCODER not in VIDEO_ENCODER_CHOICES:
    DEFAULT_VIDEO_ENCODER = VIDEO_ENCODER_AUTO
FFMPEG_ENCODER_SUPPORT_CACHE: dict[str, bool] = {}
DEFAULT_ANIMATION_REGION = os.getenv("ANIMATION_ANIMATION_REGION", "all").strip().lower() or "all"
if DEFAULT_ANIMATION_REGION not in ANIMATION_REGION_CHOICES:
    DEFAULT_ANIMATION_REGION = "all"
DEFAULT_STITCHING_ENABLED = os.getenv("ANIMATION_STITCHING_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_RELATIVE_MOTION_ENABLED = (
    os.getenv("ANIMATION_RELATIVE_MOTION_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
)
DEFAULT_PASTE_BACK_ENABLED = os.getenv("ANIMATION_PASTE_BACK_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_API_HOST = os.getenv("ANIMATION_API_HOST", "0.0.0.0").strip() or "0.0.0.0"
DEFAULT_API_PORT = max(1, int(os.getenv("ANIMATION_API_PORT", "8010").strip() or "8010"))
CURRENT_API_HOST = DEFAULT_API_HOST
CURRENT_API_PORT = DEFAULT_API_PORT
API_TOKEN_ENV_KEY = "ANIMATION_API_TOKEN"
API_TOKEN_QUERY_KEY = "token"
AUTHORIZATION_HEADER_NAME = "authorization"
AUTHORIZATION_HEADER_VALUE = "Bearer"
WEBSOCKET_UNAUTHORIZED_CLOSE_CODE = 4401
AUTH_REQUIRED_HTTP_PATH_PREFIXES = ("/api/", "/jobs/")
AUTH_FAILURE_MESSAGE = "Invalid or missing API token."
DEFAULT_API_TOKEN = os.getenv(API_TOKEN_ENV_KEY, "").strip()
API_TOKEN_ENABLED = bool(DEFAULT_API_TOKEN)
JOB_POLL_SLEEP_SEC = 0.12
VIDEO_STREAM_POLL_SLEEP_SEC = 0.02
VIDEO_STREAM_INPUT_FPS = 20.0
VIDEO_STREAM_CHUNK_SIZE = 16384
VIDEO_STREAM_BITRATE = "1500k"
VIDEO_STREAM_MAXRATE = "1800k"
VIDEO_STREAM_BUFSIZE = "3000k"
VIDEO_STREAM_GOP = "12"
VIDEO_STREAM_KEYINT_MIN = "12"
VIDEO_STREAM_X264_PRESET = "ultrafast"
VIDEO_STREAM_AUDIO_CODEC = "aac"
VIDEO_STREAM_AUDIO_BITRATE = "128k"
VIDEO_STREAM_AUDIO_SAMPLE_RATE = "48000"
VIDEO_STREAM_AUDIO_CHANNELS = "2"
VIDEO_STREAM_AUDIO_FILTER = "aresample=async=1:first_pts=0"
VIDEO_STREAM_AUDIO_SAMPLE_WIDTH_BYTES = 2
VIDEO_STREAM_AUDIO_SAMPLE_RATE_INT = int(VIDEO_STREAM_AUDIO_SAMPLE_RATE)
VIDEO_STREAM_AUDIO_CHANNELS_INT = int(VIDEO_STREAM_AUDIO_CHANNELS)
VIDEO_STREAM_AUDIO_CHUNK_SAMPLES = 960
VIDEO_STREAM_AUDIO_CHUNK_BYTES = (
    VIDEO_STREAM_AUDIO_CHUNK_SAMPLES
    * VIDEO_STREAM_AUDIO_CHANNELS_INT
    * VIDEO_STREAM_AUDIO_SAMPLE_WIDTH_BYTES
)
VIDEO_STREAM_MUX_DELAY = "0"
VIDEO_STREAM_MUX_PRELOAD = "0"
AVATAR_STREAM_MUX_MOVFLAGS = "+frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset"
MP4_BOX_HEADER_SIZE = 8
MP4_BOX_EXTENDED_SIZE = 1
MP4_BOX_TYPE_MOOF = b"moof"
VIDEO_STREAM_TERMINAL_STABLE_LOOPS = 24
VIDEO_STREAM_MAX_BACKLOG_FRAMES = 10
VIDEO_STREAM_TARGET_LATENCY_FRAMES = 2
VIDEO_STREAM_SERVER_BUFFER_FRAMES = 6
VIDEO_STREAM_REALTIME_TARGET_DELAY_SEC = 0.18
VIDEO_STREAM_INTERPOLATION_MAX_STEPS = 1
VIDEO_STREAM_TARGET_FPS_ENV_KEY = "ANIMATION_STREAM_TARGET_FPS"
VIDEO_STREAM_INTERPOLATION_TARGET_FPS = read_env_float(
    VIDEO_STREAM_TARGET_FPS_ENV_KEY,
    20.0,
    8.0,
    25.0,
)
VIDEO_STREAM_JPEG_QUALITY_ENV_KEY = "ANIMATION_STREAM_JPEG_QUALITY"
VIDEO_STREAM_INTERPOLATION_ALPHA_QUALITY = read_env_int(
    VIDEO_STREAM_JPEG_QUALITY_ENV_KEY,
    88,
    40,
    100,
)
VIDEO_STREAM_AUDIO_SAMPLES_PER_VIDEO_FRAME = VIDEO_STREAM_AUDIO_SAMPLE_RATE_INT // int(VIDEO_STREAM_INTERPOLATION_TARGET_FPS)
VIDEO_STREAM_AUDIO_BYTES_PER_VIDEO_FRAME = (
    VIDEO_STREAM_AUDIO_SAMPLES_PER_VIDEO_FRAME
    * VIDEO_STREAM_AUDIO_CHANNELS_INT
    * VIDEO_STREAM_AUDIO_SAMPLE_WIDTH_BYTES
)
VIDEO_STREAM_GENERATION_FPS_MIN = 6.0
VIDEO_STREAM_GENERATION_FPS_MAX = 25.0
VIDEO_STREAM_GENERATION_FPS_SMOOTH_ALPHA = 0.18
AVATAR_STREAM_ENCODER_DRAIN_TIMEOUT_SEC = 1.0
AVATAR_STREAM_ENCODER_EXIT_TIMEOUT_SEC = 0.35
AVATAR_STREAM_HTTP_QUEUE_MAX_CHUNKS = 96
AVATAR_CAPTURE_ROOT_REL = Path("output_fasterliveportrait/avatar_capture")
AVATAR_CAPTURE_ROOT = PROJECT_ROOT / AVATAR_CAPTURE_ROOT_REL
AVATAR_CAPTURE_DEFAULT_DURATION_SEC = 12.0
AVATAR_CAPTURE_MIN_DURATION_SEC = 2.0
AVATAR_CAPTURE_MAX_DURATION_SEC = 30.0
AVATAR_VIDEO_OUTPUT_FPS = VIDEO_STREAM_INTERPOLATION_TARGET_FPS
AVATAR_STREAM_MAX_WIDTH_ENV_KEY = "ANIMATION_AVATAR_STREAM_MAX_WIDTH"
AVATAR_STREAM_MAX_HEIGHT_ENV_KEY = "ANIMATION_AVATAR_STREAM_MAX_HEIGHT"
AVATAR_STREAM_OUTPUT_MAX_WIDTH = read_env_int(
    AVATAR_STREAM_MAX_WIDTH_ENV_KEY,
    576,
    256,
)
AVATAR_STREAM_OUTPUT_MAX_HEIGHT = read_env_int(
    AVATAR_STREAM_MAX_HEIGHT_ENV_KEY,
    324,
    256,
)
AVATAR_VIDEO_FALLBACK_WIDTH = AVATAR_STREAM_OUTPUT_MAX_WIDTH
AVATAR_VIDEO_FALLBACK_HEIGHT = AVATAR_STREAM_OUTPUT_MAX_HEIGHT
AVATAR_FALLBACK_BOUNCE_WINDOW_FRAMES = 3
JOB_STREAM_CACHE_RETENTION_MS = 3 * 60 * 1000
VIDEO_STREAM_START_MODE_QUERY_KEY = "startMode"
VIDEO_STREAM_START_MODE_BUFFERED = "buffered"
VIDEO_STREAM_START_MODE_LIVE = "live"
VIDEO_STREAM_START_PROGRESS_QUERY_KEY = "startProgress"
VIDEO_STREAM_BUFFERED_START_PROGRESS_DEFAULT = 0.35
VIDEO_STREAM_START_MODE_CHOICES = {
    VIDEO_STREAM_START_MODE_BUFFERED,
    VIDEO_STREAM_START_MODE_LIVE,
}
VIDEO_STREAM_PLAYBACK_FPS_STATUS_KEY = "fps"
STREAM_BOUNDARY = "frame"
PREVIEW_COMPOSITION_STATUS_KEY = "previewComposition"
PREVIEW_COMPOSITION_MASK_NAME = "preview_composition_mask.png"
RUN_LOG_FILE_NAME = "run.log"
RUN_REPORT_FILE_NAME = "run_report.json"
RUNNER_STATUS_FILE_NAME = "runner_status.json"
MAX_LOG_LINES = 400
JOBS_ROOT_REL = Path("output_fasterliveportrait/jobs")
JOBS_ROOT = PROJECT_ROOT / JOBS_ROOT_REL
SOURCE_TEMPLATE_PACKS_REL = Path("output_fasterliveportrait/source_template_packs")
SOURCE_TEMPLATE_PACKS_ROOT = PROJECT_ROOT / SOURCE_TEMPLATE_PACKS_REL
SOURCE_TEMPLATE_PACK_META_SUFFIX = ".json"
SOURCE_TEMPLATE_PACK_PREVIEW_SUFFIX = ".preview.png"
SOURCE_TEMPLATE_PACK_CLOSE_MOUTH_PRESET = "close_mouth_in_silence"
SOURCE_TEMPLATE_PACK_BUILD_LOGS_DIR = SOURCE_TEMPLATE_PACKS_ROOT / "_build_logs"
SOURCE_TEMPLATE_PACK_BUILD_WAIT_TIMEOUT_SEC = 60.0 * 60.0
RUNTIME_LOG_TARGET_CONTAINER = "container"
RUNTIME_LOG_TARGET_WORKER = "worker"
RUNTIME_LOG_TARGETS = {RUNTIME_LOG_TARGET_CONTAINER, RUNTIME_LOG_TARGET_WORKER}
CONTAINER_LOG_REL = Path("output_fasterliveportrait/docker_api.log")
CONTAINER_LOG_ABS = PROJECT_ROOT / CONTAINER_LOG_REL
PERSISTENT_WORKER_QUEUE_ROOT_REL = Path("output_fasterliveportrait/worker_queue")
PERSISTENT_WORKER_LOG_REL = PERSISTENT_WORKER_QUEUE_ROOT_REL / "worker.log"
PERSISTENT_WORKER_LOG_ABS = PROJECT_ROOT / PERSISTENT_WORKER_LOG_REL
RUNTIME_RESTART_DELAY_SEC = 1.0
PROCESS_STARTED_AT_MS = int(time.time() * 1000)
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
ALLOWED_SOURCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_SOURCE_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv"}
WARMUP_OUTPUT_ROOT_REL = Path("output_fasterliveportrait/warmup")
WARMUP_STREAM_SUBDIR_NAME = "stream"
WARMUP_INPUTS_SUBDIR_NAME = "inputs"
WARMUP_AUDIO_FILE_NAME = "warmup.wav"
WARMUP_AUDIO_DURATION_SEC = 0.8
WARMUP_START_DELAY_SEC = 0.75
WARMUP_STREAM_SHM_PREFIX = build_job_stream_shm_prefix("__warmup__")
WARMUP_ENABLED = os.getenv("ANIMATION_WARMUP_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
AVATAR_MODE_IDLE = "idle"
AVATAR_MODE_TALKING = "talking"
AVATAR_MODE_CHOICES = {AVATAR_MODE_IDLE, AVATAR_MODE_TALKING}
AVATAR_TRANSPORT_WEBSOCKET = "websocket"
AVATAR_TRANSPORT_WEBRTC = "webrtc"
AVATAR_SESSION_ID_HEADER_NAME = "X-Avatar-Session-Id"
AVATAR_SESSION_ID_QUERY_KEY = "sessionId"
AVATAR_SESSION_ID_MAX_LENGTH = 128
AVATAR_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
AVATAR_STREAM_SEGMENT_IDLE_KEY = "__idle__"
DEFAULT_IDLE_VIDEO_PATH = "inputs/idlevid_breath.mp4"
AVATAR_IDLE_VIDEO_REL = Path(
    os.getenv("ANIMATION_IDLE_VIDEO", DEFAULT_IDLE_VIDEO_PATH).strip() or DEFAULT_IDLE_VIDEO_PATH
)
AVATAR_IDLE_SOURCE_FRAME_REL = Path("output_fasterliveportrait/avatar_idle_source.png")
AVATAR_IDLE_SOURCE_ANCHOR_ROOT_REL = Path("output_fasterliveportrait/avatar_idle_sources")
AVATAR_IDLE_SOURCE_ANCHOR_MANIFEST_NAME = "anchors.json"
AVATAR_IDLE_SOURCE_ANCHOR_COUNT = 12
AVATAR_IDLE_MIN_HOLD_SEC = 0.35
AVATAR_RETURN_TO_IDLE_ENABLED = (
    os.getenv("ANIMATION_AVATAR_RETURN_TO_IDLE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
)
AVATAR_RETURN_TO_IDLE_DURATION_SEC_ENV_KEY = "ANIMATION_AVATAR_RETURN_TO_IDLE_DURATION_SEC"
AVATAR_RETURN_TO_IDLE_DURATION_SEC = read_env_float(
    AVATAR_RETURN_TO_IDLE_DURATION_SEC_ENV_KEY,
    0.24,
    0.0,
    2.0,
)
AVATAR_RETURN_TO_IDLE_MIN_FRAME_COUNT = 2
AVATAR_RETURN_TO_IDLE_MAX_FRAME_COUNT_ENV_KEY = "ANIMATION_AVATAR_RETURN_TO_IDLE_MAX_FRAME_COUNT"
AVATAR_RETURN_TO_IDLE_MAX_FRAME_COUNT = read_env_int(
    AVATAR_RETURN_TO_IDLE_MAX_FRAME_COUNT_ENV_KEY,
    6,
    AVATAR_RETURN_TO_IDLE_MIN_FRAME_COUNT,
    24,
)
AVATAR_IDLE_TO_TALKING_ENABLED = (
    os.getenv("ANIMATION_AVATAR_IDLE_TO_TALKING_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
)
AVATAR_IDLE_TO_TALKING_DURATION_SEC_ENV_KEY = "ANIMATION_AVATAR_IDLE_TO_TALKING_DURATION_SEC"
AVATAR_IDLE_TO_TALKING_DURATION_SEC = read_env_float(
    AVATAR_IDLE_TO_TALKING_DURATION_SEC_ENV_KEY,
    AVATAR_RETURN_TO_IDLE_DURATION_SEC,
    0.0,
    2.0,
)
AVATAR_IDLE_TO_TALKING_MIN_FRAME_COUNT = 2
AVATAR_IDLE_TO_TALKING_MAX_FRAME_COUNT_ENV_KEY = "ANIMATION_AVATAR_IDLE_TO_TALKING_MAX_FRAME_COUNT"
AVATAR_IDLE_TO_TALKING_MAX_FRAME_COUNT = read_env_int(
    AVATAR_IDLE_TO_TALKING_MAX_FRAME_COUNT_ENV_KEY,
    AVATAR_RETURN_TO_IDLE_MAX_FRAME_COUNT,
    AVATAR_IDLE_TO_TALKING_MIN_FRAME_COUNT,
    24,
)
AVATAR_RETURN_TO_IDLE_FLOW_PYR_SCALE = 0.5
AVATAR_RETURN_TO_IDLE_FLOW_LEVELS = 3
AVATAR_RETURN_TO_IDLE_FLOW_WINDOW_SIZE = 21
AVATAR_RETURN_TO_IDLE_FLOW_ITERATIONS = 5
AVATAR_RETURN_TO_IDLE_FLOW_POLY_N = 7
AVATAR_RETURN_TO_IDLE_FLOW_POLY_SIGMA = 1.5
AVATAR_RETURN_TO_IDLE_FLOW_CONSISTENCY_THRESHOLD = 1.5
AVATAR_RETURN_TO_IDLE_FLOW_CONSISTENCY_SCALE = 0.05
AVATAR_RETURN_TO_IDLE_FLOW_MASK_BLUR_KERNEL = 5
AVATAR_RETURN_TO_IDLE_FLOW_WEIGHT_EPSILON = 1e-6
AVATAR_READY_BUFFER_MIN_SEC_ENV_KEY = "ANIMATION_AVATAR_READY_BUFFER_MIN_SEC"
AVATAR_READY_BUFFER_MIN_SEC = read_env_float(
    AVATAR_READY_BUFFER_MIN_SEC_ENV_KEY,
    0.55,
    0.05,
    5.0,
)
AVATAR_READY_DYNAMIC_MARGIN_SEC_ENV_KEY = "ANIMATION_AVATAR_READY_DYNAMIC_MARGIN_SEC"
AVATAR_READY_DYNAMIC_MARGIN_SEC = read_env_float(
    AVATAR_READY_DYNAMIC_MARGIN_SEC_ENV_KEY,
    0.12,
    0.0,
    2.0,
)
AVATAR_READY_BUFFER_MAX_PROGRESS_ENV_KEY = "ANIMATION_AVATAR_READY_BUFFER_MAX_PROGRESS"
AVATAR_READY_BUFFER_MAX_PROGRESS = read_env_float(
    AVATAR_READY_BUFFER_MAX_PROGRESS_ENV_KEY,
    VIDEO_STREAM_BUFFERED_START_PROGRESS_DEFAULT,
    0.05,
    1.0,
)
AVATAR_STATE_POLL_SLEEP_SEC = 0.1
WEBRTC_OFFER_API_PATH = "/api/webrtc/offer"
WEBRTC_SESSION_POLL_SLEEP_SEC = 0.15
WEBRTC_ICE_SERVERS_ENV_KEY = "ANIMATION_WEBRTC_ICE_SERVERS_JSON"
WEBRTC_ICE_TRANSPORT_POLICY_ENV_KEY = "ANIMATION_WEBRTC_ICE_TRANSPORT_POLICY"
WEBRTC_ICE_TRANSPORT_POLICY_ALL = "all"
WEBRTC_ICE_TRANSPORT_POLICY_RELAY = "relay"
WEBRTC_ICE_TRANSPORT_POLICY_CHOICES = {
    WEBRTC_ICE_TRANSPORT_POLICY_ALL,
    WEBRTC_ICE_TRANSPORT_POLICY_RELAY,
}
WEBRTC_ICE_GATHERING_TIMEOUT_SEC = 6.0
WEBRTC_AUDIO_SAMPLE_RATE = 48000
WEBRTC_AUDIO_CHANNEL_LAYOUT = "stereo"
WEBRTC_AUDIO_SAMPLES_PER_FRAME = 960
WEBRTC_IDLE_VIDEO_FPS = 24.0
WEBRTC_VIDEO_CLOCK_RATE = 90000
WEBRTC_VIDEO_TIME_BASE = Fraction(1, WEBRTC_VIDEO_CLOCK_RATE)
DEFAULT_WEBRTC_ICE_TRANSPORT_POLICY = (
    os.getenv(WEBRTC_ICE_TRANSPORT_POLICY_ENV_KEY, WEBRTC_ICE_TRANSPORT_POLICY_ALL).strip().lower()
    or WEBRTC_ICE_TRANSPORT_POLICY_ALL
)
if DEFAULT_WEBRTC_ICE_TRANSPORT_POLICY not in WEBRTC_ICE_TRANSPORT_POLICY_CHOICES:
    DEFAULT_WEBRTC_ICE_TRANSPORT_POLICY = WEBRTC_ICE_TRANSPORT_POLICY_ALL


def now_ms() -> int:
    """
    Return current UTC timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def parse_webrtc_ice_server_payload(raw_value: str) -> list[dict[str, Any]]:
    """
    Parse ICE server JSON from one environment variable into browser/server-safe dictionaries.
    """
    normalized_value = str(raw_value or "").strip()
    if not normalized_value:
        return []
    try:
        payload = json.loads(normalized_value)
    except json.JSONDecodeError:
        print(f"[webrtc] invalid ICE server JSON in {WEBRTC_ICE_SERVERS_ENV_KEY}")
        return []
    if not isinstance(payload, list):
        print(f"[webrtc] {WEBRTC_ICE_SERVERS_ENV_KEY} must be a JSON array")
        return []
    normalized_servers: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_urls = item.get("urls")
        urls: list[str] = []
        if isinstance(raw_urls, str) and raw_urls.strip():
            urls = [raw_urls.strip()]
        elif isinstance(raw_urls, list):
            urls = [str(value).strip() for value in raw_urls if str(value).strip()]
        if not urls:
            continue
        server_payload: dict[str, Any] = {
            "urls": urls,
        }
        username = str(item.get("username") or "").strip()
        credential = str(item.get("credential") or "").strip()
        if username:
            server_payload["username"] = username
        if credential:
            server_payload["credential"] = credential
        normalized_servers.append(server_payload)
    return normalized_servers


WEBRTC_ICE_SERVER_PAYLOADS = parse_webrtc_ice_server_payload(os.getenv(WEBRTC_ICE_SERVERS_ENV_KEY, ""))


def build_webrtc_ice_servers() -> list[RTCIceServer]:
    """
    Build aiortc ICE server objects from normalized runtime configuration.
    """
    ice_servers: list[RTCIceServer] = []
    for payload in WEBRTC_ICE_SERVER_PAYLOADS:
        urls = payload.get("urls") or []
        if not isinstance(urls, list) or not urls:
            continue
        ice_servers.append(
            RTCIceServer(
                urls=urls,
                username=str(payload.get("username") or "") or None,
                credential=str(payload.get("credential") or "") or None,
            )
        )
    return ice_servers


def build_webrtc_rtc_configuration() -> RTCConfiguration:
    """
    Build server-side RTC configuration.
    """
    return RTCConfiguration(iceServers=build_webrtc_ice_servers())


def normalize_rel_path(value: str) -> str:
    """
    Normalize local path string to POSIX style for runner args.
    """
    return value.replace("\\", "/")


def is_source_video_path(path_value: Path | str | None) -> bool:
    """
    Determine whether a source path points to a video file supported by the source pipeline.
    """
    if path_value is None:
        return False
    return Path(str(path_value)).suffix.lower() in ALLOWED_SOURCE_VIDEO_EXTENSIONS


def is_video_backed_source_path(path_value: Path | str | None) -> bool:
    """
    Determine whether a source path is backed by video frames, including template packs.
    """
    if path_value is None:
        return False
    if is_source_video_path(path_value):
        return True
    if not is_source_template_pack_path(path_value):
        return False
    metadata = read_source_template_pack_metadata(Path(str(path_value)).resolve())
    return bool(metadata.get("is_source_video", False))


def resolve_source_input_kind(path_value: Path | str | None) -> str:
    """
    Resolve public source input kind for logs and payloads.
    """
    if path_value is None:
        return "unknown"
    if is_source_template_pack_path(path_value):
        return "template_pack"
    if is_source_video_path(path_value):
        return "video"
    return "image"


def resolve_source_frame_candidate(source_frame: str) -> tuple[Path, str]:
    """
    Resolve one concrete source frame path and return (absolute, runner_arg).
    """
    source_frame = str(source_frame or "").strip() or DEFAULT_SOURCE_FRAME
    if source_frame == DEFAULT_SOURCE_FRAME:
        idle_source_frame_abs = ensure_idle_source_frame_abs()
        if idle_source_frame_abs is not None:
            return idle_source_frame_abs, to_runner_source_arg(idle_source_frame_abs)
    candidate = Path(source_frame)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / source_frame).resolve()
        runner_arg = normalize_rel_path(source_frame)
    else:
        candidate = candidate.resolve()
        runner_arg = str(candidate)
    if not candidate.exists():
        if source_frame == DEFAULT_SOURCE_FRAME:
            fallback = discover_fallback_source_frame()
            if fallback is not None:
                return fallback, to_runner_source_arg(fallback)
        raise HTTPException(status_code=400, detail=f"Source frame not found: {source_frame}")
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail=f"Source frame is not a file: {source_frame}")
    return candidate, runner_arg


def resolve_configured_fixed_source_frame() -> tuple[Path, str] | None:
    """
    Resolve server-wide fixed source frame when configured.
    """
    if not FIXED_SOURCE_FRAME:
        return None
    try:
        return resolve_source_frame_candidate(FIXED_SOURCE_FRAME)
    except HTTPException as exc:
        raise RuntimeError(f"{FIXED_SOURCE_FRAME_ENV_KEY} invalid: {exc.detail}") from exc


def resolve_source_frame(source_frame: str) -> tuple[Path, str]:
    """
    Resolve effective source frame, honoring fixed-source configuration when present.
    """
    fixed_source = resolve_configured_fixed_source_frame()
    if fixed_source is not None:
        return fixed_source
    return resolve_source_frame_candidate(source_frame)


def to_runner_source_arg(source_abs_path: Path) -> str:
    """
    Convert absolute source path into runner argument (relative when inside project root).
    """
    resolved = source_abs_path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return str(resolved)
    return normalize_rel_path(str(relative))


def discover_fallback_source_frame() -> Path | None:
    """
    Discover first available source frame from output/frames directory.
    """
    frames_dir = (PROJECT_ROOT / "output/frames").resolve()
    if not frames_dir.exists() or not frames_dir.is_dir():
        return None

    primary_matches: list[Path] = []
    for extension in ALLOWED_SOURCE_IMAGE_EXTENSIONS:
        primary_matches.extend(frames_dir.glob(f"frame_*{extension}"))
    primary_matches = sorted(path for path in primary_matches if path.is_file())
    if primary_matches:
        return primary_matches[0]

    secondary_matches = sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SOURCE_IMAGE_EXTENSIONS
    )
    if secondary_matches:
        return secondary_matches[0]
    return None


def resolve_warmup_source_frame_arg() -> str | None:
    """
    Resolve warmup source frame argument with fallback when default is missing.
    """
    fixed_source = resolve_configured_fixed_source_frame()
    if fixed_source is not None:
        _, fixed_runner_arg = fixed_source
        return fixed_runner_arg

    idle_source_frame_abs = ensure_idle_source_frame_abs()
    if idle_source_frame_abs is not None:
        return to_runner_source_arg(idle_source_frame_abs)

    default_source = Path(DEFAULT_SOURCE_FRAME)
    if default_source.is_absolute():
        default_abs = default_source.resolve()
    else:
        default_abs = (PROJECT_ROOT / default_source).resolve()

    if default_abs.exists() and default_abs.is_file():
        return to_runner_source_arg(default_abs)

    fallback = discover_fallback_source_frame()
    if fallback is None:
        return None
    return to_runner_source_arg(fallback)


def read_json(path: Path) -> dict[str, Any] | None:
    """
    Read JSON file safely.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def parse_stream_cache_counter(value: Any) -> int:
    """
    Parse non-negative integer counters used by stream status payloads.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return 0


def close_job_stream_cache(cache: JobStreamMemoryCache) -> None:
    """
    Close one attached shared-memory reader and release its transport handle.
    """
    with cache.lock:
        reader = cache.reader
        cache.reader = None
    if reader is not None:
        with contextlib.suppress(Exception):
            reader.close()


def prune_expired_job_stream_caches() -> None:
    """
    Drop stale shared-memory reader handles after a short retention window.
    """
    current_time_ms = now_ms()
    stale_entries: list[JobStreamMemoryCache] = []
    with JOB_STREAM_CACHE_LOCK:
        stale_job_ids = [
            job_id
            for job_id, cache in JOB_STREAM_CACHES.items()
            if current_time_ms - int(cache.last_accessed_at_ms) > JOB_STREAM_CACHE_RETENTION_MS
        ]
        for stale_job_id in stale_job_ids:
            stale_cache = JOB_STREAM_CACHES.pop(stale_job_id, None)
            if stale_cache is not None:
                stale_entries.append(stale_cache)
    for stale_cache in stale_entries:
        close_job_stream_cache(stale_cache)


def get_or_create_job_stream_cache(job: JobRecord) -> JobStreamMemoryCache:
    """
    Resolve one per-job shared-memory reader handle.
    """
    prune_expired_job_stream_caches()
    with JOB_STREAM_CACHE_LOCK:
        cache = JOB_STREAM_CACHES.get(job.job_id)
        if cache is None:
            cache = JobStreamMemoryCache(
                job_id=job.job_id,
                stream_shm_prefix=job.stream_shm_prefix,
            )
            JOB_STREAM_CACHES[job.job_id] = cache
        cache.last_accessed_at_ms = now_ms()
        return cache


def ensure_job_stream_reader(job: JobRecord) -> JobStreamSharedMemoryReader | None:
    """
    Attach to the producer shared-memory transport when it becomes available.
    """
    cache = get_or_create_job_stream_cache(job)
    with cache.lock:
        cache.last_accessed_at_ms = now_ms()
        if cache.reader is not None:
            return cache.reader
        try:
            cache.reader = JobStreamSharedMemoryReader(cache.stream_shm_prefix)
        except (FileNotFoundError, OSError, RuntimeError):
            cache.reader = None
        return cache.reader


def read_job_stream_status(job: JobRecord) -> dict[str, Any] | None:
    """
    Read one job stream status from shared memory.
    """
    reader = ensure_job_stream_reader(job)
    if reader is not None:
        try:
            payload = reader.read_status_payload()
            if payload is not None:
                return payload
        except Exception:
            close_job_stream_cache(get_or_create_job_stream_cache(job))
    return read_runner_prepare_status(job)


def read_runner_prepare_status(job: JobRecord) -> dict[str, Any] | None:
    """
    Read one coarse pre-render status while the runner is still preparing inputs.
    """
    if job.process is None or job.process.poll() is not None:
        return None
    payload = read_json(job.output_abs / RUNNER_STATUS_FILE_NAME)
    return payload if isinstance(payload, dict) else None


def read_job_stream_frame_by_index(job: JobRecord, frame_index: int) -> bytes | None:
    """
    Read one encoded stream frame from shared memory.
    """
    reader = ensure_job_stream_reader(job)
    if reader is None:
        return None
    try:
        return reader.read_frame(frame_index)
    except Exception:
        close_job_stream_cache(get_or_create_job_stream_cache(job))
        return None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """
    Write one JSON payload atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(str(temporary_path), str(path))


def build_public_file_url(file_path: Path) -> str:
    """
    Build a browser-accessible URL for a file inside the project workspace.
    """
    resolved = file_path.resolve()
    try:
        relative_job_path = resolved.relative_to(JOBS_ROOT)
    except ValueError:
        relative_job_path = None
    if relative_job_path is not None:
        return f"/jobs/{normalize_rel_path(str(relative_job_path))}"

    try:
        relative_project_path = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return ""
    return f"/{normalize_rel_path(str(relative_project_path))}"


def build_audio_tuning_preset_values(preset_name: str) -> dict[str, Any]:
    """
    Resolve one named audio tuning preset.
    """
    normalized_preset_name = str(preset_name or "").strip().lower()
    if normalized_preset_name != SOURCE_TEMPLATE_PACK_CLOSE_MOUTH_PRESET:
        return {}
    return {
        "audioMotionTuningEnabled": True,
        "audioLipSyncAssist": True,
        "audioReanchorFirstN": 1,
        "audioMouthOpenFactor": 1.0,
        "audioPoseSmoothWindow": 3,
        "audioExpSmoothWindow": 1,
        "audioPoseJumpThreshold": 8.0,
        "audioTranslationJumpThreshold": 0.03,
        "audioLipSyncMinRatio": 0.0,
        "audioLipSyncMaxRatio": 0.38,
        "audioLipSyncSmoothWindow": 3,
        "audioLipSyncStrength": 1.35,
        "audioLipSyncPower": 0.70,
        "audioLipSyncAttack": 1.0,
        "audioLipSyncRelease": 0.55,
        "audioLipSyncOffsetMs": 0,
        "audioMouthFloorStrength": 0.0,
        "audioMouthPeakClamp": 0.0,
        "drivingMultiplier": 1.0,
        "cfgScale": 4.0,
        "animationRegion": "all",
        "stitchingEnabled": True,
        "relativeMotionEnabled": True,
    }


def build_audio_tuning_presets_payload() -> dict[str, dict[str, Any]]:
    """
    Return the public preset catalog for clients.
    """
    return {
        SOURCE_TEMPLATE_PACK_CLOSE_MOUTH_PRESET: build_audio_tuning_preset_values(
            SOURCE_TEMPLATE_PACK_CLOSE_MOUTH_PRESET
        )
    }


def resolve_source_template_pack_meta_path(template_pack_path: Path) -> Path:
    return Path(f"{template_pack_path}{SOURCE_TEMPLATE_PACK_META_SUFFIX}")


def resolve_source_template_pack_preview_path(template_pack_path: Path) -> Path:
    return template_pack_path.with_suffix(SOURCE_TEMPLATE_PACK_PREVIEW_SUFFIX)


def read_source_template_pack_metadata(template_pack_path: Path) -> dict[str, Any]:
    meta_path = resolve_source_template_pack_meta_path(template_pack_path)
    payload = read_json(meta_path)
    if isinstance(payload, dict):
        return payload
    return {}


def is_source_template_pack_path(path_value: Path | str | None) -> bool:
    safe_path = Path(str(path_value or "")).resolve() if str(path_value or "").strip() else None
    if safe_path is None or safe_path.suffix.lower() != ".pkl":
        return False
    return resolve_source_template_pack_meta_path(safe_path).exists()


def build_source_template_pack_record(template_pack_path: Path) -> dict[str, Any]:
    metadata = read_source_template_pack_metadata(template_pack_path)
    preview_path = resolve_source_template_pack_preview_path(template_pack_path)
    template_name = template_pack_path.name
    source_type = "video" if bool(metadata.get("is_source_video", False)) else "image"
    return {
        "id": template_name,
        "name": template_name,
        "fileName": template_name,
        "path": normalize_rel_path(str(template_pack_path.relative_to(PROJECT_ROOT))),
        "templatePackPath": str(template_pack_path),
        "createdAt": metadata.get("created_at", ""),
        "frameTotal": int(metadata.get("frame_total", 0) or 0),
        "sourceType": source_type,
        "sourceFps": metadata.get("source_fps"),
        "sourcePath": metadata.get("source_path", ""),
        "previewUrl": build_public_file_url(preview_path) if preview_path.exists() else "",
    }


def validate_source_template_pack_payload(template_pack_path: Path) -> tuple[bool, str]:
    """
    Validate one built source template pack structure before exposing it to clients.
    """
    try:
        with template_pack_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        return False, f"unable to load template pack payload: {exc}"
    if not isinstance(payload, dict):
        return False, "template pack payload is not a dictionary"
    src_imgs = payload.get("src_imgs")
    src_infos = payload.get("src_infos")
    if not isinstance(src_imgs, list) or not src_imgs:
        return False, "template pack payload is missing src_imgs"
    if not isinstance(src_infos, list) or not src_infos:
        return False, "template pack payload is missing src_infos"
    if len(src_imgs) != len(src_infos):
        return False, "template pack payload frame counts do not match"
    return True, ""


def list_source_template_pack_records() -> list[dict[str, Any]]:
    SOURCE_TEMPLATE_PACKS_ROOT.mkdir(parents=True, exist_ok=True)
    template_paths = sorted(
        SOURCE_TEMPLATE_PACKS_ROOT.glob("*.pkl"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    return [build_source_template_pack_record(path.resolve()) for path in template_paths]


def sanitize_source_template_pack_name(template_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(template_name or "").strip())
    safe_name = safe_name.strip("._")
    return safe_name or "source_template"


def resolve_source_template_pack_path(template_identifier: str) -> Path:
    SOURCE_TEMPLATE_PACKS_ROOT.mkdir(parents=True, exist_ok=True)
    normalized_identifier = str(template_identifier or "").strip()
    if not normalized_identifier:
        raise HTTPException(status_code=400, detail="source_template_pack is empty.")

    candidate_path = Path(normalized_identifier)
    candidate_paths: list[Path] = []
    if candidate_path.is_absolute():
        candidate_paths.append(candidate_path.resolve())
    else:
        candidate_paths.append((SOURCE_TEMPLATE_PACKS_ROOT / candidate_path.name).resolve())
        if candidate_path.suffix.lower() != ".pkl":
            candidate_paths.append((SOURCE_TEMPLATE_PACKS_ROOT / f"{candidate_path.name}.pkl").resolve())

    for resolved_path in candidate_paths:
        if resolved_path.exists() and resolved_path.is_file():
            try:
                resolved_path.relative_to(SOURCE_TEMPLATE_PACKS_ROOT)
            except ValueError:
                continue
            if is_source_template_pack_path(resolved_path):
                return resolved_path
    raise HTTPException(status_code=404, detail=f"Source template pack not found: {normalized_identifier}")


def extract_bearer_token(header_value: str) -> str:
    """
    Extract bearer token value from one Authorization header when present.
    """
    normalized_value = str(header_value or "").strip()
    if not normalized_value:
        return ""
    scheme_prefix = f"{AUTHORIZATION_HEADER_VALUE} "
    if normalized_value.lower().startswith(scheme_prefix.lower()):
        return normalized_value[len(scheme_prefix):].strip()
    return ""


def is_authentication_required_for_path(request_path: str) -> bool:
    """
    Determine whether one HTTP path must present the configured API token.
    """
    normalized_path = str(request_path or "").strip()
    return any(normalized_path.startswith(prefix) for prefix in AUTH_REQUIRED_HTTP_PATH_PREFIXES)


def is_valid_api_token(raw_token: str) -> bool:
    """
    Validate one provided API token against the configured shared secret.
    """
    if not API_TOKEN_ENABLED:
        return True
    normalized_token = str(raw_token or "").strip()
    if not normalized_token:
        return False
    return hmac.compare_digest(normalized_token, DEFAULT_API_TOKEN)


def extract_request_api_token(request: Request) -> str:
    """
    Extract API token from one HTTP request query string or Authorization header.
    """
    query_token = str(request.query_params.get(API_TOKEN_QUERY_KEY, "")).strip()
    if query_token:
        return query_token
    return extract_bearer_token(str(request.headers.get(AUTHORIZATION_HEADER_NAME, "")))


def extract_websocket_api_token(websocket: WebSocket) -> str:
    """
    Extract API token from one WebSocket query string or Authorization header.
    """
    query_token = str(websocket.query_params.get(API_TOKEN_QUERY_KEY, "")).strip()
    if query_token:
        return query_token
    return extract_bearer_token(str(websocket.headers.get(AUTHORIZATION_HEADER_NAME, "")))


def build_http_auth_error_response() -> JSONResponse:
    """
    Build one consistent HTTP 401 response for token-protected routes.
    """
    return JSONResponse(
        status_code=401,
        content={"detail": AUTH_FAILURE_MESSAGE},
        headers={"WWW-Authenticate": AUTHORIZATION_HEADER_VALUE},
    )


def authorize_http_request(request: Request) -> JSONResponse | None:
    """
    Validate token requirements for one HTTP request.
    """
    if not API_TOKEN_ENABLED:
        return None
    if not is_authentication_required_for_path(request.url.path):
        return None
    if is_valid_api_token(extract_request_api_token(request)):
        return None
    return build_http_auth_error_response()


def is_websocket_request_authorized(websocket: WebSocket) -> bool:
    """
    Validate token requirements for one WebSocket request.
    """
    return is_valid_api_token(extract_websocket_api_token(websocket))


def probe_media_duration_sec(media_path: Path) -> float:
    """
    Probe media duration in seconds using ffprobe when available.
    """
    try:
        completed = subprocess.run(
            [
                FFPROBE_BINARY,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path.resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    try:
        duration_sec = float((completed.stdout or "").strip())
    except ValueError:
        return 0.0
    return duration_sec if duration_sec > 0 else 0.0


def normalize_stream_audio_input(source_audio_abs: Path, normalized_audio_abs: Path) -> Path:
    """
    Normalize one uploaded audio file into a stable WAV input for stream muxing.
    """
    normalized_audio_abs.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                FFMPEG_BINARY,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source_audio_abs.resolve()),
                "-vn",
                "-ar",
                VIDEO_STREAM_AUDIO_SAMPLE_RATE,
                "-ac",
                VIDEO_STREAM_AUDIO_CHANNELS,
                "-c:a",
                "pcm_s16le",
                str(normalized_audio_abs.resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[audio] stream normalization failed for {source_audio_abs.name}: {exc}")
        return source_audio_abs
    if not normalized_audio_abs.exists() or normalized_audio_abs.stat().st_size <= 0:
        return source_audio_abs
    return normalized_audio_abs


def resolve_idle_video_abs() -> Path | None:
    """
    Resolve configured idle video path when available.
    """
    idle_video_path = AVATAR_IDLE_VIDEO_REL
    if not idle_video_path.is_absolute():
        idle_video_path = (PROJECT_ROOT / idle_video_path).resolve()
    else:
        idle_video_path = idle_video_path.resolve()
    if not idle_video_path.exists() or not idle_video_path.is_file():
        return None
    return idle_video_path


def resolve_idle_video_url() -> str:
    """
    Resolve browser URL for the configured idle video.
    """
    idle_video_abs = resolve_idle_video_abs()
    if idle_video_abs is None:
        return ""
    idle_video_url = build_public_file_url(idle_video_abs)
    if not idle_video_url:
        return ""
    return f"{idle_video_url}?v={int(idle_video_abs.stat().st_mtime_ns)}"


def build_idle_source_anchor_manifest_path() -> Path:
    """
    Resolve the cached anchor manifest path for the idle video.
    """
    return (PROJECT_ROOT / AVATAR_IDLE_SOURCE_ANCHOR_ROOT_REL / AVATAR_IDLE_SOURCE_ANCHOR_MANIFEST_NAME).resolve()


def build_idle_source_anchor_path(anchor_order: int, frame_index: int) -> Path:
    """
    Resolve one cached idle anchor image path.
    """
    if anchor_order <= 0:
        return (PROJECT_ROOT / AVATAR_IDLE_SOURCE_FRAME_REL).resolve()
    file_name = f"anchor_{anchor_order:02d}_frame_{max(0, int(frame_index)):06d}.png"
    return (PROJECT_ROOT / AVATAR_IDLE_SOURCE_ANCHOR_ROOT_REL / file_name).resolve()


def build_idle_source_anchor_frame_indices(frame_total: int) -> list[int]:
    """
    Build evenly distributed anchor frame indices across the idle loop.
    """
    safe_frame_total = max(1, int(frame_total))
    anchor_count = min(AVATAR_IDLE_SOURCE_ANCHOR_COUNT, safe_frame_total)
    if anchor_count <= 1:
        return [0]
    frame_indices: list[int] = []
    for anchor_order in range(anchor_count):
        frame_index = int(round((float(anchor_order) * float(safe_frame_total)) / float(anchor_count)))
        frame_indices.append(min(max(0, frame_index), safe_frame_total - 1))
    ordered_unique_indices: list[int] = []
    seen_indices: set[int] = set()
    for frame_index in frame_indices:
        if frame_index in seen_indices:
            continue
        seen_indices.add(frame_index)
        ordered_unique_indices.append(frame_index)
    if not ordered_unique_indices:
        return [0]
    return ordered_unique_indices


def load_idle_source_anchor_manifest(
    manifest_path: Path,
    idle_video_abs: Path,
    idle_video_mtime_ns: int,
) -> dict[str, Any] | None:
    """
    Load one cached idle anchor manifest when it still matches the current idle video.
    """
    manifest = read_json(manifest_path)
    if manifest is None:
        return None
    if str(manifest.get("idleVideoPath") or "") != str(idle_video_abs.resolve()):
        return None
    if int(manifest.get("idleVideoMtimeNs") or 0) != int(idle_video_mtime_ns):
        return None
    anchors = manifest.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        return None
    for anchor in anchors:
        anchor_path = Path(str(anchor.get("path") or "")).resolve()
        if not anchor_path.exists() or not anchor_path.is_file():
            return None
    return manifest


def ensure_idle_source_anchor_manifest() -> dict[str, Any] | None:
    """
    Extract and cache one reusable bank of idle anchors for avatar handoff selection.
    """
    idle_video_abs = resolve_idle_video_abs()
    if idle_video_abs is None:
        return None
    try:
        idle_video_mtime_ns = idle_video_abs.stat().st_mtime_ns
    except OSError:
        return None
    manifest_path = build_idle_source_anchor_manifest_path()
    cached_manifest = load_idle_source_anchor_manifest(manifest_path, idle_video_abs, idle_video_mtime_ns)
    if cached_manifest is not None:
        return cached_manifest

    capture = cv2.VideoCapture(str(idle_video_abs))
    if not capture.isOpened():
        return None
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = WEBRTC_IDLE_VIDEO_FPS
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_indices = build_idle_source_anchor_frame_indices(frame_total)
        anchors: list[dict[str, Any]] = []
        for anchor_order, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frame_ok, frame_bgr = capture.read()
            if not frame_ok or frame_bgr is None:
                continue
            anchor_path = build_idle_source_anchor_path(anchor_order, frame_index)
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(anchor_path), frame_bgr):
                continue
            anchors.append(
                {
                    "anchorOrder": anchor_order,
                    "frameIndex": int(frame_index),
                    "offsetSec": float(frame_index) / float(source_fps) if source_fps > 0 else 0.0,
                    "path": str(anchor_path.resolve()),
                }
            )
    finally:
        capture.release()

    if not anchors:
        return None
    duration_sec = 0.0
    if source_fps > 0 and frame_total > 0:
        duration_sec = float(frame_total) / float(source_fps)
    manifest = {
        "idleVideoPath": str(idle_video_abs.resolve()),
        "idleVideoMtimeNs": int(idle_video_mtime_ns),
        "sourceFps": float(source_fps),
        "frameTotal": int(frame_total),
        "durationSec": float(duration_sec),
        "anchors": anchors,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def ensure_idle_source_anchors() -> list[dict[str, Any]]:
    """
    Resolve the cached idle source anchors as a validated list.
    """
    manifest = ensure_idle_source_anchor_manifest()
    if manifest is None:
        return []
    anchors = manifest.get("anchors")
    if not isinstance(anchors, list):
        return []
    return [anchor for anchor in anchors if isinstance(anchor, dict)]


def ensure_idle_source_frame_abs() -> Path | None:
    """
    Extract and cache one stable source frame from the idle video.
    """
    anchors = ensure_idle_source_anchors()
    if not anchors:
        return None
    target_path = Path(str(anchors[0].get("path") or "")).resolve()
    if not target_path.exists() or not target_path.is_file():
        return None
    return target_path


def tail_log(path: Path, max_lines: int) -> str:
    """
    Return last lines from log file.
    """
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-max_lines:])


def resolve_runtime_log_path(target: str) -> Path:
    """
    Resolve one runtime log target into an absolute file path.
    """
    if target == RUNTIME_LOG_TARGET_CONTAINER:
        return CONTAINER_LOG_ABS
    if target == RUNTIME_LOG_TARGET_WORKER:
        return PERSISTENT_WORKER_LOG_ABS
    raise HTTPException(status_code=400, detail=f"Invalid log target: {target}")


def reset_runtime_restart_state() -> None:
    """
    Clear restart-in-progress markers after one failed restart attempt.
    """
    global RUNTIME_RESTARTING
    global RUNTIME_RESTART_REQUESTED_AT_MS
    with RUNTIME_RESTART_LOCK:
        RUNTIME_RESTARTING = False
        RUNTIME_RESTART_REQUESTED_AT_MS = 0


def build_host_restart_command() -> list[str]:
    """
    Build detached host restart command for local Windows execution.
    """
    return [
        str(Path(sys.executable).resolve()),
        str((PROJECT_ROOT / "realtime_stream_api.py").resolve()),
        "--host",
        str(CURRENT_API_HOST),
        "--port",
        str(CURRENT_API_PORT),
        "--backend",
        str(DEFAULT_BACKEND),
        "--trt-runtime",
        str(DEFAULT_TRT_RUNTIME),
        "--trt-precision",
        str(DEFAULT_TRT_PRECISION),
    ]


def start_runtime_restart_thread() -> None:
    """
    Schedule current API process exit so Docker can restart the container from a clean state.
    """

    def restart_runtime_process() -> None:
        time.sleep(RUNTIME_RESTART_DELAY_SEC)
        if os.name == "nt":
            try:
                creation_flags = 0
                creation_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                creation_flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
                host_stdout_path = PROJECT_ROOT / "output_fasterliveportrait" / "host_api_stdout.log"
                host_stderr_path = PROJECT_ROOT / "output_fasterliveportrait" / "host_api_stderr.log"
                host_stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_handle = host_stdout_path.open("ab")
                stderr_handle = host_stderr_path.open("ab")
                subprocess.Popen(
                    build_host_restart_command(),
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=creation_flags,
                    close_fds=True,
                )
            except Exception as exc:
                print(f"[runtime] host restart failed: {exc}")
                reset_runtime_restart_state()
                return
            os._exit(0)
        try:
            os.kill(1, signal.SIGKILL)
            return
        except Exception:
            os._exit(0)

    restart_thread = threading.Thread(target=restart_runtime_process, daemon=True)
    restart_thread.start()


def request_runtime_restart() -> int:
    """
    Mark runtime restart as requested and schedule process exit once.
    """
    global RUNTIME_RESTARTING
    global RUNTIME_RESTART_REQUESTED_AT_MS
    with RUNTIME_RESTART_LOCK:
        if RUNTIME_RESTARTING:
            return RUNTIME_RESTART_REQUESTED_AT_MS
        RUNTIME_RESTARTING = True
        RUNTIME_RESTART_REQUESTED_AT_MS = now_ms()
        start_runtime_restart_thread()
        return RUNTIME_RESTART_REQUESTED_AT_MS


def ensure_runtime_accepting_requests() -> None:
    """
    Reject mutating requests while a full runtime restart is already in progress.
    """
    if RUNTIME_RESTARTING:
        raise HTTPException(status_code=409, detail="Runtime restart already in progress.")


def build_video_stream_command(
    input_fps: float,
    audio_input_path: Path | None = None,
    include_silent_audio: bool = False,
    timestamp_offset_sec: float = 0.0,
) -> list[str]:
    """
    Build ffmpeg command that converts JPEG frames from stdin into fragmented MP4 for WebSocket transport.
    """
    safe_fps = max(1.0, float(input_fps))
    command = [
        FFMPEG_BINARY,
        "-loglevel",
        "error",
        "-nostdin",
        "-fflags",
        "+nobuffer",
        "-flags",
        "low_delay",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-framerate",
        f"{safe_fps:.6f}",
        "-i",
        "pipe:0",
    ]
    has_audio_input = audio_input_path is not None and audio_input_path.exists()
    if has_audio_input:
        command.extend(
            [
                "-i",
                str(audio_input_path.resolve()),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    elif include_silent_audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={VIDEO_STREAM_AUDIO_SAMPLE_RATE}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    else:
        command.extend(
            [
                "-an",
                "-map",
                "0:v:0",
            ]
        )
    command.extend(
        build_stream_video_codec_args()
    )
    if has_audio_input or include_silent_audio:
        command.extend(
            [
                "-c:a",
                VIDEO_STREAM_AUDIO_CODEC,
                "-b:a",
                VIDEO_STREAM_AUDIO_BITRATE,
                "-ar",
                VIDEO_STREAM_AUDIO_SAMPLE_RATE,
                "-ac",
                VIDEO_STREAM_AUDIO_CHANNELS,
                "-af",
                VIDEO_STREAM_AUDIO_FILTER,
                "-shortest",
            ]
        )
    command.extend(
        [
            "-muxdelay",
            VIDEO_STREAM_MUX_DELAY,
            "-muxpreload",
            VIDEO_STREAM_MUX_PRELOAD,
            "-movflags",
            "+frag_keyframe+frag_every_frame+empty_moov+default_base_moof+omit_tfhd_offset",
            "-flush_packets",
            "1",
        ]
    )
    if timestamp_offset_sec > 0:
        command.extend(
            [
                "-output_ts_offset",
                f"{max(0.0, float(timestamp_offset_sec)):.6f}",
            ]
        )
    command.extend(
        [
            "-f",
            "mp4",
            "pipe:1",
        ]
    )
    return command


def build_avatar_stream_command(
    input_fps: float,
    canvas_size: tuple[int, int],
    audio_pipe_fd: int | None = None,
    audio_input_path: Path | None = None,
    include_silent_audio: bool = False,
    timestamp_offset_sec: float = 0.0,
) -> list[str]:
    """
    Build ffmpeg command that converts raw BGR frames from stdin into fragmented MP4 for the continuous avatar stream.
    """
    safe_fps = max(1.0, float(input_fps))
    canvas_width = max(2, int(canvas_size[0]))
    canvas_height = max(2, int(canvas_size[1]))
    command = [
        FFMPEG_BINARY,
        "-loglevel",
        "error",
        "-nostdin",
        "-fflags",
        "+nobuffer",
        "-flags",
        "low_delay",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{canvas_width}x{canvas_height}",
        "-framerate",
        f"{safe_fps:.6f}",
        "-i",
        "pipe:0",
    ]
    has_audio_pipe = audio_pipe_fd is not None and audio_pipe_fd >= 0
    has_audio_input = audio_input_path is not None and audio_input_path.exists()
    if has_audio_pipe:
        command.extend(
            [
                "-f",
                "s16le",
                "-ar",
                VIDEO_STREAM_AUDIO_SAMPLE_RATE,
                "-ac",
                VIDEO_STREAM_AUDIO_CHANNELS,
                "-i",
                f"pipe:{int(audio_pipe_fd)}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    elif has_audio_input:
        command.extend(
            [
                "-i",
                str(audio_input_path.resolve()),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    elif include_silent_audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={VIDEO_STREAM_AUDIO_SAMPLE_RATE}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    else:
        command.extend(
            [
                "-an",
                "-map",
                "0:v:0",
            ]
        )
    command.extend(
        build_stream_video_codec_args()
    )
    if has_audio_pipe or has_audio_input or include_silent_audio:
        command.extend(
            [
                "-c:a",
                VIDEO_STREAM_AUDIO_CODEC,
                "-b:a",
                VIDEO_STREAM_AUDIO_BITRATE,
                "-ar",
                VIDEO_STREAM_AUDIO_SAMPLE_RATE,
                "-ac",
                VIDEO_STREAM_AUDIO_CHANNELS,
                "-af",
                VIDEO_STREAM_AUDIO_FILTER,
            ]
        )
        if not has_audio_pipe:
            command.append("-shortest")
    command.extend(
        [
            "-muxdelay",
            VIDEO_STREAM_MUX_DELAY,
            "-muxpreload",
            VIDEO_STREAM_MUX_PRELOAD,
            "-movflags",
            AVATAR_STREAM_MUX_MOVFLAGS,
            "-flush_packets",
            "1",
        ]
    )
    if timestamp_offset_sec > 0:
        command.extend(
            [
                "-output_ts_offset",
                f"{max(0.0, float(timestamp_offset_sec)):.6f}",
            ]
        )
    command.extend(
        [
            "-f",
            "mp4",
            "pipe:1",
        ]
    )
    return command


def fps_values_match(left_fps: float, right_fps: float, tolerance: float = 0.01) -> bool:
    """
    Compare two FPS values using a small tolerance suitable for stream restarts.
    """
    return abs(float(left_fps) - float(right_fps)) <= float(tolerance)


@dataclass
class JobRecord:
    job_id: str
    avatar_session_id: str
    created_at_ms: int
    mode: str
    source_frame_arg: str
    source_frame_abs: Path
    source_template_pack_id: str
    source_template_pack_abs: Path | None
    output_rel: Path
    output_abs: Path
    stream_rel: Path
    stream_abs: Path
    stream_shm_prefix: str
    audio_input_rel: Path
    audio_input_abs: Path
    stream_audio_input_abs: Path
    audio_original_name: str
    audio_duration_sec: float
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
    audio_lip_sync_attack: float
    audio_lip_sync_release: float
    audio_lip_sync_offset_ms: int
    audio_mouth_floor_strength: float
    audio_mouth_peak_clamp: float
    driving_multiplier: float
    cfg_scale: float
    joyvasa_inference_steps: int
    animation_region: str
    stitching_enabled: bool
    relative_motion_enabled: bool
    paste_back_enabled: bool
    defer_paste_back_enabled: bool
    log_rel: Path
    log_abs: Path
    started_at_ms: int | None = None
    process: subprocess.Popen | None = None
    log_handle: Any = None
    exit_code: int | None = None
    finished_at_ms: int | None = None
    avatar_ready_at_ms: int | None = None
    avatar_play_started_at_ms: int | None = None
    avatar_play_finished_at_ms: int | None = None

    @property
    def report_abs(self) -> Path:
        return self.output_abs / RUN_REPORT_FILE_NAME

    @property
    def result_abs(self) -> Path:
        return self.output_abs / "result.mp4"

    @property
    def result_concat_abs(self) -> Path:
        return self.output_abs / "result_concat.mp4"


@dataclass
class SourceTemplateBuildTask:
    task_id: str
    created_at_ms: int
    source_abs: Path
    template_pack_path: Path
    template_name: str
    log_abs: Path
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    exit_code: int | None = None
    error: str = ""
    result: dict[str, Any] | None = None
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)


@dataclass
class AvatarSessionState:
    avatar_session_id: str
    sequence: int = 0
    mode: str = AVATAR_MODE_IDLE
    current_job_id: str = ""
    current_job_started_at_ms: int = 0
    current_job_ends_at_ms: int = 0
    idle_started_at_ms: int = field(default_factory=now_ms)


@dataclass
class JobStreamMemoryCache:
    job_id: str
    stream_shm_prefix: str
    reader: JobStreamSharedMemoryReader | None = None
    last_accessed_at_ms: int = field(default_factory=now_ms)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class AvatarTalkingFrameState:
    job_id: str
    start_frame_index: int = 1
    next_frame_index: int = 1
    output_frame_index: int = 0
    playback_started_at_perf: float = 0.0
    virtual_source_position_zero_based: float | None = None
    last_known_frame_index: int = 0
    last_known_frame_total: int = 0
    estimated_generation_fps: float = VIDEO_STREAM_INPUT_FPS
    playback_fps: float = WEBRTC_IDLE_VIDEO_FPS
    source_frame_images: dict[int, np.ndarray] = field(default_factory=dict)
    pending_raw_frames: deque[bytes] = field(default_factory=deque)
    pending_stream_frames: deque[bytes] = field(default_factory=deque)
    pending_frame_images: deque[np.ndarray] = field(default_factory=deque)
    previous_raw_frame_bytes: bytes | None = None
    previous_source_frame_image: np.ndarray | None = None
    last_frame_bytes: bytes | None = None
    last_frame_image: np.ndarray | None = None
    last_output_source_frame_index: int = 0
    fallback_bounce_cursor: int = 0
    fallback_bounce_signature: tuple[int, ...] = field(default_factory=tuple)
    preview_composition_signature: str = ""
    preview_composition_matrix: np.ndarray | None = None
    preview_composition_mask: np.ndarray | None = None
    preview_composition_source: np.ndarray | None = None


class IdleVideoLooper:
    """
    Loop decoded idle video frames for continuous avatar streaming.
    """

    def __init__(self, idle_video_abs: Path | None) -> None:
        self.idle_video_abs = idle_video_abs
        self.capture: cv2.VideoCapture | None = None
        self.frame_width = AVATAR_VIDEO_FALLBACK_WIDTH
        self.frame_height = AVATAR_VIDEO_FALLBACK_HEIGHT
        self.source_fps = WEBRTC_IDLE_VIDEO_FPS
        self.frame_count = 0
        self._open_capture()

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.frame_width, self.frame_height

    def _open_capture(self) -> None:
        if self.idle_video_abs is None:
            return
        self.close()
        capture = cv2.VideoCapture(str(self.idle_video_abs))
        if not capture.isOpened():
            return
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width > 0 and height > 0:
            self.frame_width = width
            self.frame_height = height
        if source_fps > 0:
            self.source_fps = source_fps
        if frame_count > 0:
            self.frame_count = frame_count
        self.capture = capture

    def seek_to_frame(self, frame_index: int) -> None:
        """
        Seek the idle capture to one absolute frame index when the source is seekable.
        """
        if self.capture is None:
            return
        target_frame_index = max(0, int(frame_index))
        if self.frame_count > 0:
            target_frame_index %= self.frame_count
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)

    def seek_to_time(self, offset_sec: float) -> None:
        """
        Seek idle capture to one time offset when the source is seekable.
        """
        if self.capture is None or self.source_fps <= 0:
            return
        target_frame = int(max(0.0, offset_sec) * self.source_fps)
        self.seek_to_frame(target_frame)

    def read_next_frame(self) -> np.ndarray:
        if self.capture is None:
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        ok, frame = self.capture.read()
        if ok and frame is not None:
            return frame
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if ok and frame is not None:
            return frame
        return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class SilenceAudioStreamTrack(MediaStreamTrack):
    """
    Continuous silent audio track that keeps the avatar stream clock alive during idle periods.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = WEBRTC_AUDIO_SAMPLE_RATE
        self.samples_per_frame = WEBRTC_AUDIO_SAMPLES_PER_FRAME
        self.time_base = Fraction(1, self.sample_rate)
        self.timestamp = 0
        self.started_at_perf: float | None = None

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self.started_at_perf is None:
            self.started_at_perf = time.perf_counter()
        else:
            target_time = self.started_at_perf + (self.timestamp / self.sample_rate)
            sleep_duration = target_time - time.perf_counter()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
        frame = AudioFrame(format="s16", layout=WEBRTC_AUDIO_CHANNEL_LAYOUT, samples=self.samples_per_frame)
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.sample_rate = self.sample_rate
        frame.pts = self.timestamp
        frame.time_base = self.time_base
        self.timestamp += self.samples_per_frame
        return frame


class IdleVideoStreamTrack(MediaStreamTrack):
    """
    Loop idle video frames forever so the browser receives one uninterrupted video track.
    """

    kind = "video"

    def __init__(self, idle_video_abs: Path | None, start_offset_sec: float = 0.0) -> None:
        super().__init__()
        self.looper = IdleVideoLooper(idle_video_abs)
        self.frame_index = 0
        self.started_at_perf: float | None = None
        if start_offset_sec > 0:
            self.looper.seek_to_time(start_offset_sec)

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self.started_at_perf is None:
            self.started_at_perf = time.perf_counter()
        else:
            target_time = self.started_at_perf + (self.frame_index / WEBRTC_IDLE_VIDEO_FPS)
            sleep_duration = target_time - time.perf_counter()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
        frame_image = fit_frame_to_canvas(self.looper.read_next_frame(), self.looper.canvas_size)
        frame = VideoFrame.from_ndarray(frame_image, format="bgr24")
        frame.pts = self.frame_index
        frame.time_base = WEBRTC_VIDEO_TIME_BASE
        self.frame_index += 1
        return frame

    def stop(self) -> None:
        self.looper.close()
        super().stop()


class AvatarVideoStreamTrack(MediaStreamTrack):
    """
    Continuous avatar video track that keeps one stable WebRTC video sender alive across idle and talking modes.
    """

    kind = "video"

    def __init__(self, idle_video_abs: Path | None, start_offset_sec: float = 0.0) -> None:
        super().__init__()
        self.looper = IdleVideoLooper(idle_video_abs)
        self.canvas_size = self.looper.canvas_size
        self.mode = AVATAR_MODE_IDLE
        self.current_job: JobRecord | None = None
        self.current_talking_state: AvatarTalkingFrameState | None = None
        self.pending_idle_return_frames: deque[np.ndarray] = deque()
        self.pending_talking_intro_frames: deque[np.ndarray] = deque()
        self.frame_interval_sec = 1.0 / max(1.0, self.looper.source_fps or WEBRTC_IDLE_VIDEO_FPS)
        self.clock_started_at_perf: float | None = None
        self.next_emit_at: float | None = None
        self.last_frame_image = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        if start_offset_sec > 0:
            self.looper.seek_to_time(start_offset_sec)
        self._prime_idle_frame()

    def _prime_idle_frame(self) -> None:
        """
        Decode one idle frame immediately so the track always has a stable image available.
        """
        self.last_frame_image = fit_frame_to_canvas(self.looper.read_next_frame(), self.canvas_size)

    def _set_frame_interval(self, frames_per_second: float) -> None:
        """
        Update the pacing interval for the current avatar mode.
        """
        safe_frames_per_second = max(1.0, float(frames_per_second or WEBRTC_IDLE_VIDEO_FPS))
        self.frame_interval_sec = 1.0 / safe_frames_per_second
        self.next_emit_at = time.perf_counter()

    def switch_to_idle(self, idle_started_at_ms: int = 0) -> None:
        """
        Return the avatar track to the idle loop without changing the sender.
        """
        previous_mode = self.mode
        previous_frame_image = self.last_frame_image.copy()
        self.mode = AVATAR_MODE_IDLE
        self.current_job = None
        self.current_talking_state = None
        self.pending_idle_return_frames.clear()
        self.pending_talking_intro_frames.clear()
        if previous_mode == AVATAR_MODE_TALKING:
            self.pending_idle_return_frames.extend(
                build_avatar_return_to_idle_frame_sequence(
                    start_frame_image=previous_frame_image,
                    canvas_size=self.canvas_size,
                    frames_per_second=self.looper.source_fps or WEBRTC_IDLE_VIDEO_FPS,
                )
            )
        if self.pending_idle_return_frames:
            self.looper.seek_to_frame(0)
            self._prime_idle_frame()
        elif idle_started_at_ms > 0:
            idle_offset_sec = max(0.0, (now_ms() - idle_started_at_ms) / 1000.0)
            self.looper.seek_to_time(idle_offset_sec)
            self._prime_idle_frame()
        else:
            self._prime_idle_frame()
        self._set_frame_interval(self.looper.source_fps or WEBRTC_IDLE_VIDEO_FPS)

    def switch_to_talking(self, job: JobRecord, snapshot: dict[str, Any]) -> None:
        """
        Start consuming progressive talking frames for one job while keeping the same remote video track.
        """
        previous_mode = self.mode
        previous_frame_image = self.last_frame_image.copy()
        stream_status = read_job_stream_status(job)
        playback_fps = resolve_stream_playback_fps(stream_status)
        next_frame_index = resolve_avatar_playback_start_frame_index(snapshot, stream_status)
        self.mode = AVATAR_MODE_TALKING
        self.current_job = job
        self.current_talking_state = AvatarTalkingFrameState(
            job_id=job.job_id,
            start_frame_index=next_frame_index,
            next_frame_index=next_frame_index,
            playback_fps=max(1.0, playback_fps or WEBRTC_IDLE_VIDEO_FPS),
            playback_started_at_perf=time.perf_counter(),
        )
        self.pending_idle_return_frames.clear()
        self.pending_talking_intro_frames.clear()
        if (
            previous_mode == AVATAR_MODE_IDLE
            and resolve_avatar_idle_to_talking_frame_count(self.current_talking_state.playback_fps) > 0
        ):
            intro_frame_image, _ = resolve_webrtc_avatar_talking_frame_image(
                job,
                self.current_talking_state,
                self.canvas_size,
            )
            self.pending_talking_intro_frames.extend(
                build_avatar_idle_to_talking_frame_sequence(
                    start_frame_image=previous_frame_image,
                    end_frame_image=intro_frame_image,
                    canvas_size=self.canvas_size,
                    frames_per_second=self.current_talking_state.playback_fps,
                )
            )
            self.last_frame_image = previous_frame_image
        self._set_frame_interval(self.current_talking_state.playback_fps)

    def _resolve_idle_frame_image(self) -> np.ndarray:
        """
        Read the next idle loop frame and normalize it to the shared avatar canvas.
        """
        return fit_frame_to_canvas(self.looper.read_next_frame(), self.canvas_size)

    def _resolve_talking_frame_image(self) -> np.ndarray:
        """
        Read the next talking frame from the active job, or hold the last available frame when the buffer is empty.
        """
        if self.current_job is None or self.current_talking_state is None:
            return self.last_frame_image
        frame_image, is_finished = resolve_webrtc_avatar_talking_frame_image(
            self.current_job,
            self.current_talking_state,
            self.canvas_size,
        )
        if frame_image is not None:
            self.last_frame_image = frame_image
        if is_finished:
            mark_avatar_job_finished(self.current_job)
            return self.last_frame_image
        return self.last_frame_image

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError
        now_perf = time.perf_counter()
        if self.clock_started_at_perf is None:
            self.clock_started_at_perf = now_perf
            self.next_emit_at = now_perf
        elif self.next_emit_at is not None and now_perf < self.next_emit_at:
            await asyncio.sleep(self.next_emit_at - now_perf)
            now_perf = time.perf_counter()

        if self.mode == AVATAR_MODE_TALKING:
            if self.pending_talking_intro_frames:
                frame_image = self.pending_talking_intro_frames.popleft()
                self.last_frame_image = frame_image
            else:
                frame_image = self._resolve_talking_frame_image()
        else:
            if self.pending_idle_return_frames:
                frame_image = self.pending_idle_return_frames.popleft()
            else:
                frame_image = self._resolve_idle_frame_image()
            self.last_frame_image = frame_image

        frame = VideoFrame.from_ndarray(frame_image, format="bgr24")
        frame.pts = int(max(0.0, now_perf - self.clock_started_at_perf) * WEBRTC_VIDEO_CLOCK_RATE)
        frame.time_base = WEBRTC_VIDEO_TIME_BASE
        self.next_emit_at = now_perf + self.frame_interval_sec
        return frame

    def stop(self) -> None:
        self.looper.close()
        super().stop()


@dataclass
class AvatarStreamEncoder:
    process: asyncio.subprocess.Process
    input_fps: float
    started_at_perf: float
    submitted_frame_count: int
    stdout_task: asyncio.Task[Any]
    stderr_task: asyncio.Task[Any]


@dataclass
class Mp4InitializationFilterState:
    emit_initialization_segment: bool
    moof_forwarding_started: bool = False
    pending_bytes: bytearray = field(default_factory=bytearray)


async def forward_fragmented_mp4_chunk(
    chunk_sender: Callable[[bytes], Awaitable[None]],
    filter_state: Mp4InitializationFilterState,
    chunk: bytes,
) -> None:
    """
    Forward one MP4 chunk while optionally stripping duplicate initialization boxes.
    """
    if not chunk:
        return
    if filter_state.emit_initialization_segment or filter_state.moof_forwarding_started:
        await chunk_sender(chunk)
        return
    filter_state.pending_bytes.extend(chunk)
    while True:
        if len(filter_state.pending_bytes) < MP4_BOX_HEADER_SIZE:
            return
        header_size = MP4_BOX_HEADER_SIZE
        box_size = int.from_bytes(filter_state.pending_bytes[0:4], "big")
        if box_size == MP4_BOX_EXTENDED_SIZE:
            if len(filter_state.pending_bytes) < 16:
                return
            box_size = int.from_bytes(filter_state.pending_bytes[8:16], "big")
            header_size = 16
        box_type = bytes(filter_state.pending_bytes[4:8])
        if box_type == MP4_BOX_TYPE_MOOF:
            filter_state.moof_forwarding_started = True
            await chunk_sender(bytes(filter_state.pending_bytes))
            filter_state.pending_bytes.clear()
            return
        if box_size <= 0:
            filter_state.pending_bytes.clear()
            return
        if len(filter_state.pending_bytes) < max(header_size, box_size):
            return
        del filter_state.pending_bytes[:box_size]


def resolve_finished_avatar_encoder_exception(encoder: AvatarStreamEncoder | None) -> BaseException | None:
    """
    Return one terminal encoder task exception when the avatar stream lost its client.
    """
    if encoder is None:
        return None
    if not encoder.stdout_task.done():
        return None
    with contextlib.suppress(asyncio.CancelledError):
        return encoder.stdout_task.exception()
    return None


def resolve_avatar_idle_output_fps(idle_looper: "IdleVideoLooper") -> float:
    """
    Resolve the idle stream FPS from the source loop, falling back to the stream default.
    """
    return max(1.0, float(idle_looper.source_fps or AVATAR_VIDEO_OUTPUT_FPS))


def resolve_avatar_stream_output_fps(
    idle_looper: "IdleVideoLooper",
    talking_state: "AvatarTalkingFrameState | None",
) -> float:
    """
    Resolve the stable FPS used by the continuous avatar stream encoder.
    """
    return resolve_avatar_idle_output_fps(idle_looper)


class WebRtcOfferRequest(BaseModel):
    """
    Browser SDP offer payload.
    """

    sdp: str
    type: str


class AvatarWebRtcSession:
    """
    One peer connection that keeps a single continuous avatar stream alive for one browser client.
    """

    def __init__(self, session_id: str, avatar_session_id: str, rtc_configuration: RTCConfiguration) -> None:
        self.session_id = session_id
        self.avatar_session_id = avatar_session_id
        self.peer_connection = RTCPeerConnection(configuration=rtc_configuration)
        idle_started_at_ms = int(get_avatar_state_snapshot(self.avatar_session_id).get("idleStartedAtMs") or 0)
        idle_offset_sec = max(0.0, (now_ms() - idle_started_at_ms) / 1000.0) if idle_started_at_ms > 0 else 0.0
        self.avatar_video_track = AvatarVideoStreamTrack(resolve_idle_video_abs(), start_offset_sec=idle_offset_sec)
        self.idle_audio_track = SilenceAudioStreamTrack()
        self.video_sender = self.peer_connection.addTrack(self.avatar_video_track)
        self.audio_sender = self.peer_connection.addTrack(self.idle_audio_track)
        self.current_job_id = ""
        self.current_audio_player: MediaPlayer | None = None
        self.sync_task = asyncio.create_task(self.sync_avatar_state_loop())
        self.closed = False
        self._bind_peer_connection_events()

    def _bind_peer_connection_events(self) -> None:
        @self.peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if self.peer_connection.connectionState in {"closed", "failed"}:
                await self.close()

    def resolve_job_from_snapshot(self, snapshot: dict[str, Any]) -> JobRecord | None:
        """
        Resolve the currently active talking job for progressive avatar playback.
        """
        if str(snapshot.get("mode") or "") != AVATAR_MODE_TALKING:
            return None
        current_job_id = str(snapshot.get("currentJobId") or "")
        if not current_job_id:
            return None
        with JOBS_LOCK:
            job = JOBS.get(current_job_id)
        if job is None:
            return None
        stream_status = read_job_stream_status(job)
        if not is_job_ready_for_avatar(job, stream_status):
            return None
        start_frame_index = resolve_avatar_playback_start_frame_index(snapshot, stream_status)
        if not read_stream_frame_by_index(job, start_frame_index):
            return None
        return job

    def build_talking_audio_player(self, job: JobRecord, snapshot: dict[str, Any]) -> MediaPlayer:
        """
        Open one talking audio player at the correct avatar playback offset.
        """
        started_at_ms = int(snapshot.get("currentJobStartedAtMs") or 0)
        seek_offset_sec = max(0.0, (now_ms() - started_at_ms) / 1000.0) if started_at_ms > 0 else 0.0
        player_options: dict[str, str] | None = None
        if seek_offset_sec > 0:
            player_options = {
                "ss": f"{seek_offset_sec:.3f}",
            }
        return MediaPlayer(str(job.stream_audio_input_abs.resolve()), options=player_options)

    def stop_current_audio_player(self) -> None:
        """
        Stop the active talking audio player when the avatar returns to idle or the peer closes.
        """
        if self.current_audio_player is None:
            return
        if self.current_audio_player.audio is not None:
            self.current_audio_player.audio.stop()
        self.current_audio_player = None

    def switch_to_idle_tracks(self) -> None:
        """
        Route the peer back to the persistent idle tracks.
        """
        snapshot = get_avatar_state_snapshot(self.avatar_session_id)
        self.stop_current_audio_player()
        self.avatar_video_track.switch_to_idle(int(snapshot.get("idleStartedAtMs") or 0))
        self.audio_sender.replaceTrack(self.idle_audio_track)
        self.current_job_id = ""

    def switch_to_talking_tracks(self, job: JobRecord, snapshot: dict[str, Any]) -> None:
        """
        Route the peer to one synchronized progressive talking segment.
        """
        self.stop_current_audio_player()
        self.avatar_video_track.switch_to_talking(job, snapshot)
        player = self.build_talking_audio_player(job, snapshot)
        self.current_audio_player = player
        self.audio_sender.replaceTrack(player.audio or self.idle_audio_track)
        self.current_job_id = job.job_id

    async def sync_avatar_state_loop(self) -> None:
        """
        Keep the peer connection aligned with the global avatar scheduler.
        """
        try:
            while not self.closed:
                snapshot = get_avatar_state_snapshot(self.avatar_session_id)
                current_job = self.resolve_job_from_snapshot(snapshot)
                if current_job is None:
                    if self.current_job_id:
                        self.switch_to_idle_tracks()
                elif (
                    self.current_job_id != current_job.job_id
                    or self.current_audio_player is None
                    or self.avatar_video_track.mode != AVATAR_MODE_TALKING
                ):
                    self.switch_to_talking_tracks(current_job, snapshot)
                await asyncio.sleep(WEBRTC_SESSION_POLL_SLEEP_SEC)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[webrtc] avatar session sync failed ({self.session_id}): {exc}")
            await self.close()

    async def close(self) -> None:
        """
        Close one peer session and release every media resource.
        """
        if self.closed:
            return
        self.closed = True
        unregister_webrtc_session(self.session_id)
        self.stop_current_audio_player()
        if self.sync_task is not None and self.sync_task is not asyncio.current_task():
            self.sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.sync_task
        self.avatar_video_track.stop()
        self.idle_audio_track.stop()
        await self.peer_connection.close()


async def create_avatar_stream_encoder(
    chunk_sender: Callable[[bytes], Awaitable[None]],
    input_fps: float,
    canvas_size: tuple[int, int],
    timestamp_offset_sec: float,
    audio_pipe_fd: int | None,
    audio_input_path: Path | None,
    emit_initialization_segment: bool,
) -> AvatarStreamEncoder:
    """
    Start one fragmented MP4 encoder for the continuous avatar stream.
    """
    ffmpeg_process = await asyncio.create_subprocess_exec(
        *build_avatar_stream_command(
            input_fps,
            canvas_size=canvas_size,
            audio_pipe_fd=audio_pipe_fd,
            audio_input_path=audio_input_path,
            include_silent_audio=audio_pipe_fd is None and audio_input_path is None,
            timestamp_offset_sec=timestamp_offset_sec,
        ),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        pass_fds=tuple(fd for fd in (audio_pipe_fd,) if fd is not None and fd >= 0),
    )
    assert ffmpeg_process.stdout is not None
    assert ffmpeg_process.stderr is not None
    output_filter_state = Mp4InitializationFilterState(
        emit_initialization_segment=emit_initialization_segment,
    )

    async def pump_mp4_stdout() -> None:
        while True:
            chunk = await ffmpeg_process.stdout.read(VIDEO_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            await forward_fragmented_mp4_chunk(chunk_sender, output_filter_state, chunk)

    async def drain_stderr() -> None:
        while True:
            chunk = await ffmpeg_process.stderr.read(VIDEO_STREAM_CHUNK_SIZE)
            if not chunk:
                break

    return AvatarStreamEncoder(
        process=ffmpeg_process,
        input_fps=max(1.0, float(input_fps)),
        started_at_perf=time.perf_counter(),
        submitted_frame_count=0,
        stdout_task=asyncio.create_task(pump_mp4_stdout()),
        stderr_task=asyncio.create_task(drain_stderr()),
    )


async def stop_avatar_stream_encoder(encoder: AvatarStreamEncoder | None) -> float:
    """
    Stop one avatar stream encoder and return the media duration already emitted.
    """
    if encoder is None:
        return 0.0
    elapsed_sec = 0.0
    if encoder.submitted_frame_count > 0 and encoder.input_fps > 0:
        elapsed_sec = float(encoder.submitted_frame_count) / float(encoder.input_fps)
    with contextlib.suppress(Exception):
        if encoder.process.stdin is not None:
            encoder.process.stdin.close()
    graceful_shutdown_completed = False
    try:
        await asyncio.wait_for(
            asyncio.gather(encoder.stdout_task, encoder.stderr_task, return_exceptions=True),
            timeout=AVATAR_STREAM_ENCODER_DRAIN_TIMEOUT_SEC,
        )
        graceful_shutdown_completed = True
    except asyncio.TimeoutError:
        graceful_shutdown_completed = False
    if encoder.process.returncode is None:
        if graceful_shutdown_completed:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    encoder.process.wait(),
                    timeout=AVATAR_STREAM_ENCODER_EXIT_TIMEOUT_SEC,
                )
        if encoder.process.returncode is None:
            terminate_process_tree(encoder.process)
            for task in (encoder.stdout_task, encoder.stderr_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(encoder.stdout_task, encoder.stderr_task, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(encoder.process.wait(), timeout=AVATAR_STREAM_ENCODER_EXIT_TIMEOUT_SEC)
    return elapsed_sec


def mark_avatar_job_finished(job: JobRecord) -> None:
    """
    Mark the active avatar job as consumed and immediately return the scheduler to idle.
    """
    with JOBS_LOCK:
        job.avatar_play_finished_at_ms = now_ms()
    with AVATAR_SESSION_STATES_LOCK:
        session_state = AVATAR_SESSION_STATES.get(job.avatar_session_id)
        is_current_job = session_state is not None and session_state.current_job_id == job.job_id
    if is_current_job:
        activate_avatar_idle_mode(job.avatar_session_id)


def open_avatar_audio_wave_reader(audio_path: Path, seek_offset_sec: float) -> wave.Wave_read | None:
    """
    Open one normalized WAV reader for continuous avatar audio playback.
    """
    if audio_path.suffix.lower() != ".wav" or not audio_path.exists():
        return None
    try:
        reader = wave.open(str(audio_path), "rb")
    except (OSError, wave.Error):
        return None
    if (
        reader.getframerate() != VIDEO_STREAM_AUDIO_SAMPLE_RATE_INT
        or reader.getnchannels() != VIDEO_STREAM_AUDIO_CHANNELS_INT
        or reader.getsampwidth() != VIDEO_STREAM_AUDIO_SAMPLE_WIDTH_BYTES
    ):
        reader.close()
        return None
    seek_frame_index = max(0, int(float(seek_offset_sec) * float(reader.getframerate())))
    if reader.getnframes() > 0:
        seek_frame_index = min(seek_frame_index, max(0, reader.getnframes() - 1))
    with contextlib.suppress(Exception):
        reader.setpos(seek_frame_index)
    return reader


async def pump_continuous_avatar_audio(
    avatar_session_id: str,
    audio_write_fd: int,
    stop_event: asyncio.Event,
) -> None:
    """
    Feed one continuous PCM audio track into the persistent avatar encoder.
    """
    current_job_id = ""
    current_wave_reader: wave.Wave_read | None = None
    next_emit_at = time.perf_counter()
    silence_chunk = bytes(VIDEO_STREAM_AUDIO_CHUNK_BYTES)
    try:
        while not stop_event.is_set():
            snapshot = get_avatar_state_snapshot(avatar_session_id)
            desired_job_id = str(snapshot.get("currentJobId") or "") if snapshot.get("mode") == AVATAR_MODE_TALKING else ""
            if desired_job_id != current_job_id:
                if current_wave_reader is not None:
                    current_wave_reader.close()
                    current_wave_reader = None
                current_job_id = desired_job_id
                if current_job_id:
                    with JOBS_LOCK:
                        current_job = JOBS.get(current_job_id)
                    if current_job is not None:
                        started_at_ms = int(snapshot.get("currentJobStartedAtMs") or 0)
                        current_status = read_job_stream_status(current_job)
                        playback_fps = resolve_stream_playback_fps(current_status)
                        start_frame_index = resolve_avatar_playback_start_frame_index(snapshot, current_status)
                        if playback_fps > 0 and start_frame_index > 0:
                            seek_offset_sec = max(0.0, float(start_frame_index - 1) / float(playback_fps))
                        else:
                            seek_offset_sec = max(0.0, (now_ms() - started_at_ms) / 1000.0) if started_at_ms > 0 else 0.0
                        current_wave_reader = open_avatar_audio_wave_reader(current_job.stream_audio_input_abs, seek_offset_sec)

            audio_chunk = silence_chunk
            if current_wave_reader is not None:
                try:
                    audio_chunk = current_wave_reader.readframes(VIDEO_STREAM_AUDIO_CHUNK_SAMPLES)
                except (OSError, wave.Error):
                    audio_chunk = b""
                if not audio_chunk:
                    current_wave_reader.close()
                    current_wave_reader = None
                    audio_chunk = silence_chunk
                elif len(audio_chunk) < VIDEO_STREAM_AUDIO_CHUNK_BYTES:
                    audio_chunk = audio_chunk + bytes(VIDEO_STREAM_AUDIO_CHUNK_BYTES - len(audio_chunk))

            try:
                await asyncio.to_thread(os.write, audio_write_fd, audio_chunk)
            except (BrokenPipeError, OSError):
                break
            next_emit_at += float(VIDEO_STREAM_AUDIO_CHUNK_SAMPLES) / float(VIDEO_STREAM_AUDIO_SAMPLE_RATE_INT)
            sleep_duration = next_emit_at - time.perf_counter()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            else:
                next_emit_at = time.perf_counter()
    finally:
        if current_wave_reader is not None:
            current_wave_reader.close()


async def stop_continuous_avatar_audio(
    audio_task: asyncio.Task[Any] | None,
    stop_event: asyncio.Event | None,
    audio_write_fd: int | None,
) -> None:
    """
    Stop one continuous avatar audio pump and release its pipe.
    """
    if stop_event is not None:
        stop_event.set()
    if audio_task is not None:
        audio_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, BrokenPipeError, OSError):
            await audio_task
    if audio_write_fd is not None and audio_write_fd >= 0:
        with contextlib.suppress(OSError):
            os.close(audio_write_fd)


async def stream_continuous_avatar_video(
    avatar_session_id: str,
    chunk_sender: Callable[[bytes], Awaitable[None]],
    should_stop: Callable[[], Awaitable[bool]] | None = None,
) -> None:
    """
    Emit one continuous avatar MP4 stream through an abstract byte sink.
    """
    ensure_avatar_session_state(avatar_session_id)
    ensure_avatar_worker_started()
    idle_looper = IdleVideoLooper(resolve_idle_video_abs())
    canvas_size = resolve_avatar_stream_canvas_size(idle_looper.canvas_size)
    use_continuous_audio_pipe = os.name != "nt"
    timeline_offset_sec = 0.0
    emit_initialization_segment = True
    encoder: AvatarStreamEncoder | None = None
    encoder_output_fps = resolve_avatar_idle_output_fps(idle_looper)
    audio_write_fd: int | None = None
    audio_task: asyncio.Task[Any] | None = None
    audio_stop_event: asyncio.Event | None = None
    talking_state: AvatarTalkingFrameState | None = None
    talking_job: JobRecord | None = None
    pending_idle_return_frames: deque[np.ndarray] = deque()
    pending_talking_intro_frames: deque[np.ndarray] = deque()
    frame_interval_sec = 1.0 / max(1.0, encoder_output_fps)
    next_emit_at = time.perf_counter()
    last_output_frame_image: np.ndarray | None = None

    async def start_output_pipeline(target_fps: float) -> None:
        nonlocal encoder
        nonlocal encoder_output_fps
        nonlocal audio_write_fd
        nonlocal audio_task
        nonlocal audio_stop_event
        nonlocal emit_initialization_segment
        nonlocal frame_interval_sec
        nonlocal next_emit_at
        audio_read_fd: int | None = None
        if use_continuous_audio_pipe:
            audio_read_fd, audio_write_fd = os.pipe()
        else:
            audio_write_fd = None
        try:
            encoder_output_fps = max(1.0, float(target_fps))
            encoder = await create_avatar_stream_encoder(
                chunk_sender,
                encoder_output_fps,
                canvas_size=canvas_size,
                timestamp_offset_sec=timeline_offset_sec,
                audio_pipe_fd=audio_read_fd,
                audio_input_path=None,
                emit_initialization_segment=emit_initialization_segment,
            )
        finally:
            if audio_read_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(audio_read_fd)
        emit_initialization_segment = False
        frame_interval_sec = 1.0 / encoder_output_fps
        next_emit_at = time.perf_counter()
        if audio_write_fd is not None:
            audio_stop_event = asyncio.Event()
            audio_task = asyncio.create_task(
                pump_continuous_avatar_audio(avatar_session_id, audio_write_fd, audio_stop_event)
            )
        else:
            audio_stop_event = None
            audio_task = None

    async def ensure_output_pipeline(target_fps: float) -> None:
        nonlocal encoder
        nonlocal timeline_offset_sec

        safe_target_fps = max(1.0, float(target_fps))
        if encoder is None:
            await start_output_pipeline(safe_target_fps)
            return
        if fps_values_match(encoder_output_fps, safe_target_fps):
            return
        timeline_offset_sec += await stop_avatar_stream_encoder(encoder)
        await stop_continuous_avatar_audio(audio_task, audio_stop_event, audio_write_fd)
        encoder = None
        await start_output_pipeline(safe_target_fps)

    def arm_idle_return_transition(last_frame_image: np.ndarray | None) -> None:
        """
        Prepare one short return-to-idle bridge for the continuous avatar stream.
        """
        pending_idle_return_frames.clear()
        pending_talking_intro_frames.clear()
        pending_idle_return_frames.extend(
            build_avatar_return_to_idle_frame_sequence(
                start_frame_image=last_frame_image,
                canvas_size=canvas_size,
                frames_per_second=resolve_avatar_idle_output_fps(idle_looper),
            )
        )
        if pending_idle_return_frames:
            idle_looper.seek_to_frame(0)
            idle_looper.read_next_frame()

    def arm_talking_intro_transition(
        start_frame_image: np.ndarray | None,
        job: JobRecord,
        state: AvatarTalkingFrameState,
    ) -> None:
        """
        Prepare one short idle-to-talking bridge before the avatar starts emitting talking frames.
        """
        pending_talking_intro_frames.clear()
        stream_output_fps = resolve_avatar_stream_output_fps(idle_looper, state)
        if resolve_avatar_idle_to_talking_frame_count(stream_output_fps) <= 0:
            return
        intro_frame_image, is_finished = resolve_avatar_talking_frame(
            job,
            state,
            canvas_size,
            output_fps=stream_output_fps,
        )
        if is_finished:
            mark_avatar_job_finished(job)
            return
        pending_talking_intro_frames.extend(
            build_avatar_idle_to_talking_frame_sequence(
                start_frame_image=start_frame_image,
                end_frame_image=intro_frame_image,
                canvas_size=canvas_size,
                frames_per_second=stream_output_fps,
            )
        )

    await start_output_pipeline(encoder_output_fps)

    try:
        while True:
            if should_stop is not None and await should_stop():
                break
            encoder_exception = resolve_finished_avatar_encoder_exception(encoder)
            if encoder_exception is not None:
                raise encoder_exception

            now_perf = time.perf_counter()
            if now_perf < next_emit_at:
                await asyncio.sleep(min(VIDEO_STREAM_POLL_SLEEP_SEC, next_emit_at - now_perf))
                continue

            snapshot = get_avatar_state_snapshot(avatar_session_id)
            desired_job_id = str(snapshot["currentJobId"] or "") if snapshot["mode"] == AVATAR_MODE_TALKING else ""
            if talking_state is not None and desired_job_id != talking_state.job_id:
                arm_idle_return_transition(talking_state.last_frame_image)
                talking_state = None
                talking_job = None
            if talking_state is None and desired_job_id:
                with JOBS_LOCK:
                    candidate_job = JOBS.get(desired_job_id)
                if candidate_job is not None:
                    candidate_status = read_job_stream_status(candidate_job)
                    start_frame_index = resolve_avatar_playback_start_frame_index(snapshot, candidate_status)
                    talking_job = candidate_job
                    pending_idle_return_frames.clear()
                    pending_talking_intro_frames.clear()
                    talking_state = AvatarTalkingFrameState(
                        job_id=desired_job_id,
                        start_frame_index=max(1, start_frame_index),
                        next_frame_index=max(1, start_frame_index),
                        playback_started_at_perf=time.perf_counter(),
                    )
                    talking_state.playback_fps = max(
                        1.0,
                        float(resolve_stream_playback_fps(candidate_status) or resolve_avatar_idle_output_fps(idle_looper)),
                    )
                    await ensure_output_pipeline(resolve_avatar_stream_output_fps(idle_looper, talking_state))
                    arm_talking_intro_transition(last_output_frame_image, talking_job, talking_state)
            frame_image = None
            if talking_state is not None and talking_job is not None:
                await ensure_output_pipeline(resolve_avatar_stream_output_fps(idle_looper, talking_state))
                if pending_talking_intro_frames:
                    frame_image = pending_talking_intro_frames.popleft()
                else:
                    frame_image, is_finished = resolve_avatar_talking_frame(
                        talking_job,
                        talking_state,
                        canvas_size,
                        output_fps=encoder_output_fps,
                    )
                    if is_finished:
                        arm_idle_return_transition(talking_state.last_frame_image)
                        mark_avatar_job_finished(talking_job)
                        talking_state = None
                        talking_job = None
                        frame_image = None

            if frame_image is None:
                await ensure_output_pipeline(resolve_avatar_idle_output_fps(idle_looper))
                if pending_idle_return_frames:
                    frame_image = pending_idle_return_frames.popleft()
                else:
                    idle_frame = idle_looper.read_next_frame()
                    frame_image = fit_frame_to_canvas(idle_frame, canvas_size)

            if frame_image is not None:
                last_output_frame_image = frame_image
                try:
                    assert encoder is not None and encoder.process.stdin is not None
                    encoder.process.stdin.write(np.ascontiguousarray(frame_image, dtype=np.uint8).tobytes())
                    await encoder.process.stdin.drain()
                    encoder.submitted_frame_count += 1
                except (BrokenPipeError, ConnectionResetError, RuntimeError):
                    await stop_continuous_avatar_audio(audio_task, audio_stop_event, audio_write_fd)
                    audio_task = None
                    audio_stop_event = None
                    audio_write_fd = None
                    timeline_offset_sec += await stop_avatar_stream_encoder(encoder)
                    encoder = None
                    await start_output_pipeline(resolve_avatar_stream_output_fps(idle_looper, talking_state))
                    next_emit_at = time.perf_counter()
                    continue

            next_emit_at += frame_interval_sec
            if next_emit_at < now_perf - frame_interval_sec:
                next_emit_at = now_perf
    finally:
        idle_looper.close()
        await stop_continuous_avatar_audio(audio_task, audio_stop_event, audio_write_fd)
        await stop_avatar_stream_encoder(encoder)


async def capture_continuous_avatar_video(
    output_path: Path,
    duration_sec: float,
    avatar_session_id: str,
) -> Path:
    """
    Capture a finite slice of the continuous avatar stream into one playable MP4 file.
    """
    ensure_avatar_worker_started()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + max(AVATAR_CAPTURE_MIN_DURATION_SEC, float(duration_sec))
    if output_path.exists():
        output_path.unlink()
    output_handle = output_path.open("wb")

    async def send_chunk(chunk: bytes) -> None:
        output_handle.write(chunk)

    async def should_stop() -> bool:
        return time.perf_counter() >= deadline

    try:
        await stream_continuous_avatar_video(
            avatar_session_id=avatar_session_id,
            chunk_sender=send_chunk,
            should_stop=should_stop,
        )
    finally:
        output_handle.flush()
        output_handle.close()
    return output_path


JOBS_LOCK = threading.Lock()
JOBS: dict[str, JobRecord] = {}
JOB_STREAM_CACHE_LOCK = threading.Lock()
JOB_STREAM_CACHES: dict[str, JobStreamMemoryCache] = {}
JOB_QUEUE_CONDITION = threading.Condition()
JOB_QUEUE: deque[str] = deque()
JOB_WORKER_LOCK = threading.Lock()
JOB_WORKER_THREAD: threading.Thread | None = None
SOURCE_TEMPLATE_BUILD_TASKS_LOCK = threading.Lock()
SOURCE_TEMPLATE_BUILD_TASKS: dict[str, SourceTemplateBuildTask] = {}
QUEUE_ACTIVE_TASK_LOCK = threading.Lock()
QUEUE_ACTIVE_TASK_ID = ""
QUEUE_ACTIVE_TASK_KIND = ""
WEBRTC_SESSIONS_LOCK = threading.Lock()
WEBRTC_SESSIONS: dict[str, AvatarWebRtcSession] = {}
WARMUP_LOCK = threading.Lock()
WARMUP_LAST_STARTED_AT_MS = 0
WARMUP_RUNNING = False
WARMUP_PHASE = "idle"
WARMUP_PROGRESS = 0.0
WARMUP_MESSAGE = ""
WARMUP_ERROR = ""
RUNTIME_RESTART_LOCK = threading.Lock()
RUNTIME_RESTARTING = False
RUNTIME_RESTART_REQUESTED_AT_MS = 0
AVATAR_SESSION_STATES_LOCK = threading.Lock()
AVATAR_WORKER_LOCK = threading.Lock()
AVATAR_WORKER_THREAD: threading.Thread | None = None
AVATAR_SESSION_STATES: dict[str, AvatarSessionState] = {}


def make_job_id() -> str:
    """
    Build short unique job identifier.
    """
    return f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def register_job(job: JobRecord) -> None:
    """
    Register job in memory.
    """
    with JOBS_LOCK:
        JOBS[job.job_id] = job
    with JOB_STREAM_CACHE_LOCK:
        existing_cache = JOB_STREAM_CACHES.get(job.job_id)
        if existing_cache is None:
            JOB_STREAM_CACHES[job.job_id] = JobStreamMemoryCache(
                job_id=job.job_id,
                stream_shm_prefix=job.stream_shm_prefix,
            )
        else:
            existing_cache.last_accessed_at_ms = now_ms()
    prune_expired_job_stream_caches()


def enqueue_job(job_id: str) -> None:
    """
    Enqueue job identifier for background worker execution.
    """
    with JOB_QUEUE_CONDITION:
        JOB_QUEUE.append(job_id)
        JOB_QUEUE_CONDITION.notify()


def register_source_template_build_task(task: SourceTemplateBuildTask) -> None:
    """
    Register one queued template-build task.
    """
    with SOURCE_TEMPLATE_BUILD_TASKS_LOCK:
        SOURCE_TEMPLATE_BUILD_TASKS[task.task_id] = task


def pop_source_template_build_task(task_id: str) -> SourceTemplateBuildTask | None:
    """
    Remove one completed template-build task from memory.
    """
    with SOURCE_TEMPLATE_BUILD_TASKS_LOCK:
        return SOURCE_TEMPLATE_BUILD_TASKS.pop(task_id, None)


def get_source_template_build_task(task_id: str) -> SourceTemplateBuildTask | None:
    """
    Fetch one template-build task by identifier.
    """
    with SOURCE_TEMPLATE_BUILD_TASKS_LOCK:
        return SOURCE_TEMPLATE_BUILD_TASKS.get(task_id)


def set_active_queue_task(task_id: str, task_kind: str) -> None:
    """
    Track the queue task currently consuming the single worker slot.
    """
    global QUEUE_ACTIVE_TASK_ID
    global QUEUE_ACTIVE_TASK_KIND
    with QUEUE_ACTIVE_TASK_LOCK:
        QUEUE_ACTIVE_TASK_ID = str(task_id or "")
        QUEUE_ACTIVE_TASK_KIND = str(task_kind or "")


def clear_active_queue_task(task_id: str, task_kind: str) -> None:
    """
    Clear one active queue task marker when the expected task finishes.
    """
    global QUEUE_ACTIVE_TASK_ID
    global QUEUE_ACTIVE_TASK_KIND
    with QUEUE_ACTIVE_TASK_LOCK:
        if QUEUE_ACTIVE_TASK_ID == str(task_id or "") and QUEUE_ACTIVE_TASK_KIND == str(task_kind or ""):
            QUEUE_ACTIVE_TASK_ID = ""
            QUEUE_ACTIVE_TASK_KIND = ""


def get_active_queue_task_snapshot() -> tuple[str, str]:
    """
    Return active queue task identifier and kind.
    """
    with QUEUE_ACTIVE_TASK_LOCK:
        return QUEUE_ACTIVE_TASK_ID, QUEUE_ACTIVE_TASK_KIND


def queue_position(job_id: str) -> int:
    """
    Return 1-based queue position, or 0 when job is not queued.
    """
    with JOB_QUEUE_CONDITION:
        for index, queued_job_id in enumerate(JOB_QUEUE):
            if queued_job_id == job_id:
                return index + 1
    return 0


def get_running_job_record() -> JobRecord | None:
    """
    Return currently running job record when present.
    """
    with JOBS_LOCK:
        job_records = list(JOBS.values())
    for job in sorted(job_records, key=lambda item: item.created_at_ms):
        process = job.process
        if process is not None and process.poll() is None:
            return job
    return None


def get_head_queued_job_id() -> str:
    """
    Return first queued job identifier or empty string.
    """
    with JOB_QUEUE_CONDITION:
        if not JOB_QUEUE:
            return ""
        return str(JOB_QUEUE[0])


def get_job(job_id: str, avatar_session_id: str | None = None) -> JobRecord:
    """
    Fetch job record or raise HTTP 404.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or (avatar_session_id is not None and job.avatar_session_id != avatar_session_id):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def launch_logged_subprocess(
    command: list[str],
    cwd: Path,
    log_abs: Path,
    label: str,
) -> tuple[subprocess.Popen, Any, threading.Thread]:
    """
    Launch one subprocess whose combined output is mirrored to terminal and log file.
    """
    log_abs.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_abs.open("a", encoding="utf-8", errors="replace", buffering=1)
    command_text = format_command_for_log(command)
    launch_message = f"launch cwd={cwd} command={command_text}"
    emit_runtime_log(label, launch_message)
    log_handle.write(f"{launch_message}\n")
    log_handle.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception:
        finalize_logged_subprocess(log_handle, None)
        raise
    pump_thread = threading.Thread(
        target=pump_subprocess_output,
        args=(process.stdout, log_handle, label),
        daemon=True,
    )
    pump_thread.start()
    return process, log_handle, pump_thread


def finalize_logged_subprocess(log_handle: Any, pump_thread: threading.Thread | None) -> None:
    """
    Flush and close logging resources attached to one launched subprocess.
    """
    if pump_thread is not None:
        pump_thread.join(timeout=5.0)
    if log_handle is not None:
        try:
            log_handle.flush()
        except OSError:
            pass
        try:
            log_handle.close()
        except OSError:
            pass


def build_source_template_pack_command(source_abs: Path, template_pack_path: Path) -> list[str]:
    """
    Build runner command for one queued source template pack build.
    """
    command = [
        str(RUNNER_PYTHON),
        str(RUNNER_SCRIPT),
        "--backend",
        DEFAULT_BACKEND,
        "--trt-runtime",
        DEFAULT_TRT_RUNTIME,
        "--trt-precision",
        DEFAULT_TRT_PRECISION,
        "--source-frame",
        str(source_abs),
        "--source-template-pack-output",
        str(template_pack_path),
        "--build-source-template-pack",
        "--animation-region",
        DEFAULT_ANIMATION_REGION,
    ]
    if not DEFAULT_PASTE_BACK_ENABLED:
        command.append("--no-paste-back")
    if not DEFAULT_STITCHING_ENABLED:
        command.append("--no-stitching")
    if not DEFAULT_RELATIVE_MOTION_ENABLED:
        command.append("--no-relative-motion")
    return command


def finish_job(job: JobRecord, exit_code: int) -> None:
    """
    Mark job completed and close resources.
    """
    with JOBS_LOCK:
        job.process = None
        job.exit_code = int(exit_code)
        job.finished_at_ms = now_ms()
        job.log_handle = None
    prune_expired_job_stream_caches()


def run_job(job: JobRecord) -> None:
    """
    Execute one queued job in the background worker.
    """
    command = build_runner_command(job)
    process, log_handle, pump_thread = launch_logged_subprocess(
        command=command,
        cwd=PROJECT_ROOT,
        log_abs=job.log_abs,
        label=f"job {job.job_id}",
    )
    with JOBS_LOCK:
        job.started_at_ms = now_ms()
        job.process = process
        job.log_handle = log_handle
    try:
        exit_code = process.wait()
    except Exception:
        exit_code = -1
    finalize_logged_subprocess(log_handle, pump_thread)
    emit_runtime_log(
        f"job {job.job_id}",
        f"completed exit_code={exit_code} source_kind={resolve_source_input_kind(job.source_frame_abs)}",
    )
    finish_job(job, exit_code)


def run_source_template_build_task(task: SourceTemplateBuildTask) -> None:
    """
    Execute one queued source template build using the same single-worker queue.
    """
    command = build_source_template_pack_command(task.source_abs, task.template_pack_path)
    task.started_at_ms = now_ms()
    emit_runtime_log(
        f"template {task.task_id}",
        (
            "starting source template build "
            f"template={task.template_name} source={task.source_abs} source_kind={resolve_source_input_kind(task.source_abs)}"
        ),
    )
    try:
        process, log_handle, pump_thread = launch_logged_subprocess(
            command=command,
            cwd=PROJECT_ROOT,
            log_abs=task.log_abs,
            label=f"template {task.task_id}",
        )
        try:
            exit_code = process.wait()
        except Exception:
            exit_code = -1
        finalize_logged_subprocess(log_handle, pump_thread)
        task.exit_code = int(exit_code)
        task.finished_at_ms = now_ms()
        if exit_code != 0:
            task.error = (
                "Source template build failed. "
                f"log={task.log_abs} tail={read_text_file_tail(task.log_abs, 1600)}"
            )
            emit_runtime_log(f"template {task.task_id}", f"failed exit_code={exit_code}")
            return
        if not task.template_pack_path.exists() or not is_source_template_pack_path(task.template_pack_path):
            task.error = (
                "Source template build finished without one valid template pack. "
                f"expected={task.template_pack_path}"
            )
            emit_runtime_log(f"template {task.task_id}", task.error)
            return
        payload_valid, payload_error = validate_source_template_pack_payload(task.template_pack_path)
        if not payload_valid:
            task.error = (
                "Source template build produced one invalid template payload. "
                f"template={task.template_pack_path} error={payload_error}"
            )
            emit_runtime_log(f"template {task.task_id}", task.error)
            return
        task.result = {
            "item": build_source_template_pack_record(task.template_pack_path),
            "logPath": str(task.log_abs),
        }
        emit_runtime_log(
            f"template {task.task_id}",
            f"completed template={task.template_pack_path.name} log={task.log_abs}",
        )
    except Exception as exc:
        task.finished_at_ms = now_ms()
        task.error = f"Source template build crashed: {exc}"
        emit_runtime_log(f"template {task.task_id}", task.error)
    finally:
        task.done_event.set()


def job_worker_loop() -> None:
    """
    Persistent single-worker loop that executes queued jobs sequentially.
    """
    while True:
        with JOB_QUEUE_CONDITION:
            while not JOB_QUEUE:
                JOB_QUEUE_CONDITION.wait()
            queue_item_id = JOB_QUEUE.popleft()
        template_task = get_source_template_build_task(queue_item_id)
        if template_task is not None:
            set_active_queue_task(template_task.task_id, "template_build")
            try:
                run_source_template_build_task(template_task)
            finally:
                clear_active_queue_task(template_task.task_id, "template_build")
            continue
        with JOBS_LOCK:
            job = JOBS.get(queue_item_id)
        if job is None:
            continue
        if job.exit_code is not None:
            continue
        set_active_queue_task(job.job_id, "avatar_job")
        try:
            try:
                run_job(job)
            except Exception as exc:
                emit_runtime_log("queue", f"avatar job crashed during launch job_id={job.job_id} error={exc}")
                finish_job(job, -1)
        finally:
            clear_active_queue_task(job.job_id, "avatar_job")


def ensure_job_worker_started() -> None:
    """
    Start persistent job worker thread once.
    """
    global JOB_WORKER_THREAD
    with JOB_WORKER_LOCK:
        if JOB_WORKER_THREAD is not None and JOB_WORKER_THREAD.is_alive():
            return
        JOB_WORKER_THREAD = threading.Thread(target=job_worker_loop, daemon=True)
        JOB_WORKER_THREAD.start()


def register_webrtc_session(session: AvatarWebRtcSession) -> None:
    """
    Register one live WebRTC avatar session.
    """
    with WEBRTC_SESSIONS_LOCK:
        WEBRTC_SESSIONS[session.session_id] = session


def unregister_webrtc_session(session_id: str) -> None:
    """
    Remove one WebRTC avatar session from the live registry.
    """
    with WEBRTC_SESSIONS_LOCK:
        WEBRTC_SESSIONS.pop(session_id, None)


async def close_webrtc_session(session: AvatarWebRtcSession | None) -> None:
    """
    Close one WebRTC session and unregister it.
    """
    if session is None:
        return
    unregister_webrtc_session(session.session_id)
    await session.close()


async def close_all_webrtc_sessions() -> None:
    """
    Close every active WebRTC session during API shutdown.
    """
    with WEBRTC_SESSIONS_LOCK:
        sessions = list(WEBRTC_SESSIONS.values())
        WEBRTC_SESSIONS.clear()
    for session in sessions:
        with contextlib.suppress(Exception):
            await session.close()


def normalize_avatar_session_id(raw_value: Any) -> str:
    """
    Normalize one public avatar session identifier.
    """
    normalized_value = str(raw_value or "").strip()
    if not normalized_value:
        raise ValueError("Missing avatar session id.")
    if len(normalized_value) > AVATAR_SESSION_ID_MAX_LENGTH:
        raise ValueError("Avatar session id is too long.")
    if AVATAR_SESSION_ID_PATTERN.fullmatch(normalized_value) is None:
        raise ValueError("Avatar session id contains unsupported characters.")
    return normalized_value


def issue_avatar_session_id() -> str:
    """
    Issue one new public avatar session identifier for client use.
    """
    return normalize_avatar_session_id(f"avatar_{uuid.uuid4().hex}")


def ensure_avatar_session_state(avatar_session_id: str) -> AvatarSessionState:
    """
    Create or fetch one avatar scheduler state bucket.
    """
    normalized_avatar_session_id = normalize_avatar_session_id(avatar_session_id)
    with AVATAR_SESSION_STATES_LOCK:
        session_state = AVATAR_SESSION_STATES.get(normalized_avatar_session_id)
        if session_state is None:
            session_state = AvatarSessionState(avatar_session_id=normalized_avatar_session_id)
            AVATAR_SESSION_STATES[normalized_avatar_session_id] = session_state
        return session_state


def get_avatar_session_id_from_request(request: Request) -> str:
    """
    Resolve the avatar interaction session id from one HTTP request.
    """
    raw_session_id = request.headers.get(AVATAR_SESSION_ID_HEADER_NAME) or request.query_params.get(
        AVATAR_SESSION_ID_QUERY_KEY
    )
    try:
        return ensure_avatar_session_state(str(raw_session_id or "")).avatar_session_id
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_optional_avatar_session_id_from_request(request: Request) -> str | None:
    """
    Resolve the optional avatar interaction session id from one HTTP request.
    """
    raw_session_id = request.headers.get(AVATAR_SESSION_ID_HEADER_NAME) or request.query_params.get(
        AVATAR_SESSION_ID_QUERY_KEY
    )
    if raw_session_id is None or not str(raw_session_id).strip():
        return None
    try:
        return ensure_avatar_session_state(str(raw_session_id or "")).avatar_session_id
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_avatar_session_id_from_websocket(websocket: WebSocket) -> str:
    """
    Resolve the avatar interaction session id from one websocket handshake.
    """
    raw_session_id = websocket.headers.get(AVATAR_SESSION_ID_HEADER_NAME) or websocket.query_params.get(
        AVATAR_SESSION_ID_QUERY_KEY
    )
    return ensure_avatar_session_state(str(raw_session_id or "")).avatar_session_id


async def wait_for_ice_gathering_complete(peer_connection: RTCPeerConnection) -> None:
    """
    Wait until aiortc finishes gathering local ICE candidates for the SDP answer.
    """
    deadline = time.perf_counter() + WEBRTC_ICE_GATHERING_TIMEOUT_SEC
    while peer_connection.iceGatheringState != "complete" and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)


def get_avatar_state_snapshot(avatar_session_id: str) -> dict[str, Any]:
    """
    Read current avatar scheduler state atomically for one interaction session.
    """
    session_state = ensure_avatar_session_state(avatar_session_id)
    with AVATAR_SESSION_STATES_LOCK:
        current_job_id = session_state.current_job_id
        current_mode = session_state.mode
        current_started_at_ms = session_state.current_job_started_at_ms
        current_ends_at_ms = session_state.current_job_ends_at_ms
        sequence = session_state.sequence
        idle_started_at_ms = session_state.idle_started_at_ms
    current_job = None
    current_job_stream_status = None
    if current_job_id:
        with JOBS_LOCK:
            current_job = JOBS.get(current_job_id)
        if current_job is not None and current_job.avatar_session_id != avatar_session_id:
            current_job = None
        if current_job is not None:
            current_job_stream_status = read_job_stream_status(current_job)
    buffered_start_progress = (
        resolve_avatar_minimum_ready_progress(current_job.audio_duration_sec)
        if current_job is not None
        else 0.0
    )
    return {
        "mode": current_mode,
        "sequence": sequence,
        "currentJobId": current_job_id,
        "currentJobStartedAtMs": current_started_at_ms,
        "currentJobEndsAtMs": current_ends_at_ms,
        "idleStartedAtMs": idle_started_at_ms,
        "idleVideoUrl": resolve_idle_video_url(),
        "bufferedStartProgress": buffered_start_progress,
        "avatarSessionId": avatar_session_id,
        "currentJobVideoWsUrl": f"/ws/jobs/{current_job_id}/video" if current_job_id else "",
        "currentJobStatusWsUrl": f"/ws/jobs/{current_job_id}" if current_job_id else "",
        "currentJobAudioDurationSec": current_job.audio_duration_sec if current_job is not None else 0.0,
        "currentJobSourceFrameUrl": build_public_file_url(current_job.source_frame_abs) if current_job is not None else "",
        "currentJobSourceMediaType": (
            "video" if current_job is not None and is_video_backed_source_path(current_job.source_frame_abs) else "image"
        ),
        "currentJobPreviewComposition": (
            build_public_preview_composition_payload(current_job, current_job_stream_status)
            if current_job is not None
            else None
        ),
    }


def count_pending_avatar_jobs(avatar_session_id: str, current_job_id: str) -> int:
    """
    Count queued or ready avatar jobs still pending for one interaction session.
    """
    with JOBS_LOCK:
        job_records = list(JOBS.values())
    pending_count = 0
    for job in job_records:
        if job.avatar_session_id != avatar_session_id:
            continue
        if job.job_id == current_job_id:
            continue
        if job.avatar_play_finished_at_ms is not None:
            continue
        stream_status = read_job_stream_status(job)
        state = determine_job_state(job, stream_status)
        if state in {"error", "canceled"}:
            continue
        pending_count += 1
    return pending_count


def build_avatar_payload(avatar_session_id: str) -> dict[str, Any]:
    """
    Build one public avatar state payload for the requested interaction session.
    """
    snapshot = get_avatar_state_snapshot(avatar_session_id)
    queue_depth = count_pending_avatar_jobs(avatar_session_id, str(snapshot["currentJobId"] or ""))
    running_job = get_running_job_record()
    current_job: JobRecord | None = None
    current_job_id = str(snapshot["currentJobId"] or "")
    if current_job_id:
        with JOBS_LOCK:
            current_job = JOBS.get(current_job_id)
        if current_job is not None and current_job.avatar_session_id != avatar_session_id:
            current_job = None
    with WEBRTC_SESSIONS_LOCK:
        active_webrtc_sessions = sum(
            1 for session in WEBRTC_SESSIONS.values() if session.avatar_session_id == avatar_session_id
        )
    return {
        **snapshot,
        "queueDepth": queue_depth,
        "runningJobId": running_job.job_id if running_job is not None else "",
        "idleVideoAvailable": bool(snapshot["idleVideoUrl"]),
        "avatarVideoWsUrl": "/ws/avatar/video",
        "avatarVideoHttpUrl": "/api/avatar/video.mp4",
        "avatarWebrtcOfferUrl": WEBRTC_OFFER_API_PATH,
        "avatarTransport": AVATAR_TRANSPORT_WEBSOCKET,
        "webrtcEnabled": False,
        "webrtcIceServers": WEBRTC_ICE_SERVER_PAYLOADS,
        "webrtcIceTransportPolicy": DEFAULT_WEBRTC_ICE_TRANSPORT_POLICY,
        "activeWebrtcSessions": active_webrtc_sessions,
        "currentJobDrivingMediaUrl": build_public_file_url(current_job.audio_input_abs) if current_job is not None else "",
        "status": "ok",
    }


def build_public_avatar_health_payload() -> dict[str, Any]:
    """
    Build one session-neutral avatar payload for infrastructure health checks.
    """
    with WEBRTC_SESSIONS_LOCK:
        active_webrtc_sessions = len(WEBRTC_SESSIONS)
    idle_video_url = resolve_idle_video_url()
    return {
        "avatarSessionId": "",
        "mode": AVATAR_MODE_IDLE,
        "sequence": 0,
        "currentJobId": "",
        "currentJobStartedAtMs": 0,
        "currentJobEndsAtMs": 0,
        "idleStartedAtMs": PROCESS_STARTED_AT_MS,
        "idleVideoUrl": idle_video_url,
        "bufferedStartProgress": 0.0,
        "currentJobVideoWsUrl": "",
        "currentJobStatusWsUrl": "",
        "currentJobAudioDurationSec": 0.0,
        "currentJobSourceFrameUrl": "",
        "currentJobPreviewComposition": None,
        "queueDepth": 0,
        "runningJobId": "",
        "idleVideoAvailable": bool(idle_video_url),
        "avatarVideoWsUrl": "/ws/avatar/video",
        "avatarVideoHttpUrl": "/api/avatar/video.mp4",
        "avatarWebrtcOfferUrl": WEBRTC_OFFER_API_PATH,
        "avatarTransport": AVATAR_TRANSPORT_WEBSOCKET,
        "webrtcEnabled": False,
        "webrtcIceServers": WEBRTC_ICE_SERVER_PAYLOADS,
        "webrtcIceTransportPolicy": DEFAULT_WEBRTC_ICE_TRANSPORT_POLICY,
        "activeWebrtcSessions": active_webrtc_sessions,
        "currentJobDrivingMediaUrl": "",
        "status": "ok",
    }


def resolve_avatar_ready_buffer_sec_for_duration(audio_duration_sec: float) -> float:
    """
    Resolve the baseline talking prebuffer window from one audio duration.
    """
    safe_audio_duration_sec = max(0.0, float(audio_duration_sec))
    if safe_audio_duration_sec <= 0:
        return float(AVATAR_READY_BUFFER_MIN_SEC)
    progress_limited_buffer_sec = safe_audio_duration_sec * float(AVATAR_READY_BUFFER_MAX_PROGRESS)
    if progress_limited_buffer_sec <= 0:
        return float(AVATAR_READY_BUFFER_MIN_SEC)
    return min(float(AVATAR_READY_BUFFER_MIN_SEC), progress_limited_buffer_sec)


def resolve_avatar_minimum_ready_progress(audio_duration_sec: float) -> float:
    """
    Resolve the display-only progress ratio implied by the startup safety window.
    """
    safe_audio_duration_sec = max(0.0, float(audio_duration_sec))
    if safe_audio_duration_sec <= 0:
        return 0.0
    return clamp_float(
        resolve_avatar_ready_buffer_sec_for_duration(safe_audio_duration_sec) / safe_audio_duration_sec,
        0.0,
        1.0,
    )


def resolve_avatar_ready_buffer_sec(job: JobRecord, stream_status: dict[str, Any] | None) -> float:
    """
    Resolve the baseline talking prebuffer window before the avatar starts one job.
    """
    return resolve_avatar_ready_buffer_sec_for_duration(job.audio_duration_sec)


def resolve_avatar_required_ready_frame_count(
    job: JobRecord,
    stream_status: dict[str, Any] | None,
) -> int:
    """
    Resolve the minimum generated frame count required before the avatar can
    start one talking job without outrunning the renderer.
    """
    frame_total = parse_status_int(stream_status, "frameTotal")
    playback_fps = resolve_stream_playback_fps(stream_status)
    baseline_duration_sec = resolve_avatar_ready_buffer_sec(job, stream_status)
    baseline_frame_count = 0
    if playback_fps > 0 and baseline_duration_sec > 0:
        baseline_frame_count = int(math.ceil(baseline_duration_sec * playback_fps))
    if frame_total <= 0 or playback_fps <= 0:
        return max(1, baseline_frame_count)

    maximum_progress_frame_count = 0
    if AVATAR_READY_BUFFER_MAX_PROGRESS > 0:
        maximum_progress_frame_count = max(
            1,
            int(math.ceil(float(frame_total) * float(AVATAR_READY_BUFFER_MAX_PROGRESS))),
        )

    estimated_generation_fps = estimate_generation_fps(stream_status, 0.0)
    safety_frame_count = 0
    if estimated_generation_fps > 0 and playback_fps > 0 and AVATAR_READY_DYNAMIC_MARGIN_SEC > 0:
        generation_gap_ratio = clamp_float(
            max(0.0, float(playback_fps) - float(estimated_generation_fps)) / float(playback_fps),
            0.0,
            1.0,
        )
        if generation_gap_ratio > 0:
            safety_frame_count = int(
                math.ceil(float(playback_fps) * float(AVATAR_READY_DYNAMIC_MARGIN_SEC) * generation_gap_ratio)
            )

    required_frame_count = max(1, min(frame_total, baseline_frame_count + safety_frame_count))
    if maximum_progress_frame_count > 0:
        required_frame_count = min(required_frame_count, maximum_progress_frame_count)
    return required_frame_count


def resolve_avatar_idle_anchor_source_frame(
    audio_duration_sec: float,
    avatar_session_id: str,
) -> tuple[Path, str] | None:
    """
    Select the idle anchor whose loop position is nearest to the predicted talking handoff moment.
    """
    anchors = ensure_idle_source_anchors()
    if not anchors:
        return None
    manifest = ensure_idle_source_anchor_manifest()
    duration_sec = float((manifest or {}).get("durationSec") or 0.0)
    if duration_sec <= 0:
        first_anchor_path = Path(str(anchors[0].get("path") or "")).resolve()
        return first_anchor_path, to_runner_source_arg(first_anchor_path)

    snapshot = get_avatar_state_snapshot(avatar_session_id)
    if str(snapshot.get("mode") or "") != AVATAR_MODE_IDLE:
        first_anchor_path = Path(str(anchors[0].get("path") or "")).resolve()
        return first_anchor_path, to_runner_source_arg(first_anchor_path)

    idle_started_at_ms = int(snapshot.get("idleStartedAtMs") or 0)
    future_offset_sec = resolve_avatar_ready_buffer_sec_for_duration(audio_duration_sec)
    elapsed_idle_sec = max(0.0, (now_ms() - idle_started_at_ms) / 1000.0) if idle_started_at_ms > 0 else 0.0
    target_offset_sec = (elapsed_idle_sec + future_offset_sec) % duration_sec

    selected_anchor = anchors[0]
    selected_anchor_distance_sec = duration_sec
    for anchor in anchors:
        anchor_offset_sec = float(anchor.get("offsetSec") or 0.0) % duration_sec
        forward_distance_sec = abs(anchor_offset_sec - target_offset_sec)
        circular_distance_sec = min(forward_distance_sec, duration_sec - forward_distance_sec)
        if circular_distance_sec < selected_anchor_distance_sec:
            selected_anchor = anchor
            selected_anchor_distance_sec = circular_distance_sec

    selected_anchor_path = Path(str(selected_anchor.get("path") or "")).resolve()
    if not selected_anchor_path.exists() or not selected_anchor_path.is_file():
        return None
    return selected_anchor_path, to_runner_source_arg(selected_anchor_path)


def resolve_avatar_playback_start_frame_index(
    snapshot: dict[str, Any],
    stream_status: dict[str, Any] | None,
) -> int:
    """
    Resolve the frame index that should be visible at the current avatar playback offset.
    """
    playback_fps = resolve_stream_playback_fps(stream_status)
    started_at_ms = int(snapshot.get("currentJobStartedAtMs") or 0)
    elapsed_sec = max(0.0, (now_ms() - started_at_ms) / 1000.0) if started_at_ms > 0 else 0.0
    start_frame_index = 1
    if playback_fps > 0 and elapsed_sec > 0:
        start_frame_index = max(1, int(elapsed_sec * playback_fps) + 1)
    frame_total = parse_status_int(stream_status, "frameTotal")
    if frame_total > 0:
        start_frame_index = min(start_frame_index, frame_total)
    return start_frame_index


def is_job_ready_for_avatar(job: JobRecord, stream_status: dict[str, Any] | None) -> bool:
    """
    Determine whether a job has enough buffered frames to start synchronized avatar playback.
    """
    state = determine_job_state(job, stream_status)
    if state in {"error", "canceled"}:
        return False
    frame_index = parse_status_int(stream_status, "frameIndex")
    if state == "done":
        return frame_index > 0
    playback_fps = resolve_stream_playback_fps(stream_status)
    if frame_index <= 0 or playback_fps <= 0:
        return False
    required_frame_count = resolve_avatar_required_ready_frame_count(job, stream_status)
    return frame_index >= required_frame_count


def select_next_avatar_job(avatar_session_id: str) -> JobRecord | None:
    """
    Select the oldest ready job that has not been played yet for one interaction session.
    """
    with JOBS_LOCK:
        job_records = sorted(JOBS.values(), key=lambda item: item.created_at_ms)
    for job in job_records:
        if job.avatar_session_id != avatar_session_id:
            continue
        if job.avatar_play_finished_at_ms is not None or job.avatar_play_started_at_ms is not None:
            continue
        stream_status = read_job_stream_status(job)
        state = determine_job_state(job, stream_status)
        if state in {"error", "canceled"}:
            with JOBS_LOCK:
                job.avatar_play_finished_at_ms = now_ms()
            continue
        if not is_job_ready_for_avatar(job, stream_status):
            continue
        with JOBS_LOCK:
            if job.avatar_ready_at_ms is None:
                job.avatar_ready_at_ms = now_ms()
        return job
    return None


def activate_avatar_idle_mode(avatar_session_id: str) -> None:
    """
    Switch one interaction session back to idle and advance its transition sequence.
    """
    session_state = ensure_avatar_session_state(avatar_session_id)
    with AVATAR_SESSION_STATES_LOCK:
        session_state.mode = AVATAR_MODE_IDLE
        session_state.current_job_id = ""
        session_state.current_job_started_at_ms = 0
        session_state.current_job_ends_at_ms = 0
        session_state.idle_started_at_ms = now_ms()
        session_state.sequence += 1


def activate_avatar_job(avatar_session_id: str, job: JobRecord) -> None:
    """
    Switch one interaction session to a talking job and record playback timestamps.
    """
    started_at_ms = now_ms()
    with JOBS_LOCK:
        job.avatar_play_started_at_ms = started_at_ms
    session_state = ensure_avatar_session_state(avatar_session_id)
    with AVATAR_SESSION_STATES_LOCK:
        session_state.mode = AVATAR_MODE_TALKING
        session_state.current_job_id = job.job_id
        session_state.current_job_started_at_ms = started_at_ms
        session_state.current_job_ends_at_ms = 0
        session_state.sequence += 1


def resolve_avatar_job_expected_end_at_ms(
    started_at_ms: int,
    job: JobRecord,
    stream_status: dict[str, Any] | None,
) -> int:
    """
    Resolve one deterministic avatar cutoff timestamp, preferring audio duration over frame totals.
    """
    if started_at_ms <= 0:
        return 0
    audio_duration_sec = max(0.0, float(job.audio_duration_sec or 0.0))
    if audio_duration_sec > 0:
        return started_at_ms + int(round(audio_duration_sec * 1000.0))
    playback_fps = resolve_stream_playback_fps(stream_status)
    frame_total = parse_status_int(stream_status, "frameTotal")
    if playback_fps <= 0 or frame_total <= 0:
        return 0
    clip_duration_ms = int(round((float(frame_total) / float(playback_fps)) * 1000.0))
    if clip_duration_ms <= 0:
        return 0
    return started_at_ms + clip_duration_ms


def advance_avatar_state_machine(avatar_session_id: str) -> None:
    """
    Advance the avatar scheduler state for one interaction session.
    """
    session_state = ensure_avatar_session_state(avatar_session_id)
    with AVATAR_SESSION_STATES_LOCK:
        current_mode = session_state.mode
        current_job_id = session_state.current_job_id
        current_job_started_at_ms = session_state.current_job_started_at_ms
        idle_started_at_ms = session_state.idle_started_at_ms

    now_timestamp_ms = now_ms()
    if current_mode == AVATAR_MODE_TALKING and current_job_id:
        with JOBS_LOCK:
            current_job = JOBS.get(current_job_id)
        if current_job is None or current_job.avatar_session_id != avatar_session_id:
            activate_avatar_idle_mode(avatar_session_id)
            return
        if current_job.avatar_play_finished_at_ms is not None:
            activate_avatar_idle_mode(avatar_session_id)
            return
        current_status = read_job_stream_status(current_job)
        current_state = determine_job_state(current_job, current_status)
        if current_state in {"error", "canceled"}:
            with JOBS_LOCK:
                current_job.avatar_play_finished_at_ms = now_timestamp_ms
            activate_avatar_idle_mode(avatar_session_id)
            return
        expected_end_at_ms = resolve_avatar_job_expected_end_at_ms(
            current_job_started_at_ms,
            current_job,
            current_status,
        )
        if expected_end_at_ms > 0:
            with AVATAR_SESSION_STATES_LOCK:
                if session_state.current_job_id == current_job.job_id:
                    session_state.current_job_ends_at_ms = expected_end_at_ms
        if expected_end_at_ms > 0 and now_timestamp_ms >= expected_end_at_ms:
            with JOBS_LOCK:
                current_job.avatar_play_finished_at_ms = now_timestamp_ms
            activate_avatar_idle_mode(avatar_session_id)
            return
        return

    if current_mode != AVATAR_MODE_IDLE:
        activate_avatar_idle_mode(avatar_session_id)
        return
    if now_timestamp_ms - idle_started_at_ms < int(AVATAR_IDLE_MIN_HOLD_SEC * 1000.0):
        return

    next_job = select_next_avatar_job(avatar_session_id)
    if next_job is None:
        return
    activate_avatar_job(avatar_session_id, next_job)


def avatar_worker_loop() -> None:
    """
    Persistent scheduler that advances every active avatar interaction session.
    """
    while True:
        try:
            with AVATAR_SESSION_STATES_LOCK:
                avatar_session_ids = list(AVATAR_SESSION_STATES.keys())
            for avatar_session_id in avatar_session_ids:
                advance_avatar_state_machine(avatar_session_id)
        except Exception as exc:
            print(f"[avatar] scheduler error: {exc}")
        time.sleep(AVATAR_STATE_POLL_SLEEP_SEC)


def ensure_avatar_worker_started() -> None:
    """
    Start persistent avatar scheduler thread once.
    """
    global AVATAR_WORKER_THREAD
    with AVATAR_WORKER_LOCK:
        if AVATAR_WORKER_THREAD is not None and AVATAR_WORKER_THREAD.is_alive():
            return
        AVATAR_WORKER_THREAD = threading.Thread(target=avatar_worker_loop, daemon=True)
        AVATAR_WORKER_THREAD.start()


def should_defer_preview_paste_back(
    mode: str,
    paste_back_enabled: bool,
    stitching_enabled: bool,
    source_media_path: Path | str | None = None,
) -> bool:
    """
    Defer paste-back only for preview jobs that still require stitched full-frame output.
    """
    if is_video_backed_source_path(source_media_path):
        return False
    return bool(str(mode or "").strip().lower() == "preview" and paste_back_enabled and stitching_enabled)


def build_runner_command(job: JobRecord) -> list[str]:
    """
    Build runner command for a specific job.
    """
    command = [
        str(RUNNER_PYTHON),
        str(RUNNER_SCRIPT),
        "--backend",
        DEFAULT_BACKEND,
        "--trt-runtime",
        DEFAULT_TRT_RUNTIME,
        "--trt-precision",
        DEFAULT_TRT_PRECISION,
        "--mode",
        job.mode,
        "--source-frame",
        job.source_frame_arg,
        "--driving-audio",
        normalize_rel_path(str(job.audio_input_rel)),
        "--audio-motion-stride",
        str(job.audio_motion_stride),
        "--audio-eye-soft-factor",
        f"{float(job.audio_eye_soft_factor):.6f}",
        "--audio-eye-hard-factor",
        f"{float(job.audio_eye_hard_factor):.6f}",
        "--audio-eye-hard-dy-min",
        f"{float(job.audio_eye_hard_dy_min):.6f}",
        "--audio-eye-hard-dy-max",
        f"{float(job.audio_eye_hard_dy_max):.6f}",
        "--audio-reanchor-first-n",
        str(int(job.audio_reanchor_first_n)),
        "--audio-mouth-open-factor",
        f"{float(job.audio_mouth_open_factor):.6f}",
        "--audio-pose-smooth-window",
        str(int(job.audio_pose_smooth_window)),
        "--audio-exp-smooth-window",
        str(int(job.audio_exp_smooth_window)),
        "--audio-pose-jump-threshold",
        f"{float(job.audio_pose_jump_threshold):.6f}",
        "--audio-translation-jump-threshold",
        f"{float(job.audio_translation_jump_threshold):.6f}",
        "--audio-lip-sync-min-ratio",
        f"{float(job.audio_lip_sync_min_ratio):.6f}",
        "--audio-lip-sync-max-ratio",
        f"{float(job.audio_lip_sync_max_ratio):.6f}",
        "--audio-lip-sync-smooth-window",
        str(int(job.audio_lip_sync_smooth_window)),
        "--audio-lip-sync-strength",
        f"{float(job.audio_lip_sync_strength):.6f}",
        "--audio-lip-sync-power",
        f"{float(job.audio_lip_sync_power):.6f}",
        "--audio-lip-sync-attack",
        f"{float(job.audio_lip_sync_attack):.6f}",
        "--audio-lip-sync-release",
        f"{float(job.audio_lip_sync_release):.6f}",
        "--audio-lip-sync-offset-ms",
        str(int(job.audio_lip_sync_offset_ms)),
        "--audio-mouth-floor-strength",
        f"{float(job.audio_mouth_floor_strength):.6f}",
        "--audio-mouth-peak-clamp",
        f"{float(job.audio_mouth_peak_clamp):.6f}",
        "--driving-multiplier",
        f"{float(job.driving_multiplier):.6f}",
        "--cfg-scale",
        f"{float(job.cfg_scale):.6f}",
        "--joyvasa-inference-steps",
        str(int(job.joyvasa_inference_steps)),
        "--render-batch-size",
        str(DEFAULT_RENDER_BATCH_SIZE),
        "--trt-engine-batch-size",
        str(DEFAULT_TRT_ENGINE_BATCH_SIZE),
        "--video-encoder",
        DEFAULT_VIDEO_ENCODER,
        "--output-dir",
        normalize_rel_path(str(job.output_rel)),
        "--stream-dir",
        normalize_rel_path(str(job.stream_rel)),
        "--stream-shm-prefix",
        job.stream_shm_prefix,
        "--animation-region",
        job.animation_region,
    ]
    if DEFAULT_DOCKER_GPU_DEVICE:
        command.extend(
            [
                "--docker-gpu-device",
                DEFAULT_DOCKER_GPU_DEVICE,
            ]
        )
    if job.generation_frame_count is not None:
        command.extend(
            [
                "--generation-frame-count",
                str(job.generation_frame_count),
            ]
        )
    command.append("--audio-eye-tamed-preset" if job.audio_eye_tamed_preset else "--no-audio-eye-tamed-preset")
    command.append("--audio-motion-tuning-enabled" if job.audio_motion_tuning_enabled else "--no-audio-motion-tuning")
    command.append("--audio-lip-sync-assist" if job.audio_lip_sync_assist else "--no-audio-lip-sync-assist")
    if job.defer_paste_back_enabled:
        command.append("--defer-paste-back")
    elif not job.paste_back_enabled:
        command.append("--no-paste-back")
    if not job.stitching_enabled:
        command.append("--no-stitching")
    if not job.relative_motion_enabled:
        command.append("--no-relative-motion")
    if DEFAULT_SKIP_TRT_ENGINE_BUILD:
        command.append("--skip-trt-engine-build")
    return command


def determine_job_state(job: JobRecord, stream_status: dict[str, Any] | None) -> str:
    """
    Resolve high-level job state.
    """
    if job.exit_code is not None:
        if int(job.exit_code) != 0:
            return "error"
        if stream_status and isinstance(stream_status.get("state"), str):
            stream_state = str(stream_status["state"]).lower()
            if stream_state in {"done", "error"}:
                return stream_state
        return "done"
    process = job.process
    if process is not None and process.poll() is None:
        return "running"
    if stream_status and isinstance(stream_status.get("state"), str):
        return str(stream_status["state"])
    if process is None:
        return "queued"
    return "running"


def read_updated_latest_frame(job: JobRecord, last_seen_frame_index: int) -> tuple[bytes | None, int, bool]:
    """
    Read latest JPEG frame only when the producer publishes a newer frame index.
    """
    reader = ensure_job_stream_reader(job)
    if reader is None:
        return None, last_seen_frame_index, False
    try:
        latest_frame = reader.read_latest_frame()
    except Exception:
        close_job_stream_cache(get_or_create_job_stream_cache(job))
        return None, last_seen_frame_index, False
    if latest_frame is None:
        return None, last_seen_frame_index, False
    latest_frame_index, frame_bytes = latest_frame
    if latest_frame_index == last_seen_frame_index or not frame_bytes:
        return None, last_seen_frame_index, False
    return frame_bytes, latest_frame_index, True


def parse_status_int(stream_status: dict[str, Any] | None, key: str) -> int:
    """
    Parse integer values from stream status payload.
    """
    if not stream_status:
        return 0
    value = stream_status.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return 0


def parse_status_float(stream_status: dict[str, Any] | None, key: str, fallback: float) -> float:
    """
    Parse float values from stream status payload.
    """
    if not stream_status:
        return fallback
    value = stream_status.get(key)
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0 else fallback
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
            return parsed if parsed > 0 else fallback
        except ValueError:
            return fallback
    return fallback


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    """
    Clamp float to inclusive range.
    """
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def estimate_generation_fps(
    stream_status: dict[str, Any] | None,
    previous_estimate: float,
) -> float:
    """
    Estimate generator throughput FPS from frame index and elapsed seconds.
    """
    frame_index = parse_status_int(stream_status, "frameIndex")
    elapsed_sec = parse_status_float(stream_status, "elapsedSec", 0.0)
    if frame_index <= 0 or elapsed_sec <= 0:
        return previous_estimate
    measured_fps = frame_index / elapsed_sec
    bounded_fps = clamp_float(measured_fps, VIDEO_STREAM_GENERATION_FPS_MIN, VIDEO_STREAM_GENERATION_FPS_MAX)
    if previous_estimate <= 0:
        return bounded_fps
    return (
        previous_estimate * (1.0 - VIDEO_STREAM_GENERATION_FPS_SMOOTH_ALPHA)
        + bounded_fps * VIDEO_STREAM_GENERATION_FPS_SMOOTH_ALPHA
    )


def resolve_video_stream_start_mode(websocket: WebSocket) -> str:
    """
    Resolve requested stream start mode from WebSocket query params.
    """
    raw_value = str(websocket.query_params.get(VIDEO_STREAM_START_MODE_QUERY_KEY, "")).strip().lower()
    if raw_value in VIDEO_STREAM_START_MODE_CHOICES:
        return raw_value
    return VIDEO_STREAM_START_MODE_LIVE


def resolve_video_stream_start_progress(websocket: WebSocket) -> float:
    """
    Resolve buffered stream start threshold from WebSocket query params.
    """
    raw_value = str(websocket.query_params.get(VIDEO_STREAM_START_PROGRESS_QUERY_KEY, "")).strip()
    if not raw_value:
        return VIDEO_STREAM_BUFFERED_START_PROGRESS_DEFAULT
    try:
        parsed_value = float(raw_value)
    except ValueError:
        return VIDEO_STREAM_BUFFERED_START_PROGRESS_DEFAULT
    return clamp_float(parsed_value, 0.0, 1.0)


def resolve_stream_progress_ratio(
    stream_status: dict[str, Any] | None,
    frame_index: int,
    frame_total: int,
) -> float:
    """
    Resolve normalized progress ratio using explicit progress or frame counters.
    """
    explicit_progress = parse_status_float(stream_status, "progress", -1.0)
    if explicit_progress >= 0:
        return clamp_float(explicit_progress, 0.0, 1.0)
    if frame_total > 0:
        return clamp_float(float(frame_index) / float(frame_total), 0.0, 1.0)
    return 0.0


def resolve_stream_playback_fps(stream_status: dict[str, Any] | None) -> float:
    """
    Resolve intended clip playback FPS from stream status.
    """
    return parse_status_float(stream_status, VIDEO_STREAM_PLAYBACK_FPS_STATUS_KEY, 0.0)


def resolve_interpolation_steps(estimated_generation_fps: float) -> int:
    """
    Resolve interpolation step count based on current generation FPS.
    """
    safe_generation_fps = max(1.0, estimated_generation_fps)
    target_ratio = VIDEO_STREAM_INTERPOLATION_TARGET_FPS / safe_generation_fps
    target_steps = int(round(target_ratio)) - 1
    if target_steps < 0:
        return 0
    if target_steps > VIDEO_STREAM_INTERPOLATION_MAX_STEPS:
        return VIDEO_STREAM_INTERPOLATION_MAX_STEPS
    return target_steps


def decode_jpeg_frame(frame_bytes: bytes) -> np.ndarray | None:
    """
    Decode JPEG bytes into BGR image.
    """
    if not frame_bytes:
        return None
    encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
    if encoded.size == 0:
        return None
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return frame


def encode_jpeg_frame(frame_image: np.ndarray) -> bytes:
    """
    Encode BGR image into JPEG bytes for stream transport.
    """
    ok, encoded = cv2.imencode(".jpg", frame_image, [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_STREAM_INTERPOLATION_ALPHA_QUALITY])
    if not ok:
        return b""
    return encoded.tobytes()


def fit_frame_to_canvas(frame_image: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
    """
    Resize one frame into a stable canvas while preserving aspect ratio.
    """
    canvas_width = max(1, int(canvas_size[0]))
    canvas_height = max(1, int(canvas_size[1]))
    if frame_image.ndim != 3 or frame_image.shape[2] != 3:
        return np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    source_height, source_width = frame_image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        return np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    if source_width == canvas_width and source_height == canvas_height:
        return frame_image
    scale = min(float(canvas_width) / float(source_width), float(canvas_height) / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame_image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    offset_x = max(0, (canvas_width - resized_width) // 2)
    offset_y = max(0, (canvas_height - resized_height) // 2)
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def resolve_avatar_stream_canvas_size(source_canvas_size: tuple[int, int]) -> tuple[int, int]:
    """
    Downscale the avatar transport canvas for remote streaming while preserving aspect ratio.
    """
    source_width = max(1, int(source_canvas_size[0]))
    source_height = max(1, int(source_canvas_size[1]))
    if source_width <= AVATAR_STREAM_OUTPUT_MAX_WIDTH and source_height <= AVATAR_STREAM_OUTPUT_MAX_HEIGHT:
        return source_width, source_height
    scale = min(
        float(AVATAR_STREAM_OUTPUT_MAX_WIDTH) / float(source_width),
        float(AVATAR_STREAM_OUTPUT_MAX_HEIGHT) / float(source_height),
    )
    scaled_width = max(2, int(round(source_width * scale)))
    scaled_height = max(2, int(round(source_height * scale)))
    if scaled_width % 2 != 0:
        scaled_width -= 1
    if scaled_height % 2 != 0:
        scaled_height -= 1
    return max(2, scaled_width), max(2, scaled_height)


def normalize_jpeg_frame_to_canvas(frame_bytes: bytes, canvas_size: tuple[int, int]) -> bytes:
    """
    Decode JPEG bytes, resize into the shared avatar canvas, and encode back to JPEG.
    """
    frame_image = decode_jpeg_frame(frame_bytes)
    if frame_image is None:
        return b""
    return encode_jpeg_frame(fit_frame_to_canvas(frame_image, canvas_size))


def normalize_jpeg_frame_image_to_canvas(frame_bytes: bytes, canvas_size: tuple[int, int]) -> np.ndarray | None:
    """
    Decode JPEG bytes and resize the frame into the shared avatar canvas without re-encoding it.
    """
    frame_image = decode_jpeg_frame(frame_bytes)
    if frame_image is None:
        return None
    return fit_frame_to_canvas(frame_image, canvas_size)


def resolve_preview_composition_status(stream_status: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Extract one normalized preview-composition payload from stream status.
    """
    if not isinstance(stream_status, dict):
        return None
    preview_composition = stream_status.get(PREVIEW_COMPOSITION_STATUS_KEY)
    if not isinstance(preview_composition, dict):
        return None
    if not bool(preview_composition.get("enabled")):
        return None
    return preview_composition


def build_preview_composition_signature(
    job: JobRecord,
    preview_composition: dict[str, Any],
    canvas_size: tuple[int, int],
) -> str:
    """
    Build one cache key for avatar-side preview composition assets.
    """
    matrix_value = preview_composition.get("matrix")
    mask_image_name = str(preview_composition.get("maskImage") or PREVIEW_COMPOSITION_MASK_NAME)
    return json.dumps(
        {
            "jobId": job.job_id,
            "maskImage": mask_image_name,
            "matrix": matrix_value,
            "canvasWidth": int(canvas_size[0]),
            "canvasHeight": int(canvas_size[1]),
            "sourceFrame": str(job.source_frame_abs),
        },
        sort_keys=True,
    )


def clear_avatar_preview_composition_state(state: AvatarTalkingFrameState) -> None:
    """
    Drop cached preview-composition assets when the active job changes.
    """
    state.preview_composition_signature = ""
    state.preview_composition_matrix = None
    state.preview_composition_mask = None
    state.preview_composition_source = None


def load_avatar_preview_composition_state(
    job: JobRecord,
    state: AvatarTalkingFrameState,
    stream_status: dict[str, Any] | None,
    canvas_size: tuple[int, int],
) -> bool:
    """
    Load one cached preview-composition state for the continuous avatar stream.
    """
    preview_composition = resolve_preview_composition_status(stream_status)
    if preview_composition is None:
        return bool(
            state.preview_composition_matrix is not None
            and state.preview_composition_mask is not None
            and state.preview_composition_source is not None
        )

    signature = build_preview_composition_signature(job, preview_composition, canvas_size)
    if (
        signature == state.preview_composition_signature
        and state.preview_composition_matrix is not None
        and state.preview_composition_mask is not None
        and state.preview_composition_source is not None
    ):
        return True

    source_frame_image = cv2.imread(str(job.source_frame_abs), cv2.IMREAD_COLOR)
    if source_frame_image is None:
        clear_avatar_preview_composition_state(state)
        return False
    source_height, source_width = source_frame_image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        clear_avatar_preview_composition_state(state)
        return False

    mask_image_name = str(preview_composition.get("maskImage") or PREVIEW_COMPOSITION_MASK_NAME)
    mask_image_path = job.stream_abs / mask_image_name
    mask_image = cv2.imread(str(mask_image_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None or mask_image.ndim != 3 or mask_image.shape[2] < 4:
        clear_avatar_preview_composition_state(state)
        return False

    matrix_value = preview_composition.get("matrix")
    if not isinstance(matrix_value, list):
        clear_avatar_preview_composition_state(state)
        return False
    transform_matrix = np.asarray(matrix_value, dtype=np.float32)
    if transform_matrix.shape != (2, 3):
        clear_avatar_preview_composition_state(state)
        return False

    canvas_width = max(1, int(canvas_size[0]))
    canvas_height = max(1, int(canvas_size[1]))
    scale = min(float(canvas_width) / float(source_width), float(canvas_height) / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    offset_x = float(max(0, (canvas_width - resized_width) // 2))
    offset_y = float(max(0, (canvas_height - resized_height) // 2))
    scaled_transform = transform_matrix.copy()
    scaled_transform[:, :2] *= scale
    scaled_transform[0, 2] = (transform_matrix[0, 2] * scale) + offset_x
    scaled_transform[1, 2] = (transform_matrix[1, 2] * scale) + offset_y

    source_canvas_image = fit_frame_to_canvas(source_frame_image, canvas_size)
    mask_canvas_image = fit_frame_to_canvas(
        cv2.cvtColor(mask_image[..., 3], cv2.COLOR_GRAY2BGR),
        canvas_size,
    )[..., 0]
    mask_canvas_float = np.repeat(
        (mask_canvas_image.astype(np.float32) / 255.0)[..., None],
        3,
        axis=2,
    )

    state.preview_composition_signature = signature
    state.preview_composition_matrix = scaled_transform
    state.preview_composition_mask = mask_canvas_float
    state.preview_composition_source = source_canvas_image
    return True


def compose_avatar_preview_frame(
    job: JobRecord,
    state: AvatarTalkingFrameState,
    stream_status: dict[str, Any] | None,
    frame_image: np.ndarray,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """
    Compose one crop-only frame into the shared avatar canvas using cached preview metadata.
    """
    if not load_avatar_preview_composition_state(job, state, stream_status, canvas_size):
        return fit_frame_to_canvas(frame_image, canvas_size)

    assert state.preview_composition_matrix is not None
    assert state.preview_composition_mask is not None
    assert state.preview_composition_source is not None

    canvas_width = max(1, int(canvas_size[0]))
    canvas_height = max(1, int(canvas_size[1]))
    warped_frame_image = cv2.warpAffine(
        frame_image,
        state.preview_composition_matrix,
        (canvas_width, canvas_height),
    )
    return np.clip(
        (state.preview_composition_mask * warped_frame_image)
        + ((1.0 - state.preview_composition_mask) * state.preview_composition_source),
        0.0,
        255.0,
    ).astype(np.uint8)


def normalize_job_stream_frame_image_to_canvas(
    job: JobRecord,
    state: AvatarTalkingFrameState,
    frame_bytes: bytes,
    canvas_size: tuple[int, int],
    stream_status: dict[str, Any] | None,
) -> np.ndarray | None:
    """
    Decode one job frame and compose it into the avatar canvas when preview metadata is present.
    """
    frame_image = decode_jpeg_frame(frame_bytes)
    if frame_image is None:
        return None
    return compose_avatar_preview_frame(job, state, stream_status, frame_image, canvas_size)


def refresh_avatar_source_frame_images(
    job: JobRecord,
    state: AvatarTalkingFrameState,
    canvas_size: tuple[int, int],
    required_frame_index: int,
) -> dict[str, Any] | None:
    """
    Load the source frame images needed for one time-aligned avatar talking frame.
    """
    stream_status = read_job_stream_status(job)
    state.estimated_generation_fps = estimate_generation_fps(stream_status, state.estimated_generation_fps)
    playback_fps = resolve_stream_playback_fps(stream_status)
    if playback_fps > 0:
        state.playback_fps = playback_fps
    status_frame_index = parse_status_int(stream_status, "frameIndex")
    status_frame_total = parse_status_int(stream_status, "frameTotal")
    if status_frame_index > state.last_known_frame_index:
        state.last_known_frame_index = status_frame_index
    if status_frame_total > state.last_known_frame_total:
        state.last_known_frame_total = status_frame_total

    max_frame_to_load = min(
        state.last_known_frame_index,
        max(required_frame_index, state.start_frame_index) + VIDEO_STREAM_SERVER_BUFFER_FRAMES,
    )
    while state.next_frame_index <= max_frame_to_load:
        frame_bytes = read_stream_frame_by_index(job, state.next_frame_index)
        if not frame_bytes:
            break
        frame_image = normalize_job_stream_frame_image_to_canvas(
            job,
            state,
            frame_bytes,
            canvas_size,
            stream_status,
        )
        if frame_image is None:
            break
        state.source_frame_images[state.next_frame_index] = frame_image
        state.next_frame_index += 1

    minimum_cached_frame_index = max(
        state.start_frame_index,
        required_frame_index - AVATAR_FALLBACK_BOUNCE_WINDOW_FRAMES,
    )
    stale_frame_indices = [
        frame_index
        for frame_index in state.source_frame_images.keys()
        if frame_index < minimum_cached_frame_index
    ]
    for stale_frame_index in stale_frame_indices:
        state.source_frame_images.pop(stale_frame_index, None)
    return stream_status


def blend_avatar_source_frames(
    base_frame_image: np.ndarray,
    next_frame_image: np.ndarray | None,
    blend_alpha: float,
) -> np.ndarray:
    """
    Blend two adjacent source frames for one time-aligned talking output frame.
    """
    if next_frame_image is None:
        return base_frame_image
    if base_frame_image.shape != next_frame_image.shape:
        return base_frame_image
    safe_alpha = clamp_float(float(blend_alpha), 0.0, 1.0)
    if safe_alpha <= 0.0:
        return base_frame_image
    if safe_alpha >= 1.0:
        return next_frame_image
    return cv2.addWeighted(base_frame_image, 1.0 - safe_alpha, next_frame_image, safe_alpha, 0.0)


def reset_avatar_bounce_fallback_state(state: AvatarTalkingFrameState) -> None:
    """
    Clear the short reverse-forward fallback cursor once forward playback resumes.
    """
    state.fallback_bounce_cursor = 0
    state.fallback_bounce_signature = ()


def build_avatar_bounce_cycle(frame_indices: list[int]) -> tuple[int, ...]:
    """
    Build one short reverse-forward cycle from the newest available frame indices.
    """
    if not frame_indices:
        return ()
    ordered_frame_indices = sorted({int(frame_index) for frame_index in frame_indices if int(frame_index) > 0})
    if not ordered_frame_indices:
        return ()
    descending_frame_indices = list(reversed(ordered_frame_indices))
    if len(descending_frame_indices) <= 2:
        return tuple(descending_frame_indices)
    return tuple(descending_frame_indices + ordered_frame_indices[1:-1])


def resolve_avatar_bounce_frame_image(
    state: AvatarTalkingFrameState,
    requested_frame_index: int,
) -> np.ndarray | None:
    """
    Return one brief ping-pong fallback frame when generation is momentarily late.
    """
    fallback_frame_ceiling = max(
        1,
        min(
            max(state.last_known_frame_index, 1),
            max(state.last_output_source_frame_index, requested_frame_index - 1),
        ),
    )
    recent_frame_indices = sorted(
        frame_index
        for frame_index in state.source_frame_images.keys()
        if frame_index <= fallback_frame_ceiling
    )[-AVATAR_FALLBACK_BOUNCE_WINDOW_FRAMES:]
    bounce_cycle = build_avatar_bounce_cycle(recent_frame_indices)
    if not bounce_cycle:
        return state.last_frame_image
    if bounce_cycle != state.fallback_bounce_signature:
        state.fallback_bounce_signature = bounce_cycle
        state.fallback_bounce_cursor = 0
    bounce_frame_index = bounce_cycle[state.fallback_bounce_cursor % len(bounce_cycle)]
    state.fallback_bounce_cursor = (state.fallback_bounce_cursor + 1) % len(bounce_cycle)
    bounce_frame_image = state.source_frame_images.get(bounce_frame_index)
    if bounce_frame_image is not None:
        state.last_output_source_frame_index = bounce_frame_index
        return bounce_frame_image
    return state.last_frame_image


def update_avatar_talking_frame_state(job: JobRecord, state: AvatarTalkingFrameState) -> dict[str, Any] | None:
    """
    Refresh one talking job frame buffer so the continuous avatar stream can emit frames sequentially.
    """
    stream_status = read_job_stream_status(job)
    state.estimated_generation_fps = estimate_generation_fps(stream_status, state.estimated_generation_fps)
    status_frame_index = parse_status_int(stream_status, "frameIndex")
    status_frame_total = parse_status_int(stream_status, "frameTotal")
    if status_frame_index > state.last_known_frame_index:
        state.last_known_frame_index = status_frame_index
    if status_frame_total > state.last_known_frame_total:
        state.last_known_frame_total = status_frame_total

    while (
        len(state.pending_raw_frames) < VIDEO_STREAM_SERVER_BUFFER_FRAMES
        and state.next_frame_index <= state.last_known_frame_index
    ):
        frame_bytes = read_stream_frame_by_index(job, state.next_frame_index)
        if not frame_bytes:
            break
        state.pending_raw_frames.append(frame_bytes)
        state.next_frame_index += 1

    interpolation_steps = resolve_interpolation_steps(state.estimated_generation_fps)
    while state.pending_raw_frames and len(state.pending_stream_frames) < VIDEO_STREAM_SERVER_BUFFER_FRAMES * (
        VIDEO_STREAM_INTERPOLATION_MAX_STEPS + 1
    ):
        current_raw_frame = state.pending_raw_frames.popleft()
        sequence = build_interpolated_sequence(
            previous_frame_bytes=state.previous_raw_frame_bytes,
            current_frame_bytes=current_raw_frame,
            interpolation_steps=interpolation_steps,
        )
        for stream_frame in sequence:
            state.pending_stream_frames.append(stream_frame)
        state.previous_raw_frame_bytes = current_raw_frame
    return stream_status


def resolve_avatar_talking_frame(
    job: JobRecord,
    state: AvatarTalkingFrameState,
    canvas_size: tuple[int, int],
    output_fps: float,
) -> tuple[np.ndarray | None, bool]:
    """
    Resolve the next frame for one talking job inside the continuous avatar stream.
    """
    stream_status = refresh_avatar_source_frame_images(job, state, canvas_size, state.start_frame_index + 1)
    elapsed_talking_sec = max(0.0, time.perf_counter() - float(state.playback_started_at_perf))
    if float(job.audio_duration_sec or 0.0) > 0.0 and elapsed_talking_sec >= float(job.audio_duration_sec):
        return None, True
    desired_source_position_zero_based = float(state.start_frame_index - 1) + (
        elapsed_talking_sec * float(state.playback_fps)
    )
    if state.virtual_source_position_zero_based is None:
        state.virtual_source_position_zero_based = float(state.start_frame_index - 1)
    source_position_step = float(state.playback_fps) / float(max(1.0, float(output_fps)))
    candidate_source_position_zero_based = min(
        desired_source_position_zero_based,
        float(state.virtual_source_position_zero_based) + max(source_position_step, 0.0),
    )
    maximum_available_source_position_zero_based = max(0.0, float(state.last_known_frame_index - 1))
    source_position_zero_based = min(
        candidate_source_position_zero_based,
        maximum_available_source_position_zero_based,
    )
    source_position_advanced = (
        source_position_zero_based > (float(state.virtual_source_position_zero_based) + 1e-6)
    )
    state.virtual_source_position_zero_based = source_position_zero_based
    base_frame_index = int(math.floor(source_position_zero_based)) + 1
    next_frame_index = base_frame_index + 1
    blend_alpha = source_position_zero_based - math.floor(source_position_zero_based)
    stream_status = refresh_avatar_source_frame_images(job, state, canvas_size, next_frame_index)

    job_state = determine_job_state(job, stream_status)
    running = bool(job.process is not None and job.process.poll() is None)
    final_frame_total = state.last_known_frame_total or state.last_known_frame_index
    if (
        job_state in {"error", "canceled"}
        or (
            job_state == "done"
            and not running
            and final_frame_total > 0
            and source_position_zero_based > float(final_frame_total - 1)
        )
    ):
        return None, True

    base_frame_image = state.source_frame_images.get(base_frame_index)
    if base_frame_image is None:
        fallback_frame_image = resolve_avatar_bounce_frame_image(state, base_frame_index)
        if fallback_frame_image is None:
            return None, False
        state.last_frame_image = fallback_frame_image
        state.output_frame_index += 1
        return fallback_frame_image, False

    next_frame_image = state.source_frame_images.get(next_frame_index)
    if not source_position_advanced and state.last_output_source_frame_index == base_frame_index:
        fallback_frame_image = resolve_avatar_bounce_frame_image(state, base_frame_index)
        if fallback_frame_image is not None:
            state.last_frame_image = fallback_frame_image
            state.output_frame_index += 1
            return fallback_frame_image, False
    frame_image = blend_avatar_source_frames(base_frame_image, next_frame_image, blend_alpha)
    reset_avatar_bounce_fallback_state(state)
    state.last_frame_image = frame_image
    state.last_output_source_frame_index = base_frame_index
    state.output_frame_index += 1
    return frame_image, False


def update_webrtc_avatar_talking_frame_state(
    job: JobRecord,
    state: AvatarTalkingFrameState,
    canvas_size: tuple[int, int],
) -> dict[str, Any] | None:
    """
    Refresh the progressive talking frame buffer used by the continuous WebRTC avatar track.
    """
    stream_status = read_job_stream_status(job)
    state.estimated_generation_fps = estimate_generation_fps(stream_status, state.estimated_generation_fps)
    playback_fps = resolve_stream_playback_fps(stream_status)
    if playback_fps > 0:
        state.playback_fps = playback_fps
    status_frame_index = parse_status_int(stream_status, "frameIndex")
    status_frame_total = parse_status_int(stream_status, "frameTotal")
    if status_frame_index > state.last_known_frame_index:
        state.last_known_frame_index = status_frame_index
    if status_frame_total > state.last_known_frame_total:
        state.last_known_frame_total = status_frame_total

    while (
        len(state.pending_frame_images) < VIDEO_STREAM_SERVER_BUFFER_FRAMES
        and state.next_frame_index <= state.last_known_frame_index
    ):
        frame_bytes = read_stream_frame_by_index(job, state.next_frame_index)
        if not frame_bytes:
            break
        normalized_frame_image = normalize_job_stream_frame_image_to_canvas(
            job,
            state,
            frame_bytes,
            canvas_size,
            stream_status,
        )
        if normalized_frame_image is None:
            break
        state.pending_frame_images.append(normalized_frame_image)
        state.next_frame_index += 1
    return stream_status


def resolve_webrtc_avatar_talking_frame_image(
    job: JobRecord,
    state: AvatarTalkingFrameState,
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray | None, bool]:
    """
    Resolve the next progressive talking frame for the persistent WebRTC avatar video track.
    """
    stream_status = update_webrtc_avatar_talking_frame_state(job, state, canvas_size)
    elapsed_talking_sec = max(0.0, time.perf_counter() - float(state.playback_started_at_perf))
    if float(job.audio_duration_sec or 0.0) > 0.0 and elapsed_talking_sec >= float(job.audio_duration_sec):
        return state.last_frame_image, True
    if state.pending_frame_images:
        frame_image = state.pending_frame_images.popleft()
        state.last_frame_image = frame_image
        return frame_image, False

    job_state = determine_job_state(job, stream_status)
    running = bool(job.process is not None and job.process.poll() is None)
    final_frame_total = state.last_known_frame_total or state.last_known_frame_index
    is_finished = (
        job_state in {"error", "canceled"}
        or (
            job_state == "done"
            and not running
            and final_frame_total > 0
            and state.next_frame_index > final_frame_total
            and not state.pending_frame_images
        )
    )
    return state.last_frame_image, is_finished


def build_interpolated_sequence(
    previous_frame_bytes: bytes | None,
    current_frame_bytes: bytes,
    interpolation_steps: int,
) -> list[bytes]:
    """
    Create interpolated frame sequence from previous and current JPEG frames.
    """
    if interpolation_steps <= 0 or not previous_frame_bytes:
        return [current_frame_bytes]
    previous_frame = decode_jpeg_frame(previous_frame_bytes)
    current_frame = decode_jpeg_frame(current_frame_bytes)
    if previous_frame is None or current_frame is None:
        return [current_frame_bytes]
    if previous_frame.shape != current_frame.shape:
        return [current_frame_bytes]

    sequence: list[bytes] = []
    total_steps = interpolation_steps + 1
    for step in range(1, total_steps):
        alpha = float(step) / float(total_steps)
        beta = 1.0 - alpha
        blended = cv2.addWeighted(previous_frame, beta, current_frame, alpha, 0.0)
        blended_bytes = encode_jpeg_frame(blended)
        if blended_bytes:
            sequence.append(blended_bytes)
    sequence.append(current_frame_bytes)
    return sequence


def build_interpolated_image_sequence(
    previous_frame_image: np.ndarray | None,
    current_frame_image: np.ndarray,
    interpolation_steps: int,
) -> list[np.ndarray]:
    """
    Create interpolated frame sequence from previous and current frame images.
    """
    if interpolation_steps <= 0 or previous_frame_image is None:
        return [current_frame_image]
    if previous_frame_image.shape != current_frame_image.shape:
        return [current_frame_image]

    sequence: list[np.ndarray] = []
    total_steps = interpolation_steps + 1
    for step in range(1, total_steps):
        alpha = float(step) / float(total_steps)
        beta = 1.0 - alpha
        sequence.append(cv2.addWeighted(previous_frame_image, beta, current_frame_image, alpha, 0.0))
    sequence.append(current_frame_image)
    return sequence


def build_avatar_transition_pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one reusable pixel-coordinate grid for dense avatar frame warping.
    """
    return np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))


def compute_avatar_transition_optical_flow(
    gray_frame_a: np.ndarray,
    gray_frame_b: np.ndarray,
) -> np.ndarray:
    """
    Compute dense optical flow between two grayscale avatar frames.
    """
    return cv2.calcOpticalFlowFarneback(
        gray_frame_a,
        gray_frame_b,
        None,
        pyr_scale=AVATAR_RETURN_TO_IDLE_FLOW_PYR_SCALE,
        levels=AVATAR_RETURN_TO_IDLE_FLOW_LEVELS,
        winsize=AVATAR_RETURN_TO_IDLE_FLOW_WINDOW_SIZE,
        iterations=AVATAR_RETURN_TO_IDLE_FLOW_ITERATIONS,
        poly_n=AVATAR_RETURN_TO_IDLE_FLOW_POLY_N,
        poly_sigma=AVATAR_RETURN_TO_IDLE_FLOW_POLY_SIGMA,
        flags=0,
    )


def warp_avatar_transition_image_with_flow(
    image: np.ndarray,
    flow: np.ndarray,
    factor: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """
    Warp one avatar frame using a scaled optical-flow field.
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


def build_avatar_transition_flow_consistency_mask(
    flow_forward: np.ndarray,
    flow_backward: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """
    Estimate one soft validity mask for bidirectional avatar optical flow.
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
    validity_mask = (
        residual
        <= (
            float(AVATAR_RETURN_TO_IDLE_FLOW_CONSISTENCY_THRESHOLD)
            + (float(AVATAR_RETURN_TO_IDLE_FLOW_CONSISTENCY_SCALE) * magnitude)
        )
    ).astype(np.float32)
    blur_kernel = int(AVATAR_RETURN_TO_IDLE_FLOW_MASK_BLUR_KERNEL)
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        validity_mask = cv2.GaussianBlur(validity_mask, (blur_kernel, blur_kernel), 0)
    return np.clip(validity_mask, 0.0, 1.0)


def synthesize_avatar_transition_frame(
    start_frame_image: np.ndarray,
    end_frame_image: np.ndarray,
    flow_forward: np.ndarray,
    flow_backward: np.ndarray,
    mask_forward: np.ndarray,
    mask_backward: np.ndarray,
    alpha: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """
    Synthesize one intermediate return-to-idle frame with bidirectional optical flow.
    """
    safe_alpha = clamp_float(float(alpha), 0.0, 1.0)
    warped_start_frame = warp_avatar_transition_image_with_flow(
        start_frame_image,
        flow_forward,
        safe_alpha,
        grid_x,
        grid_y,
    ).astype(np.float32)
    warped_end_frame = warp_avatar_transition_image_with_flow(
        end_frame_image,
        flow_backward,
        1.0 - safe_alpha,
        grid_x,
        grid_y,
    ).astype(np.float32)
    warped_start_mask = warp_avatar_transition_image_with_flow(
        mask_forward,
        flow_forward,
        safe_alpha,
        grid_x,
        grid_y,
    ).astype(np.float32)
    warped_end_mask = warp_avatar_transition_image_with_flow(
        mask_backward,
        flow_backward,
        1.0 - safe_alpha,
        grid_x,
        grid_y,
    ).astype(np.float32)
    start_weight = ((1.0 - safe_alpha) * warped_start_mask)[..., None]
    end_weight = (safe_alpha * warped_end_mask)[..., None]
    weight_sum = np.maximum(start_weight + end_weight, AVATAR_RETURN_TO_IDLE_FLOW_WEIGHT_EPSILON)
    blended = ((warped_start_frame * start_weight) + (warped_end_frame * end_weight)) / weight_sum
    fallback_frame = blend_avatar_source_frames(start_frame_image, end_frame_image, safe_alpha).astype(np.float32)
    transition_confidence = np.clip(start_weight + end_weight, 0.0, 1.0)
    stabilized_frame = (
        (blended * transition_confidence)
        + (fallback_frame * (1.0 - transition_confidence))
    )
    return np.clip(stabilized_frame, 0, 255).astype(np.uint8)


def resolve_avatar_idle_reference_frame_image(canvas_size: tuple[int, int]) -> np.ndarray | None:
    """
    Resolve the first idle-loop frame resized into the shared avatar canvas.
    """
    idle_source_frame_abs = ensure_idle_source_frame_abs()
    if idle_source_frame_abs is not None:
        idle_source_frame_image = cv2.imread(str(idle_source_frame_abs), cv2.IMREAD_COLOR)
        if idle_source_frame_image is not None:
            return fit_frame_to_canvas(idle_source_frame_image, canvas_size)
    idle_video_abs = resolve_idle_video_abs()
    if idle_video_abs is None:
        return None
    capture = cv2.VideoCapture(str(idle_video_abs))
    if not capture.isOpened():
        return None
    try:
        frame_ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not frame_ok or frame_bgr is None:
        return None
    return fit_frame_to_canvas(frame_bgr, canvas_size)


def resolve_avatar_return_to_idle_frame_count(frames_per_second: float) -> int:
    """
    Resolve how many frames the return-to-idle motion should emit.
    """
    if not AVATAR_RETURN_TO_IDLE_ENABLED or AVATAR_RETURN_TO_IDLE_DURATION_SEC <= 0:
        return 0
    return resolve_avatar_transition_frame_count(
        frames_per_second=frames_per_second,
        duration_sec=AVATAR_RETURN_TO_IDLE_DURATION_SEC,
        minimum_frame_count=AVATAR_RETURN_TO_IDLE_MIN_FRAME_COUNT,
        maximum_frame_count=AVATAR_RETURN_TO_IDLE_MAX_FRAME_COUNT,
    )


def resolve_avatar_idle_to_talking_frame_count(frames_per_second: float) -> int:
    """
    Resolve how many frames the idle-to-talking motion should emit.
    """
    if not AVATAR_IDLE_TO_TALKING_ENABLED or AVATAR_IDLE_TO_TALKING_DURATION_SEC <= 0:
        return 0
    return resolve_avatar_transition_frame_count(
        frames_per_second=frames_per_second,
        duration_sec=AVATAR_IDLE_TO_TALKING_DURATION_SEC,
        minimum_frame_count=AVATAR_IDLE_TO_TALKING_MIN_FRAME_COUNT,
        maximum_frame_count=AVATAR_IDLE_TO_TALKING_MAX_FRAME_COUNT,
    )


def resolve_avatar_transition_frame_count(
    frames_per_second: float,
    duration_sec: float,
    minimum_frame_count: int,
    maximum_frame_count: int,
) -> int:
    """
    Resolve one bounded avatar transition frame count from duration and playback FPS.
    """
    safe_frames_per_second = max(1.0, float(frames_per_second or AVATAR_VIDEO_OUTPUT_FPS))
    estimated_frame_count = int(round(float(duration_sec) * safe_frames_per_second))
    bounded_frame_count = max(int(minimum_frame_count), estimated_frame_count)
    return min(int(maximum_frame_count), bounded_frame_count)


def build_avatar_transition_frame_sequence(
    start_frame_image: np.ndarray | None,
    end_frame_image: np.ndarray | None,
    canvas_size: tuple[int, int],
    target_frame_count: int,
) -> list[np.ndarray]:
    """
    Build one short avatar motion bridge between two concrete frames.
    """
    if start_frame_image is None or end_frame_image is None or target_frame_count <= 0:
        return []
    normalized_start_frame_image = fit_frame_to_canvas(start_frame_image, canvas_size)
    normalized_end_frame_image = fit_frame_to_canvas(end_frame_image, canvas_size)
    interpolation_steps = max(0, int(target_frame_count) - 1)
    fallback_sequence = build_interpolated_image_sequence(
        normalized_start_frame_image,
        normalized_end_frame_image,
        interpolation_steps,
    )
    if normalized_start_frame_image.shape != normalized_end_frame_image.shape:
        return fallback_sequence
    try:
        gray_start_frame = cv2.cvtColor(normalized_start_frame_image, cv2.COLOR_BGR2GRAY)
        gray_end_frame = cv2.cvtColor(normalized_end_frame_image, cv2.COLOR_BGR2GRAY)
        flow_forward = compute_avatar_transition_optical_flow(gray_start_frame, gray_end_frame)
        flow_backward = compute_avatar_transition_optical_flow(gray_end_frame, gray_start_frame)
        frame_height, frame_width = normalized_start_frame_image.shape[:2]
        grid_x, grid_y = build_avatar_transition_pixel_grid(frame_width, frame_height)
        mask_forward = build_avatar_transition_flow_consistency_mask(
            flow_forward=flow_forward,
            flow_backward=flow_backward,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        mask_backward = build_avatar_transition_flow_consistency_mask(
            flow_forward=flow_backward,
            flow_backward=flow_forward,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        transition_sequence: list[np.ndarray] = []
        for step in range(1, interpolation_steps + 1):
            alpha = float(step) / float(interpolation_steps + 1)
            transition_sequence.append(
                synthesize_avatar_transition_frame(
                    start_frame_image=normalized_start_frame_image,
                    end_frame_image=normalized_end_frame_image,
                    flow_forward=flow_forward,
                    flow_backward=flow_backward,
                    mask_forward=mask_forward,
                    mask_backward=mask_backward,
                    alpha=alpha,
                    grid_x=grid_x,
                    grid_y=grid_y,
                )
            )
        transition_sequence.append(normalized_end_frame_image.copy())
        return transition_sequence
    except cv2.error:
        return fallback_sequence


def build_avatar_return_to_idle_frame_sequence(
    start_frame_image: np.ndarray | None,
    canvas_size: tuple[int, int],
    frames_per_second: float,
) -> list[np.ndarray]:
    """
    Build one short motion bridge from the last talking frame back to the idle-loop first frame.
    """
    if start_frame_image is None:
        return []
    target_frame_count = resolve_avatar_return_to_idle_frame_count(frames_per_second)
    if target_frame_count <= 0:
        return []
    idle_reference_frame_image = resolve_avatar_idle_reference_frame_image(canvas_size)
    return build_avatar_transition_frame_sequence(
        start_frame_image=start_frame_image,
        end_frame_image=idle_reference_frame_image,
        canvas_size=canvas_size,
        target_frame_count=target_frame_count,
    )


def build_avatar_idle_to_talking_frame_sequence(
    start_frame_image: np.ndarray | None,
    end_frame_image: np.ndarray | None,
    canvas_size: tuple[int, int],
    frames_per_second: float,
) -> list[np.ndarray]:
    """
    Build one short motion bridge from the current idle frame into the first talking frame.
    """
    target_frame_count = resolve_avatar_idle_to_talking_frame_count(frames_per_second)
    return build_avatar_transition_frame_sequence(
        start_frame_image=start_frame_image,
        end_frame_image=end_frame_image,
        canvas_size=canvas_size,
        target_frame_count=target_frame_count,
    )


def read_stream_frame_by_index(job: JobRecord, frame_index: int) -> bytes | None:
    """
    Read sequential stream frame by index.
    """
    return read_job_stream_frame_by_index(job, frame_index)


def resolve_driving_media_url(job: JobRecord) -> str:
    """
    Resolve driving media URL for UI playback.
    """
    for extension in (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4"):
        candidate = job.output_abs / f"driving{extension}"
        if candidate.exists():
            return f"/jobs/{job.job_id}/{candidate.name}"
    return f"/jobs/{job.job_id}/inputs/{job.audio_input_abs.name}"


def build_public_preview_composition_payload(
    job: JobRecord,
    stream_status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Convert preview-composition status metadata into public API URLs.
    """
    preview_composition = resolve_preview_composition_status(stream_status)
    if preview_composition is None:
        return None
    payload = dict(preview_composition)
    mask_image_name = str(payload.get("maskImage") or PREVIEW_COMPOSITION_MASK_NAME)
    payload["maskImage"] = mask_image_name
    payload["maskImageUrl"] = f"/jobs/{job.job_id}/stream/{mask_image_name}"
    payload["sourceFrameUrl"] = build_public_file_url(job.source_frame_abs)
    return payload


def build_warmup_command(audio_rel_path: Path) -> list[str]:
    """
    Build warmup runner command that primes container, models and kernels.
    """
    source_frame_arg = resolve_warmup_source_frame_arg()
    if not source_frame_arg:
        raise RuntimeError("Warmup source frame not found. Expected an image in output/frames.")

    output_root_rel = WARMUP_OUTPUT_ROOT_REL
    stream_rel = output_root_rel / WARMUP_STREAM_SUBDIR_NAME
    command = [
        str(RUNNER_PYTHON),
        str(RUNNER_SCRIPT),
        "--backend",
        DEFAULT_BACKEND,
        "--trt-runtime",
        DEFAULT_TRT_RUNTIME,
        "--trt-precision",
        DEFAULT_TRT_PRECISION,
        "--mode",
        "preview",
        "--source-frame",
        source_frame_arg,
        "--driving-audio",
        normalize_rel_path(str(audio_rel_path)),
        "--audio-motion-stride",
        str(DEFAULT_AUDIO_MOTION_STRIDE),
        "--render-batch-size",
        str(DEFAULT_RENDER_BATCH_SIZE),
        "--trt-engine-batch-size",
        str(DEFAULT_TRT_ENGINE_BATCH_SIZE),
        "--video-encoder",
        DEFAULT_VIDEO_ENCODER,
        "--output-dir",
        normalize_rel_path(str(output_root_rel)),
        "--stream-dir",
        normalize_rel_path(str(stream_rel)),
        "--stream-shm-prefix",
        WARMUP_STREAM_SHM_PREFIX,
        "--animation-region",
        DEFAULT_ANIMATION_REGION,
    ]
    if should_defer_preview_paste_back(
        "preview",
        DEFAULT_PASTE_BACK_ENABLED,
        DEFAULT_STITCHING_ENABLED,
        source_frame_arg,
    ):
        command.append("--defer-paste-back")
    elif not DEFAULT_PASTE_BACK_ENABLED:
        command.append("--no-paste-back")
    if not DEFAULT_STITCHING_ENABLED:
        command.append("--no-stitching")
    if not DEFAULT_RELATIVE_MOTION_ENABLED:
        command.append("--no-relative-motion")
    if DEFAULT_SKIP_TRT_ENGINE_BUILD:
        command.append("--skip-trt-engine-build")
    return command


def set_warmup_state(phase: str, progress: float, message: str = "", error: str = "") -> None:
    """
    Update the public warmup state exposed through the health payload.
    """
    global WARMUP_PHASE
    global WARMUP_PROGRESS
    global WARMUP_MESSAGE
    global WARMUP_ERROR
    with WARMUP_LOCK:
        WARMUP_PHASE = str(phase or "idle")
        WARMUP_PROGRESS = clamp_float(float(progress), 0.0, 1.0)
        WARMUP_MESSAGE = str(message or "")
        WARMUP_ERROR = str(error or "")


def read_warmup_stream_status() -> dict[str, Any] | None:
    """
    Read the live warmup stream status when the warmup runner has started emitting frames.
    """
    try:
        reader = JobStreamSharedMemoryReader(WARMUP_STREAM_SHM_PREFIX)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    try:
        return reader.read_status_payload()
    except Exception:
        return None
    finally:
        with contextlib.suppress(Exception):
            reader.close()


def ensure_warmup_audio_file(audio_abs_path: Path) -> None:
    """
    Ensure warmup silence WAV exists for startup preheat run.
    """
    if audio_abs_path.exists():
        return
    audio_abs_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG_BINARY,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-t",
        f"{WARMUP_AUDIO_DURATION_SEC:.3f}",
        "-acodec",
        "pcm_s16le",
        str(audio_abs_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def run_startup_warmup_once() -> None:
    """
    Execute one non-streaming warmup generation if enabled.
    """
    global WARMUP_RUNNING
    global WARMUP_LAST_STARTED_AT_MS
    if not WARMUP_ENABLED:
        return
    with WARMUP_LOCK:
        if WARMUP_RUNNING:
            return
        WARMUP_RUNNING = True
        WARMUP_LAST_STARTED_AT_MS = now_ms()
    set_warmup_state("starting", 0.0, "starting warmup")
    try:
        time.sleep(WARMUP_START_DELAY_SEC)
        output_root_abs = PROJECT_ROOT / WARMUP_OUTPUT_ROOT_REL
        input_dir_abs = output_root_abs / WARMUP_INPUTS_SUBDIR_NAME
        warmup_audio_abs = input_dir_abs / WARMUP_AUDIO_FILE_NAME
        warmup_audio_rel = warmup_audio_abs.relative_to(PROJECT_ROOT)
        stream_dir_abs = output_root_abs / WARMUP_STREAM_SUBDIR_NAME
        stream_dir_abs.mkdir(parents=True, exist_ok=True)
        set_warmup_state("prepare-audio", 0.05, "preparing warmup audio")
        ensure_warmup_audio_file(warmup_audio_abs)
        command = build_warmup_command(warmup_audio_rel)
        print("[warmup] starting runtime warmup job")
        set_warmup_state("launch-runner", 0.1, "launching warmup runner")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
        )
        last_status_message = "initializing models"
        while True:
            stream_status = read_warmup_stream_status()
            if isinstance(stream_status, dict):
                progress_ratio = float(stream_status.get("progress") or 0.0)
                frame_index = parse_status_int(stream_status, "frameIndex")
                frame_total = parse_status_int(stream_status, "frameTotal")
                latest_message = str(stream_status.get("message") or "rendering")
                if frame_total > 0:
                    last_status_message = f"{latest_message} ({frame_index}/{frame_total})"
                else:
                    last_status_message = latest_message
                set_warmup_state(
                    "rendering",
                    0.1 + (clamp_float(progress_ratio, 0.0, 1.0) * 0.85),
                    last_status_message,
                )
            elif process.poll() is None:
                set_warmup_state("initializing-models", 0.1, last_status_message)

            if process.poll() is not None:
                break
            time.sleep(max(0.1, JOB_POLL_SLEEP_SEC))

        if process.returncode != 0:
            error_message = f"warmup runner exited with code {process.returncode}"
            set_warmup_state("error", 0.0, "warmup failed", error_message)
            raise RuntimeError(error_message)
        set_warmup_state("completed", 1.0, "warmup completed")
        print("[warmup] completed runtime warmup job")
    except Exception as exc:
        set_warmup_state("error", 0.0, "warmup failed", str(exc))
        print(f"[warmup] failed: {exc}")
    finally:
        with WARMUP_LOCK:
            WARMUP_RUNNING = False


def build_job_payload(job: JobRecord) -> dict[str, Any]:
    """
    Build API response payload for job.
    """
    stream_status = read_job_stream_status(job)
    report = read_json(job.report_abs)
    process = job.process
    running = bool(job.exit_code is None and process is not None and process.poll() is None)
    state = determine_job_state(job, stream_status)
    exit_code = job.exit_code if job.exit_code is not None else (process.poll() if process is not None else None)
    queued_position = queue_position(job.job_id)
    public_stream_status = dict(stream_status or {})
    public_preview_composition = build_public_preview_composition_payload(job, stream_status)
    if public_preview_composition is not None:
        public_stream_status[PREVIEW_COMPOSITION_STATUS_KEY] = public_preview_composition
    source_preview_url = build_public_file_url(job.source_frame_abs)
    source_media_type = "video" if is_video_backed_source_path(job.source_frame_abs) else "image"
    source_input_kind = resolve_source_input_kind(job.source_frame_abs)
    source_template_payload: dict[str, Any] | None = None
    if job.source_template_pack_abs is not None:
        source_template_payload = build_source_template_pack_record(job.source_template_pack_abs)
        source_preview_url = str(source_template_payload.get("previewUrl", "") or source_preview_url)
        source_media_type = "video" if str(source_template_payload.get("sourceType", "")) == "video" else "image"
    payload: dict[str, Any] = {
        "jobId": job.job_id,
        "avatarSessionId": job.avatar_session_id,
        "state": state,
        "running": running,
        "exitCode": exit_code,
        "createdAtMs": job.created_at_ms,
        "startedAtMs": job.started_at_ms,
        "finishedAtMs": job.finished_at_ms,
        "queuePosition": queued_position,
        "mode": job.mode,
        "audioDurationSec": job.audio_duration_sec,
        "audioMotionStride": job.audio_motion_stride,
        "generationFrameCount": job.generation_frame_count,
        "audioEyeTamedPreset": job.audio_eye_tamed_preset,
        "audioEyeSoftFactor": job.audio_eye_soft_factor,
        "audioEyeHardFactor": job.audio_eye_hard_factor,
        "audioEyeHardDyMin": job.audio_eye_hard_dy_min,
        "audioEyeHardDyMax": job.audio_eye_hard_dy_max,
        "audioMotionTuningEnabled": job.audio_motion_tuning_enabled,
        "audioReanchorFirstN": job.audio_reanchor_first_n,
        "audioMouthOpenFactor": job.audio_mouth_open_factor,
        "audioPoseSmoothWindow": job.audio_pose_smooth_window,
        "audioExpSmoothWindow": job.audio_exp_smooth_window,
        "audioPoseJumpThreshold": job.audio_pose_jump_threshold,
        "audioTranslationJumpThreshold": job.audio_translation_jump_threshold,
        "audioLipSyncAssist": job.audio_lip_sync_assist,
        "audioLipSyncMinRatio": job.audio_lip_sync_min_ratio,
        "audioLipSyncMaxRatio": job.audio_lip_sync_max_ratio,
        "audioLipSyncSmoothWindow": job.audio_lip_sync_smooth_window,
        "audioLipSyncStrength": job.audio_lip_sync_strength,
        "audioLipSyncPower": job.audio_lip_sync_power,
        "audioLipSyncAttack": job.audio_lip_sync_attack,
        "audioLipSyncRelease": job.audio_lip_sync_release,
        "audioLipSyncOffsetMs": job.audio_lip_sync_offset_ms,
        "audioMouthFloorStrength": job.audio_mouth_floor_strength,
        "audioMouthPeakClamp": job.audio_mouth_peak_clamp,
        "drivingMultiplier": job.driving_multiplier,
        "cfgScale": job.cfg_scale,
        "joyvasaInferenceSteps": job.joyvasa_inference_steps,
        "animationRegion": job.animation_region,
        "stitchingEnabled": job.stitching_enabled,
        "relativeMotionEnabled": job.relative_motion_enabled,
        "pasteBackEnabled": job.paste_back_enabled,
        "deferPasteBackEnabled": job.defer_paste_back_enabled,
        "sourceFrame": job.source_frame_arg,
        "sourceFrameUrl": source_preview_url,
        "sourceMediaType": source_media_type,
        "sourceInputKind": source_input_kind,
        "sourceTemplatePackId": job.source_template_pack_id,
        "status": public_stream_status,
        "previewComposition": public_preview_composition,
        "streamUrl": f"/api/jobs/{job.job_id}/stream.mjpg",
        "wsUrl": f"/ws/jobs/{job.job_id}",
        "videoWsUrl": f"/ws/jobs/{job.job_id}/video",
        "statusUrl": f"/api/jobs/{job.job_id}/status",
        "logUrl": f"/api/jobs/{job.job_id}/log",
        "reportUrl": f"/api/jobs/{job.job_id}/report",
        "resultVideoUrl": f"/jobs/{job.job_id}/result.mp4" if job.result_abs.exists() else "",
        "resultConcatUrl": f"/jobs/{job.job_id}/result_concat.mp4" if job.result_concat_abs.exists() else "",
        "drivingMediaUrl": resolve_driving_media_url(job),
        "avatarReadyAtMs": job.avatar_ready_at_ms,
        "avatarPlayStartedAtMs": job.avatar_play_started_at_ms,
        "avatarPlayFinishedAtMs": job.avatar_play_finished_at_ms,
    }
    if source_template_payload is not None:
        payload["sourceTemplatePack"] = source_template_payload
    if report is not None:
        payload["report"] = report
    return payload


async def save_upload_file(upload: UploadFile, target_path: Path) -> None:
    """
    Save uploaded file in chunks.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


async def resolve_requested_source_frame(
    source_frame: str,
    source_template_pack: str,
    source_image: UploadFile | None,
    source_video: UploadFile | None,
    output_abs: Path,
    output_rel: Path,
    audio_duration_sec: float,
    avatar_session_id: str,
) -> tuple[Path, str]:
    """
    Resolve source frame from optional uploaded image or local path input.
    """
    has_source_image_upload = bool(source_image is not None and source_image.filename)
    has_source_video_upload = bool(source_video is not None and source_video.filename)
    if has_source_image_upload and has_source_video_upload:
        raise HTTPException(status_code=400, detail="Provide either source_image or source_video, not both.")
    normalized_source_frame = str(source_frame or "").strip() or DEFAULT_SOURCE_FRAME
    normalized_source_template_pack = str(source_template_pack or "").strip()

    if normalized_source_template_pack:
        if has_source_image_upload or has_source_video_upload:
            emit_runtime_log(
                "source",
                f"source_template_pack takes precedence over uploaded source media template={normalized_source_template_pack}",
            )
        source_template_pack_abs = resolve_source_template_pack_path(normalized_source_template_pack)
        source_template_pack_rel = normalize_rel_path(str(source_template_pack_abs.relative_to(PROJECT_ROOT)))
        emit_runtime_log(
            "source",
            (
                f"resolved source template pack template={source_template_pack_abs.name} "
                f"video_backed={is_video_backed_source_path(source_template_pack_abs)}"
            ),
        )
        return source_template_pack_abs, source_template_pack_rel

    if not has_source_image_upload and not has_source_video_upload:
        try:
            fixed_source = resolve_configured_fixed_source_frame()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if fixed_source is not None and normalized_source_frame == DEFAULT_SOURCE_FRAME:
            emit_runtime_log("source", f"using fixed source frame source={fixed_source[1]}")
            return fixed_source
        if normalized_source_frame == DEFAULT_SOURCE_FRAME:
            idle_anchor_source = resolve_avatar_idle_anchor_source_frame(audio_duration_sec, avatar_session_id)
            if idle_anchor_source is not None:
                emit_runtime_log("source", f"using idle anchor source source={idle_anchor_source[1]}")
                return idle_anchor_source
        resolved_source = resolve_source_frame_candidate(normalized_source_frame)
        emit_runtime_log("source", f"using requested source frame source={resolved_source[1]}")
        return resolved_source

    if has_source_video_upload:
        extension = Path(str(source_video.filename)).suffix.lower()
        if extension not in ALLOWED_SOURCE_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported source video extension '{extension}'. "
                    f"Allowed: {sorted(ALLOWED_SOURCE_VIDEO_EXTENSIONS)}"
                ),
            )
        source_video_rel = output_rel / "inputs" / f"source{extension}"
        source_video_abs = (PROJECT_ROOT / source_video_rel).resolve()
        output_abs.mkdir(parents=True, exist_ok=True)
        await save_upload_file(source_video, source_video_abs)
        emit_runtime_log("source", f"saved uploaded source video path={source_video_abs}")
        return source_video_abs, normalize_rel_path(str(source_video_rel))

    extension = Path(str(source_image.filename)).suffix.lower()
    if extension not in ALLOWED_SOURCE_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source image extension '{extension}'. Allowed: {sorted(ALLOWED_SOURCE_IMAGE_EXTENSIONS)}",
        )
    source_image_rel = output_rel / "inputs" / f"source{extension}"
    source_image_abs = (PROJECT_ROOT / source_image_rel).resolve()
    output_abs.mkdir(parents=True, exist_ok=True)
    await save_upload_file(source_image, source_image_abs)
    emit_runtime_log("source", f"saved uploaded source image path={source_image_abs}")
    return source_image_abs, normalize_rel_path(str(source_image_rel))


async def create_and_enqueue_audio_job(
    avatar_session_id: str,
    audio: UploadFile,
    source_image: UploadFile | None,
    source_video: UploadFile | None,
    source_frame: str,
    source_template_pack: str,
    audio_tuning_preset: str,
    mode: str,
    motion_stride: int,
    generation_frame_count: int | None,
    audio_eye_tamed_preset: bool,
    audio_eye_soft_factor: float,
    audio_eye_hard_factor: float,
    audio_eye_hard_dy_min: float,
    audio_eye_hard_dy_max: float,
    audio_motion_tuning_enabled: bool,
    audio_reanchor_first_n: int,
    audio_mouth_open_factor: float,
    audio_pose_smooth_window: int,
    audio_exp_smooth_window: int,
    audio_pose_jump_threshold: float,
    audio_translation_jump_threshold: float,
    audio_lip_sync_assist: bool,
    audio_lip_sync_min_ratio: float,
    audio_lip_sync_max_ratio: float,
    audio_lip_sync_smooth_window: int,
    audio_lip_sync_strength: float,
    audio_lip_sync_power: float,
    audio_lip_sync_attack: float,
    audio_lip_sync_release: float,
    audio_lip_sync_offset_ms: int,
    audio_mouth_floor_strength: float,
    audio_mouth_peak_clamp: float,
    driving_multiplier: float,
    cfg_scale: float,
    joyvasa_inference_steps: int,
    animation_region: str,
    stitching: bool,
    relative_motion: bool,
    paste_back: bool,
) -> dict[str, Any]:
    """
    Validate one audio generation request, register the job, and enqueue it for playback/rendering.
    """
    ensure_avatar_session_state(avatar_session_id)
    ensure_runtime_accepting_requests()
    ensure_job_worker_started()
    ensure_avatar_worker_started()
    extension = Path(audio.filename or "").suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio extension '{extension}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )
    if mode not in {"preview", "full"}:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    if int(motion_stride) < 1 or int(motion_stride) > 6:
        raise HTTPException(status_code=400, detail="Invalid motion_stride. Allowed range: 1..6")
    normalized_generation_frame_count: int | None = None
    if generation_frame_count is not None:
        normalized_generation_frame_count = int(generation_frame_count)
        if (
            normalized_generation_frame_count < GENERATION_FRAME_COUNT_MIN
            or normalized_generation_frame_count > GENERATION_FRAME_COUNT_MAX
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid generation_frame_count. "
                    f"Allowed range: {GENERATION_FRAME_COUNT_MIN}..{GENERATION_FRAME_COUNT_MAX}"
                ),
            )
    normalized_audio_eye_soft_factor = float(audio_eye_soft_factor)
    normalized_audio_eye_hard_factor = float(audio_eye_hard_factor)
    if not math.isfinite(normalized_audio_eye_soft_factor):
        raise HTTPException(status_code=400, detail="Invalid audio_eye_soft_factor.")
    if not math.isfinite(normalized_audio_eye_hard_factor):
        raise HTTPException(status_code=400, detail="Invalid audio_eye_hard_factor.")
    normalized_audio_eye_soft_factor = min(1.0, max(0.0, normalized_audio_eye_soft_factor))
    normalized_audio_eye_hard_factor = min(1.0, max(0.0, normalized_audio_eye_hard_factor))
    raw_audio_eye_hard_dy_min = float(audio_eye_hard_dy_min)
    raw_audio_eye_hard_dy_max = float(audio_eye_hard_dy_max)
    if not math.isfinite(raw_audio_eye_hard_dy_min):
        raise HTTPException(status_code=400, detail="Invalid audio_eye_hard_dy_min.")
    if not math.isfinite(raw_audio_eye_hard_dy_max):
        raise HTTPException(status_code=400, detail="Invalid audio_eye_hard_dy_max.")
    normalized_audio_eye_hard_dy_min = min(raw_audio_eye_hard_dy_min, raw_audio_eye_hard_dy_max)
    normalized_audio_eye_hard_dy_max = max(raw_audio_eye_hard_dy_min, raw_audio_eye_hard_dy_max)
    normalized_audio_reanchor_first_n = int(audio_reanchor_first_n)
    normalized_audio_reanchor_first_n = min(15, max(1, normalized_audio_reanchor_first_n))
    normalized_audio_mouth_open_factor = float(audio_mouth_open_factor)
    if not math.isfinite(normalized_audio_mouth_open_factor):
        raise HTTPException(status_code=400, detail="Invalid audio_mouth_open_factor.")
    normalized_audio_mouth_open_factor = min(3.0, max(0.0, normalized_audio_mouth_open_factor))
    normalized_audio_pose_smooth_window = min(21, max(0, int(audio_pose_smooth_window)))
    normalized_audio_exp_smooth_window = min(21, max(0, int(audio_exp_smooth_window)))
    normalized_audio_pose_jump_threshold = float(audio_pose_jump_threshold)
    if not math.isfinite(normalized_audio_pose_jump_threshold):
        raise HTTPException(status_code=400, detail="Invalid audio_pose_jump_threshold.")
    normalized_audio_pose_jump_threshold = min(60.0, max(0.0, normalized_audio_pose_jump_threshold))
    normalized_audio_translation_jump_threshold = float(audio_translation_jump_threshold)
    if not math.isfinite(normalized_audio_translation_jump_threshold):
        raise HTTPException(status_code=400, detail="Invalid audio_translation_jump_threshold.")
    normalized_audio_translation_jump_threshold = min(
        1.0,
        max(0.0, normalized_audio_translation_jump_threshold),
    )
    normalized_audio_lip_sync_min_ratio = float(audio_lip_sync_min_ratio)
    normalized_audio_lip_sync_max_ratio = float(audio_lip_sync_max_ratio)
    if not math.isfinite(normalized_audio_lip_sync_min_ratio):
        raise HTTPException(status_code=400, detail="Invalid audio_lip_sync_min_ratio.")
    if not math.isfinite(normalized_audio_lip_sync_max_ratio):
        raise HTTPException(status_code=400, detail="Invalid audio_lip_sync_max_ratio.")
    normalized_audio_lip_sync_min_ratio = min(1.0, max(0.0, normalized_audio_lip_sync_min_ratio))
    normalized_audio_lip_sync_max_ratio = min(1.0, max(normalized_audio_lip_sync_min_ratio, normalized_audio_lip_sync_max_ratio))
    normalized_audio_lip_sync_smooth_window = min(21, max(0, int(audio_lip_sync_smooth_window)))
    normalized_audio_lip_sync_strength = float(audio_lip_sync_strength)
    if not math.isfinite(normalized_audio_lip_sync_strength):
        raise HTTPException(status_code=400, detail="Invalid audio_lip_sync_strength.")
    normalized_audio_lip_sync_strength = min(4.0, max(0.0, normalized_audio_lip_sync_strength))
    normalized_audio_lip_sync_power = float(audio_lip_sync_power)
    if not math.isfinite(normalized_audio_lip_sync_power):
        raise HTTPException(status_code=400, detail="Invalid audio_lip_sync_power.")
    normalized_audio_lip_sync_power = min(4.0, max(0.001, normalized_audio_lip_sync_power))
    normalized_audio_lip_sync_attack = float(audio_lip_sync_attack)
    if not math.isfinite(normalized_audio_lip_sync_attack):
        raise HTTPException(status_code=400, detail="Invalid audio_lip_sync_attack.")
    normalized_audio_lip_sync_attack = min(1.0, max(0.0, normalized_audio_lip_sync_attack))
    normalized_audio_lip_sync_release = float(audio_lip_sync_release)
    if not math.isfinite(normalized_audio_lip_sync_release):
        raise HTTPException(status_code=400, detail="Invalid audio_lip_sync_release.")
    normalized_audio_lip_sync_release = min(1.0, max(0.0, normalized_audio_lip_sync_release))
    normalized_audio_lip_sync_offset_ms = min(1000, max(-1000, int(audio_lip_sync_offset_ms)))
    normalized_audio_mouth_floor_strength = float(audio_mouth_floor_strength)
    if not math.isfinite(normalized_audio_mouth_floor_strength):
        raise HTTPException(status_code=400, detail="Invalid audio_mouth_floor_strength.")
    normalized_audio_mouth_floor_strength = min(4.0, max(0.0, normalized_audio_mouth_floor_strength))
    normalized_audio_mouth_peak_clamp = float(audio_mouth_peak_clamp)
    if not math.isfinite(normalized_audio_mouth_peak_clamp):
        raise HTTPException(status_code=400, detail="Invalid audio_mouth_peak_clamp.")
    normalized_audio_mouth_peak_clamp = min(4.0, max(0.0, normalized_audio_mouth_peak_clamp))
    normalized_driving_multiplier = float(driving_multiplier)
    if not math.isfinite(normalized_driving_multiplier):
        raise HTTPException(status_code=400, detail="Invalid driving_multiplier.")
    normalized_driving_multiplier = min(2.0, max(0.0, normalized_driving_multiplier))
    normalized_cfg_scale = float(cfg_scale)
    if not math.isfinite(normalized_cfg_scale):
        raise HTTPException(status_code=400, detail="Invalid cfg_scale.")
    normalized_cfg_scale = min(10.0, max(0.0, normalized_cfg_scale))
    normalized_joyvasa_inference_steps = int(joyvasa_inference_steps)
    normalized_joyvasa_inference_steps = min(100, max(1, normalized_joyvasa_inference_steps))
    normalized_animation_region = str(animation_region or "").strip().lower()
    if normalized_animation_region not in ANIMATION_REGION_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid animation_region. Allowed values: {sorted(ANIMATION_REGION_CHOICES)}",
        )
    normalized_audio_tuning_preset = str(audio_tuning_preset or "").strip().lower()
    preset_values = build_audio_tuning_preset_values(normalized_audio_tuning_preset)
    if normalized_audio_tuning_preset and not preset_values:
        raise HTTPException(status_code=400, detail=f"Unsupported audio_tuning_preset: {audio_tuning_preset}")
    if preset_values:
        normalized_audio_reanchor_first_n = int(preset_values["audioReanchorFirstN"])
        normalized_audio_mouth_open_factor = float(preset_values["audioMouthOpenFactor"])
        normalized_audio_pose_smooth_window = int(preset_values["audioPoseSmoothWindow"])
        normalized_audio_exp_smooth_window = int(preset_values["audioExpSmoothWindow"])
        normalized_audio_pose_jump_threshold = float(preset_values["audioPoseJumpThreshold"])
        normalized_audio_translation_jump_threshold = float(preset_values["audioTranslationJumpThreshold"])
        normalized_audio_lip_sync_min_ratio = float(preset_values["audioLipSyncMinRatio"])
        normalized_audio_lip_sync_max_ratio = float(preset_values["audioLipSyncMaxRatio"])
        normalized_audio_lip_sync_smooth_window = int(preset_values["audioLipSyncSmoothWindow"])
        normalized_audio_lip_sync_strength = float(preset_values["audioLipSyncStrength"])
        normalized_audio_lip_sync_power = float(preset_values["audioLipSyncPower"])
        normalized_audio_lip_sync_attack = float(preset_values["audioLipSyncAttack"])
        normalized_audio_lip_sync_release = float(preset_values["audioLipSyncRelease"])
        normalized_audio_lip_sync_offset_ms = int(preset_values["audioLipSyncOffsetMs"])
        normalized_audio_mouth_floor_strength = float(preset_values["audioMouthFloorStrength"])
        normalized_audio_mouth_peak_clamp = float(preset_values["audioMouthPeakClamp"])
        normalized_driving_multiplier = float(preset_values["drivingMultiplier"])
        normalized_cfg_scale = float(preset_values["cfgScale"])
        normalized_animation_region = str(preset_values["animationRegion"])
        stitching = bool(preset_values["stitchingEnabled"])
        relative_motion = bool(preset_values["relativeMotionEnabled"])
        audio_motion_tuning_enabled = bool(preset_values["audioMotionTuningEnabled"])
        audio_lip_sync_assist = bool(preset_values["audioLipSyncAssist"])

    job_id = make_job_id()
    output_rel = JOBS_ROOT_REL / job_id
    output_abs = PROJECT_ROOT / output_rel
    stream_rel = output_rel / "stream"
    stream_abs = PROJECT_ROOT / stream_rel
    stream_shm_prefix = build_job_stream_shm_prefix(job_id)
    log_rel = output_rel / RUN_LOG_FILE_NAME
    log_abs = PROJECT_ROOT / log_rel
    input_rel = output_rel / "inputs" / f"driving{extension}"
    input_abs = PROJECT_ROOT / input_rel
    stream_audio_abs = PROJECT_ROOT / output_rel / "inputs" / "driving_stream.wav"
    output_abs.mkdir(parents=True, exist_ok=True)
    await save_upload_file(audio, input_abs)
    stream_audio_input_abs = normalize_stream_audio_input(input_abs, stream_audio_abs)
    audio_duration_sec = probe_media_duration_sec(input_abs)
    source_frame_abs, source_frame_arg = await resolve_requested_source_frame(
        source_frame=source_frame,
        source_template_pack=source_template_pack,
        source_image=source_image,
        source_video=source_video,
        output_abs=output_abs,
        output_rel=output_rel,
        audio_duration_sec=audio_duration_sec,
        avatar_session_id=avatar_session_id,
    )
    source_template_pack_abs = source_frame_abs if is_source_template_pack_path(source_frame_abs) else None
    source_template_pack_id = source_template_pack_abs.name if source_template_pack_abs is not None else ""
    emit_runtime_log(
        "queue",
        (
            f"enqueue avatar job source_kind={resolve_source_input_kind(source_frame_abs)} "
            f"template_pack={source_template_pack_id or '-'} "
            f"audio={normalize_rel_path(str(input_rel))}"
        ),
    )

    job = JobRecord(
        job_id=job_id,
        avatar_session_id=avatar_session_id,
        created_at_ms=now_ms(),
        mode=mode,
        source_frame_arg=source_frame_arg,
        source_frame_abs=source_frame_abs,
        source_template_pack_id=source_template_pack_id,
        source_template_pack_abs=source_template_pack_abs,
        output_rel=output_rel,
        output_abs=output_abs,
        stream_rel=stream_rel,
        stream_abs=stream_abs,
        stream_shm_prefix=stream_shm_prefix,
        audio_input_rel=input_rel,
        audio_input_abs=input_abs,
        stream_audio_input_abs=stream_audio_input_abs,
        audio_original_name=audio.filename or input_abs.name,
        audio_duration_sec=audio_duration_sec,
        audio_motion_stride=int(motion_stride),
        generation_frame_count=normalized_generation_frame_count,
        audio_eye_tamed_preset=bool(audio_eye_tamed_preset),
        audio_eye_soft_factor=normalized_audio_eye_soft_factor,
        audio_eye_hard_factor=normalized_audio_eye_hard_factor,
        audio_eye_hard_dy_min=normalized_audio_eye_hard_dy_min,
        audio_eye_hard_dy_max=normalized_audio_eye_hard_dy_max,
        audio_motion_tuning_enabled=bool(audio_motion_tuning_enabled),
        audio_reanchor_first_n=normalized_audio_reanchor_first_n,
        audio_mouth_open_factor=normalized_audio_mouth_open_factor,
        audio_pose_smooth_window=normalized_audio_pose_smooth_window,
        audio_exp_smooth_window=normalized_audio_exp_smooth_window,
        audio_pose_jump_threshold=normalized_audio_pose_jump_threshold,
        audio_translation_jump_threshold=normalized_audio_translation_jump_threshold,
        audio_lip_sync_assist=bool(audio_lip_sync_assist),
        audio_lip_sync_min_ratio=normalized_audio_lip_sync_min_ratio,
        audio_lip_sync_max_ratio=normalized_audio_lip_sync_max_ratio,
        audio_lip_sync_smooth_window=normalized_audio_lip_sync_smooth_window,
        audio_lip_sync_strength=normalized_audio_lip_sync_strength,
        audio_lip_sync_power=normalized_audio_lip_sync_power,
        audio_lip_sync_attack=normalized_audio_lip_sync_attack,
        audio_lip_sync_release=normalized_audio_lip_sync_release,
        audio_lip_sync_offset_ms=normalized_audio_lip_sync_offset_ms,
        audio_mouth_floor_strength=normalized_audio_mouth_floor_strength,
        audio_mouth_peak_clamp=normalized_audio_mouth_peak_clamp,
        driving_multiplier=normalized_driving_multiplier,
        cfg_scale=normalized_cfg_scale,
        joyvasa_inference_steps=normalized_joyvasa_inference_steps,
        animation_region=normalized_animation_region,
        stitching_enabled=bool(stitching),
        relative_motion_enabled=bool(relative_motion),
        paste_back_enabled=bool(paste_back),
        defer_paste_back_enabled=should_defer_preview_paste_back(
            mode,
            bool(paste_back),
            bool(stitching),
            source_frame_abs,
        ),
        log_rel=log_rel,
        log_abs=log_abs,
    )
    register_job(job)
    enqueue_job(job.job_id)
    emit_runtime_log(
        "queue",
        f"avatar job enqueued job_id={job.job_id} queue_position={queue_position(job.job_id)}",
    )
    return build_job_payload(job)


def create_app() -> FastAPI:
    """
    Build FastAPI application.
    """
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="FasterLivePortrait Streaming API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def require_api_token(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        auth_error_response = authorize_http_request(request)
        if auth_error_response is not None:
            return auth_error_response
        return await call_next(request)

    @app.on_event("startup")
    async def startup_warmup() -> None:
        ensure_job_worker_started()
        ensure_avatar_worker_started()
        if not WARMUP_ENABLED:
            return
        warmup_thread = threading.Thread(target=run_startup_warmup_once, daemon=True)
        warmup_thread.start()

    @app.on_event("shutdown")
    async def shutdown_webrtc_sessions() -> None:
        await close_all_webrtc_sessions()

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        request_avatar_session_id = get_optional_avatar_session_id_from_request(request)
        response_avatar_session_id = request_avatar_session_id or issue_avatar_session_id()
        with JOB_QUEUE_CONDITION:
            queue_depth = len(JOB_QUEUE)
        worker_alive = bool(JOB_WORKER_THREAD is not None and JOB_WORKER_THREAD.is_alive())
        running_job = get_running_job_record()
        head_queued_job_id = get_head_queued_job_id()
        active_queue_task_id, active_queue_task_kind = get_active_queue_task_snapshot()
        warmup_source_frame_arg = ""
        fixed_source_frame_arg = ""
        fixed_source_error = ""
        try:
            fixed_source = resolve_configured_fixed_source_frame()
            if fixed_source is not None:
                _, fixed_source_frame_arg = fixed_source
            warmup_source_frame_arg = resolve_warmup_source_frame_arg() or ""
        except RuntimeError as exc:
            fixed_source_error = str(exc)
        avatar_payload = (
            build_avatar_payload(request_avatar_session_id)
            if request_avatar_session_id is not None
            else build_public_avatar_health_payload()
        )
        with WARMUP_LOCK:
            warmup_phase = WARMUP_PHASE
            warmup_progress = WARMUP_PROGRESS
            warmup_message = WARMUP_MESSAGE
            warmup_error = WARMUP_ERROR
        return {
            "status": "ok",
            "backend": DEFAULT_BACKEND,
            "trtRuntime": DEFAULT_TRT_RUNTIME,
            "trtPrecision": DEFAULT_TRT_PRECISION,
            "skipTrtEngineBuild": DEFAULT_SKIP_TRT_ENGINE_BUILD,
            "defaultAudioMotionStride": DEFAULT_AUDIO_MOTION_STRIDE,
            "defaultMode": DEFAULT_MODE,
            "generationFrameCountSupported": True,
            "generationFrameCountMin": GENERATION_FRAME_COUNT_MIN,
            "generationFrameCountMax": GENERATION_FRAME_COUNT_MAX,
            "defaultRenderBatchSize": DEFAULT_RENDER_BATCH_SIZE,
            "defaultTrtEngineBatchSize": DEFAULT_TRT_ENGINE_BATCH_SIZE,
            "defaultAnimationRegion": DEFAULT_ANIMATION_REGION,
            "defaultStitchingEnabled": DEFAULT_STITCHING_ENABLED,
            "defaultRelativeMotionEnabled": DEFAULT_RELATIVE_MOTION_ENABLED,
            "defaultPasteBackEnabled": DEFAULT_PASTE_BACK_ENABLED,
            "defaultAudioEyeTamedPreset": DEFAULT_AUDIO_EYE_TAMED_PRESET,
            "defaultAudioEyeSoftFactor": DEFAULT_AUDIO_EYE_SOFT_FACTOR,
            "defaultAudioEyeHardFactor": DEFAULT_AUDIO_EYE_HARD_FACTOR,
            "defaultAudioEyeHardDyMin": DEFAULT_AUDIO_EYE_HARD_DY_MIN,
            "defaultAudioEyeHardDyMax": DEFAULT_AUDIO_EYE_HARD_DY_MAX,
            "defaultAudioMotionTuningEnabled": DEFAULT_AUDIO_MOTION_TUNING_ENABLED,
            "defaultAudioReanchorFirstN": DEFAULT_AUDIO_REANCHOR_FIRST_N,
            "defaultAudioMouthOpenFactor": DEFAULT_AUDIO_MOUTH_OPEN_FACTOR,
            "defaultAudioPoseSmoothWindow": DEFAULT_AUDIO_POSE_SMOOTH_WINDOW,
            "defaultAudioExpSmoothWindow": DEFAULT_AUDIO_EXP_SMOOTH_WINDOW,
            "defaultAudioPoseJumpThreshold": DEFAULT_AUDIO_POSE_JUMP_THRESHOLD,
            "defaultAudioTranslationJumpThreshold": DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD,
            "defaultAudioLipSyncAssist": DEFAULT_AUDIO_LIP_SYNC_ASSIST,
            "defaultAudioLipSyncMinRatio": DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO,
            "defaultAudioLipSyncMaxRatio": DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO,
            "defaultAudioLipSyncSmoothWindow": DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW,
            "defaultAudioLipSyncStrength": DEFAULT_AUDIO_LIP_SYNC_STRENGTH,
            "defaultAudioLipSyncPower": DEFAULT_AUDIO_LIP_SYNC_POWER,
            "defaultAudioLipSyncAttack": DEFAULT_AUDIO_LIP_SYNC_ATTACK,
            "defaultAudioLipSyncRelease": DEFAULT_AUDIO_LIP_SYNC_RELEASE,
            "defaultAudioLipSyncOffsetMs": DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS,
            "defaultAudioMouthFloorStrength": DEFAULT_AUDIO_MOUTH_FLOOR_STRENGTH,
            "defaultAudioMouthPeakClamp": DEFAULT_AUDIO_MOUTH_PEAK_CLAMP,
            "defaultAvatarReadyBufferMinSec": AVATAR_READY_BUFFER_MIN_SEC,
            "defaultAvatarReadyDynamicMarginSec": AVATAR_READY_DYNAMIC_MARGIN_SEC,
            "defaultAvatarBufferedStartProgress": AVATAR_READY_BUFFER_MAX_PROGRESS,
            "defaultDrivingMultiplier": DEFAULT_DRIVING_MULTIPLIER,
            "defaultCfgScale": DEFAULT_CFG_SCALE,
            "defaultJoyvasaInferenceSteps": DEFAULT_JOYVASA_INFERENCE_STEPS,
            "audioTuningPresets": build_audio_tuning_presets_payload(),
            "sourceTemplatePacksUrl": "/api/source-templates",
            "createSourceTemplatePackUrl": "/api/source-templates",
            "defaultVideoEncoder": DEFAULT_VIDEO_ENCODER,
            "authEnabled": API_TOKEN_ENABLED,
            "authHeaderName": "Authorization",
            "authScheme": AUTHORIZATION_HEADER_VALUE,
            "authQueryParam": API_TOKEN_QUERY_KEY,
            "enqueueAudioUrl": "/api/avatar/enqueue",
            "fixedSourceEnabled": bool(FIXED_SOURCE_FRAME),
            "fixedSourceFrame": fixed_source_frame_arg,
            "fixedSourceError": fixed_source_error,
            "allowCustomSourceFrame": not bool(FIXED_SOURCE_FRAME),
            "warmupEnabled": WARMUP_ENABLED,
            "warmupRunning": WARMUP_RUNNING,
            "warmupLastStartedAtMs": WARMUP_LAST_STARTED_AT_MS,
            "warmupSourceFrame": warmup_source_frame_arg,
            "warmupPhase": warmup_phase,
            "warmupProgress": warmup_progress,
            "warmupMessage": warmup_message,
            "warmupError": warmup_error,
            "runtimeRestarting": RUNTIME_RESTARTING,
            "runtimeRestartRequestedAtMs": RUNTIME_RESTART_REQUESTED_AT_MS,
            "processStartedAtMs": PROCESS_STARTED_AT_MS,
            "runningJobId": running_job.job_id if running_job is not None else "",
            "runningJobState": determine_job_state(running_job, read_job_stream_status(running_job)) if running_job is not None else "",
            "headQueuedJobId": head_queued_job_id,
            "jobWorkerAlive": worker_alive,
            "jobQueueDepth": queue_depth,
            "activeQueueTaskId": active_queue_task_id,
            "activeQueueTaskKind": active_queue_task_kind,
            "avatarSessionId": response_avatar_session_id,
            "avatarMode": avatar_payload["mode"],
            "avatarSequence": avatar_payload["sequence"],
            "avatarCurrentJobId": avatar_payload["currentJobId"],
            "avatarCurrentJobStartedAtMs": avatar_payload["currentJobStartedAtMs"],
            "avatarCurrentJobEndsAtMs": avatar_payload["currentJobEndsAtMs"],
            "avatarIdleVideoUrl": avatar_payload["idleVideoUrl"],
            "avatarBufferedStartProgress": avatar_payload["bufferedStartProgress"],
            "avatarVideoWsUrl": avatar_payload["avatarVideoWsUrl"],
            "avatarVideoHttpUrl": avatar_payload["avatarVideoHttpUrl"],
            "avatarTransport": avatar_payload["avatarTransport"],
            "avatarWebrtcOfferUrl": avatar_payload["avatarWebrtcOfferUrl"],
            "avatarQueueDepth": avatar_payload["queueDepth"],
            "webrtcEnabled": avatar_payload["webrtcEnabled"],
            "webrtcIceServers": avatar_payload["webrtcIceServers"],
            "webrtcIceTransportPolicy": avatar_payload["webrtcIceTransportPolicy"],
            "activeWebrtcSessions": avatar_payload["activeWebrtcSessions"],
            "containerLogPath": str(CONTAINER_LOG_REL),
            "workerLogPath": str(PERSISTENT_WORKER_LOG_REL),
        }

    @app.get("/api/source-templates")
    async def list_source_templates() -> dict[str, Any]:
        return {
            "items": list_source_template_pack_records(),
            "presetIds": [SOURCE_TEMPLATE_PACK_CLOSE_MOUTH_PRESET],
        }

    @app.post("/api/source-templates")
    async def create_source_template_pack_endpoint(
        source_image: UploadFile | None = File(None),
        source_video: UploadFile | None = File(None),
        source_frame: str = Form(""),
        template_name: str = Form(""),
    ) -> dict[str, Any]:
        ensure_runtime_accepting_requests()
        ensure_job_worker_started()
        has_source_image_upload = bool(source_image is not None and source_image.filename)
        has_source_video_upload = bool(source_video is not None and source_video.filename)
        if has_source_image_upload and has_source_video_upload:
            raise HTTPException(status_code=400, detail="Provide either source_image or source_video, not both.")

        build_request_id = uuid.uuid4().hex
        build_root = (SOURCE_TEMPLATE_PACKS_ROOT / "_build_inputs" / build_request_id).resolve()
        build_root.mkdir(parents=True, exist_ok=True)

        if has_source_video_upload:
            extension = Path(str(source_video.filename)).suffix.lower()
            if extension not in ALLOWED_SOURCE_VIDEO_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported source video extension '{extension}'. "
                        f"Allowed: {sorted(ALLOWED_SOURCE_VIDEO_EXTENSIONS)}"
                    ),
                )
            source_abs = (build_root / f"source{extension}").resolve()
            await save_upload_file(source_video, source_abs)
        elif has_source_image_upload:
            extension = Path(str(source_image.filename)).suffix.lower()
            if extension not in ALLOWED_SOURCE_IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported source image extension '{extension}'. "
                        f"Allowed: {sorted(ALLOWED_SOURCE_IMAGE_EXTENSIONS)}"
                    ),
                )
            source_abs = (build_root / f"source{extension}").resolve()
            await save_upload_file(source_image, source_abs)
        else:
            source_abs, _ = resolve_source_frame_candidate(str(source_frame or "").strip() or DEFAULT_SOURCE_FRAME)

        safe_template_name = sanitize_source_template_pack_name(
            template_name or Path(source_abs).stem or "source_template"
        )
        template_pack_path = (SOURCE_TEMPLATE_PACKS_ROOT / f"{safe_template_name}.pkl").resolve()
        if template_pack_path.exists():
            template_pack_path = (
                SOURCE_TEMPLATE_PACKS_ROOT
                / f"{safe_template_name}-{time.strftime('%Y%m%d-%H%M%S')}.pkl"
            ).resolve()

        build_task_id = f"template_build_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        build_log_abs = (SOURCE_TEMPLATE_PACK_BUILD_LOGS_DIR / f"{build_task_id}.log").resolve()
        build_task = SourceTemplateBuildTask(
            task_id=build_task_id,
            created_at_ms=now_ms(),
            source_abs=source_abs,
            template_pack_path=template_pack_path,
            template_name=safe_template_name,
            log_abs=build_log_abs,
        )
        register_source_template_build_task(build_task)
        enqueue_job(build_task.task_id)
        emit_runtime_log(
            "queue",
            (
                f"template build enqueued task_id={build_task.task_id} "
                f"template={template_pack_path.name} queue_position={queue_position(build_task.task_id)}"
            ),
        )
        completed_in_time = await asyncio.to_thread(
            build_task.done_event.wait,
            SOURCE_TEMPLATE_PACK_BUILD_WAIT_TIMEOUT_SEC,
        )
        try:
            if not completed_in_time:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Source template build timed out while waiting in or through the shared queue. "
                        f"log={build_log_abs}"
                    ),
                )
            if build_task.error:
                raise HTTPException(status_code=500, detail=build_task.error)
            if build_task.result is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Source template build completed without one public result payload. "
                        f"log={build_log_abs}"
                    ),
                )
            return build_task.result
        finally:
            if build_task.done_event.is_set():
                pop_source_template_build_task(build_task.task_id)

    @app.get("/api/avatar/status")
    async def avatar_status(request: Request) -> JSONResponse:
        ensure_avatar_worker_started()
        avatar_session_id = get_avatar_session_id_from_request(request)
        return JSONResponse(build_avatar_payload(avatar_session_id))

    @app.post(WEBRTC_OFFER_API_PATH)
    async def webrtc_offer(request: Request, offer: WebRtcOfferRequest) -> JSONResponse:
        ensure_avatar_worker_started()
        if str(offer.type or "").strip().lower() != "offer":
            raise HTTPException(status_code=400, detail="WebRTC request type must be 'offer'.")
        avatar_session_id = get_avatar_session_id_from_request(request)
        session = AvatarWebRtcSession(str(uuid.uuid4()), avatar_session_id, build_webrtc_rtc_configuration())
        register_webrtc_session(session)
        try:
            await session.peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=offer.sdp, type=offer.type)
            )
            answer = await session.peer_connection.createAnswer()
            await session.peer_connection.setLocalDescription(answer)
            await wait_for_ice_gathering_complete(session.peer_connection)
            local_description = session.peer_connection.localDescription
            if local_description is None:
                raise HTTPException(status_code=500, detail="WebRTC answer was not generated.")
            return JSONResponse(
                {
                    "type": local_description.type,
                    "sdp": local_description.sdp,
                    "sessionId": session.session_id,
                }
            )
        except HTTPException:
            await close_webrtc_session(session)
            raise
        except Exception as exc:
            await close_webrtc_session(session)
            raise HTTPException(status_code=500, detail=f"WebRTC negotiation failed: {exc}") from exc

    @app.post("/api/warmup")
    async def warmup() -> JSONResponse:
        ensure_runtime_accepting_requests()
        if not WARMUP_ENABLED:
            return JSONResponse({"status": "disabled"})
        warmup_thread = threading.Thread(target=run_startup_warmup_once, daemon=True)
        warmup_thread.start()
        with WARMUP_LOCK:
            warmup_phase = WARMUP_PHASE
            warmup_progress = WARMUP_PROGRESS
            warmup_message = WARMUP_MESSAGE
            warmup_error = WARMUP_ERROR
        return JSONResponse(
            {
                "status": "started",
                "startedAtMs": now_ms(),
                "warmupPhase": warmup_phase,
                "warmupProgress": warmup_progress,
                "warmupMessage": warmup_message,
                "warmupError": warmup_error,
            }
        )

    @app.post("/api/runtime/restart")
    async def runtime_restart() -> JSONResponse:
        requested_at_ms = request_runtime_restart()
        print("[runtime] restart requested; terminating container process for clean restart")
        return JSONResponse(
            {
                "status": "restarting",
                "requestedAtMs": requested_at_ms,
                "restartDelaySec": RUNTIME_RESTART_DELAY_SEC,
                "warmupWillRun": WARMUP_ENABLED,
            }
        )

    @app.get("/api/runtime/logs")
    async def runtime_logs(target: str = RUNTIME_LOG_TARGET_CONTAINER, lines: int = 200) -> JSONResponse:
        normalized_target = str(target).strip().lower()
        if normalized_target not in RUNTIME_LOG_TARGETS:
            raise HTTPException(status_code=400, detail=f"Invalid log target: {target}")
        safe_lines = min(MAX_LOG_LINES, max(20, int(lines)))
        log_path = resolve_runtime_log_path(normalized_target)
        return JSONResponse(
            {
                "target": normalized_target,
                "lines": safe_lines,
                "path": str(log_path),
                "available": log_path.exists(),
                "content": tail_log(log_path, safe_lines),
            }
        )

    @app.post("/api/generate")
    async def generate(
        request: Request,
        audio: UploadFile = File(...),
        source_image: UploadFile | None = File(None),
        source_video: UploadFile | None = File(None),
        source_frame: str = Form(DEFAULT_SOURCE_FRAME),
        source_template_pack: str = Form(""),
        audio_tuning_preset: str = Form(""),
        mode: str = Form(DEFAULT_MODE),
        motion_stride: int = Form(DEFAULT_AUDIO_MOTION_STRIDE),
        generation_frame_count: int | None = Form(None),
        audio_eye_tamed_preset: bool = Form(DEFAULT_AUDIO_EYE_TAMED_PRESET),
        audio_eye_soft_factor: float = Form(DEFAULT_AUDIO_EYE_SOFT_FACTOR),
        audio_eye_hard_factor: float = Form(DEFAULT_AUDIO_EYE_HARD_FACTOR),
        audio_eye_hard_dy_min: float = Form(DEFAULT_AUDIO_EYE_HARD_DY_MIN),
        audio_eye_hard_dy_max: float = Form(DEFAULT_AUDIO_EYE_HARD_DY_MAX),
        audio_motion_tuning_enabled: bool = Form(DEFAULT_AUDIO_MOTION_TUNING_ENABLED),
        audio_reanchor_first_n: int = Form(DEFAULT_AUDIO_REANCHOR_FIRST_N),
        audio_mouth_open_factor: float = Form(DEFAULT_AUDIO_MOUTH_OPEN_FACTOR),
        audio_pose_smooth_window: int = Form(DEFAULT_AUDIO_POSE_SMOOTH_WINDOW),
        audio_exp_smooth_window: int = Form(DEFAULT_AUDIO_EXP_SMOOTH_WINDOW),
        audio_pose_jump_threshold: float = Form(DEFAULT_AUDIO_POSE_JUMP_THRESHOLD),
        audio_translation_jump_threshold: float = Form(DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD),
        audio_lip_sync_assist: bool = Form(DEFAULT_AUDIO_LIP_SYNC_ASSIST),
        audio_lip_sync_min_ratio: float = Form(DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO),
        audio_lip_sync_max_ratio: float = Form(DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO),
        audio_lip_sync_smooth_window: int = Form(DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW),
        audio_lip_sync_strength: float = Form(DEFAULT_AUDIO_LIP_SYNC_STRENGTH),
        audio_lip_sync_power: float = Form(DEFAULT_AUDIO_LIP_SYNC_POWER),
        audio_lip_sync_attack: float = Form(DEFAULT_AUDIO_LIP_SYNC_ATTACK),
        audio_lip_sync_release: float = Form(DEFAULT_AUDIO_LIP_SYNC_RELEASE),
        audio_lip_sync_offset_ms: int = Form(DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS),
        audio_mouth_floor_strength: float = Form(DEFAULT_AUDIO_MOUTH_FLOOR_STRENGTH),
        audio_mouth_peak_clamp: float = Form(DEFAULT_AUDIO_MOUTH_PEAK_CLAMP),
        driving_multiplier: float = Form(DEFAULT_DRIVING_MULTIPLIER),
        cfg_scale: float = Form(DEFAULT_CFG_SCALE),
        joyvasa_inference_steps: int = Form(DEFAULT_JOYVASA_INFERENCE_STEPS),
        animation_region: str = Form(DEFAULT_ANIMATION_REGION),
        stitching: bool = Form(DEFAULT_STITCHING_ENABLED),
        relative_motion: bool = Form(DEFAULT_RELATIVE_MOTION_ENABLED),
        paste_back: bool = Form(DEFAULT_PASTE_BACK_ENABLED),
    ) -> JSONResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        payload = await create_and_enqueue_audio_job(
            avatar_session_id=avatar_session_id,
            audio=audio,
            source_image=source_image,
            source_video=source_video,
            source_frame=source_frame,
            source_template_pack=source_template_pack,
            audio_tuning_preset=audio_tuning_preset,
            mode=mode,
            motion_stride=motion_stride,
            generation_frame_count=generation_frame_count,
            audio_eye_tamed_preset=audio_eye_tamed_preset,
            audio_eye_soft_factor=audio_eye_soft_factor,
            audio_eye_hard_factor=audio_eye_hard_factor,
            audio_eye_hard_dy_min=audio_eye_hard_dy_min,
            audio_eye_hard_dy_max=audio_eye_hard_dy_max,
            audio_motion_tuning_enabled=audio_motion_tuning_enabled,
            audio_reanchor_first_n=audio_reanchor_first_n,
            audio_mouth_open_factor=audio_mouth_open_factor,
            audio_pose_smooth_window=audio_pose_smooth_window,
            audio_exp_smooth_window=audio_exp_smooth_window,
            audio_pose_jump_threshold=audio_pose_jump_threshold,
            audio_translation_jump_threshold=audio_translation_jump_threshold,
            audio_lip_sync_assist=audio_lip_sync_assist,
            audio_lip_sync_min_ratio=audio_lip_sync_min_ratio,
            audio_lip_sync_max_ratio=audio_lip_sync_max_ratio,
            audio_lip_sync_smooth_window=audio_lip_sync_smooth_window,
            audio_lip_sync_strength=audio_lip_sync_strength,
            audio_lip_sync_power=audio_lip_sync_power,
            audio_lip_sync_attack=audio_lip_sync_attack,
            audio_lip_sync_release=audio_lip_sync_release,
            audio_lip_sync_offset_ms=audio_lip_sync_offset_ms,
            audio_mouth_floor_strength=audio_mouth_floor_strength,
            audio_mouth_peak_clamp=audio_mouth_peak_clamp,
            driving_multiplier=driving_multiplier,
            cfg_scale=cfg_scale,
            joyvasa_inference_steps=joyvasa_inference_steps,
            animation_region=animation_region,
            stitching=stitching,
            relative_motion=relative_motion,
            paste_back=paste_back,
        )
        return JSONResponse(payload)

    @app.post("/api/avatar/enqueue")
    async def enqueue_avatar_audio(
        request: Request,
        audio: UploadFile = File(...),
        source_image: UploadFile | None = File(None),
        source_video: UploadFile | None = File(None),
        source_frame: str = Form(DEFAULT_SOURCE_FRAME),
        source_template_pack: str = Form(""),
        audio_tuning_preset: str = Form(""),
        mode: str = Form(DEFAULT_MODE),
        motion_stride: int = Form(DEFAULT_AUDIO_MOTION_STRIDE),
        generation_frame_count: int | None = Form(None),
        audio_eye_tamed_preset: bool = Form(DEFAULT_AUDIO_EYE_TAMED_PRESET),
        audio_eye_soft_factor: float = Form(DEFAULT_AUDIO_EYE_SOFT_FACTOR),
        audio_eye_hard_factor: float = Form(DEFAULT_AUDIO_EYE_HARD_FACTOR),
        audio_eye_hard_dy_min: float = Form(DEFAULT_AUDIO_EYE_HARD_DY_MIN),
        audio_eye_hard_dy_max: float = Form(DEFAULT_AUDIO_EYE_HARD_DY_MAX),
        audio_motion_tuning_enabled: bool = Form(DEFAULT_AUDIO_MOTION_TUNING_ENABLED),
        audio_reanchor_first_n: int = Form(DEFAULT_AUDIO_REANCHOR_FIRST_N),
        audio_mouth_open_factor: float = Form(DEFAULT_AUDIO_MOUTH_OPEN_FACTOR),
        audio_pose_smooth_window: int = Form(DEFAULT_AUDIO_POSE_SMOOTH_WINDOW),
        audio_exp_smooth_window: int = Form(DEFAULT_AUDIO_EXP_SMOOTH_WINDOW),
        audio_pose_jump_threshold: float = Form(DEFAULT_AUDIO_POSE_JUMP_THRESHOLD),
        audio_translation_jump_threshold: float = Form(DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD),
        audio_lip_sync_assist: bool = Form(DEFAULT_AUDIO_LIP_SYNC_ASSIST),
        audio_lip_sync_min_ratio: float = Form(DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO),
        audio_lip_sync_max_ratio: float = Form(DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO),
        audio_lip_sync_smooth_window: int = Form(DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW),
        audio_lip_sync_strength: float = Form(DEFAULT_AUDIO_LIP_SYNC_STRENGTH),
        audio_lip_sync_power: float = Form(DEFAULT_AUDIO_LIP_SYNC_POWER),
        audio_lip_sync_attack: float = Form(DEFAULT_AUDIO_LIP_SYNC_ATTACK),
        audio_lip_sync_release: float = Form(DEFAULT_AUDIO_LIP_SYNC_RELEASE),
        audio_lip_sync_offset_ms: int = Form(DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS),
        audio_mouth_floor_strength: float = Form(DEFAULT_AUDIO_MOUTH_FLOOR_STRENGTH),
        audio_mouth_peak_clamp: float = Form(DEFAULT_AUDIO_MOUTH_PEAK_CLAMP),
        driving_multiplier: float = Form(DEFAULT_DRIVING_MULTIPLIER),
        cfg_scale: float = Form(DEFAULT_CFG_SCALE),
        joyvasa_inference_steps: int = Form(DEFAULT_JOYVASA_INFERENCE_STEPS),
        animation_region: str = Form(DEFAULT_ANIMATION_REGION),
        stitching: bool = Form(DEFAULT_STITCHING_ENABLED),
        relative_motion: bool = Form(DEFAULT_RELATIVE_MOTION_ENABLED),
        paste_back: bool = Form(DEFAULT_PASTE_BACK_ENABLED),
    ) -> JSONResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        payload = await create_and_enqueue_audio_job(
            avatar_session_id=avatar_session_id,
            audio=audio,
            source_image=source_image,
            source_video=source_video,
            source_frame=source_frame,
            source_template_pack=source_template_pack,
            audio_tuning_preset=audio_tuning_preset,
            mode=mode,
            motion_stride=motion_stride,
            generation_frame_count=generation_frame_count,
            audio_eye_tamed_preset=audio_eye_tamed_preset,
            audio_eye_soft_factor=audio_eye_soft_factor,
            audio_eye_hard_factor=audio_eye_hard_factor,
            audio_eye_hard_dy_min=audio_eye_hard_dy_min,
            audio_eye_hard_dy_max=audio_eye_hard_dy_max,
            audio_motion_tuning_enabled=audio_motion_tuning_enabled,
            audio_reanchor_first_n=audio_reanchor_first_n,
            audio_mouth_open_factor=audio_mouth_open_factor,
            audio_pose_smooth_window=audio_pose_smooth_window,
            audio_exp_smooth_window=audio_exp_smooth_window,
            audio_pose_jump_threshold=audio_pose_jump_threshold,
            audio_translation_jump_threshold=audio_translation_jump_threshold,
            audio_lip_sync_assist=audio_lip_sync_assist,
            audio_lip_sync_min_ratio=audio_lip_sync_min_ratio,
            audio_lip_sync_max_ratio=audio_lip_sync_max_ratio,
            audio_lip_sync_smooth_window=audio_lip_sync_smooth_window,
            audio_lip_sync_strength=audio_lip_sync_strength,
            audio_lip_sync_power=audio_lip_sync_power,
            audio_lip_sync_attack=audio_lip_sync_attack,
            audio_lip_sync_release=audio_lip_sync_release,
            audio_lip_sync_offset_ms=audio_lip_sync_offset_ms,
            audio_mouth_floor_strength=audio_mouth_floor_strength,
            audio_mouth_peak_clamp=audio_mouth_peak_clamp,
            driving_multiplier=driving_multiplier,
            cfg_scale=cfg_scale,
            joyvasa_inference_steps=joyvasa_inference_steps,
            animation_region=animation_region,
            stitching=stitching,
            relative_motion=relative_motion,
            paste_back=paste_back,
        )
        return JSONResponse(payload)

    @app.get("/api/jobs/{job_id}/status")
    async def job_status(job_id: str, request: Request) -> JSONResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        job = get_job(job_id, avatar_session_id)
        return JSONResponse(build_job_payload(job))

    @app.get("/api/jobs/{job_id}/report")
    async def job_report(job_id: str, request: Request) -> JSONResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        job = get_job(job_id, avatar_session_id)
        report = read_json(job.report_abs)
        if report is None:
            raise HTTPException(status_code=404, detail="run_report.json not available yet")
        return JSONResponse(report)

    @app.get("/api/jobs/{job_id}/log")
    async def job_log(job_id: str, request: Request, lines: int = 200) -> JSONResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        job = get_job(job_id, avatar_session_id)
        safe_lines = min(MAX_LOG_LINES, max(10, int(lines)))
        return JSONResponse(
            {
                "jobId": job_id,
                "logPath": f"/jobs/{job_id}/{RUN_LOG_FILE_NAME}",
                "tail": tail_log(job.log_abs, safe_lines),
            }
        )

    @app.get("/api/jobs/{job_id}/stream.mjpg")
    async def job_stream(job_id: str, request: Request) -> StreamingResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        job = get_job(job_id, avatar_session_id)

        async def stream_generator() -> Any:
            last_seen_frame_index = -1
            stable_loops = 0
            while True:
                if await request.is_disconnected():
                    break
                frame_bytes, updated_signature, has_new_frame = read_updated_latest_frame(job, last_seen_frame_index)
                if has_new_frame and frame_bytes:
                    last_seen_frame_index = updated_signature
                    stable_loops = 0
                    header = (
                        f"--{STREAM_BOUNDARY}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame_bytes)}\r\n\r\n"
                    ).encode("utf-8")
                    yield header + frame_bytes + b"\r\n"
                else:
                    stable_loops += 1

                stream_status = read_job_stream_status(job)
                state = determine_job_state(job, stream_status)
                if state in {"done", "error"} and stable_loops >= 25:
                    break
                await asyncio.sleep(JOB_POLL_SLEEP_SEC)
            yield f"--{STREAM_BOUNDARY}--\r\n".encode("utf-8")

        return StreamingResponse(
            stream_generator(),
            media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.websocket("/ws/jobs/{job_id}")
    async def job_stream_ws(websocket: WebSocket, job_id: str) -> None:
        if not is_websocket_request_authorized(websocket):
            await websocket.close(code=WEBSOCKET_UNAUTHORIZED_CLOSE_CODE, reason=AUTH_FAILURE_MESSAGE)
            return
        try:
            avatar_session_id = get_avatar_session_id_from_websocket(websocket)
        except ValueError as exc:
            await websocket.close(code=4400, reason=str(exc))
            return
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None or job.avatar_session_id != avatar_session_id:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        last_status_signature = ""

        try:
            while True:
                stream_status = read_job_stream_status(job)
                payload = build_job_payload(job)
                signature_source = {
                    "state": payload.get("state"),
                    "running": payload.get("running"),
                    "exitCode": payload.get("exitCode"),
                    "resultVideoUrl": payload.get("resultVideoUrl"),
                    "resultConcatUrl": payload.get("resultConcatUrl"),
                    "progress": (payload.get("status") or {}).get("progress"),
                    "frameIndex": (payload.get("status") or {}).get("frameIndex"),
                    "frameTotal": (payload.get("status") or {}).get("frameTotal"),
                    "updatedAtMs": (payload.get("status") or {}).get("updatedAtMs"),
                }
                status_signature = json.dumps(signature_source, sort_keys=True)
                if status_signature != last_status_signature:
                    await websocket.send_json({"type": "status", "payload": payload})
                    last_status_signature = status_signature

                state = determine_job_state(job, stream_status)
                running = bool(job.process is not None and job.process.poll() is None)
                if state in {"done", "error"} and not running:
                    await websocket.send_json({"type": "terminal", "payload": payload})
                    break

                await asyncio.sleep(JOB_POLL_SLEEP_SEC)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.get("/api/avatar/video.mp4")
    async def avatar_video_http(request: Request) -> StreamingResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        ensure_avatar_worker_started()
        chunk_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=AVATAR_STREAM_HTTP_QUEUE_MAX_CHUNKS)

        async def send_chunk(chunk: bytes) -> None:
            await chunk_queue.put(chunk)

        async def should_stop() -> bool:
            return await request.is_disconnected()

        async def producer() -> None:
            try:
                await stream_continuous_avatar_video(
                    avatar_session_id=avatar_session_id,
                    chunk_sender=send_chunk,
                    should_stop=should_stop,
                )
            finally:
                with contextlib.suppress(Exception):
                    chunk_queue.put_nowait(None)

        producer_task = asyncio.create_task(producer())

        async def stream_generator() -> Any:
            try:
                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer_task

        return StreamingResponse(
            stream_generator(),
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/avatar/capture.mp4")
    async def avatar_video_capture(request: Request, seconds: float = AVATAR_CAPTURE_DEFAULT_DURATION_SEC) -> FileResponse:
        avatar_session_id = get_avatar_session_id_from_request(request)
        ensure_avatar_worker_started()
        capture_duration_sec = clamp_float(
            float(seconds),
            AVATAR_CAPTURE_MIN_DURATION_SEC,
            AVATAR_CAPTURE_MAX_DURATION_SEC,
        )
        capture_path = AVATAR_CAPTURE_ROOT / f"avatar_capture_{uuid.uuid4().hex}.mp4"
        await capture_continuous_avatar_video(capture_path, capture_duration_sec, avatar_session_id)
        return FileResponse(
            str(capture_path),
            media_type="video/mp4",
            filename=capture_path.name,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            },
        )

    @app.websocket("/ws/avatar/video")
    async def avatar_video_ws(websocket: WebSocket) -> None:
        if not is_websocket_request_authorized(websocket):
            await websocket.close(code=WEBSOCKET_UNAUTHORIZED_CLOSE_CODE, reason=AUTH_FAILURE_MESSAGE)
            return
        try:
            avatar_session_id = get_avatar_session_id_from_websocket(websocket)
        except ValueError as exc:
            await websocket.close(code=4400, reason=str(exc))
            return
        await websocket.accept()

        async def send_chunk(chunk: bytes) -> None:
            await websocket.send_bytes(chunk)

        try:
            await stream_continuous_avatar_video(avatar_session_id=avatar_session_id, chunk_sender=send_chunk)
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.websocket("/ws/avatar")
    async def avatar_status_ws(websocket: WebSocket) -> None:
        if not is_websocket_request_authorized(websocket):
            await websocket.close(code=WEBSOCKET_UNAUTHORIZED_CLOSE_CODE, reason=AUTH_FAILURE_MESSAGE)
            return
        try:
            avatar_session_id = get_avatar_session_id_from_websocket(websocket)
        except ValueError as exc:
            await websocket.close(code=4400, reason=str(exc))
            return
        ensure_avatar_worker_started()
        await websocket.accept()
        last_signature = ""
        try:
            while True:
                payload = build_avatar_payload(avatar_session_id)
                signature = json.dumps(payload, sort_keys=True)
                if signature != last_signature:
                    await websocket.send_json({"type": "status", "payload": payload})
                    last_signature = signature
                await asyncio.sleep(JOB_POLL_SLEEP_SEC)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.websocket("/ws/jobs/{job_id}/video")
    async def job_video_ws(websocket: WebSocket, job_id: str) -> None:
        if not is_websocket_request_authorized(websocket):
            await websocket.close(code=WEBSOCKET_UNAUTHORIZED_CLOSE_CODE, reason=AUTH_FAILURE_MESSAGE)
            return
        try:
            avatar_session_id = get_avatar_session_id_from_websocket(websocket)
        except ValueError as exc:
            await websocket.close(code=4400, reason=str(exc))
            return
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None or job.avatar_session_id != avatar_session_id:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        stream_start_mode = resolve_video_stream_start_mode(websocket)
        stream_start_progress = resolve_video_stream_start_progress(websocket)
        use_buffered_start = stream_start_mode == VIDEO_STREAM_START_MODE_BUFFERED
        ffmpeg_process: asyncio.subprocess.Process | None = None
        ffmpeg_input_fps = 0.0

        next_frame_index = 1
        last_known_frame_index = 0
        last_known_frame_total = 0
        estimated_generation_fps = VIDEO_STREAM_INPUT_FPS
        frame_interval_sec = 1.0
        next_emit_at = 0.0
        pending_raw_frames: deque[bytes] = deque()
        pending_stream_frames: deque[bytes] = deque()
        previous_raw_frame_bytes: bytes | None = None
        buffered_start_ready = not use_buffered_start
        stable_loops = 0
        stdout_task: asyncio.Task[Any] | None = None
        stderr_task: asyncio.Task[Any] | None = None

        async def pump_mp4_stdout() -> None:
            """
            Send encoded fragmented MP4 chunks to WebSocket.
            """
            assert ffmpeg_process is not None
            assert ffmpeg_process.stdout is not None
            while True:
                chunk = await ffmpeg_process.stdout.read(VIDEO_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                await websocket.send_bytes(chunk)

        async def drain_stderr() -> None:
            """
            Drain ffmpeg stderr to avoid pipe backpressure.
            """
            assert ffmpeg_process is not None
            assert ffmpeg_process.stderr is not None
            while True:
                chunk = await ffmpeg_process.stderr.read(VIDEO_STREAM_CHUNK_SIZE)
                if not chunk:
                    break

        async def ensure_ffmpeg_process(input_fps: float) -> None:
            """
            Start FFmpeg subprocess lazily using the requested playback FPS.
            """
            nonlocal ffmpeg_process
            nonlocal ffmpeg_input_fps
            nonlocal frame_interval_sec
            nonlocal next_emit_at
            nonlocal stdout_task
            nonlocal stderr_task
            if ffmpeg_process is not None:
                return
            ffmpeg_input_fps = max(1.0, float(input_fps))
            frame_interval_sec = 1.0 / ffmpeg_input_fps
            next_emit_at = time.perf_counter()
            ffmpeg_process = await asyncio.create_subprocess_exec(
                *build_video_stream_command(ffmpeg_input_fps, job.stream_audio_input_abs),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert ffmpeg_process.stdin is not None
            assert ffmpeg_process.stdout is not None
            assert ffmpeg_process.stderr is not None
            stdout_task = asyncio.create_task(pump_mp4_stdout())
            stderr_task = asyncio.create_task(drain_stderr())

        try:
            while True:
                stream_status = read_job_stream_status(job)
                status_frame_index = parse_status_int(stream_status, "frameIndex")
                status_frame_total = parse_status_int(stream_status, "frameTotal")
                playback_fps = resolve_stream_playback_fps(stream_status)
                estimated_generation_fps = estimate_generation_fps(stream_status, estimated_generation_fps)
                if status_frame_index > last_known_frame_index:
                    last_known_frame_index = status_frame_index
                if status_frame_total > last_known_frame_total:
                    last_known_frame_total = status_frame_total
                progress_ratio = resolve_stream_progress_ratio(stream_status, status_frame_index, status_frame_total)

                if use_buffered_start and not buffered_start_ready:
                    state = determine_job_state(job, stream_status)
                    running = bool(job.process is not None and job.process.poll() is None)
                    if progress_ratio < stream_start_progress and running and state not in {"done", "error"}:
                        await asyncio.sleep(VIDEO_STREAM_POLL_SLEEP_SEC)
                        continue
                    buffered_start_ready = True

                if ffmpeg_process is None:
                    state = determine_job_state(job, stream_status)
                    running = bool(job.process is not None and job.process.poll() is None)
                    if playback_fps <= 0 and running and state not in {"done", "error"}:
                        await asyncio.sleep(VIDEO_STREAM_POLL_SLEEP_SEC)
                        continue
                    await ensure_ffmpeg_process(
                        playback_fps if playback_fps > 0 else VIDEO_STREAM_INTERPOLATION_TARGET_FPS
                    )

                # Fill a small server-side buffer with newest available frames.
                while (
                    len(pending_raw_frames) < VIDEO_STREAM_SERVER_BUFFER_FRAMES
                    and next_frame_index <= last_known_frame_index
                ):
                    frame_bytes = read_stream_frame_by_index(job, next_frame_index)
                    if not frame_bytes:
                        break
                    pending_raw_frames.append(frame_bytes)
                    next_frame_index += 1

                # Prevent queue growth by dropping stale buffered frames when behind.
                while len(pending_raw_frames) > VIDEO_STREAM_SERVER_BUFFER_FRAMES:
                    pending_raw_frames.popleft()

                interpolation_steps = 0
                while pending_raw_frames and len(pending_stream_frames) < VIDEO_STREAM_SERVER_BUFFER_FRAMES * (
                    VIDEO_STREAM_INTERPOLATION_MAX_STEPS + 1
                ):
                    current_raw_frame = pending_raw_frames.popleft()
                    sequence = build_interpolated_sequence(
                        previous_frame_bytes=previous_raw_frame_bytes,
                        current_frame_bytes=current_raw_frame,
                        interpolation_steps=interpolation_steps,
                    )
                    for stream_frame in sequence:
                        pending_stream_frames.append(stream_frame)
                    previous_raw_frame_bytes = current_raw_frame

                while len(pending_stream_frames) > VIDEO_STREAM_SERVER_BUFFER_FRAMES * (
                    VIDEO_STREAM_INTERPOLATION_MAX_STEPS + 1
                ):
                    pending_stream_frames.popleft()

                sent_any_frame = False
                now_perf = time.perf_counter()
                if ffmpeg_process is not None and pending_stream_frames and now_perf >= next_emit_at:
                    assert ffmpeg_process.stdin is not None
                    ffmpeg_process.stdin.write(pending_stream_frames.popleft())
                    await ffmpeg_process.stdin.drain()
                    sent_any_frame = True
                    next_emit_at += frame_interval_sec
                    if next_emit_at < now_perf - frame_interval_sec:
                        next_emit_at = now_perf

                if sent_any_frame or pending_stream_frames or pending_raw_frames:
                    stable_loops = 0
                else:
                    stable_loops += 1

                state = determine_job_state(job, stream_status)
                running = bool(job.process is not None and job.process.poll() is None)
                if state in {"done", "error"} and not running:
                    final_frame_total = last_known_frame_total or last_known_frame_index
                    if (
                        final_frame_total > 0
                        and next_frame_index > final_frame_total
                        and not pending_raw_frames
                        and not pending_stream_frames
                        and stable_loops >= 2
                    ):
                        break
                    if final_frame_total <= 0 and stable_loops >= VIDEO_STREAM_TERMINAL_STABLE_LOOPS:
                        break
                await asyncio.sleep(VIDEO_STREAM_POLL_SLEEP_SEC)
        except WebSocketDisconnect:
            pass
        finally:
            try:
                if ffmpeg_process is not None and ffmpeg_process.stdin is not None:
                    ffmpeg_process.stdin.close()
            except Exception:
                pass
            try:
                if stdout_task is not None:
                    await asyncio.wait_for(stdout_task, timeout=5.0)
            except Exception:
                if stdout_task is not None:
                    stdout_task.cancel()
            try:
                if stderr_task is not None:
                    await asyncio.wait_for(stderr_task, timeout=5.0)
            except Exception:
                if stderr_task is not None:
                    stderr_task.cancel()
            try:
                if ffmpeg_process is not None:
                    await asyncio.wait_for(ffmpeg_process.wait(), timeout=5.0)
            except Exception:
                try:
                    if ffmpeg_process is not None:
                        ffmpeg_process.kill()
                except Exception:
                    pass
            try:
                await websocket.close()
            except Exception:
                pass

    app.mount("/jobs", StaticFiles(directory=str(JOBS_ROOT), html=False), name="jobs")
    app.mount("/", StaticFiles(directory=str(PROJECT_ROOT), html=True), name="static")
    return app


def parse_args() -> argparse.Namespace:
    """
    Parse API server CLI options.
    """
    parser = argparse.ArgumentParser(description="Run realtime streaming API for FasterLivePortrait.")
    parser.add_argument("--host", default=DEFAULT_API_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--backend", choices=sorted(BACKEND_CHOICES), default=DEFAULT_BACKEND)
    parser.add_argument("--trt-runtime", choices=sorted(TRT_RUNTIME_CHOICES), default=DEFAULT_TRT_RUNTIME)
    parser.add_argument("--trt-precision", choices=["fp32", "fp16", "int8"], default=DEFAULT_TRT_PRECISION)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Run uvicorn app server.
    """
    global DEFAULT_BACKEND
    global DEFAULT_TRT_RUNTIME
    global DEFAULT_TRT_PRECISION
    global CURRENT_API_HOST
    global CURRENT_API_PORT
    global WARMUP_ENABLED
    args = parse_args()
    DEFAULT_BACKEND = str(args.backend).strip().lower() or DEFAULT_BACKEND
    DEFAULT_TRT_RUNTIME = str(args.trt_runtime).strip().lower() or DEFAULT_TRT_RUNTIME
    DEFAULT_TRT_PRECISION = str(args.trt_precision).strip().lower() or DEFAULT_TRT_PRECISION
    CURRENT_API_HOST = str(args.host).strip() or DEFAULT_API_HOST
    CURRENT_API_PORT = int(args.port)
    if args.no_warmup:
        WARMUP_ENABLED = False
    app_target: Any = "realtime_stream_api:create_app"
    factory_mode = True
    if not args.reload:
        app_target = create_app()
        factory_mode = False
    uvicorn.run(
        app_target,
        host=CURRENT_API_HOST,
        port=CURRENT_API_PORT,
        factory=factory_mode,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
