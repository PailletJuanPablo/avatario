"""
API server for real-time FasterLivePortrait generation with WebSocket status and video streaming.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_SCRIPT = PROJECT_ROOT / "faster_liveportrait_runner.py"
RUNNER_PYTHON = PROJECT_ROOT / ".venv-liveportrait" / "Scripts" / "python.exe"
if not RUNNER_PYTHON.exists():
    RUNNER_PYTHON = Path(sys.executable)

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
DEFAULT_SKIP_TRT_ENGINE_BUILD = (
    os.getenv("ANIMATION_SKIP_TRT_ENGINE_BUILD", "0").strip().lower() in {"1", "true", "yes"}
)
DEFAULT_AUDIO_MOTION_STRIDE = max(
    1,
    int(os.getenv("ANIMATION_AUDIO_MOTION_STRIDE", "1").strip() or "1"),
)
DEFAULT_ANIMATION_REGION = os.getenv("ANIMATION_ANIMATION_REGION", "all").strip().lower() or "all"
if DEFAULT_ANIMATION_REGION not in ANIMATION_REGION_CHOICES:
    DEFAULT_ANIMATION_REGION = "all"
DEFAULT_STITCHING_ENABLED = os.getenv("ANIMATION_STITCHING_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_RELATIVE_MOTION_ENABLED = (
    os.getenv("ANIMATION_RELATIVE_MOTION_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
)
DEFAULT_PASTE_BACK_ENABLED = os.getenv("ANIMATION_PASTE_BACK_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_API_HOST = os.getenv("ANIMATION_API_HOST", "0.0.0.0").strip() or "0.0.0.0"
DEFAULT_API_PORT = 8010
JOB_POLL_SLEEP_SEC = 0.12
VIDEO_STREAM_POLL_SLEEP_SEC = 0.02
VIDEO_STREAM_INPUT_FPS = 20.0
VIDEO_STREAM_CHUNK_SIZE = 16384
VIDEO_STREAM_BITRATE = "1400k"
VIDEO_STREAM_MAXRATE = "1400k"
VIDEO_STREAM_BUFSIZE = "700k"
VIDEO_STREAM_GOP = "12"
VIDEO_STREAM_KEYINT_MIN = "12"
VIDEO_STREAM_AUDIO_CODEC = "aac"
VIDEO_STREAM_AUDIO_BITRATE = "128k"
VIDEO_STREAM_AUDIO_SAMPLE_RATE = "48000"
VIDEO_STREAM_AUDIO_CHANNELS = "2"
VIDEO_STREAM_AUDIO_FILTER = "aresample=async=1:first_pts=0"
VIDEO_STREAM_MUX_DELAY = "0"
VIDEO_STREAM_MUX_PRELOAD = "0"
VIDEO_STREAM_TERMINAL_STABLE_LOOPS = 24
VIDEO_STREAM_MAX_BACKLOG_FRAMES = 10
VIDEO_STREAM_TARGET_LATENCY_FRAMES = 2
VIDEO_STREAM_SERVER_BUFFER_FRAMES = 6
VIDEO_STREAM_REALTIME_TARGET_DELAY_SEC = 0.18
VIDEO_STREAM_INTERPOLATION_MAX_STEPS = 1
VIDEO_STREAM_INTERPOLATION_TARGET_FPS = 24.0
VIDEO_STREAM_INTERPOLATION_ALPHA_QUALITY = 96
VIDEO_STREAM_GENERATION_FPS_MIN = 6.0
VIDEO_STREAM_GENERATION_FPS_MAX = 24.0
VIDEO_STREAM_GENERATION_FPS_SMOOTH_ALPHA = 0.18
AVATAR_VIDEO_OUTPUT_FPS = VIDEO_STREAM_INTERPOLATION_TARGET_FPS
AVATAR_VIDEO_FALLBACK_WIDTH = 1280
AVATAR_VIDEO_FALLBACK_HEIGHT = 720
VIDEO_STREAM_START_MODE_QUERY_KEY = "startMode"
VIDEO_STREAM_START_MODE_BUFFERED = "buffered"
VIDEO_STREAM_START_MODE_LIVE = "live"
VIDEO_STREAM_START_PROGRESS_QUERY_KEY = "startProgress"
VIDEO_STREAM_BUFFERED_START_PROGRESS_DEFAULT = 0.25
VIDEO_STREAM_START_MODE_CHOICES = {
    VIDEO_STREAM_START_MODE_BUFFERED,
    VIDEO_STREAM_START_MODE_LIVE,
}
VIDEO_STREAM_PLAYBACK_FPS_STATUS_KEY = "fps"
STREAM_BOUNDARY = "frame"
STREAM_STATUS_FILE_NAME = "status.json"
STREAM_IMAGE_FILE_NAME = "latest.jpg"
STREAM_FRAME_NAME_PATTERN = "frame_{:06d}.jpg"
RUN_LOG_FILE_NAME = "run.log"
RUN_REPORT_FILE_NAME = "run_report.json"
MAX_LOG_LINES = 400
JOBS_ROOT_REL = Path("output_fasterliveportrait/jobs")
JOBS_ROOT = PROJECT_ROOT / JOBS_ROOT_REL
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
WARMUP_OUTPUT_ROOT_REL = Path("output_fasterliveportrait/warmup")
WARMUP_STREAM_SUBDIR_NAME = "stream"
WARMUP_INPUTS_SUBDIR_NAME = "inputs"
WARMUP_AUDIO_FILE_NAME = "warmup.wav"
WARMUP_AUDIO_DURATION_SEC = 0.8
WARMUP_START_DELAY_SEC = 0.75
WARMUP_ENABLED = os.getenv("ANIMATION_WARMUP_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
AVATAR_MODE_IDLE = "idle"
AVATAR_MODE_TALKING = "talking"
AVATAR_MODE_CHOICES = {AVATAR_MODE_IDLE, AVATAR_MODE_TALKING}
AVATAR_IDLE_VIDEO_REL = Path(os.getenv("ANIMATION_IDLE_VIDEO", "inputs/idlevid.mp4").strip() or "inputs/idlevid.mp4")
AVATAR_IDLE_SOURCE_FRAME_REL = Path("output_fasterliveportrait/avatar_idle_source.png")
AVATAR_IDLE_MIN_HOLD_SEC = 0.35
AVATAR_READY_PROGRESS = 0.25
AVATAR_STATE_POLL_SLEEP_SEC = 0.1


def now_ms() -> int:
    """
    Return current UTC timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def normalize_rel_path(value: str) -> str:
    """
    Normalize local path string to POSIX style for runner args.
    """
    return value.replace("\\", "/")


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


def probe_media_duration_sec(media_path: Path) -> float:
    """
    Probe media duration in seconds using ffprobe when available.
    """
    try:
        completed = subprocess.run(
            [
                "ffprobe",
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
    return build_public_file_url(idle_video_abs)


def ensure_idle_source_frame_abs() -> Path | None:
    """
    Extract and cache one stable source frame from the idle video.
    """
    idle_video_abs = resolve_idle_video_abs()
    if idle_video_abs is None:
        return None
    target_path = (PROJECT_ROOT / AVATAR_IDLE_SOURCE_FRAME_REL).resolve()
    try:
        idle_video_mtime_ns = idle_video_abs.stat().st_mtime_ns
    except OSError:
        return None
    if target_path.exists():
        try:
            target_mtime_ns = target_path.stat().st_mtime_ns
        except OSError:
            target_mtime_ns = -1
        if target_mtime_ns >= idle_video_mtime_ns:
            return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(idle_video_abs),
                "-frames:v",
                "1",
                str(target_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
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


def start_runtime_restart_thread() -> None:
    """
    Schedule current API process exit so Docker can restart the container from a clean state.
    """

    def restart_runtime_process() -> None:
        time.sleep(RUNTIME_RESTART_DELAY_SEC)
        try:
            os.kill(1, signal.SIGKILL)
            return
        except OSError:
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
) -> list[str]:
    """
    Build ffmpeg command that converts JPEG frames from stdin into fragmented MP4 for WebSocket transport.
    """
    safe_fps = max(1.0, float(input_fps))
    command = [
        "ffmpeg",
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
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
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
            "-f",
            "mp4",
            "pipe:1",
        ]
    )
    return command


@dataclass
class JobRecord:
    job_id: str
    created_at_ms: int
    mode: str
    source_frame_arg: str
    source_frame_abs: Path
    output_rel: Path
    output_abs: Path
    stream_rel: Path
    stream_abs: Path
    audio_input_rel: Path
    audio_input_abs: Path
    audio_original_name: str
    audio_duration_sec: float
    audio_motion_stride: int
    animation_region: str
    stitching_enabled: bool
    relative_motion_enabled: bool
    paste_back_enabled: bool
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
    def status_abs(self) -> Path:
        return self.stream_abs / STREAM_STATUS_FILE_NAME

    @property
    def latest_frame_abs(self) -> Path:
        return self.stream_abs / STREAM_IMAGE_FILE_NAME

    def stream_frame_abs(self, frame_index: int) -> Path:
        return self.stream_abs / STREAM_FRAME_NAME_PATTERN.format(int(frame_index))

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
class AvatarTalkingFrameState:
    job_id: str
    next_frame_index: int = 1
    last_known_frame_index: int = 0
    last_known_frame_total: int = 0
    estimated_generation_fps: float = VIDEO_STREAM_INPUT_FPS
    pending_raw_frames: deque[bytes] = field(default_factory=deque)
    pending_stream_frames: deque[bytes] = field(default_factory=deque)
    previous_raw_frame_bytes: bytes | None = None
    last_frame_bytes: bytes | None = None


class IdleVideoLooper:
    """
    Loop decoded idle video frames for continuous avatar streaming.
    """

    def __init__(self, idle_video_abs: Path | None) -> None:
        self.idle_video_abs = idle_video_abs
        self.capture: cv2.VideoCapture | None = None
        self.frame_width = AVATAR_VIDEO_FALLBACK_WIDTH
        self.frame_height = AVATAR_VIDEO_FALLBACK_HEIGHT
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
        if width > 0 and height > 0:
            self.frame_width = width
            self.frame_height = height
        self.capture = capture

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


JOBS_LOCK = threading.Lock()
JOBS: dict[str, JobRecord] = {}
JOB_QUEUE_CONDITION = threading.Condition()
JOB_QUEUE: deque[str] = deque()
JOB_WORKER_LOCK = threading.Lock()
JOB_WORKER_THREAD: threading.Thread | None = None
WARMUP_LOCK = threading.Lock()
WARMUP_LAST_STARTED_AT_MS = 0
WARMUP_RUNNING = False
RUNTIME_RESTART_LOCK = threading.Lock()
RUNTIME_RESTARTING = False
RUNTIME_RESTART_REQUESTED_AT_MS = 0
AVATAR_STATE_LOCK = threading.Lock()
AVATAR_WORKER_LOCK = threading.Lock()
AVATAR_WORKER_THREAD: threading.Thread | None = None
AVATAR_STATE_SEQUENCE = 0
AVATAR_MODE = AVATAR_MODE_IDLE
AVATAR_CURRENT_JOB_ID = ""
AVATAR_CURRENT_JOB_STARTED_AT_MS = 0
AVATAR_CURRENT_JOB_ENDS_AT_MS = 0
AVATAR_LAST_IDLE_STARTED_AT_MS = PROCESS_STARTED_AT_MS


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


def enqueue_job(job_id: str) -> None:
    """
    Enqueue job identifier for background worker execution.
    """
    with JOB_QUEUE_CONDITION:
        JOB_QUEUE.append(job_id)
        JOB_QUEUE_CONDITION.notify()


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


def get_job(job_id: str) -> JobRecord:
    """
    Fetch job record or raise HTTP 404.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def finish_job(job: JobRecord, exit_code: int) -> None:
    """
    Mark job completed and close resources.
    """
    with JOBS_LOCK:
        job.process = None
        job.exit_code = int(exit_code)
        job.finished_at_ms = now_ms()
        if job.log_handle is not None:
            try:
                job.log_handle.close()
            except OSError:
                pass
            job.log_handle = None


def run_job(job: JobRecord) -> None:
    """
    Execute one queued job in the background worker.
    """
    command = build_runner_command(job)
    job.log_abs.parent.mkdir(parents=True, exist_ok=True)
    log_handle = job.log_abs.open("ab")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    with JOBS_LOCK:
        job.started_at_ms = now_ms()
        job.process = process
        job.log_handle = log_handle
    try:
        exit_code = process.wait()
    except Exception:
        exit_code = -1
    finish_job(job, exit_code)


def job_worker_loop() -> None:
    """
    Persistent single-worker loop that executes queued jobs sequentially.
    """
    while True:
        with JOB_QUEUE_CONDITION:
            while not JOB_QUEUE:
                JOB_QUEUE_CONDITION.wait()
            job_id = JOB_QUEUE.popleft()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            continue
        if job.exit_code is not None:
            continue
        run_job(job)


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


def get_avatar_state_snapshot() -> dict[str, Any]:
    """
    Read current avatar scheduler state atomically.
    """
    with AVATAR_STATE_LOCK:
        current_job_id = AVATAR_CURRENT_JOB_ID
        current_mode = AVATAR_MODE
        current_started_at_ms = AVATAR_CURRENT_JOB_STARTED_AT_MS
        current_ends_at_ms = AVATAR_CURRENT_JOB_ENDS_AT_MS
        sequence = AVATAR_STATE_SEQUENCE
        idle_started_at_ms = AVATAR_LAST_IDLE_STARTED_AT_MS
    current_job = None
    if current_job_id:
        with JOBS_LOCK:
            current_job = JOBS.get(current_job_id)
    return {
        "mode": current_mode,
        "sequence": sequence,
        "currentJobId": current_job_id,
        "currentJobStartedAtMs": current_started_at_ms,
        "currentJobEndsAtMs": current_ends_at_ms,
        "idleStartedAtMs": idle_started_at_ms,
        "idleVideoUrl": resolve_idle_video_url(),
        "bufferedStartProgress": AVATAR_READY_PROGRESS,
        "currentJobVideoWsUrl": f"/ws/jobs/{current_job_id}/video" if current_job_id else "",
        "currentJobStatusWsUrl": f"/ws/jobs/{current_job_id}" if current_job_id else "",
        "currentJobAudioDurationSec": current_job.audio_duration_sec if current_job is not None else 0.0,
        "currentJobSourceFrameUrl": build_public_file_url(current_job.source_frame_abs) if current_job is not None else "",
    }


def count_pending_avatar_jobs(current_job_id: str) -> int:
    """
    Count queued or ready avatar jobs that still have not been played.
    """
    with JOBS_LOCK:
        job_records = list(JOBS.values())
    pending_count = 0
    for job in job_records:
        if job.job_id == current_job_id:
            continue
        if job.avatar_play_finished_at_ms is not None:
            continue
        stream_status = read_json(job.status_abs)
        state = determine_job_state(job, stream_status)
        if state in {"error", "canceled"}:
            continue
        pending_count += 1
    return pending_count


def build_avatar_payload() -> dict[str, Any]:
    """
    Build one public avatar state payload for the UI.
    """
    snapshot = get_avatar_state_snapshot()
    queue_depth = count_pending_avatar_jobs(str(snapshot["currentJobId"] or ""))
    running_job = get_running_job_record()
    current_job: JobRecord | None = None
    current_job_id = str(snapshot["currentJobId"] or "")
    if current_job_id:
        with JOBS_LOCK:
            current_job = JOBS.get(current_job_id)
    return {
        **snapshot,
        "queueDepth": queue_depth,
        "runningJobId": running_job.job_id if running_job is not None else "",
        "idleVideoAvailable": bool(snapshot["idleVideoUrl"]),
        "avatarVideoWsUrl": "/ws/avatar/video",
        "currentJobDrivingMediaUrl": build_public_file_url(current_job.audio_input_abs) if current_job is not None else "",
        "status": "ok",
    }


def is_job_ready_for_avatar(job: JobRecord, stream_status: dict[str, Any] | None) -> bool:
    """
    Determine whether a job has buffered enough frames to start avatar playback.
    """
    state = determine_job_state(job, stream_status)
    if state in {"error", "canceled"}:
        return False
    if state == "done":
        return True
    frame_index = parse_status_int(stream_status, "frameIndex")
    frame_total = parse_status_int(stream_status, "frameTotal")
    progress_ratio = resolve_stream_progress_ratio(stream_status, frame_index, frame_total)
    return progress_ratio >= AVATAR_READY_PROGRESS


def select_next_avatar_job() -> JobRecord | None:
    """
    Select the oldest ready job that has not been played yet.
    """
    with JOBS_LOCK:
        job_records = sorted(JOBS.values(), key=lambda item: item.created_at_ms)
    for job in job_records:
        if job.avatar_play_finished_at_ms is not None or job.avatar_play_started_at_ms is not None:
            continue
        stream_status = read_json(job.status_abs)
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


def activate_avatar_idle_mode() -> None:
    """
    Switch avatar state to idle and advance the transition sequence.
    """
    global AVATAR_MODE
    global AVATAR_CURRENT_JOB_ID
    global AVATAR_CURRENT_JOB_STARTED_AT_MS
    global AVATAR_CURRENT_JOB_ENDS_AT_MS
    global AVATAR_LAST_IDLE_STARTED_AT_MS
    global AVATAR_STATE_SEQUENCE
    with AVATAR_STATE_LOCK:
        AVATAR_MODE = AVATAR_MODE_IDLE
        AVATAR_CURRENT_JOB_ID = ""
        AVATAR_CURRENT_JOB_STARTED_AT_MS = 0
        AVATAR_CURRENT_JOB_ENDS_AT_MS = 0
        AVATAR_LAST_IDLE_STARTED_AT_MS = now_ms()
        AVATAR_STATE_SEQUENCE += 1


def activate_avatar_job(job: JobRecord) -> None:
    """
    Switch avatar state to one talking job and record playback timestamps.
    """
    global AVATAR_MODE
    global AVATAR_CURRENT_JOB_ID
    global AVATAR_CURRENT_JOB_STARTED_AT_MS
    global AVATAR_CURRENT_JOB_ENDS_AT_MS
    global AVATAR_STATE_SEQUENCE
    started_at_ms = now_ms()
    ends_at_ms = started_at_ms + int(max(0.1, job.audio_duration_sec) * 1000.0)
    with JOBS_LOCK:
        job.avatar_play_started_at_ms = started_at_ms
    with AVATAR_STATE_LOCK:
        AVATAR_MODE = AVATAR_MODE_TALKING
        AVATAR_CURRENT_JOB_ID = job.job_id
        AVATAR_CURRENT_JOB_STARTED_AT_MS = started_at_ms
        AVATAR_CURRENT_JOB_ENDS_AT_MS = ends_at_ms
        AVATAR_STATE_SEQUENCE += 1


def advance_avatar_state_machine() -> None:
    """
    Advance avatar scheduler state between idle and talking modes.
    """
    with AVATAR_STATE_LOCK:
        current_mode = AVATAR_MODE
        current_job_id = AVATAR_CURRENT_JOB_ID
        current_job_ends_at_ms = AVATAR_CURRENT_JOB_ENDS_AT_MS
        idle_started_at_ms = AVATAR_LAST_IDLE_STARTED_AT_MS

    now_timestamp_ms = now_ms()
    if current_mode == AVATAR_MODE_TALKING and current_job_id:
        with JOBS_LOCK:
            current_job = JOBS.get(current_job_id)
        if current_job is None:
            activate_avatar_idle_mode()
            return
        current_status = read_json(current_job.status_abs)
        current_state = determine_job_state(current_job, current_status)
        if current_state in {"error", "canceled"} or now_timestamp_ms >= current_job_ends_at_ms:
            with JOBS_LOCK:
                current_job.avatar_play_finished_at_ms = now_timestamp_ms
            activate_avatar_idle_mode()
        return

    if current_mode != AVATAR_MODE_IDLE:
        activate_avatar_idle_mode()
        return
    if now_timestamp_ms - idle_started_at_ms < int(AVATAR_IDLE_MIN_HOLD_SEC * 1000.0):
        return

    next_job = select_next_avatar_job()
    if next_job is None:
        return
    activate_avatar_job(next_job)


def avatar_worker_loop() -> None:
    """
    Persistent scheduler that alternates the avatar between idle and talking jobs.
    """
    activate_avatar_idle_mode()
    while True:
        try:
            advance_avatar_state_machine()
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
        "--output-dir",
        normalize_rel_path(str(job.output_rel)),
        "--stream-dir",
        normalize_rel_path(str(job.stream_rel)),
        "--animation-region",
        job.animation_region,
    ]
    if not job.paste_back_enabled:
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


def read_updated_latest_frame(job: JobRecord, last_mtime_ns: int) -> tuple[bytes | None, int, bool]:
    """
    Read latest JPEG frame only when file mtime changes.
    """
    frame_path = job.latest_frame_abs
    if not frame_path.exists():
        return None, last_mtime_ns, False
    try:
        mtime_ns = frame_path.stat().st_mtime_ns
    except OSError:
        return None, last_mtime_ns, False
    if mtime_ns == last_mtime_ns:
        return None, last_mtime_ns, False
    try:
        frame_bytes = frame_path.read_bytes()
    except OSError:
        return None, last_mtime_ns, False
    if not frame_bytes:
        return None, last_mtime_ns, False
    return frame_bytes, mtime_ns, True


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


def normalize_jpeg_frame_to_canvas(frame_bytes: bytes, canvas_size: tuple[int, int]) -> bytes:
    """
    Decode JPEG bytes, resize into the shared avatar canvas, and encode back to JPEG.
    """
    frame_image = decode_jpeg_frame(frame_bytes)
    if frame_image is None:
        return b""
    return encode_jpeg_frame(fit_frame_to_canvas(frame_image, canvas_size))


def update_avatar_talking_frame_state(job: JobRecord, state: AvatarTalkingFrameState) -> dict[str, Any] | None:
    """
    Refresh one talking job frame buffer so the continuous avatar stream can emit frames sequentially.
    """
    stream_status = read_json(job.status_abs)
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
) -> tuple[bytes | None, bool]:
    """
    Resolve the next frame for one talking job inside the continuous avatar stream.
    """
    stream_status = update_avatar_talking_frame_state(job, state)
    if state.pending_stream_frames:
        frame_bytes = state.pending_stream_frames.popleft()
        normalized_frame_bytes = normalize_jpeg_frame_to_canvas(frame_bytes, canvas_size)
        if normalized_frame_bytes:
            state.last_frame_bytes = normalized_frame_bytes
            return normalized_frame_bytes, False

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
            and not state.pending_raw_frames
            and not state.pending_stream_frames
        )
    )
    if is_finished:
        return None, True
    if state.last_frame_bytes:
        return state.last_frame_bytes, False
    return None, False


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


def read_stream_frame_by_index(job: JobRecord, frame_index: int) -> bytes | None:
    """
    Read sequential stream frame by index.
    """
    frame_path = job.stream_frame_abs(frame_index)
    if not frame_path.exists():
        return None
    try:
        frame_bytes = frame_path.read_bytes()
    except OSError:
        return None
    if not frame_bytes:
        return None
    return frame_bytes


def resolve_driving_media_url(job: JobRecord) -> str:
    """
    Resolve driving media URL for UI playback.
    """
    for extension in (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4"):
        candidate = job.output_abs / f"driving{extension}"
        if candidate.exists():
            return f"/jobs/{job.job_id}/{candidate.name}"
    return f"/jobs/{job.job_id}/inputs/{job.audio_input_abs.name}"


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
        "--output-dir",
        normalize_rel_path(str(output_root_rel)),
        "--stream-dir",
        normalize_rel_path(str(stream_rel)),
        "--disable-stream",
        "--animation-region",
        DEFAULT_ANIMATION_REGION,
    ]
    if not DEFAULT_PASTE_BACK_ENABLED:
        command.append("--no-paste-back")
    if not DEFAULT_STITCHING_ENABLED:
        command.append("--no-stitching")
    if not DEFAULT_RELATIVE_MOTION_ENABLED:
        command.append("--no-relative-motion")
    if DEFAULT_SKIP_TRT_ENGINE_BUILD:
        command.append("--skip-trt-engine-build")
    return command


def ensure_warmup_audio_file(audio_abs_path: Path) -> None:
    """
    Ensure warmup silence WAV exists for startup preheat run.
    """
    if audio_abs_path.exists():
        return
    audio_abs_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
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
    try:
        time.sleep(WARMUP_START_DELAY_SEC)
        output_root_abs = PROJECT_ROOT / WARMUP_OUTPUT_ROOT_REL
        input_dir_abs = output_root_abs / WARMUP_INPUTS_SUBDIR_NAME
        warmup_audio_abs = input_dir_abs / WARMUP_AUDIO_FILE_NAME
        warmup_audio_rel = warmup_audio_abs.relative_to(PROJECT_ROOT)
        ensure_warmup_audio_file(warmup_audio_abs)
        command = build_warmup_command(warmup_audio_rel)
        print("[warmup] starting runtime warmup job")
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        print("[warmup] completed runtime warmup job")
    except Exception as exc:
        print(f"[warmup] failed: {exc}")
    finally:
        with WARMUP_LOCK:
            WARMUP_RUNNING = False


def build_job_payload(job: JobRecord) -> dict[str, Any]:
    """
    Build API response payload for job.
    """
    stream_status = read_json(job.status_abs)
    report = read_json(job.report_abs)
    process = job.process
    running = bool(job.exit_code is None and process is not None and process.poll() is None)
    state = determine_job_state(job, stream_status)
    exit_code = job.exit_code if job.exit_code is not None else (process.poll() if process is not None else None)
    queued_position = queue_position(job.job_id)
    payload: dict[str, Any] = {
        "jobId": job.job_id,
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
        "animationRegion": job.animation_region,
        "stitchingEnabled": job.stitching_enabled,
        "relativeMotionEnabled": job.relative_motion_enabled,
        "pasteBackEnabled": job.paste_back_enabled,
        "sourceFrame": job.source_frame_arg,
        "sourceFrameUrl": build_public_file_url(job.source_frame_abs),
        "status": stream_status or {},
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
    source_image: UploadFile | None,
    output_abs: Path,
    output_rel: Path,
) -> tuple[Path, str]:
    """
    Resolve source frame from optional uploaded image or local path input.
    """
    try:
        fixed_source = resolve_configured_fixed_source_frame()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if fixed_source is not None:
        return fixed_source
    normalized_source_frame = str(source_frame or "").strip() or DEFAULT_SOURCE_FRAME
    if source_image is None or not source_image.filename:
        return resolve_source_frame_candidate(normalized_source_frame)

    extension = Path(source_image.filename).suffix.lower()
    if extension not in ALLOWED_SOURCE_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source image extension '{extension}'. Allowed: {sorted(ALLOWED_SOURCE_IMAGE_EXTENSIONS)}",
        )
    source_image_rel = output_rel / "inputs" / f"source{extension}"
    source_image_abs = (PROJECT_ROOT / source_image_rel).resolve()
    output_abs.mkdir(parents=True, exist_ok=True)
    await save_upload_file(source_image, source_image_abs)
    return source_image_abs, normalize_rel_path(str(source_image_rel))


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

    @app.on_event("startup")
    async def startup_warmup() -> None:
        ensure_job_worker_started()
        ensure_avatar_worker_started()
        if not WARMUP_ENABLED:
            return
        warmup_thread = threading.Thread(target=run_startup_warmup_once, daemon=True)
        warmup_thread.start()

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        with JOB_QUEUE_CONDITION:
            queue_depth = len(JOB_QUEUE)
        worker_alive = bool(JOB_WORKER_THREAD is not None and JOB_WORKER_THREAD.is_alive())
        running_job = get_running_job_record()
        head_queued_job_id = get_head_queued_job_id()
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
        avatar_payload = build_avatar_payload()
        return {
            "status": "ok",
            "backend": DEFAULT_BACKEND,
            "trtRuntime": DEFAULT_TRT_RUNTIME,
            "trtPrecision": DEFAULT_TRT_PRECISION,
            "skipTrtEngineBuild": DEFAULT_SKIP_TRT_ENGINE_BUILD,
            "defaultAudioMotionStride": DEFAULT_AUDIO_MOTION_STRIDE,
            "defaultAnimationRegion": DEFAULT_ANIMATION_REGION,
            "defaultStitchingEnabled": DEFAULT_STITCHING_ENABLED,
            "defaultRelativeMotionEnabled": DEFAULT_RELATIVE_MOTION_ENABLED,
            "defaultPasteBackEnabled": DEFAULT_PASTE_BACK_ENABLED,
            "fixedSourceEnabled": bool(FIXED_SOURCE_FRAME),
            "fixedSourceFrame": fixed_source_frame_arg,
            "fixedSourceError": fixed_source_error,
            "allowCustomSourceFrame": not bool(FIXED_SOURCE_FRAME),
            "warmupEnabled": WARMUP_ENABLED,
            "warmupRunning": WARMUP_RUNNING,
            "warmupLastStartedAtMs": WARMUP_LAST_STARTED_AT_MS,
            "warmupSourceFrame": warmup_source_frame_arg,
            "runtimeRestarting": RUNTIME_RESTARTING,
            "runtimeRestartRequestedAtMs": RUNTIME_RESTART_REQUESTED_AT_MS,
            "processStartedAtMs": PROCESS_STARTED_AT_MS,
            "runningJobId": running_job.job_id if running_job is not None else "",
            "runningJobState": determine_job_state(running_job, read_json(running_job.status_abs)) if running_job is not None else "",
            "headQueuedJobId": head_queued_job_id,
            "jobWorkerAlive": worker_alive,
            "jobQueueDepth": queue_depth,
            "avatarMode": avatar_payload["mode"],
            "avatarSequence": avatar_payload["sequence"],
            "avatarCurrentJobId": avatar_payload["currentJobId"],
            "avatarCurrentJobStartedAtMs": avatar_payload["currentJobStartedAtMs"],
            "avatarCurrentJobEndsAtMs": avatar_payload["currentJobEndsAtMs"],
            "avatarIdleVideoUrl": avatar_payload["idleVideoUrl"],
            "avatarBufferedStartProgress": avatar_payload["bufferedStartProgress"],
            "containerLogPath": str(CONTAINER_LOG_REL),
            "workerLogPath": str(PERSISTENT_WORKER_LOG_REL),
        }

    @app.get("/api/avatar/status")
    async def avatar_status() -> JSONResponse:
        ensure_avatar_worker_started()
        return JSONResponse(build_avatar_payload())

    @app.post("/api/warmup")
    async def warmup() -> JSONResponse:
        ensure_runtime_accepting_requests()
        if not WARMUP_ENABLED:
            return JSONResponse({"status": "disabled"})
        warmup_thread = threading.Thread(target=run_startup_warmup_once, daemon=True)
        warmup_thread.start()
        return JSONResponse({"status": "started", "startedAtMs": now_ms()})

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
        audio: UploadFile = File(...),
        source_image: UploadFile | None = File(None),
        source_frame: str = Form(DEFAULT_SOURCE_FRAME),
        mode: str = Form(DEFAULT_MODE),
        motion_stride: int = Form(DEFAULT_AUDIO_MOTION_STRIDE),
        animation_region: str = Form(DEFAULT_ANIMATION_REGION),
        stitching: bool = Form(DEFAULT_STITCHING_ENABLED),
        relative_motion: bool = Form(DEFAULT_RELATIVE_MOTION_ENABLED),
        paste_back: bool = Form(DEFAULT_PASTE_BACK_ENABLED),
    ) -> JSONResponse:
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
        normalized_animation_region = str(animation_region or "").strip().lower()
        if normalized_animation_region not in ANIMATION_REGION_CHOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid animation_region. Allowed values: {sorted(ANIMATION_REGION_CHOICES)}",
            )

        job_id = make_job_id()
        output_rel = JOBS_ROOT_REL / job_id
        output_abs = PROJECT_ROOT / output_rel
        stream_rel = output_rel / "stream"
        stream_abs = PROJECT_ROOT / stream_rel
        log_rel = output_rel / RUN_LOG_FILE_NAME
        log_abs = PROJECT_ROOT / log_rel
        input_rel = output_rel / "inputs" / f"driving{extension}"
        input_abs = PROJECT_ROOT / input_rel
        output_abs.mkdir(parents=True, exist_ok=True)
        source_frame_abs, source_frame_arg = await resolve_requested_source_frame(
            source_frame=source_frame,
            source_image=source_image,
            output_abs=output_abs,
            output_rel=output_rel,
        )
        await save_upload_file(audio, input_abs)
        audio_duration_sec = probe_media_duration_sec(input_abs)

        job = JobRecord(
            job_id=job_id,
            created_at_ms=now_ms(),
            mode=mode,
            source_frame_arg=source_frame_arg,
            source_frame_abs=source_frame_abs,
            output_rel=output_rel,
            output_abs=output_abs,
            stream_rel=stream_rel,
            stream_abs=stream_abs,
            audio_input_rel=input_rel,
            audio_input_abs=input_abs,
            audio_original_name=audio.filename or input_abs.name,
            audio_duration_sec=audio_duration_sec,
            audio_motion_stride=int(motion_stride),
            animation_region=normalized_animation_region,
            stitching_enabled=bool(stitching),
            relative_motion_enabled=bool(relative_motion),
            paste_back_enabled=bool(paste_back),
            log_rel=log_rel,
            log_abs=log_abs,
        )
        register_job(job)
        enqueue_job(job.job_id)
        return JSONResponse(build_job_payload(job))

    @app.get("/api/jobs/{job_id}/status")
    async def job_status(job_id: str) -> JSONResponse:
        job = get_job(job_id)
        return JSONResponse(build_job_payload(job))

    @app.get("/api/jobs/{job_id}/report")
    async def job_report(job_id: str) -> JSONResponse:
        job = get_job(job_id)
        report = read_json(job.report_abs)
        if report is None:
            raise HTTPException(status_code=404, detail="run_report.json not available yet")
        return JSONResponse(report)

    @app.get("/api/jobs/{job_id}/log")
    async def job_log(job_id: str, lines: int = 200) -> JSONResponse:
        job = get_job(job_id)
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
        job = get_job(job_id)

        async def stream_generator() -> Any:
            last_mtime_ns = -1
            stable_loops = 0
            while True:
                if await request.is_disconnected():
                    break
                frame_path = job.latest_frame_abs
                if frame_path.exists():
                    try:
                        mtime_ns = frame_path.stat().st_mtime_ns
                    except OSError:
                        mtime_ns = -1
                    if mtime_ns != last_mtime_ns:
                        last_mtime_ns = mtime_ns
                        stable_loops = 0
                        try:
                            frame_bytes = frame_path.read_bytes()
                        except OSError:
                            frame_bytes = b""
                        if frame_bytes:
                            header = (
                                f"--{STREAM_BOUNDARY}\r\n"
                                "Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(frame_bytes)}\r\n\r\n"
                            ).encode("utf-8")
                            yield header + frame_bytes + b"\r\n"
                    else:
                        stable_loops += 1
                else:
                    stable_loops += 1

                stream_status = read_json(job.status_abs)
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
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        last_status_signature = ""

        try:
            while True:
                stream_status = read_json(job.status_abs)
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

    @app.websocket("/ws/avatar/video")
    async def avatar_video_ws(websocket: WebSocket) -> None:
        ensure_avatar_worker_started()
        await websocket.accept()
        idle_looper = IdleVideoLooper(resolve_idle_video_abs())
        canvas_size = idle_looper.canvas_size
        ffmpeg_process = await asyncio.create_subprocess_exec(
            *build_video_stream_command(AVATAR_VIDEO_OUTPUT_FPS, include_silent_audio=True),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert ffmpeg_process.stdin is not None
        assert ffmpeg_process.stdout is not None
        assert ffmpeg_process.stderr is not None
        talking_state: AvatarTalkingFrameState | None = None
        talking_job: JobRecord | None = None
        frame_interval_sec = 1.0 / max(1.0, AVATAR_VIDEO_OUTPUT_FPS)
        next_emit_at = time.perf_counter()

        async def pump_mp4_stdout() -> None:
            while True:
                chunk = await ffmpeg_process.stdout.read(VIDEO_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                await websocket.send_bytes(chunk)

        async def drain_stderr() -> None:
            while True:
                chunk = await ffmpeg_process.stderr.read(VIDEO_STREAM_CHUNK_SIZE)
                if not chunk:
                    break

        stdout_task = asyncio.create_task(pump_mp4_stdout())
        stderr_task = asyncio.create_task(drain_stderr())

        try:
            while True:
                now_perf = time.perf_counter()
                if now_perf < next_emit_at:
                    await asyncio.sleep(min(VIDEO_STREAM_POLL_SLEEP_SEC, next_emit_at - now_perf))
                    continue

                snapshot = get_avatar_state_snapshot()
                desired_job_id = str(snapshot["currentJobId"] or "") if snapshot["mode"] == AVATAR_MODE_TALKING else ""
                if talking_state is not None and desired_job_id != talking_state.job_id:
                    talking_state = None
                    talking_job = None
                if talking_state is None and desired_job_id:
                    with JOBS_LOCK:
                        candidate_job = JOBS.get(desired_job_id)
                    if candidate_job is not None:
                        talking_job = candidate_job
                        talking_state = AvatarTalkingFrameState(job_id=desired_job_id)

                frame_bytes = None
                if talking_state is not None and talking_job is not None:
                    frame_bytes, is_finished = resolve_avatar_talking_frame(talking_job, talking_state, canvas_size)
                    if is_finished:
                        talking_state = None
                        talking_job = None
                        frame_bytes = None

                if not frame_bytes:
                    idle_frame = idle_looper.read_next_frame()
                    frame_bytes = encode_jpeg_frame(fit_frame_to_canvas(idle_frame, canvas_size))

                if frame_bytes:
                    ffmpeg_process.stdin.write(frame_bytes)
                    await ffmpeg_process.stdin.drain()

                next_emit_at += frame_interval_sec
                if next_emit_at < now_perf - frame_interval_sec:
                    next_emit_at = now_perf
        except WebSocketDisconnect:
            pass
        finally:
            idle_looper.close()
            try:
                ffmpeg_process.stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(stdout_task, timeout=5.0)
            except Exception:
                stdout_task.cancel()
            try:
                await asyncio.wait_for(stderr_task, timeout=5.0)
            except Exception:
                stderr_task.cancel()
            try:
                await asyncio.wait_for(ffmpeg_process.wait(), timeout=5.0)
            except Exception:
                try:
                    ffmpeg_process.kill()
                except Exception:
                    pass
            try:
                await websocket.close()
            except Exception:
                pass

    @app.websocket("/ws/avatar")
    async def avatar_status_ws(websocket: WebSocket) -> None:
        ensure_avatar_worker_started()
        await websocket.accept()
        last_signature = ""
        try:
            while True:
                payload = build_avatar_payload()
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
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        stream_start_mode = resolve_video_stream_start_mode(websocket)
        stream_start_progress = resolve_video_stream_start_progress(websocket)
        use_buffered_start = stream_start_mode == VIDEO_STREAM_START_MODE_BUFFERED
        ffmpeg_process: asyncio.subprocess.Process | None = None
        ffmpeg_input_fps = VIDEO_STREAM_INTERPOLATION_TARGET_FPS

        next_frame_index = 1
        last_known_frame_index = 0
        last_known_frame_total = 0
        estimated_generation_fps = VIDEO_STREAM_INPUT_FPS
        frame_interval_sec = 1.0 / max(1.0, ffmpeg_input_fps)
        next_emit_at = 0.0
        stream_started_at = time.perf_counter()
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
                *build_video_stream_command(ffmpeg_input_fps, job.audio_input_abs),
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
            if not use_buffered_start:
                await ensure_ffmpeg_process(VIDEO_STREAM_INTERPOLATION_TARGET_FPS)
            while True:
                stream_status = read_json(job.status_abs)
                status_frame_index = parse_status_int(stream_status, "frameIndex")
                status_frame_total = parse_status_int(stream_status, "frameTotal")
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

                if use_buffered_start and ffmpeg_process is None:
                    playback_fps = resolve_stream_playback_fps(stream_status)
                    state = determine_job_state(job, stream_status)
                    running = bool(job.process is not None and job.process.poll() is None)
                    if playback_fps <= 0 and running and state not in {"done", "error"}:
                        await asyncio.sleep(VIDEO_STREAM_POLL_SLEEP_SEC)
                        continue
                    await ensure_ffmpeg_process(
                        playback_fps if playback_fps > 0 else VIDEO_STREAM_INTERPOLATION_TARGET_FPS
                    )

                if not use_buffered_start:
                    # Keep stream near live edge by syncing frame index to wall clock.
                    elapsed_sec = max(0.0, time.perf_counter() - stream_started_at)
                    expected_live_index = int(
                        max(0.0, elapsed_sec - VIDEO_STREAM_REALTIME_TARGET_DELAY_SEC) * estimated_generation_fps
                    ) + 1
                    if expected_live_index > next_frame_index:
                        desired_live_index = min(expected_live_index, max(1, last_known_frame_index))
                        if desired_live_index > next_frame_index:
                            next_frame_index = desired_live_index

                    # Keep stream near live edge by skipping stale backlog when producer lags.
                    backlog_frames = last_known_frame_index - next_frame_index + 1
                    if backlog_frames > VIDEO_STREAM_MAX_BACKLOG_FRAMES:
                        next_frame_index = max(
                            1,
                            last_known_frame_index - VIDEO_STREAM_TARGET_LATENCY_FRAMES + 1,
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

                interpolation_steps = 0 if use_buffered_start else resolve_interpolation_steps(estimated_generation_fps)
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
    global WARMUP_ENABLED
    args = parse_args()
    DEFAULT_BACKEND = str(args.backend).strip().lower() or DEFAULT_BACKEND
    DEFAULT_TRT_RUNTIME = str(args.trt_runtime).strip().lower() or DEFAULT_TRT_RUNTIME
    DEFAULT_TRT_PRECISION = str(args.trt_precision).strip().lower() or DEFAULT_TRT_PRECISION
    if args.no_warmup:
        WARMUP_ENABLED = False
    uvicorn.run(
        "realtime_stream_api:create_app",
        host=args.host,
        port=args.port,
        factory=True,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
