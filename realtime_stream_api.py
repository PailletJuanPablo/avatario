"""
API server for real-time FasterLivePortrait generation with WebSocket status and video streaming.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
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
DEFAULT_MODE = "preview"
BACKEND_TRT = "trt"
BACKEND_ONNX = "onnx"
TRT_RUNTIME_DOCKER = "docker"
TRT_RUNTIME_LOCAL = "local"
BACKEND_CHOICES = {BACKEND_TRT, BACKEND_ONNX}
TRT_RUNTIME_CHOICES = {TRT_RUNTIME_DOCKER, TRT_RUNTIME_LOCAL}

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
DEFAULT_API_HOST = "127.0.0.1"
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
STREAM_BOUNDARY = "frame"
STREAM_STATUS_FILE_NAME = "status.json"
STREAM_IMAGE_FILE_NAME = "latest.jpg"
STREAM_FRAME_NAME_PATTERN = "frame_{:06d}.jpg"
RUN_LOG_FILE_NAME = "run.log"
RUN_REPORT_FILE_NAME = "run_report.json"
MAX_LOG_LINES = 400
JOBS_ROOT_REL = Path("output_fasterliveportrait/jobs")
JOBS_ROOT = PROJECT_ROOT / JOBS_ROOT_REL
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
ALLOWED_SOURCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
WARMUP_OUTPUT_ROOT_REL = Path("output_fasterliveportrait/warmup")
WARMUP_STREAM_SUBDIR_NAME = "stream"
WARMUP_INPUTS_SUBDIR_NAME = "inputs"
WARMUP_AUDIO_FILE_NAME = "warmup.wav"
WARMUP_AUDIO_DURATION_SEC = 0.8
WARMUP_START_DELAY_SEC = 0.75
WARMUP_ENABLED = os.getenv("ANIMATION_WARMUP_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


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


def resolve_source_frame(source_frame: str) -> tuple[Path, str]:
    """
    Resolve source frame from user input and return (absolute, runner_arg).
    """
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


def build_video_stream_command(input_fps: float) -> list[str]:
    """
    Build ffmpeg command that converts JPEG frames from stdin into fragmented MP4 for WebSocket transport.
    """
    safe_fps = max(1.0, float(input_fps))
    return [
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
        "-an",
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
        "-movflags",
        "+frag_keyframe+frag_every_frame+empty_moov+default_base_moof+omit_tfhd_offset",
        "-flush_packets",
        "1",
        "-f",
        "mp4",
        "pipe:1",
    ]


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
    audio_motion_stride: int
    log_rel: Path
    log_abs: Path
    started_at_ms: int | None = None
    process: subprocess.Popen | None = None
    log_handle: Any = None
    exit_code: int | None = None
    finished_at_ms: int | None = None

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


JOBS_LOCK = threading.Lock()
JOBS: dict[str, JobRecord] = {}
JOB_QUEUE_CONDITION = threading.Condition()
JOB_QUEUE: deque[str] = deque()
JOB_WORKER_LOCK = threading.Lock()
JOB_WORKER_THREAD: threading.Thread | None = None
WARMUP_LOCK = threading.Lock()
WARMUP_LAST_STARTED_AT_MS = 0
WARMUP_RUNNING = False


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
    ]
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
    ]
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
        "audioMotionStride": job.audio_motion_stride,
        "sourceFrame": str(job.source_frame_abs),
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


def create_app() -> FastAPI:
    """
    Build FastAPI application.
    """
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="FasterLivePortrait Streaming API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_warmup() -> None:
        ensure_job_worker_started()
        if not WARMUP_ENABLED:
            return
        warmup_thread = threading.Thread(target=run_startup_warmup_once, daemon=True)
        warmup_thread.start()

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        with JOB_QUEUE_CONDITION:
            queue_depth = len(JOB_QUEUE)
        worker_alive = bool(JOB_WORKER_THREAD is not None and JOB_WORKER_THREAD.is_alive())
        warmup_source_frame_arg = resolve_warmup_source_frame_arg()
        return {
            "status": "ok",
            "backend": DEFAULT_BACKEND,
            "trtRuntime": DEFAULT_TRT_RUNTIME,
            "trtPrecision": DEFAULT_TRT_PRECISION,
            "skipTrtEngineBuild": DEFAULT_SKIP_TRT_ENGINE_BUILD,
            "defaultAudioMotionStride": DEFAULT_AUDIO_MOTION_STRIDE,
            "warmupEnabled": WARMUP_ENABLED,
            "warmupRunning": WARMUP_RUNNING,
            "warmupLastStartedAtMs": WARMUP_LAST_STARTED_AT_MS,
            "warmupSourceFrame": warmup_source_frame_arg or "",
            "jobWorkerAlive": worker_alive,
            "jobQueueDepth": queue_depth,
        }

    @app.post("/api/warmup")
    async def warmup() -> JSONResponse:
        if not WARMUP_ENABLED:
            return JSONResponse({"status": "disabled"})
        warmup_thread = threading.Thread(target=run_startup_warmup_once, daemon=True)
        warmup_thread.start()
        return JSONResponse({"status": "started", "startedAtMs": now_ms()})

    @app.post("/api/generate")
    async def generate(
        audio: UploadFile = File(...),
        source_frame: str = Form(DEFAULT_SOURCE_FRAME),
        mode: str = Form(DEFAULT_MODE),
        motion_stride: int = Form(DEFAULT_AUDIO_MOTION_STRIDE),
    ) -> JSONResponse:
        ensure_job_worker_started()
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
        source_frame_abs, source_frame_arg = resolve_source_frame(source_frame)

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
        await save_upload_file(audio, input_abs)

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
            audio_motion_stride=int(motion_stride),
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

    @app.websocket("/ws/jobs/{job_id}/video")
    async def job_video_ws(websocket: WebSocket, job_id: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        ffmpeg_process = await asyncio.create_subprocess_exec(
            *build_video_stream_command(VIDEO_STREAM_INTERPOLATION_TARGET_FPS),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert ffmpeg_process.stdin is not None
        assert ffmpeg_process.stdout is not None
        assert ffmpeg_process.stderr is not None

        next_frame_index = 1
        last_known_frame_index = 0
        last_known_frame_total = 0
        estimated_generation_fps = VIDEO_STREAM_INPUT_FPS
        frame_interval_sec = 1.0 / max(1.0, VIDEO_STREAM_INTERPOLATION_TARGET_FPS)
        next_emit_at = time.perf_counter()
        stream_started_at = time.perf_counter()
        pending_raw_frames: deque[bytes] = deque()
        pending_stream_frames: deque[bytes] = deque()
        previous_raw_frame_bytes: bytes | None = None
        stable_loops = 0

        async def pump_mp4_stdout() -> None:
            """
            Send encoded fragmented MP4 chunks to WebSocket.
            """
            while True:
                chunk = await ffmpeg_process.stdout.read(VIDEO_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                await websocket.send_bytes(chunk)

        async def drain_stderr() -> None:
            """
            Drain ffmpeg stderr to avoid pipe backpressure.
            """
            while True:
                chunk = await ffmpeg_process.stderr.read(VIDEO_STREAM_CHUNK_SIZE)
                if not chunk:
                    break

        stdout_task = asyncio.create_task(pump_mp4_stdout())
        stderr_task = asyncio.create_task(drain_stderr())

        try:
            while True:
                stream_status = read_json(job.status_abs)
                status_frame_index = parse_status_int(stream_status, "frameIndex")
                status_frame_total = parse_status_int(stream_status, "frameTotal")
                estimated_generation_fps = estimate_generation_fps(stream_status, estimated_generation_fps)
                if status_frame_index > last_known_frame_index:
                    last_known_frame_index = status_frame_index
                if status_frame_total > last_known_frame_total:
                    last_known_frame_total = status_frame_total

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

                interpolation_steps = resolve_interpolation_steps(estimated_generation_fps)
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
                if pending_stream_frames and now_perf >= next_emit_at:
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
    )


if __name__ == "__main__":
    main()
