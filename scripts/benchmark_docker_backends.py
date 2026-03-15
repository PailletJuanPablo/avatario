#!/usr/bin/env python3
"""
Run the same Dockerized avatar generation job against ONNX and TensorRT and compare timings.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_fasterliveportrait" / "backend_benchmark"
DEFAULT_BACKENDS = ("onnx", "trt")
DEFAULT_SOURCE_FRAME = "output/frames/frame_00061.png"
DEFAULT_SERVICE_NAME = "animation-api"
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8010
DEFAULT_JOB_MODE = "preview"
DEFAULT_MOTION_STRIDE = 2
DEFAULT_STARTUP_TIMEOUT_SEC = 900.0
DEFAULT_JOB_TIMEOUT_SEC = 7200.0
DEFAULT_HEALTH_POLL_INTERVAL_SEC = 2.0
DEFAULT_JOB_POLL_INTERVAL_SEC = 1.5
API_HEALTH_PATH = "/api/health"
API_ENQUEUE_PATH = "/api/avatar/enqueue"
API_JOB_STATUS_PATH_TEMPLATE = "/api/jobs/{job_id}/status"
API_JOB_REPORT_PATH_TEMPLATE = "/api/jobs/{job_id}/report"
AUTHORIZATION_HEADER_NAME = "Authorization"
AUTHORIZATION_SCHEME = "Bearer"
ENV_FILE_NAME = ".env"
ENV_API_PORT_KEY = "ANIMATION_API_PORT"
ENV_API_TOKEN_KEY = "ANIMATION_API_TOKEN"
ENV_BACKEND_KEY = "ANIMATION_BACKEND"
ENV_TRT_RUNTIME_KEY = "ANIMATION_TRT_RUNTIME"
ENV_TRT_PRECISION_KEY = "ANIMATION_TRT_PRECISION"
ENV_WARMUP_ENABLED_KEY = "ANIMATION_WARMUP_ENABLED"
JOB_STATE_DONE = "done"
JOB_STATE_ERROR = "error"
JOB_STATE_CANCELED = "canceled"
TERMINAL_JOB_STATES = {JOB_STATE_DONE, JOB_STATE_ERROR, JOB_STATE_CANCELED}


class BenchmarkError(RuntimeError):
    """Raised when one benchmark step fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Benchmark Dockerized ONNX and TRT avatar generation.")
    parser.add_argument("--audio", required=True, help="Path to the driving audio file.")
    parser.add_argument(
        "--source-image",
        default="",
        help="Optional path to a source image upload. Overrides --source-frame when provided.",
    )
    parser.add_argument(
        "--source-frame",
        default=DEFAULT_SOURCE_FRAME,
        help="Source frame path relative to the project root when no source image is uploaded.",
    )
    parser.add_argument("--mode", default=DEFAULT_JOB_MODE, choices=["preview", "full"])
    parser.add_argument("--motion-stride", type=int, default=DEFAULT_MOTION_STRIDE)
    parser.add_argument("--api-host", default=DEFAULT_API_HOST)
    parser.add_argument("--api-port", type=int, default=0, help="Defaults to ANIMATION_API_PORT or 8010.")
    parser.add_argument(
        "--token",
        default="",
        help="Defaults to ANIMATION_API_TOKEN from .env or current environment.",
    )
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE))
    parser.add_argument("--service", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_SEC)
    parser.add_argument("--job-timeout", type=float, default=DEFAULT_JOB_TIMEOUT_SEC)
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=sorted(DEFAULT_BACKENDS),
        help="Benchmark only specific backend(s). Repeat to select more than one.",
    )
    parser.add_argument("--trt-runtime", default="")
    parser.add_argument("--trt-precision", default="")
    parser.add_argument("--warmup-enabled", default="")
    parser.add_argument("--build", action="store_true", help="Build the Docker image before running benchmarks.")
    parser.add_argument(
        "--keep-last-container",
        action="store_true",
        help="Leave the last benchmarked backend container running.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional explicit output path for the benchmark JSON summary.",
    )
    return parser.parse_args()


def read_env_file(path: Path) -> dict[str, str]:
    """Read a simple KEY=VALUE .env file."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_value(cli_value: str, env_values: dict[str, str], key: str, fallback: str) -> str:
    """Resolve one runtime value from CLI, environment, .env, or fallback."""
    normalized_cli_value = str(cli_value or "").strip()
    if normalized_cli_value:
        return normalized_cli_value
    normalized_env_value = str(os.environ.get(key, "")).strip()
    if normalized_env_value:
        return normalized_env_value
    normalized_file_value = str(env_values.get(key, "")).strip()
    if normalized_file_value:
        return normalized_file_value
    return fallback


def resolve_api_port(args: argparse.Namespace, env_values: dict[str, str]) -> int:
    """Resolve API port from CLI, environment, .env, or default."""
    if int(args.api_port or 0) > 0:
        return int(args.api_port)
    raw_port = resolve_value("", env_values, ENV_API_PORT_KEY, str(DEFAULT_API_PORT))
    return max(1, int(raw_port))


def ensure_file_exists(path: Path, label: str) -> Path:
    """Validate that one required file exists."""
    resolved_path = path.resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise BenchmarkError(f"{label} not found: {resolved_path}")
    return resolved_path


def resolve_host_path(path_value: str, project_root: Path) -> Path:
    """Resolve one host path, relative to the project root when needed."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def build_api_base_url(host: str, port: int) -> str:
    """Build API base URL for local HTTP calls."""
    return f"http://{host}:{port}"


def build_auth_headers(token: str) -> dict[str, str]:
    """Build optional Authorization header payload."""
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return {}
    return {AUTHORIZATION_HEADER_NAME: f"{AUTHORIZATION_SCHEME} {normalized_token}"}


def api_request(
    method: str,
    url: str,
    token: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Send one JSON API request and decode the response."""
    request_headers = build_auth_headers(token)
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url=url, method=method.upper(), data=body, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = str(response.headers.get("Content-Type", ""))
            payload = response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {exc.code} for {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"Request failed for {url}: {exc}") from exc
    except OSError as exc:
        raise BenchmarkError(f"Socket request failed for {url}: {exc}") from exc

    if "application/json" not in content_type:
        return payload
    return json.loads(payload.decode("utf-8"))


def build_multipart_form_data(
    fields: dict[str, str],
    file_fields: dict[str, tuple[Path, str]],
) -> tuple[bytes, str]:
    """Build a multipart/form-data body using only the Python standard library."""
    boundary = f"----animation-benchmark-{uuid.uuid4().hex}"
    payload = bytearray()
    line_break = b"\r\n"

    for field_name, field_value in fields.items():
        payload.extend(f"--{boundary}".encode("utf-8"))
        payload.extend(line_break)
        payload.extend(f'Content-Disposition: form-data; name="{field_name}"'.encode("utf-8"))
        payload.extend(line_break)
        payload.extend(line_break)
        payload.extend(str(field_value).encode("utf-8"))
        payload.extend(line_break)

    for field_name, (file_path, content_type) in file_fields.items():
        payload.extend(f"--{boundary}".encode("utf-8"))
        payload.extend(line_break)
        payload.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"'.encode("utf-8")
        )
        payload.extend(line_break)
        payload.extend(f"Content-Type: {content_type}".encode("utf-8"))
        payload.extend(line_break)
        payload.extend(line_break)
        payload.extend(file_path.read_bytes())
        payload.extend(line_break)

    payload.extend(f"--{boundary}--".encode("utf-8"))
    payload.extend(line_break)
    return bytes(payload), f"multipart/form-data; boundary={boundary}"


def compose_command(*parts: str) -> list[str]:
    """Build one docker compose command list."""
    return ["docker", "compose", *parts]


def run_compose(project_root: Path, env: dict[str, str], *parts: str) -> None:
    """Run one docker compose command."""
    command = compose_command(*parts)
    subprocess.run(command, cwd=str(project_root), env=env, check=True)


def wait_for_backend_ready(
    base_url: str,
    token: str,
    timeout_sec: float,
    poll_interval_sec: float,
) -> tuple[dict[str, Any], float, float]:
    """
    Wait until the API is healthy and warmup is completed when enabled.

    Returns the final health payload, time-to-health, and warmup wait time.
    """
    started_at = time.perf_counter()
    health_ready_at: float | None = None
    deadline = started_at + timeout_sec
    last_error = ""

    while time.perf_counter() < deadline:
        try:
            payload = api_request("GET", f"{base_url}{API_HEALTH_PATH}", token)
        except BenchmarkError as exc:
            last_error = str(exc)
            time.sleep(poll_interval_sec)
            continue

        if str(payload.get("status", "")).lower() != "ok":
            last_error = f"Unexpected health status: {payload}"
            time.sleep(poll_interval_sec)
            continue

        if health_ready_at is None:
            health_ready_at = time.perf_counter()

        warmup_enabled = bool(payload.get("warmupEnabled"))
        warmup_running = bool(payload.get("warmupRunning"))
        if warmup_enabled and warmup_running:
            time.sleep(poll_interval_sec)
            continue

        ready_at = time.perf_counter()
        health_ready_seconds = (health_ready_at or ready_at) - started_at
        warmup_wait_seconds = ready_at - (health_ready_at or started_at)
        return payload, round(health_ready_seconds, 3), round(warmup_wait_seconds, 3)

    raise BenchmarkError(f"Backend did not become ready within {timeout_sec:.1f}s. Last error: {last_error}")


def enqueue_job(
    base_url: str,
    token: str,
    audio_path: Path,
    source_image_path: Path | None,
    source_frame: str,
    mode: str,
    motion_stride: int,
) -> tuple[dict[str, Any], float]:
    """Submit one avatar generation job and return the API payload plus round-trip seconds."""
    fields = {
        "source_frame": source_frame,
        "mode": mode,
        "motion_stride": str(int(motion_stride)),
    }
    file_fields: dict[str, tuple[Path, str]] = {
        "audio": (audio_path, mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream")
    }
    if source_image_path is not None:
        file_fields["source_image"] = (
            source_image_path,
            mimetypes.guess_type(source_image_path.name)[0] or "application/octet-stream",
        )
    request_body, content_type = build_multipart_form_data(fields, file_fields)
    started_at = time.perf_counter()
    payload = api_request(
        "POST",
        f"{base_url}{API_ENQUEUE_PATH}",
        token,
        body=request_body,
        headers={"Content-Type": content_type},
    )
    return payload, round(time.perf_counter() - started_at, 3)


def wait_for_job_completion(
    base_url: str,
    token: str,
    job_id: str,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    """Poll one job until it reaches a terminal state."""
    deadline = time.perf_counter() + timeout_sec
    status_url = f"{base_url}{API_JOB_STATUS_PATH_TEMPLATE.format(job_id=urllib.parse.quote(job_id))}"
    last_payload: dict[str, Any] | None = None

    while time.perf_counter() < deadline:
        payload = api_request("GET", status_url, token)
        last_payload = payload
        state = str(payload.get("state", "")).strip().lower()
        if state in TERMINAL_JOB_STATES:
            return payload
        time.sleep(poll_interval_sec)

    raise BenchmarkError(f"Job {job_id} did not finish within {timeout_sec:.1f}s. Last payload: {last_payload}")


def fetch_job_report(base_url: str, token: str, job_id: str) -> dict[str, Any]:
    """Fetch one completed job report."""
    report_url = f"{base_url}{API_JOB_REPORT_PATH_TEMPLATE.format(job_id=urllib.parse.quote(job_id))}"
    return api_request("GET", report_url, token)


def safe_seconds_from_ms(end_ms: Any, start_ms: Any) -> float | None:
    """Convert two millisecond timestamps into seconds when both are present."""
    if end_ms is None or start_ms is None:
        return None
    try:
        delta_ms = int(end_ms) - int(start_ms)
    except (TypeError, ValueError):
        return None
    return round(delta_ms / 1000.0, 3)


def safe_float(value: Any) -> float | None:
    """Parse one optional float value."""
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def summarise_backend_run(
    backend: str,
    health_payload: dict[str, Any],
    enqueue_payload: dict[str, Any],
    final_status_payload: dict[str, Any],
    report_payload: dict[str, Any],
    health_ready_seconds: float,
    warmup_wait_seconds: float,
    api_ack_seconds: float,
) -> dict[str, Any]:
    """Build one normalized backend benchmark summary."""
    created_at_ms = enqueue_payload.get("createdAtMs")
    started_at_ms = final_status_payload.get("startedAtMs")
    finished_at_ms = final_status_payload.get("finishedAtMs")
    queue_wait_seconds = safe_seconds_from_ms(started_at_ms, created_at_ms)
    worker_runtime_seconds = safe_seconds_from_ms(finished_at_ms, started_at_ms)
    job_wall_seconds = safe_seconds_from_ms(finished_at_ms, created_at_ms)
    summary = {
        "backend": backend,
        "jobId": str(enqueue_payload.get("jobId", "")),
        "jobState": str(final_status_payload.get("state", "")),
        "healthReadySeconds": health_ready_seconds,
        "warmupWaitSeconds": warmup_wait_seconds,
        "apiAckSeconds": api_ack_seconds,
        "queueWaitSeconds": queue_wait_seconds,
        "workerRuntimeSeconds": worker_runtime_seconds,
        "jobWallSeconds": job_wall_seconds,
        "runnerElapsedSeconds": safe_float(report_payload.get("elapsedSeconds")),
        "phaseTimingsSeconds": report_payload.get("phaseTimingsSeconds", {}),
        "queuePosition": enqueue_payload.get("queuePosition"),
        "defaultRenderBatchSize": health_payload.get("defaultRenderBatchSize"),
        "defaultTrtEngineBatchSize": health_payload.get("defaultTrtEngineBatchSize"),
        "runtimeConfig": report_payload.get("runtimeConfig", {}),
        "report": report_payload,
    }
    return summary


def format_seconds(value: float | None) -> str:
    """Format one optional second value for console output."""
    if value is None:
        return "-"
    return f"{value:.3f}"


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a compact benchmark table."""
    headers = (
        "backend",
        "health",
        "warmup",
        "api_ack",
        "queue",
        "worker",
        "runner",
        "job_wall",
    )
    rows = [
        (
            item["backend"],
            format_seconds(item.get("healthReadySeconds")),
            format_seconds(item.get("warmupWaitSeconds")),
            format_seconds(item.get("apiAckSeconds")),
            format_seconds(item.get("queueWaitSeconds")),
            format_seconds(item.get("workerRuntimeSeconds")),
            format_seconds(item.get("runnerElapsedSeconds")),
            format_seconds(item.get("jobWallSeconds")),
        )
        for item in results
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    def format_row(values: tuple[Any, ...]) -> str:
        return " | ".join(str(value).ljust(width) for value, width in zip(values, widths))

    print()
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))

    by_backend = {item["backend"]: item for item in results}
    onnx_summary = by_backend.get("onnx")
    trt_summary = by_backend.get("trt")
    if onnx_summary is None or trt_summary is None:
        return
    onnx_runner = onnx_summary.get("runnerElapsedSeconds")
    trt_runner = trt_summary.get("runnerElapsedSeconds")
    if onnx_runner is None or trt_runner is None or float(trt_runner) <= 0:
        return
    speedup = round(float(onnx_runner) / float(trt_runner), 3)
    print()
    print(f"TRT speedup vs ONNX (runnerElapsedSeconds): {speedup}x")


def build_output_path(requested_path: str) -> Path:
    """Resolve final benchmark JSON output path."""
    if requested_path:
        return Path(requested_path).resolve()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_DIR / f"docker_backend_benchmark_{timestamp}.json").resolve()


def build_compose_environment(
    base_env: dict[str, str],
    backend: str,
    args: argparse.Namespace,
    token: str,
    api_port: int,
) -> dict[str, str]:
    """Build docker compose environment overrides for one backend run."""
    env = dict(base_env)
    env[ENV_BACKEND_KEY] = backend
    env[ENV_API_PORT_KEY] = str(api_port)
    if token:
        env[ENV_API_TOKEN_KEY] = token
    if args.trt_runtime:
        env[ENV_TRT_RUNTIME_KEY] = str(args.trt_runtime).strip()
    if args.trt_precision:
        env[ENV_TRT_PRECISION_KEY] = str(args.trt_precision).strip()
    if args.warmup_enabled:
        env[ENV_WARMUP_ENABLED_KEY] = str(args.warmup_enabled).strip()
    return env


def benchmark_backend(
    backend: str,
    args: argparse.Namespace,
    project_root: Path,
    compose_file: Path,
    base_url: str,
    token: str,
    audio_path: Path,
    source_image_path: Path | None,
    base_env: dict[str, str],
    api_port: int,
) -> dict[str, Any]:
    """Benchmark one backend by starting Docker, running one job, and collecting metrics."""
    compose_env = build_compose_environment(base_env, backend, args, token, api_port)
    print()
    print(f"[benchmark] starting backend={backend}")
    run_compose(project_root, compose_env, "-f", str(compose_file), "down", "--remove-orphans")
    try:
        up_command = ["-f", str(compose_file), "up"]
        up_command.extend(["-d", args.service])
        run_compose(project_root, compose_env, *up_command)
        health_payload, health_ready_seconds, warmup_wait_seconds = wait_for_backend_ready(
            base_url=base_url,
            token=token,
            timeout_sec=float(args.startup_timeout),
            poll_interval_sec=DEFAULT_HEALTH_POLL_INTERVAL_SEC,
        )
        enqueue_payload, api_ack_seconds = enqueue_job(
            base_url=base_url,
            token=token,
            audio_path=audio_path,
            source_image_path=source_image_path,
            source_frame=str(args.source_frame),
            mode=str(args.mode),
            motion_stride=int(args.motion_stride),
        )
        job_id = str(enqueue_payload.get("jobId", "")).strip()
        if not job_id:
            raise BenchmarkError(f"Backend {backend} did not return a jobId: {enqueue_payload}")
        final_status_payload = wait_for_job_completion(
            base_url=base_url,
            token=token,
            job_id=job_id,
            timeout_sec=float(args.job_timeout),
            poll_interval_sec=DEFAULT_JOB_POLL_INTERVAL_SEC,
        )
        final_state = str(final_status_payload.get("state", "")).strip().lower()
        if final_state != JOB_STATE_DONE:
            raise BenchmarkError(f"Backend {backend} finished with state '{final_state}': {final_status_payload}")
        report_payload = fetch_job_report(base_url=base_url, token=token, job_id=job_id)
        return summarise_backend_run(
            backend=backend,
            health_payload=health_payload,
            enqueue_payload=enqueue_payload,
            final_status_payload=final_status_payload,
            report_payload=report_payload,
            health_ready_seconds=health_ready_seconds,
            warmup_wait_seconds=warmup_wait_seconds,
            api_ack_seconds=api_ack_seconds,
        )
    finally:
        if not args.keep_last_container or backend != (args.backends or list(DEFAULT_BACKENDS))[-1]:
            run_compose(project_root, compose_env, "-f", str(compose_file), "down", "--remove-orphans")


def main() -> None:
    """Execute Docker backend benchmarks."""
    args = parse_args()
    project_root = PROJECT_ROOT
    env_values = read_env_file(project_root / ENV_FILE_NAME)
    compose_file = ensure_file_exists(resolve_host_path(str(args.compose_file), project_root), "Docker compose file")
    audio_path = ensure_file_exists(resolve_host_path(str(args.audio), project_root), "Driving audio")
    source_image_path = (
        ensure_file_exists(resolve_host_path(str(args.source_image), project_root), "Source image")
        if args.source_image
        else None
    )
    if source_image_path is None:
        source_frame_path = Path(str(args.source_frame))
        if source_frame_path.is_absolute():
            raise BenchmarkError("--source-frame must be relative to the project root when using Docker.")
        ensure_file_exists(project_root / source_frame_path, "Source frame")
    token = resolve_value(args.token, env_values, ENV_API_TOKEN_KEY, "")
    api_port = resolve_api_port(args, env_values)
    base_url = build_api_base_url(str(args.api_host), api_port)
    selected_backends = list(args.backends or DEFAULT_BACKENDS)
    base_env = dict(os.environ)

    if args.build:
        run_compose(project_root, base_env, "-f", str(compose_file), "build", args.service)

    results: list[dict[str, Any]] = []
    for backend in selected_backends:
        result = benchmark_backend(
            backend=backend,
            args=args,
            project_root=project_root,
            compose_file=compose_file,
            base_url=base_url,
            token=token,
            audio_path=audio_path,
            source_image_path=source_image_path,
            base_env=base_env,
            api_port=api_port,
        )
        results.append(result)

    output_path = build_output_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "apiBaseUrl": base_url,
        "composeFile": str(compose_file),
        "audio": str(audio_path),
        "sourceImage": str(source_image_path) if source_image_path is not None else "",
        "sourceFrame": str(args.source_frame),
        "mode": str(args.mode),
        "motionStride": int(args.motion_stride),
        "backends": selected_backends,
        "results": results,
    }
    output_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print_summary_table(results)
    print()
    print(f"[ok] benchmark summary -> {output_path}")


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[error] command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        sys.exit(exc.returncode or 1)
