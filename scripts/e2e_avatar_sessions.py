from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus


API_TOKEN_QUERY_KEY = "token"
AVATAR_SESSION_HEADER_NAME = "X-Avatar-Session-Id"
AVATAR_SESSION_QUERY_KEY = "sessionId"
DEFAULT_BASE_URL = "http://127.0.0.1:8010"
DEFAULT_IDLE_CHUNKS = 2
DEFAULT_IDLE_TIMEOUT_SEC = 20.0
DEFAULT_TALKING_START_TIMEOUT_SEC = 300.0
DEFAULT_ISOLATION_OBSERVATION_SEC = 12.0
DEFAULT_COMPLETION_TIMEOUT_SEC = 180.0
DEFAULT_HTTP_TIMEOUT_SEC = 30.0
DEFAULT_RUNTIME_IDLE_TIMEOUT_SEC = 300.0
POLL_INTERVAL_SEC = 0.5
WEBSOCKET_CONNECT_KWARGS = {
    "max_size": None,
    "ping_interval": None,
    "ping_timeout": None,
}


class E2EFailure(RuntimeError):
    """Raised when the real end-to-end validation fails."""


@dataclass(frozen=True)
class AvatarClient:
    name: str
    session_id: str


def log(message: str) -> None:
    print(f"[e2e] {message}", flush=True)


def create_short_e2e_audio_fixture() -> Path:
    sample_rate_hz = 16000
    duration_sec = 0.6
    total_samples = int(sample_rate_hz * duration_sec)
    frequency_hz = 220.0
    amplitude = 12000
    temp_file = NamedTemporaryFile(prefix="avatar_e2e_", suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    with wave.open(str(temp_path), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate_hz)
        for sample_index in range(total_samples):
            sample_value = int(amplitude * math.sin((2.0 * math.pi * frequency_hz * sample_index) / sample_rate_hz))
            wave_file.writeframesraw(sample_value.to_bytes(2, byteorder="little", signed=True))
    return temp_path


def normalize_base_url(raw_value: str) -> str:
    normalized_value = str(raw_value or "").strip().rstrip("/")
    if not normalized_value:
        raise E2EFailure("Base URL is required.")
    return normalized_value


def build_http_headers(session_id: str | None, api_token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    if session_id:
        headers[AVATAR_SESSION_HEADER_NAME] = session_id
    return headers


def build_ws_url(base_url: str, path: str, session_id: str, api_token: str | None) -> str:
    target_url = urljoin(f"{base_url}/", path.lstrip("/"))
    parsed_url = urlparse(target_url)
    query_params = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
    query_params[AVATAR_SESSION_QUERY_KEY] = session_id
    if api_token:
        query_params[API_TOKEN_QUERY_KEY] = api_token
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    return urlunparse(
        (
            ws_scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            urlencode(query_params),
            parsed_url.fragment,
        )
    )


def build_api_url(base_url: str, path: str) -> str:
    return urljoin(f"{base_url}/", path.lstrip("/"))


def http_get_json(
    session: requests.Session,
    base_url: str,
    path: str,
    *,
    session_id: str | None,
    api_token: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    response = session.get(
        build_api_url(base_url, path),
        headers=build_http_headers(session_id, api_token),
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return dict(response.json())


def enqueue_audio_job(
    session: requests.Session,
    base_url: str,
    client: AvatarClient,
    audio_path: Path,
    api_token: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    mime_type = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"
    with audio_path.open("rb") as audio_handle:
        response = session.post(
            build_api_url(base_url, "/api/avatar/enqueue"),
            headers=build_http_headers(client.session_id, api_token),
            files={"audio": (audio_path.name, audio_handle, mime_type)},
            data={},
            timeout=timeout_sec,
        )
    response.raise_for_status()
    payload = dict(response.json())
    if str(payload.get("avatarSessionId") or "") != client.session_id:
        raise E2EFailure(
            f"{client.name} enqueue returned avatarSessionId={payload.get('avatarSessionId')!r}, expected {client.session_id!r}"
        )
    return payload


def wait_for_http_condition(
    session: requests.Session,
    base_url: str,
    path: str,
    *,
    session_id: str | None,
    api_token: str | None,
    timeout_sec: float,
    description: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_payload = http_get_json(
            session,
            base_url,
            path,
            session_id=session_id,
            api_token=api_token,
            timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
        )
        if predicate(last_payload):
            return last_payload
        time.sleep(POLL_INTERVAL_SEC)
    raise E2EFailure(f"Timed out waiting for {description}. Last payload: {json.dumps(last_payload or {}, ensure_ascii=True)}")


async def receive_status_payload(
    websocket: websockets.WebSocketClientProtocol,
    *,
    predicate: Callable[[dict[str, Any]], bool],
    timeout_sec: float,
    description: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        raw_message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        if isinstance(raw_message, bytes):
            continue
        message = json.loads(raw_message)
        if str(message.get("type") or "") != "status":
            continue
        last_payload = dict(message.get("payload") or {})
        if predicate(last_payload):
            return last_payload
    raise E2EFailure(
        f"Timed out waiting for websocket status {description}. Last payload: {json.dumps(last_payload or {}, ensure_ascii=True)}"
    )


def assert_no_http_status_transition(
    session: requests.Session,
    base_url: str,
    *,
    client: AvatarClient,
    api_token: str | None,
    forbidden_job_id: str,
    observation_sec: float,
) -> None:
    deadline = time.monotonic() + observation_sec
    while time.monotonic() < deadline:
        payload = http_get_json(
            session,
            base_url,
            "/api/avatar/status",
            session_id=client.session_id,
            api_token=api_token,
            timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
        )
        if str(payload.get("currentJobId") or "") == forbidden_job_id or str(payload.get("mode") or "") == "talking":
            raise E2EFailure(
                f"{client.name} unexpectedly received talking status for foreign job {forbidden_job_id}: "
                f"{json.dumps(payload, ensure_ascii=True)}"
            )
        time.sleep(POLL_INTERVAL_SEC)


async def assert_binary_stream_active(
    ws_url: str,
    *,
    chunk_count: int,
    timeout_sec: float,
    label: str,
) -> int:
    total_bytes = 0
    async with websockets.connect(ws_url, **WEBSOCKET_CONNECT_KWARGS) as websocket:
        deadline = time.monotonic() + timeout_sec
        received_chunks = 0
        while received_chunks < chunk_count and time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            if not isinstance(message, bytes):
                continue
            total_bytes += len(message)
            received_chunks += 1
    if total_bytes <= 0:
        raise E2EFailure(f"{label} did not emit binary payloads.")
    return total_bytes


async def open_status_socket(
    base_url: str,
    client: AvatarClient,
    api_token: str | None,
) -> websockets.WebSocketClientProtocol:
    status_ws_url = build_ws_url(base_url, "/ws/avatar", client.session_id, api_token)
    return await websockets.connect(status_ws_url, **WEBSOCKET_CONNECT_KWARGS)


async def run_e2e(args: argparse.Namespace) -> None:
    base_url = normalize_base_url(args.base_url)
    audio_path = Path(args.audio_file).resolve() if args.audio_file else create_short_e2e_audio_fixture()
    generated_audio_fixture = not bool(args.audio_file)
    if not audio_path.exists():
        raise E2EFailure(f"Audio file not found: {audio_path}")

    client_a = AvatarClient(name="client-a", session_id=f"avatar_e2e_a_{uuid.uuid4().hex[:16]}")
    client_b = AvatarClient(name="client-b", session_id=f"avatar_e2e_b_{uuid.uuid4().hex[:16]}")
    log(f"base_url={base_url}")
    log(f"audio_file={audio_path}")
    log(f"{client_a.name}.session_id={client_a.session_id}")
    log(f"{client_b.name}.session_id={client_b.session_id}")

    http_session = requests.Session()
    try:
        public_health = http_get_json(
            http_session,
            base_url,
            "/api/health",
            session_id=None,
            api_token=args.api_token,
            timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
        )
        if str(public_health.get("status") or "") != "ok":
            raise E2EFailure(f"/api/health returned unexpected status: {json.dumps(public_health, ensure_ascii=True)}")
        log("public health endpoint responded")

        wait_for_http_condition(
            http_session,
            base_url,
            "/api/health",
            session_id=None,
            api_token=args.api_token,
            timeout_sec=args.runtime_idle_timeout_sec,
            description="runtime idle state before starting the e2e",
            predicate=lambda payload: not str(payload.get("runningJobId") or "") and int(payload.get("jobQueueDepth") or 0) == 0,
        )
        log("runtime is idle before starting the e2e")

        health_a = http_get_json(
            http_session,
            base_url,
            "/api/health",
            session_id=client_a.session_id,
            api_token=args.api_token,
            timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
        )
        health_b = http_get_json(
            http_session,
            base_url,
            "/api/health",
            session_id=client_b.session_id,
            api_token=args.api_token,
            timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
        )
        if str(health_a.get("avatarSessionId") or "") != client_a.session_id:
            raise E2EFailure(f"{client_a.name} /api/health did not echo its session id.")
        if str(health_b.get("avatarSessionId") or "") != client_b.session_id:
            raise E2EFailure(f"{client_b.name} /api/health did not echo its session id.")
        if client_a.session_id == client_b.session_id:
            raise E2EFailure("Generated session ids collided.")
        log("session-scoped health payloads are distinct")

        async with await open_status_socket(base_url, client_a, args.api_token) as status_ws_a, await open_status_socket(
            base_url, client_b, args.api_token
        ) as status_ws_b:
            initial_status_a = await receive_status_payload(
                status_ws_a,
                predicate=lambda payload: str(payload.get("avatarSessionId") or "") == client_a.session_id,
                timeout_sec=args.talking_start_timeout_sec,
                description=f"initial status for {client_a.name}",
            )
            initial_status_b = await receive_status_payload(
                status_ws_b,
                predicate=lambda payload: str(payload.get("avatarSessionId") or "") == client_b.session_id,
                timeout_sec=args.talking_start_timeout_sec,
                description=f"initial status for {client_b.name}",
            )
            if str(initial_status_a.get("mode") or "") != "idle":
                raise E2EFailure(f"{client_a.name} initial mode is not idle: {json.dumps(initial_status_a, ensure_ascii=True)}")
            if str(initial_status_b.get("mode") or "") != "idle":
                raise E2EFailure(f"{client_b.name} initial mode is not idle: {json.dumps(initial_status_b, ensure_ascii=True)}")
            log("status websocket is alive for both clients")
            await status_ws_a.close()
            await status_ws_b.close()

            avatar_video_ws_url_a = build_ws_url(base_url, "/ws/avatar/video", client_a.session_id, args.api_token)
            avatar_video_ws_url_b = build_ws_url(base_url, "/ws/avatar/video", client_b.session_id, args.api_token)
            idle_bytes_a, idle_bytes_b = await asyncio.gather(
                assert_binary_stream_active(
                    avatar_video_ws_url_a,
                    chunk_count=args.idle_chunk_count,
                    timeout_sec=args.idle_timeout_sec,
                    label=f"{client_a.name} idle avatar stream",
                ),
                assert_binary_stream_active(
                    avatar_video_ws_url_b,
                    chunk_count=args.idle_chunk_count,
                    timeout_sec=args.idle_timeout_sec,
                    label=f"{client_b.name} idle avatar stream",
                ),
            )
            log(f"idle avatar video emitted binary chunks: {client_a.name}={idle_bytes_a} bytes {client_b.name}={idle_bytes_b} bytes")

            enqueue_payload = enqueue_audio_job(
                http_session,
                base_url,
                client_a,
                audio_path,
                args.api_token,
                timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
            )
            job_id = str(enqueue_payload.get("jobId") or "")
            if not job_id:
                raise E2EFailure(f"{client_a.name} enqueue response did not include jobId: {json.dumps(enqueue_payload, ensure_ascii=True)}")
            log(f"enqueued job {job_id} for {client_a.name}")

            talking_status_a = wait_for_http_condition(
                http_session,
                base_url,
                "/api/avatar/status",
                session_id=client_a.session_id,
                api_token=args.api_token,
                timeout_sec=args.talking_start_timeout_sec,
                description=f"{client_a.name} talking transition",
                predicate=lambda payload: str(payload.get("currentJobId") or "") == job_id and str(payload.get("mode") or "") == "talking",
            )
            log(f"{client_a.name} entered talking with job {job_id}")

            status_b_during_a = http_get_json(
                http_session,
                base_url,
                "/api/avatar/status",
                session_id=client_b.session_id,
                api_token=args.api_token,
                timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
            )
            if str(status_b_during_a.get("mode") or "") != "idle" or str(status_b_during_a.get("currentJobId") or ""):
                raise E2EFailure(
                    f"{client_b.name} left idle after {client_a.name} enqueue: {json.dumps(status_b_during_a, ensure_ascii=True)}"
                )
            assert_no_http_status_transition(
                http_session,
                base_url,
                client=client_b,
                api_token=args.api_token,
                forbidden_job_id=job_id,
                observation_sec=args.isolation_observation_sec,
            )
            log(f"{client_b.name} stayed isolated while {client_a.name} was talking")

            own_job_payload = http_get_json(
                http_session,
                base_url,
                f"/api/jobs/{job_id}/status",
                session_id=client_a.session_id,
                api_token=args.api_token,
                timeout_sec=DEFAULT_HTTP_TIMEOUT_SEC,
            )
            if str(own_job_payload.get("jobId") or "") != job_id:
                raise E2EFailure(f"{client_a.name} could not read its own job payload.")
            foreign_job_response = http_session.get(
                build_api_url(base_url, f"/api/jobs/{job_id}/status"),
                headers=build_http_headers(client_b.session_id, args.api_token),
                timeout=DEFAULT_HTTP_TIMEOUT_SEC,
            )
            if foreign_job_response.status_code != 404:
                raise E2EFailure(
                    f"{client_b.name} unexpectedly accessed foreign job {job_id}: HTTP {foreign_job_response.status_code} {foreign_job_response.text}"
                )
            log("job access control is enforced")

            audio_duration_sec = float(enqueue_payload.get("audioDurationSec") or talking_status_a.get("currentJobAudioDurationSec") or 0.0)
            completion_timeout_sec = max(args.completion_timeout_sec, audio_duration_sec + 30.0)
            completion_payload = wait_for_http_condition(
                http_session,
                base_url,
                f"/api/jobs/{job_id}/status",
                session_id=client_a.session_id,
                api_token=args.api_token,
                timeout_sec=completion_timeout_sec,
                description=f"job completion for {job_id}",
                predicate=lambda payload: str(payload.get("state") or "") in {"done", "error", "canceled"},
            )
            final_state = str(completion_payload.get("state") or "")
            if final_state != "done":
                raise E2EFailure(f"Job {job_id} did not finish successfully: {json.dumps(completion_payload, ensure_ascii=True)}")
            log(f"job {job_id} finished successfully")

            wait_for_http_condition(
                http_session,
                base_url,
                "/api/avatar/status",
                session_id=client_a.session_id,
                api_token=args.api_token,
                timeout_sec=completion_timeout_sec,
                description=f"{client_a.name} idle return",
                predicate=lambda payload: str(payload.get("mode") or "") == "idle" and not str(payload.get("currentJobId") or ""),
            )
            log(f"{client_a.name} returned to idle after job {job_id}")

            idle_bytes_after_completion = await assert_binary_stream_active(
                avatar_video_ws_url_a,
                chunk_count=args.idle_chunk_count,
                timeout_sec=args.idle_timeout_sec,
                label=f"{client_a.name} avatar stream after completion",
            )
            log(f"{client_a.name} avatar stream stayed alive after completion: {idle_bytes_after_completion} bytes")
    finally:
        http_session.close()
        if generated_audio_fixture:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real end-to-end validation of per-session avatar isolation using HTTP and WebSocket flows."
    )
    parser.add_argument("--base-url", default=os.environ.get("E2E_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-token", default=os.environ.get("E2E_API_TOKEN", "").strip() or None)
    parser.add_argument("--audio-file", default=os.environ.get("E2E_AUDIO_FILE", "").strip() or None)
    parser.add_argument("--idle-chunk-count", type=int, default=DEFAULT_IDLE_CHUNKS)
    parser.add_argument("--idle-timeout-sec", type=float, default=DEFAULT_IDLE_TIMEOUT_SEC)
    parser.add_argument("--talking-start-timeout-sec", type=float, default=DEFAULT_TALKING_START_TIMEOUT_SEC)
    parser.add_argument("--isolation-observation-sec", type=float, default=DEFAULT_ISOLATION_OBSERVATION_SEC)
    parser.add_argument("--completion-timeout-sec", type=float, default=DEFAULT_COMPLETION_TIMEOUT_SEC)
    parser.add_argument("--runtime-idle-timeout-sec", type=float, default=DEFAULT_RUNTIME_IDLE_TIMEOUT_SEC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run_e2e(args))
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    except (E2EFailure, requests.RequestException, ConnectionClosed, InvalidStatus, TimeoutError, asyncio.TimeoutError) as exc:
        log(f"FAILED: {exc}")
        return 1
    except Exception as exc:
        log(f"UNEXPECTED FAILURE: {exc}")
        return 1
    log("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
