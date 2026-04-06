"""
Smoke-test the local avatar HTTP flow against a real running API instance.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test avatar HTTP E2E flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--api-token", default=os.getenv("ANIMATION_API_TOKEN", "dev-token"))
    parser.add_argument("--source-image", default="")
    parser.add_argument("--source-video", default="")
    parser.add_argument("--source-frame", default="")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--template-name", default=f"smoke-{int(time.time())}")
    parser.add_argument("--timeout-sec", type=float, default=420.0)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    return parser.parse_args()


def build_headers(api_token: str, avatar_session_id: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if str(api_token or "").strip():
        headers["Authorization"] = f"Bearer {api_token.strip()}"
    if str(avatar_session_id or "").strip():
        headers["X-Avatar-Session-Id"] = str(avatar_session_id).strip()
    return headers


def require_file(path_value: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def resolve_url(base_url: str, relative_or_absolute_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", str(relative_or_absolute_url or "").lstrip("/"))


def get_json(session: requests.Session, url: str, headers: dict[str, str], timeout_sec: float = 30.0) -> dict[str, Any]:
    response = session.get(url, headers=headers, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def post_source_template(
    session: requests.Session,
    base_url: str,
    headers: dict[str, str],
    source_image: Path | None,
    source_video: Path | None,
    source_frame: str,
    template_name: str,
    timeout_sec: float,
) -> dict[str, Any]:
    form_data = {"template_name": template_name}
    files: dict[str, Any] = {}
    if source_image is not None:
        files["source_image"] = (source_image.name, source_image.open("rb"), "application/octet-stream")
    elif source_video is not None:
        files["source_video"] = (source_video.name, source_video.open("rb"), "application/octet-stream")
    else:
        form_data["source_frame"] = str(source_frame or "").strip()
    try:
        response = session.post(
            resolve_url(base_url, "/api/source-templates"),
            headers=headers,
            data=form_data,
            files=files or None,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        return response.json()
    finally:
        for file_entry in files.values():
            file_entry[1].close()


def enqueue_audio(
    session: requests.Session,
    base_url: str,
    headers: dict[str, str],
    audio_path: Path,
    source_template_pack: str,
    timeout_sec: float,
) -> dict[str, Any]:
    with audio_path.open("rb") as audio_handle:
        response = session.post(
            resolve_url(base_url, "/api/avatar/enqueue"),
            headers=headers,
            data={"source_template_pack": source_template_pack},
            files={"audio": (audio_path.name, audio_handle, "application/octet-stream")},
            timeout=timeout_sec,
        )
    response.raise_for_status()
    return response.json()


def read_stream_bytes(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    minimum_bytes: int = 4096,
    timeout_sec: float = 20.0,
) -> int:
    total_bytes = 0
    deadline = time.time() + timeout_sec
    with session.get(url, headers=headers, stream=True, timeout=(10.0, timeout_sec)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                if time.time() > deadline:
                    break
                continue
            total_bytes += len(chunk)
            if total_bytes >= minimum_bytes:
                return total_bytes
            if time.time() > deadline:
                break
    return total_bytes


def wait_for_job_state(
    session: requests.Session,
    job_status_url: str,
    headers: dict[str, str],
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        payload = get_json(session, job_status_url, headers)
        last_payload = payload
        if str(payload.get("state") or "").lower() in {"processing", "done", "error"}:
            return payload
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"Timed out waiting for job to start. Last payload: {last_payload}")


def wait_for_talking_state(
    session: requests.Session,
    avatar_status_url: str,
    headers: dict[str, str],
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        payload = get_json(session, avatar_status_url, headers)
        last_payload = payload
        if str(payload.get("mode") or "").lower() == "talking":
            return payload
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"Timed out waiting for talking state. Last payload: {last_payload}")


def wait_for_job_done(
    session: requests.Session,
    job_status_url: str,
    headers: dict[str, str],
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        payload = get_json(session, job_status_url, headers)
        last_payload = payload
        state = str(payload.get("state") or "").lower()
        if state == "done":
            return payload
        if state == "error":
            raise RuntimeError(f"Job failed: {payload}")
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"Timed out waiting for job completion. Last payload: {last_payload}")


def main() -> int:
    args = parse_args()
    source_image = require_file(args.source_image, "Source image") if str(args.source_image).strip() else None
    source_video = require_file(args.source_video, "Source video") if str(args.source_video).strip() else None
    if source_image is not None and source_video is not None:
        raise ValueError("Provide either --source-image or --source-video, not both.")
    if source_image is None and source_video is None and not str(args.source_frame).strip():
        raise ValueError("Provide --source-image, --source-video or --source-frame.")
    audio_path = require_file(args.audio, "Audio")

    session = requests.Session()
    base_url = args.base_url.rstrip("/")

    health = get_json(session, resolve_url(base_url, "/api/health"), build_headers(args.api_token))
    avatar_session_id = str(health.get("avatarSessionId") or "").strip()
    if not avatar_session_id:
        raise RuntimeError(f"/api/health did not return avatarSessionId: {health}")

    session_headers = build_headers(args.api_token, avatar_session_id)
    avatar_status_url = resolve_url(base_url, "/api/avatar/status")
    avatar_stream_url = resolve_url(base_url, str(health.get("avatarVideoHttpUrl") or "/api/avatar/video.mp4"))

    print(f"[smoke] session={avatar_session_id}")
    print(f"[smoke] avatar transport={health.get('avatarTransport')}")
    print(f"[smoke] runner python={health.get('runnerPython')}")
    print(f"[smoke] runner repo={health.get('runnerFasterRepoDir')}")
    print(f"[smoke] avatar stream={avatar_stream_url}")

    idle_status = get_json(session, avatar_status_url, session_headers)
    print(f"[smoke] idle mode={idle_status.get('mode')} queueDepth={idle_status.get('queueDepth')}")

    idle_stream_bytes = read_stream_bytes(session, avatar_stream_url, session_headers)
    if idle_stream_bytes <= 0:
        raise RuntimeError("Avatar HTTP stream did not return idle bytes.")
    print(f"[smoke] idle stream bytes={idle_stream_bytes}")

    template_response = post_source_template(
        session=session,
        base_url=base_url,
        headers=session_headers,
        source_image=source_image,
        source_video=source_video,
        source_frame=args.source_frame,
        template_name=args.template_name,
        timeout_sec=args.timeout_sec,
    )
    template_item = dict(template_response.get("item") or {})
    template_pack_id = str(template_item.get("id") or template_item.get("name") or "").strip()
    if not template_pack_id:
        raise RuntimeError(f"Template build did not return an id: {template_response}")
    print(f"[smoke] template pack={template_pack_id}")

    enqueue_response = enqueue_audio(
        session=session,
        base_url=base_url,
        headers=session_headers,
        audio_path=audio_path,
        source_template_pack=template_pack_id,
        timeout_sec=args.timeout_sec,
    )
    job_id = str(enqueue_response.get("jobId") or "").strip()
    if not job_id:
        raise RuntimeError(f"Enqueue response missing jobId: {enqueue_response}")
    print(f"[smoke] job id={job_id}")

    job_status_url = resolve_url(base_url, f"/api/jobs/{job_id}/status")
    started_payload = wait_for_job_state(
        session=session,
        job_status_url=job_status_url,
        headers=session_headers,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    print(
        "[smoke] job started "
        f"state={started_payload.get('state')} progress={dict(started_payload.get('status') or {}).get('progress')}"
    )

    talking_payload = wait_for_talking_state(
        session=session,
        avatar_status_url=avatar_status_url,
        headers=session_headers,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    print(
        "[smoke] avatar talking "
        f"job={talking_payload.get('currentJobId')} sequence={talking_payload.get('sequence')}"
    )

    talking_stream_bytes = read_stream_bytes(session, avatar_stream_url, session_headers)
    if talking_stream_bytes <= 0:
        raise RuntimeError("Avatar HTTP stream did not return bytes while talking.")
    print(f"[smoke] talking stream bytes={talking_stream_bytes}")

    finished_payload = wait_for_job_done(
        session=session,
        job_status_url=job_status_url,
        headers=session_headers,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    result_video_url = str(finished_payload.get("resultVideoUrl") or "").strip()
    if not result_video_url:
        raise RuntimeError(f"Job completed without resultVideoUrl: {finished_payload}")

    result_url = resolve_url(base_url, result_video_url)
    result_response = session.get(result_url, headers=session_headers, timeout=60.0)
    result_response.raise_for_status()
    result_size = len(result_response.content)
    if result_size <= 0:
        raise RuntimeError("Downloaded result video is empty.")

    print(f"[smoke] result video bytes={result_size} url={result_url}")
    print("[smoke] E2E flow OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
