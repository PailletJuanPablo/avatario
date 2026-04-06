"""
Supervisor wrapper for FasterLivePortrait persistent workers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


HEARTBEAT_FILE_NAME = "worker_heartbeat.json"


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Persistent worker supervisor", add_help=True)
    parser.add_argument("--worker-script", required=True, type=str)
    parser.add_argument("--heartbeat-interval-sec", dest="wrapper_heartbeat_interval_sec", type=float, default=1.0)
    return parser.parse_known_args()


def extract_option_value(arguments: list[str], option_name: str, default_value: str = "") -> str:
    for index, token in enumerate(arguments):
        if token != option_name:
            continue
        if index + 1 < len(arguments):
            return str(arguments[index + 1])
    return default_value


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    json_text = json.dumps(payload, indent=2)
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(json_text)
    replace_deadline = time.time() + 2.0
    last_error: PermissionError | None = None
    while True:
        try:
            os.replace(str(tmp_path), str(path))
            return
        except PermissionError as exc:
            last_error = exc
            if time.time() >= replace_deadline:
                break
            time.sleep(0.05)
    with contextlib.suppress(Exception):
        tmp_path.unlink(missing_ok=True)
    if last_error is not None:
        raise last_error


def build_heartbeat_payload(queue_dir: Path, child_pid: int, forwarded_args: list[str]) -> dict:
    return {
        "state": "alive",
        "pid": int(child_pid),
        "updatedAtMs": now_ms(),
        "queueDir": str(queue_dir),
        "supervised": True,
        "forwardedArgs": forwarded_args,
    }


def main() -> int:
    wrapper_args, forwarded_args = parse_wrapper_args()
    worker_script = Path(str(wrapper_args.worker_script)).resolve()
    if not worker_script.exists():
        raise FileNotFoundError(f"Persistent worker script not found: {worker_script}")

    queue_dir = Path(extract_option_value(forwarded_args, "--queue_dir")).resolve()
    if not str(queue_dir):
        raise RuntimeError("Missing forwarded --queue_dir for persistent worker supervisor.")
    heartbeat_interval_sec = max(
        0.2,
        float(extract_option_value(forwarded_args, "--heartbeat_interval_sec", str(wrapper_args.wrapper_heartbeat_interval_sec))),
    )
    heartbeat_path = queue_dir / HEARTBEAT_FILE_NAME

    child_command = [sys.executable, "-u", str(worker_script), *forwarded_args]
    child_env = dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    print(
        f"[worker-supervisor] launch queue_dir={queue_dir} heartbeat={heartbeat_path} command={subprocess.list2cmdline(child_command)}",
        flush=True,
    )
    child_process = subprocess.Popen(
        child_command,
        cwd=str(worker_script.parent),
        env=child_env,
    )
    try:
        while True:
            write_json_atomic(
                heartbeat_path,
                build_heartbeat_payload(queue_dir, child_process.pid, forwarded_args),
            )
            child_exit_code = child_process.poll()
            if child_exit_code is not None:
                print(
                    f"[worker-supervisor] child exited exit_code={child_exit_code} pid={child_process.pid}",
                    flush=True,
                )
                return int(child_exit_code)
            time.sleep(heartbeat_interval_sec)
    finally:
        if child_process.poll() is None:
            with contextlib.suppress(Exception):
                child_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
