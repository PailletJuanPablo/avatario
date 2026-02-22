"""
Render viseme clips from prebuilt motion templates (.pkl) using FasterLivePortrait.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_motion_manifest.json"
DEFAULT_CFG_PATH = "third_party/FasterLivePortrait/configs/trt_infer.yaml"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/clips"
DEFAULT_OUTPUT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_clip_manifest.json"
DEFAULT_SOURCE_CACHE_DIR = "output_fasterliveportrait/source_preprocess_cache"
DEFAULT_DOCKER_CONTAINER = "animation_api"
DEFAULT_DOCKER_SERVICE = "animation-api"
DEFAULT_DOCKER_PYTHON = "/root/miniconda3/bin/python"
DEFAULT_CONTAINER_APP_ROOT = "/app"
DEFAULT_CONTAINER_FASTER_REPO = "/app/third_party/FasterLivePortrait"
DEFAULT_RUNTIME = "docker"
DEFAULT_PARALLEL_JOBS = 1
DEFAULT_VALIDATE_MOUTH_OPEN = True
RUNTIME_DOCKER = "docker"
RUNTIME_LOCAL = "local"
DEFAULT_MOUTH_MIN_MEAN_BY_VISEME = {
    "AA": 0.0120,
    "E": 0.0090,
    "I": 0.0070,
    "O": 0.0080,
    "U": 0.0055,
}
DEFAULT_MOUTH_MIN_MAX_BY_VISEME = {
    "AA": 0.0200,
    "E": 0.0200,
    "I": 0.0120,
    "O": 0.0150,
    "U": 0.0090,
}


@dataclass(frozen=True)
class ClipBuildResult:
    """
    Per-viseme rendered clip outputs.
    """

    viseme: str
    pkl_path: Path
    org_clip_path: Path
    crop_clip_path: Path
    org_duration_sec: float
    crop_duration_sec: float
    elapsed_seconds: float
    mouth_metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(description="Render viseme mp4 clips from pkl motion templates.")
    parser.add_argument("--motion-manifest", default=DEFAULT_MOTION_MANIFEST_PATH)
    parser.add_argument("--cfg", default=DEFAULT_CFG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST_PATH)
    parser.add_argument("--source-cache-dir", default=DEFAULT_SOURCE_CACHE_DIR)
    parser.add_argument("--base-image", default="", help="Override base image path from motion manifest.")
    parser.add_argument("--runtime", choices=[RUNTIME_DOCKER, RUNTIME_LOCAL], default=DEFAULT_RUNTIME)
    parser.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    parser.add_argument("--docker-service", default=DEFAULT_DOCKER_SERVICE)
    parser.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--auto-start-container", action="store_true", default=True)
    parser.add_argument("--no-auto-start-container", dest="auto_start_container", action="store_false")
    parser.add_argument("--paste-back", action="store_true", default=True)
    parser.add_argument("--no-paste-back", dest="paste_back", action="store_false")
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_PARALLEL_JOBS,
        help="Parallel workers for viseme clip rendering.",
    )
    parser.add_argument(
        "--validate-mouth-open",
        action="store_true",
        default=DEFAULT_VALIDATE_MOUTH_OPEN,
        help="Validate mouth openness for base vowel visemes and fail on invalid clips.",
    )
    parser.add_argument(
        "--no-validate-mouth-open",
        dest="validate_mouth_open",
        action="store_false",
        help="Disable mouth openness validation.",
    )
    parser.add_argument(
        "--mouth-min-mean-json",
        default="",
        help="JSON object that overrides per-viseme minimum mouth-open mean ratios.",
    )
    parser.add_argument(
        "--mouth-min-max-json",
        default="",
        help="JSON object that overrides per-viseme minimum mouth-open peak ratios.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve path relative to project root when needed.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """
    Convert path to project-relative POSIX format when possible.
    """
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def to_container_path(path_value: Path) -> str:
    """
    Convert project path to /app path inside Docker container.
    """
    resolved = path_value.resolve()
    relative = resolved.relative_to(PROJECT_ROOT)
    return f"{DEFAULT_CONTAINER_APP_ROOT}/{relative.as_posix()}"


def run_command(command: list[str]) -> None:
    """
    Execute command and raise on non-zero return code.
    """
    print(f"[cmd] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def run_command_capture(command: list[str]) -> str:
    """
    Execute command and return stdout.
    """
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")
    return (result.stdout or "").strip()


def parse_threshold_override(raw_value: str, default_map: dict[str, float], field_name: str) -> dict[str, float]:
    """Parse JSON threshold override map."""
    if not str(raw_value).strip():
        return dict(default_map)
    payload = json.loads(str(raw_value))
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    merged = dict(default_map)
    for key, value in payload.items():
        merged[str(key).strip()] = float(value)
    return merged


def should_validate_viseme(viseme: str, min_mean_by_viseme: dict[str, float], min_max_by_viseme: dict[str, float]) -> bool:
    """Return True when viseme requires mouth openness validation."""
    if "_to_" in viseme:
        return False
    return viseme in min_mean_by_viseme or viseme in min_max_by_viseme


def compute_mouth_open_metrics(video_path: Path) -> dict[str, float]:
    """Compute mouth-open ratio metrics using MediaPipe FaceMesh."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Mouth validation requires 'opencv-python' and 'mediapipe' installed on host.") from exc
    import statistics

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for mouth validation: {video_path}")
    values: list[float] = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)
            if not result.multi_face_landmarks:
                continue
            landmarks = result.multi_face_landmarks[0].landmark
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_left = landmarks[78]
            mouth_right = landmarks[308]
            open_distance = ((upper_lip.x - lower_lip.x) ** 2 + (upper_lip.y - lower_lip.y) ** 2) ** 0.5
            mouth_width = ((mouth_left.x - mouth_right.x) ** 2 + (mouth_left.y - mouth_right.y) ** 2) ** 0.5
            if mouth_width > 1e-6:
                values.append(open_distance / mouth_width)
    finally:
        cap.release()
        face_mesh.close()
    if not values:
        raise RuntimeError(f"No face landmarks detected for mouth validation: {video_path}")
    return {
        "count": float(len(values)),
        "mean": float(statistics.mean(values)),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def validate_mouth_metrics(
    viseme: str,
    metrics: dict[str, float],
    min_mean_by_viseme: dict[str, float],
    min_max_by_viseme: dict[str, float],
) -> None:
    """Validate mouth metrics against per-viseme thresholds."""
    observed_mean = float(metrics.get("mean", 0.0))
    observed_max = float(metrics.get("max", 0.0))
    expected_min_mean = float(min_mean_by_viseme.get(viseme, 0.0))
    expected_min_max = float(min_max_by_viseme.get(viseme, 0.0))
    if observed_mean < expected_min_mean or observed_max < expected_min_max:
        raise ValueError(
            f"mouth validation failed for {viseme}: "
            f"mean={observed_mean:.6f} minMean={expected_min_mean:.6f} "
            f"max={observed_max:.6f} minMax={expected_min_max:.6f}"
        )


def is_container_running(container_name: str) -> bool:
    """
    Check if Docker container is running.
    """
    try:
        stdout = run_command_capture(["docker", "inspect", "-f", "{{.State.Running}}", container_name])
    except RuntimeError:
        return False
    return stdout.strip().lower() == "true"


def ensure_container_running(container_name: str, service_name: str, auto_start: bool) -> None:
    """
    Ensure Docker runtime container is available.
    """
    if is_container_running(container_name):
        return
    if not auto_start:
        raise RuntimeError(f"Docker container is not running: {container_name}")
    print(f"[info] starting container via docker compose: {service_name}")
    run_command(["docker", "compose", "up", "-d", service_name])
    if not is_container_running(container_name):
        raise RuntimeError(f"Docker container failed to start: {container_name}")


def read_json(path_value: Path) -> dict[str, Any]:
    """
    Read JSON object from file.
    """
    with path_value.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def load_motion_manifest(
    motion_manifest_path: Path,
    base_image_override: str,
) -> tuple[Path, list[dict[str, str]]]:
    """
    Load base image and viseme pkl entries from motion manifest.
    """
    payload = read_json(motion_manifest_path)
    base_image_rel = base_image_override.strip() or str(payload.get("baseImage", "")).strip()
    if not base_image_rel:
        raise ValueError("Missing baseImage in motion manifest and no --base-image override provided.")
    base_image_path = resolve_path(base_image_rel)
    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")

    visemes = payload.get("visemes")
    if not isinstance(visemes, list) or not visemes:
        raise ValueError(f"Motion manifest has no viseme entries: {motion_manifest_path}")

    entries: list[dict[str, str]] = []
    for item in visemes:
        if not isinstance(item, dict):
            raise ValueError("Invalid viseme entry type in motion manifest.")
        viseme = str(item.get("viseme", "")).strip()
        pkl_rel = str(item.get("pkl", "")).strip()
        if not viseme:
            raise ValueError("Viseme entry missing 'viseme'.")
        if not pkl_rel:
            raise ValueError(f"Viseme {viseme} missing 'pkl'.")
        entries.append({"viseme": viseme, "pkl": pkl_rel})
    return base_image_path, entries


def get_video_duration_sec(video_path: Path) -> float:
    """
    Read media duration using ffprobe.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    stdout = run_command_capture(command)
    return float(stdout.strip() or 0.0)


def find_generated_clip(raw_dir: Path, suffix: str) -> Path:
    """
    Find latest generated clip with expected suffix.
    """
    matches = sorted(
        (path for path in raw_dir.glob(f"*{suffix}.mp4") if path.is_file()),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No generated clip with suffix {suffix} in {raw_dir}")
    return matches[0]


def build_local_command(
    base_image_path: Path,
    pkl_path: Path,
    cfg_path: Path,
    source_cache_dir: Path,
    save_dir: Path,
    paste_back: bool,
) -> list[str]:
    """
    Build local run.py command.
    """
    command = [
        "python",
        str((PROJECT_ROOT / "third_party/FasterLivePortrait/run.py").resolve()),
        "--src_image",
        str(base_image_path),
        "--dri_video",
        str(pkl_path),
        "--cfg",
        str(cfg_path),
        "--source_cache_dir",
        str(source_cache_dir),
        "--save_dir",
        str(save_dir),
    ]
    if paste_back:
        command.append("--paste_back")
    return command


def build_docker_command(
    container_name: str,
    container_python: str,
    base_image_path: Path,
    pkl_path: Path,
    cfg_path: Path,
    source_cache_dir: Path,
    save_dir: Path,
    paste_back: bool,
) -> list[str]:
    """
    Build docker exec run.py command.
    """
    command = [
        "docker",
        "exec",
        "-w",
        DEFAULT_CONTAINER_FASTER_REPO,
        container_name,
        container_python,
        "run.py",
        "--src_image",
        to_container_path(base_image_path),
        "--dri_video",
        to_container_path(pkl_path),
        "--cfg",
        to_container_path(cfg_path),
        "--source_cache_dir",
        to_container_path(source_cache_dir),
        "--save_dir",
        to_container_path(save_dir),
    ]
    if paste_back:
        command.append("--paste_back")
    return command


def write_output_manifest(
    output_manifest_path: Path,
    motion_manifest_path: Path,
    base_image_path: Path,
    output_dir: Path,
    results: list[ClipBuildResult],
) -> None:
    """
    Write clip manifest JSON.
    """
    payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceMotionManifest": to_project_relative(motion_manifest_path),
        "baseImage": to_project_relative(base_image_path),
        "outputDir": to_project_relative(output_dir),
        "visemeCount": len(results),
        "visemes": [
            {
                "viseme": result.viseme,
                "pkl": to_project_relative(result.pkl_path),
                "clipOrg": to_project_relative(result.org_clip_path),
                "clipCrop": to_project_relative(result.crop_clip_path),
                "orgDurationSec": round(result.org_duration_sec, 3),
                "cropDurationSec": round(result.crop_duration_sec, 3),
                "elapsedSec": round(result.elapsed_seconds, 3),
                "mouthMetrics": result.mouth_metrics,
            }
            for result in results
        ],
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_single_clip_entry(
    entry: dict[str, str],
    args: argparse.Namespace,
    base_image_path: Path,
    cfg_path: Path,
    output_dir: Path,
    source_cache_dir: Path,
    validate_mouth_open: bool,
    mouth_min_mean_by_viseme: dict[str, float],
    mouth_min_max_by_viseme: dict[str, float],
) -> ClipBuildResult:
    """
    Render clips for one viseme entry.
    """
    viseme = entry["viseme"]
    pkl_path = resolve_path(entry["pkl"])
    if not pkl_path.exists():
        raise FileNotFoundError(f"missing pkl for {viseme}: {pkl_path}")

    viseme_dir = output_dir / viseme
    raw_dir = viseme_dir / "raw"
    org_clip_path = viseme_dir / "result_org.mp4"
    crop_clip_path = viseme_dir / "result_crop.mp4"
    viseme_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if org_clip_path.exists() and crop_clip_path.exists() and not args.overwrite:
        org_duration = get_video_duration_sec(org_clip_path)
        crop_duration = get_video_duration_sec(crop_clip_path)
        mouth_metrics: dict[str, float] = {}
        if validate_mouth_open and should_validate_viseme(viseme, mouth_min_mean_by_viseme, mouth_min_max_by_viseme):
            mouth_metrics = compute_mouth_open_metrics(org_clip_path)
            validate_mouth_metrics(viseme, mouth_metrics, mouth_min_mean_by_viseme, mouth_min_max_by_viseme)
        print(f"[skip] {viseme} clips already exist: {viseme_dir}")
        return ClipBuildResult(
            viseme=viseme,
            pkl_path=pkl_path,
            org_clip_path=org_clip_path,
            crop_clip_path=crop_clip_path,
            org_duration_sec=org_duration,
            crop_duration_sec=crop_duration,
            elapsed_seconds=0.0,
            mouth_metrics=mouth_metrics,
        )

    for stale in raw_dir.glob("*"):
        if stale.is_file():
            stale.unlink()

    entry_cache_dir = source_cache_dir / f"cache_{viseme}"
    entry_cache_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    if args.runtime == RUNTIME_DOCKER:
        command = build_docker_command(
            container_name=str(args.docker_container),
            container_python=str(args.docker_python),
            base_image_path=base_image_path,
            pkl_path=pkl_path,
            cfg_path=cfg_path,
            source_cache_dir=entry_cache_dir,
            save_dir=raw_dir,
            paste_back=bool(args.paste_back),
        )
    else:
        command = build_local_command(
            base_image_path=base_image_path,
            pkl_path=pkl_path,
            cfg_path=cfg_path,
            source_cache_dir=entry_cache_dir,
            save_dir=raw_dir,
            paste_back=bool(args.paste_back),
        )

    run_command(command)
    generated_org = find_generated_clip(raw_dir, "-org")
    generated_crop = find_generated_clip(raw_dir, "-crop")
    shutil.copy2(generated_org, org_clip_path)
    shutil.copy2(generated_crop, crop_clip_path)
    org_duration = get_video_duration_sec(org_clip_path)
    crop_duration = get_video_duration_sec(crop_clip_path)
    mouth_metrics: dict[str, float] = {}
    if validate_mouth_open and should_validate_viseme(viseme, mouth_min_mean_by_viseme, mouth_min_max_by_viseme):
        mouth_metrics = compute_mouth_open_metrics(org_clip_path)
        validate_mouth_metrics(viseme, mouth_metrics, mouth_min_mean_by_viseme, mouth_min_max_by_viseme)
    elapsed = time.perf_counter() - started_at
    print(
        f"[ok] {viseme} -> {viseme_dir} "
        f"(org={org_duration:.2f}s crop={crop_duration:.2f}s sec={elapsed:.2f})"
    )
    return ClipBuildResult(
        viseme=viseme,
        pkl_path=pkl_path,
        org_clip_path=org_clip_path,
        crop_clip_path=crop_clip_path,
        org_duration_sec=org_duration,
        crop_duration_sec=crop_duration,
        elapsed_seconds=elapsed,
        mouth_metrics=mouth_metrics,
    )


def main() -> None:
    """
    Render all viseme clips from pkl templates.
    """
    args = parse_args()
    motion_manifest_path = resolve_path(args.motion_manifest)
    cfg_path = resolve_path(args.cfg)
    output_dir = resolve_path(args.output_dir)
    output_manifest_path = resolve_path(args.output_manifest)
    source_cache_dir = resolve_path(args.source_cache_dir)
    parallel_jobs = max(1, int(args.jobs))
    mouth_min_mean_by_viseme = parse_threshold_override(
        raw_value=str(args.mouth_min_mean_json),
        default_map=DEFAULT_MOUTH_MIN_MEAN_BY_VISEME,
        field_name="mouth-min-mean-json",
    )
    mouth_min_max_by_viseme = parse_threshold_override(
        raw_value=str(args.mouth_min_max_json),
        default_map=DEFAULT_MOUTH_MIN_MAX_BY_VISEME,
        field_name="mouth-min-max-json",
    )

    if not motion_manifest_path.exists():
        raise FileNotFoundError(f"Motion manifest not found: {motion_manifest_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    if args.runtime == RUNTIME_DOCKER:
        ensure_container_running(
            container_name=str(args.docker_container),
            service_name=str(args.docker_service),
            auto_start=bool(args.auto_start_container),
        )

    base_image_path, entries = load_motion_manifest(motion_manifest_path, str(args.base_image))
    output_dir.mkdir(parents=True, exist_ok=True)
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    viseme_order = {entry["viseme"]: index for index, entry in enumerate(entries)}
    print(f"[info] entries={len(entries)} jobs={parallel_jobs}")

    built_results: list[ClipBuildResult] = []
    failures: list[str] = []
    if parallel_jobs == 1:
        for entry in entries:
            viseme = entry["viseme"]
            try:
                result = build_single_clip_entry(
                    entry=entry,
                    args=args,
                    base_image_path=base_image_path,
                    cfg_path=cfg_path,
                    output_dir=output_dir,
                    source_cache_dir=source_cache_dir,
                    validate_mouth_open=bool(args.validate_mouth_open),
                    mouth_min_mean_by_viseme=mouth_min_mean_by_viseme,
                    mouth_min_max_by_viseme=mouth_min_max_by_viseme,
                )
                built_results.append(result)
            except Exception as exc:  # noqa: BLE001
                message = f"[error] {viseme} clip render failed: {exc}"
                print(message)
                failures.append(message)
                if not args.continue_on_error:
                    raise
    else:
        future_to_viseme: dict[concurrent.futures.Future[ClipBuildResult], str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            for entry in entries:
                future = executor.submit(
                    build_single_clip_entry,
                    entry,
                    args,
                    base_image_path,
                    cfg_path,
                    output_dir,
                    source_cache_dir,
                    bool(args.validate_mouth_open),
                    mouth_min_mean_by_viseme,
                    mouth_min_max_by_viseme,
                )
                future_to_viseme[future] = entry["viseme"]

            for future in concurrent.futures.as_completed(future_to_viseme):
                viseme = future_to_viseme[future]
                try:
                    result = future.result()
                    built_results.append(result)
                except Exception as exc:  # noqa: BLE001
                    message = f"[error] {viseme} clip render failed: {exc}"
                    print(message)
                    failures.append(message)
                    if not args.continue_on_error:
                        for pending in future_to_viseme:
                            pending.cancel()
                        raise

    built_results.sort(key=lambda item: viseme_order.get(item.viseme, 10**9))

    write_output_manifest(
        output_manifest_path=output_manifest_path,
        motion_manifest_path=motion_manifest_path,
        base_image_path=base_image_path,
        output_dir=output_dir,
        results=built_results,
    )
    print(f"[ok] clip manifest -> {output_manifest_path}")
    print(f"[ok] generated viseme clips -> {len(built_results)}")

    if failures:
        print(f"[warn] failures -> {len(failures)}")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
