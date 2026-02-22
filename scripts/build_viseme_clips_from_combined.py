"""
Build viseme/transition clips from one combined FastLivePortrait render.
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
DEFAULT_COMBINED_AUDIO_PATH = "output_fasterliveportrait/viseme_library/viseme_all_segments.wav"
DEFAULT_SEGMENT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_all_segments_manifest.json"
DEFAULT_BASE_IMAGE_PATH = "output/frames/frame_00095.png"
DEFAULT_WORK_DIR = "output_fasterliveportrait/viseme_library/combined_single_run"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/clips_from_combined_single"
DEFAULT_OUTPUT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_clip_from_combined_single_manifest.json"
DEFAULT_FASTER_REPO_DIR = "third_party/FasterLivePortrait"
DEFAULT_CFG_PATH = "third_party/FasterLivePortrait/configs/trt_infer.yaml"
DEFAULT_AUDIO_TO_PKL_SCRIPT = "faster_liveportrait_audio_to_pkl.py"
DEFAULT_SOURCE_CACHE_DIR = "output_fasterliveportrait/source_preprocess_cache/combined_single"
DEFAULT_DOCKER_CONTAINER = "animation_api"
DEFAULT_DOCKER_SERVICE = "animation-api"
DEFAULT_DOCKER_PYTHON = "/root/miniconda3/bin/python"
DEFAULT_CONTAINER_APP_ROOT = "/app"
DEFAULT_CONTAINER_FASTER_REPO = "/app/third_party/FasterLivePortrait"
DEFAULT_RUNTIME = "docker"
DEFAULT_JOBS = 4
DEFAULT_SEED = 1234
RUNTIME_DOCKER = "docker"
RUNTIME_LOCAL = "local"


@dataclass(frozen=True)
class SegmentSpec:
    """
    One viseme/transition segment to cut.
    """

    index: int
    kind: str
    viseme: str
    from_viseme: str
    to_viseme: str
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass(frozen=True)
class SegmentClipResult:
    """
    Output clip paths for one segment.
    """

    segment: SegmentSpec
    clip_org_path: Path
    clip_crop_path: Path
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(
        description="Run FastLivePortrait once for combined audio and cut viseme/transition clips by segment times."
    )
    parser.add_argument("--combined-audio", default=DEFAULT_COMBINED_AUDIO_PATH)
    parser.add_argument("--segment-manifest", default=DEFAULT_SEGMENT_MANIFEST_PATH)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE_PATH)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST_PATH)
    parser.add_argument("--faster-repo-dir", default=DEFAULT_FASTER_REPO_DIR)
    parser.add_argument("--cfg", default=DEFAULT_CFG_PATH)
    parser.add_argument("--audio-to-pkl-script", default=DEFAULT_AUDIO_TO_PKL_SCRIPT)
    parser.add_argument("--source-cache-dir", default=DEFAULT_SOURCE_CACHE_DIR)
    parser.add_argument("--runtime", choices=[RUNTIME_DOCKER, RUNTIME_LOCAL], default=DEFAULT_RUNTIME)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    parser.add_argument("--docker-service", default=DEFAULT_DOCKER_SERVICE)
    parser.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--auto-start-container", action="store_true", default=True)
    parser.add_argument("--no-auto-start-container", dest="auto_start_container", action="store_false")
    parser.add_argument("--paste-back", action="store_true", default=True)
    parser.add_argument("--no-paste-back", dest="paste_back", action="store_false")
    parser.add_argument("--skip-pkl-build", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
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
    Convert absolute path to project-relative POSIX representation when possible.
    """
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def to_container_path(path_value: Path) -> str:
    """
    Convert project path into /app path used inside Docker container.
    """
    resolved = path_value.resolve()
    relative = resolved.relative_to(PROJECT_ROOT)
    return f"{DEFAULT_CONTAINER_APP_ROOT}/{relative.as_posix()}"


def run_command(command: list[str]) -> None:
    """
    Run command and raise with stderr on non-zero exit.
    """
    print(f"[cmd] {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")


def run_command_capture(command: list[str]) -> str:
    """
    Run command and return stdout.
    """
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")
    return (result.stdout or "").strip()


def ensure_ffmpeg_available() -> None:
    """
    Ensure ffmpeg is installed and available in PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not found in PATH.")


def read_json(path_value: Path) -> dict[str, Any]:
    """
    Read and validate JSON object.
    """
    payload = json.loads(path_value.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def is_container_running(container_name: str) -> bool:
    """
    Return True when Docker container is running.
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
    run_command(["docker", "compose", "up", "-d", service_name])
    if not is_container_running(container_name):
        raise RuntimeError(f"Docker container failed to start: {container_name}")


def parse_transition_viseme(viseme_key: str) -> tuple[str, str]:
    """
    Parse FROM_to_TO transition key.
    """
    if "_to_" not in viseme_key:
        return "", ""
    from_viseme, to_viseme = viseme_key.split("_to_", 1)
    return from_viseme.strip(), to_viseme.strip()


def load_segment_specs(segment_manifest_path: Path) -> list[SegmentSpec]:
    """
    Load viseme segment definitions from combined-audio manifest.
    """
    payload = read_json(segment_manifest_path)
    raw_segments = payload.get("segments")
    sample_rate_hz = int(payload.get("sampleRateHz", 0) or 0)
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"No segments in manifest: {segment_manifest_path}")

    segments: list[SegmentSpec] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise ValueError("Invalid segment entry in manifest.")
        index = int(item.get("index", -1))
        kind = str(item.get("kind", "")).strip()
        viseme = str(item.get("viseme", "")).strip()
        raw_start_sec = item.get("startSec")
        raw_end_sec = item.get("endSec")
        raw_duration_sec = item.get("durationSec")
        start_sec = float(raw_start_sec) if raw_start_sec is not None else -1.0
        end_sec = float(raw_end_sec) if raw_end_sec is not None else -1.0
        duration_sec = float(raw_duration_sec) if raw_duration_sec is not None else 0.0
        if start_sec < 0.0 or end_sec <= start_sec:
            start_sample = int(item.get("startSample", -1))
            end_sample = int(item.get("endSample", -1))
            if sample_rate_hz <= 0 or start_sample < 0 or end_sample <= start_sample:
                raise ValueError(f"Invalid timing for segment {viseme}")
            start_sec = float(start_sample) / float(sample_rate_hz)
            end_sec = float(end_sample) / float(sample_rate_hz)
            duration_sec = end_sec - start_sec
        if index < 0:
            raise ValueError("Segment index must be >= 0.")
        if kind not in {"base", "transition"}:
            kind = "transition" if "_to_" in viseme else "base"
        if not viseme:
            raise ValueError("Segment entry missing viseme.")
        from_viseme, to_viseme = parse_transition_viseme(viseme)
        if kind == "base":
            from_viseme = viseme
            to_viseme = viseme
        segments.append(
            SegmentSpec(
                index=index,
                kind=kind,
                viseme=viseme,
                from_viseme=from_viseme,
                to_viseme=to_viseme,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                duration_sec=float(max(duration_sec, end_sec - start_sec)),
            )
        )
    segments.sort(key=lambda item: item.index)
    expected_indices = list(range(len(segments)))
    actual_indices = [item.index for item in segments]
    if actual_indices != expected_indices:
        raise ValueError("Segment indices must be contiguous from 0.")
    return segments


def find_generated_clip(raw_dir: Path, suffix: str) -> Path:
    """
    Find latest generated run.py clip with expected suffix.
    """
    matches = sorted(
        (path for path in raw_dir.glob(f"*{suffix}.mp4") if path.is_file()),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No generated clip with suffix {suffix} in {raw_dir}")
    return matches[0]


def build_pkl(
    args: argparse.Namespace,
    combined_audio_path: Path,
    output_pkl_path: Path,
    faster_repo_dir: Path,
    cfg_path: Path,
    audio_to_pkl_script: Path,
) -> None:
    """
    Build one motion pkl from combined audio.
    """
    if output_pkl_path.exists() and not bool(args.overwrite):
        print(f"[skip] combined pkl already exists: {output_pkl_path}")
        return
    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    if args.runtime == RUNTIME_DOCKER:
        command = [
            "docker",
            "exec",
            str(args.docker_container),
            str(args.docker_python),
            to_container_path(audio_to_pkl_script),
            "--faster-repo-dir",
            to_container_path(faster_repo_dir),
            "--cfg",
            to_container_path(cfg_path),
            "--driving-audio",
            to_container_path(combined_audio_path),
            "--output-pkl",
            to_container_path(output_pkl_path),
            "--seed",
            str(int(args.seed)),
        ]
    else:
        command = [
            str(args.python_executable),
            str(audio_to_pkl_script),
            "--faster-repo-dir",
            str(faster_repo_dir),
            "--cfg",
            str(cfg_path),
            "--driving-audio",
            str(combined_audio_path),
            "--output-pkl",
            str(output_pkl_path),
            "--seed",
            str(int(args.seed)),
        ]
    run_command(command)
    print(f"[ok] combined pkl -> {output_pkl_path}")


def render_combined_clip(
    args: argparse.Namespace,
    base_image_path: Path,
    combined_pkl_path: Path,
    cfg_path: Path,
    source_cache_dir: Path,
    render_raw_dir: Path,
    render_org_path: Path,
    render_crop_path: Path,
) -> None:
    """
    Render combined org/crop clips from one combined pkl.
    """
    if render_org_path.exists() and render_crop_path.exists() and not bool(args.overwrite):
        print(f"[skip] combined render already exists: {render_org_path} / {render_crop_path}")
        return

    render_raw_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.overwrite):
        for stale in render_raw_dir.glob("*"):
            if stale.is_file():
                stale.unlink()

    source_cache_dir.mkdir(parents=True, exist_ok=True)
    if args.runtime == RUNTIME_DOCKER:
        command = [
            "docker",
            "exec",
            "-w",
            DEFAULT_CONTAINER_FASTER_REPO,
            str(args.docker_container),
            str(args.docker_python),
            "run.py",
            "--src_image",
            to_container_path(base_image_path),
            "--dri_video",
            to_container_path(combined_pkl_path),
            "--cfg",
            to_container_path(cfg_path),
            "--source_cache_dir",
            to_container_path(source_cache_dir),
            "--save_dir",
            to_container_path(render_raw_dir),
        ]
    else:
        command = [
            str(args.python_executable),
            str((PROJECT_ROOT / "third_party/FasterLivePortrait/run.py").resolve()),
            "--src_image",
            str(base_image_path),
            "--dri_video",
            str(combined_pkl_path),
            "--cfg",
            str(cfg_path),
            "--source_cache_dir",
            str(source_cache_dir),
            "--save_dir",
            str(render_raw_dir),
        ]
    if bool(args.paste_back):
        command.append("--paste_back")

    run_command(command)
    generated_org = find_generated_clip(render_raw_dir, "-org")
    generated_crop = find_generated_clip(render_raw_dir, "-crop")
    render_org_path.parent.mkdir(parents=True, exist_ok=True)
    render_crop_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_org, render_org_path)
    shutil.copy2(generated_crop, render_crop_path)
    print(f"[ok] combined org clip -> {render_org_path}")
    print(f"[ok] combined crop clip -> {render_crop_path}")


def trim_video_segment(
    source_clip_path: Path,
    output_clip_path: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool,
) -> None:
    """
    Cut one video segment with exact trim and reset timestamps.
    """
    if output_clip_path.exists() and not overwrite:
        return
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_clip_path),
        "-vf",
        f"trim=start={float(start_sec):.6f}:end={float(end_sec):.6f},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_clip_path),
    ]
    run_command(command)


def build_one_segment_clip(
    segment: SegmentSpec,
    combined_org_path: Path,
    combined_crop_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> SegmentClipResult:
    """
    Build org/crop clips for one viseme segment.
    """
    viseme_dir = output_dir / segment.viseme
    org_clip_path = viseme_dir / "result_org.mp4"
    crop_clip_path = viseme_dir / "result_crop.mp4"
    started_at = time.perf_counter()
    trim_video_segment(
        source_clip_path=combined_org_path,
        output_clip_path=org_clip_path,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        overwrite=overwrite,
    )
    trim_video_segment(
        source_clip_path=combined_crop_path,
        output_clip_path=crop_clip_path,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        overwrite=overwrite,
    )
    elapsed = time.perf_counter() - started_at
    print(f"[ok] {segment.viseme} -> {viseme_dir}")
    return SegmentClipResult(
        segment=segment,
        clip_org_path=org_clip_path,
        clip_crop_path=crop_clip_path,
        elapsed_seconds=elapsed,
    )


def write_output_manifest(
    output_manifest_path: Path,
    output_dir: Path,
    base_image_path: Path,
    combined_audio_path: Path,
    segment_manifest_path: Path,
    combined_pkl_path: Path,
    combined_org_path: Path,
    combined_crop_path: Path,
    results: list[SegmentClipResult],
) -> None:
    """
    Write clip manifest compatible with extract_viseme_frames.py.
    """
    payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceSegmentManifest": to_project_relative(segment_manifest_path),
        "sourceCombinedAudio": to_project_relative(combined_audio_path),
        "sourceCombinedPkl": to_project_relative(combined_pkl_path),
        "sourceCombinedOrgClip": to_project_relative(combined_org_path),
        "sourceCombinedCropClip": to_project_relative(combined_crop_path),
        "baseImage": to_project_relative(base_image_path),
        "outputDir": to_project_relative(output_dir),
        "visemeCount": len(results),
        "visemes": [
            {
                "viseme": result.segment.viseme,
                "kind": result.segment.kind,
                "index": result.segment.index,
                "fromViseme": result.segment.from_viseme,
                "toViseme": result.segment.to_viseme,
                "startSec": round(float(result.segment.start_sec), 6),
                "endSec": round(float(result.segment.end_sec), 6),
                "durationSec": round(float(result.segment.duration_sec), 6),
                "clipOrg": to_project_relative(result.clip_org_path),
                "clipCrop": to_project_relative(result.clip_crop_path),
                "orgDurationSec": round(float(result.segment.duration_sec), 6),
                "cropDurationSec": round(float(result.segment.duration_sec), 6),
                "elapsedSec": round(float(result.elapsed_seconds), 3),
            }
            for result in results
        ],
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """
    Program entry point.
    """
    args = parse_args()
    combined_audio_path = resolve_path(str(args.combined_audio))
    segment_manifest_path = resolve_path(str(args.segment_manifest))
    base_image_path = resolve_path(str(args.base_image))
    work_dir = resolve_path(str(args.work_dir))
    output_dir = resolve_path(str(args.output_dir))
    output_manifest_path = resolve_path(str(args.output_manifest))
    faster_repo_dir = resolve_path(str(args.faster_repo_dir))
    cfg_path = resolve_path(str(args.cfg))
    audio_to_pkl_script = resolve_path(str(args.audio_to_pkl_script))
    source_cache_dir = resolve_path(str(args.source_cache_dir))
    parallel_jobs = max(1, int(args.jobs))

    if not combined_audio_path.exists():
        raise FileNotFoundError(f"Combined audio not found: {combined_audio_path}")
    if not segment_manifest_path.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest_path}")
    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    if not faster_repo_dir.exists():
        raise FileNotFoundError(f"Faster repo not found: {faster_repo_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not audio_to_pkl_script.exists():
        raise FileNotFoundError(f"Audio->PKL script not found: {audio_to_pkl_script}")

    segments = load_segment_specs(segment_manifest_path)
    if not segments:
        raise ValueError("No segments available for clip cutting.")
    print(f"[info] segments={len(segments)} jobs={parallel_jobs}")
    ensure_ffmpeg_available()

    if args.runtime == RUNTIME_DOCKER:
        ensure_container_running(
            container_name=str(args.docker_container),
            service_name=str(args.docker_service),
            auto_start=bool(args.auto_start_container),
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_pkl_path = work_dir / "motion_combined.pkl"
    render_raw_dir = work_dir / "render_raw"
    combined_org_path = work_dir / "result_org.mp4"
    combined_crop_path = work_dir / "result_crop.mp4"

    if not bool(args.skip_pkl_build):
        build_pkl(
            args=args,
            combined_audio_path=combined_audio_path,
            output_pkl_path=combined_pkl_path,
            faster_repo_dir=faster_repo_dir,
            cfg_path=cfg_path,
            audio_to_pkl_script=audio_to_pkl_script,
        )
    elif not combined_pkl_path.exists():
        raise FileNotFoundError(f"--skip-pkl-build set but pkl not found: {combined_pkl_path}")

    if not bool(args.skip_render):
        render_combined_clip(
            args=args,
            base_image_path=base_image_path,
            combined_pkl_path=combined_pkl_path,
            cfg_path=cfg_path,
            source_cache_dir=source_cache_dir,
            render_raw_dir=render_raw_dir,
            render_org_path=combined_org_path,
            render_crop_path=combined_crop_path,
        )
    elif not combined_org_path.exists() or not combined_crop_path.exists():
        raise FileNotFoundError(
            f"--skip-render set but combined render clips are missing: {combined_org_path} / {combined_crop_path}"
        )

    failures: list[str] = []
    results: list[SegmentClipResult] = []
    if parallel_jobs == 1:
        for segment in segments:
            try:
                results.append(
                    build_one_segment_clip(
                        segment=segment,
                        combined_org_path=combined_org_path,
                        combined_crop_path=combined_crop_path,
                        output_dir=output_dir,
                        overwrite=bool(args.overwrite),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                message = f"[error] {segment.viseme} cut failed: {exc}"
                print(message)
                failures.append(message)
                if not bool(args.continue_on_error):
                    raise
    else:
        future_to_segment: dict[concurrent.futures.Future[SegmentClipResult], SegmentSpec] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            for segment in segments:
                future = executor.submit(
                    build_one_segment_clip,
                    segment,
                    combined_org_path,
                    combined_crop_path,
                    output_dir,
                    bool(args.overwrite),
                )
                future_to_segment[future] = segment
            for future in concurrent.futures.as_completed(future_to_segment):
                segment = future_to_segment[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    message = f"[error] {segment.viseme} cut failed: {exc}"
                    print(message)
                    failures.append(message)
                    if not bool(args.continue_on_error):
                        for pending in future_to_segment:
                            pending.cancel()
                        raise

    results.sort(key=lambda item: item.segment.index)
    write_output_manifest(
        output_manifest_path=output_manifest_path,
        output_dir=output_dir,
        base_image_path=base_image_path,
        combined_audio_path=combined_audio_path,
        segment_manifest_path=segment_manifest_path,
        combined_pkl_path=combined_pkl_path,
        combined_org_path=combined_org_path,
        combined_crop_path=combined_crop_path,
        results=results,
    )
    print(f"[ok] clip manifest -> {output_manifest_path}")
    print(f"[ok] generated viseme/transition clips -> {len(results)}")

    if failures:
        print(f"[warn] failures -> {len(failures)}")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
