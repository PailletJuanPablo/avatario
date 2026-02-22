"""
Build JoyVASA motion templates (.pkl) for each viseme audio clip.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import pickle
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_audio_manifest.json"
DEFAULT_OUTPUT_DIR = "output_fasterliveportrait/viseme_library/pkl"
DEFAULT_OUTPUT_MANIFEST_PATH = "output_fasterliveportrait/viseme_library/viseme_motion_manifest.json"
DEFAULT_FASTER_REPO_DIR = "third_party/FasterLivePortrait"
DEFAULT_CFG_PATH = "third_party/FasterLivePortrait/configs/trt_infer.yaml"
DEFAULT_PYTHON_EXECUTABLE = "python"
DEFAULT_AUDIO_TO_PKL_SCRIPT = "faster_liveportrait_audio_to_pkl.py"
DEFAULT_DOCKER_CONTAINER = "animation_api"
DEFAULT_DOCKER_SERVICE = "animation-api"
DEFAULT_DOCKER_PYTHON = "/root/miniconda3/bin/python"
DEFAULT_CONTAINER_APP_ROOT = "/app"
DEFAULT_RUNTIME = "docker"
DEFAULT_MOTION_UPSAMPLE_FACTOR = 1
DEFAULT_PARALLEL_JOBS = 1
DEFAULT_RANDOM_SEED = 1234
DEFAULT_VOWEL_OPEN_BASE_DELTA = 0.0105
DEFAULT_ENABLE_EYE_TAMED_PRESET = True
DEFAULT_EYE_SOFT_FACTOR = 0.45
DEFAULT_EYE_HARD_FACTOR = 0.18
DEFAULT_EYE_HARD_DY_MIN = -0.0045
DEFAULT_EYE_HARD_DY_MAX = 0.0035
RUNTIME_DOCKER = "docker"
RUNTIME_LOCAL = "local"
VOWEL_VISEME_KEYS = ("AA", "E", "I", "O", "U")
EYE_TAMED_SOFT_INDICES = (0, 1, 2, 3, 4, 5, 7, 10, 13)
EYE_TAMED_HARD_INDICES = (11, 15)
DEFAULT_VOWEL_OPEN_GAIN_BY_VISEME = {
    "AA": 1.35,
    "E": 0.90,
    "I": 0.70,
    "O": 1.20,
    "U": 1.05,
}


@dataclass(frozen=True)
class BuildResult:
    """
    Single viseme build result payload.
    """

    viseme: str
    audio_path: Path
    pkl_path: Path
    elapsed_seconds: float
    stats: dict[str, Any]
    from_viseme: str
    to_viseme: str


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(description="Build .pkl templates for all visemes from audio manifest.")
    parser.add_argument("--audio-manifest", default=DEFAULT_AUDIO_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST_PATH)
    parser.add_argument("--faster-repo-dir", default=DEFAULT_FASTER_REPO_DIR)
    parser.add_argument("--cfg", default=DEFAULT_CFG_PATH)
    parser.add_argument("--runtime", choices=[RUNTIME_DOCKER, RUNTIME_LOCAL], default=DEFAULT_RUNTIME)
    parser.add_argument("--python-executable", default=DEFAULT_PYTHON_EXECUTABLE)
    parser.add_argument("--audio-to-pkl-script", default=DEFAULT_AUDIO_TO_PKL_SCRIPT)
    parser.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    parser.add_argument("--docker-service", default=DEFAULT_DOCKER_SERVICE)
    parser.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--motion-upsample-factor",
        type=int,
        default=DEFAULT_MOTION_UPSAMPLE_FACTOR,
        help="Temporal upsampling factor for generated pkl motion (1=disabled).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_PARALLEL_JOBS,
        help="Parallel workers for viseme pkl generation.",
    )
    parser.add_argument(
        "--enable-vowel-open-boost",
        action="store_true",
        default=True,
        help="Boost lip opening in vowel viseme templates (AA/E/I/O/U).",
    )
    parser.add_argument(
        "--disable-vowel-open-boost",
        dest="enable_vowel_open_boost",
        action="store_false",
    )
    parser.add_argument(
        "--vowel-open-base-delta",
        type=float,
        default=DEFAULT_VOWEL_OPEN_BASE_DELTA,
        help="Base additive lip-opening delta applied to vowel visemes.",
    )
    parser.add_argument(
        "--vowel-open-gains-json",
        default="",
        help="JSON object overriding per-vowel opening gains.",
    )
    parser.add_argument(
        "--enable-eye-tamed-preset",
        action="store_true",
        default=DEFAULT_ENABLE_EYE_TAMED_PRESET,
        help="Apply eye/upper-face damping preset to reduce excessive eye motion.",
    )
    parser.add_argument(
        "--disable-eye-tamed-preset",
        dest="enable_eye_tamed_preset",
        action="store_false",
        help="Disable eye/upper-face damping preset.",
    )
    parser.add_argument(
        "--eye-soft-factor",
        type=float,
        default=DEFAULT_EYE_SOFT_FACTOR,
        help="Soft damping factor [0..1] for upper-face indices.",
    )
    parser.add_argument(
        "--eye-hard-factor",
        type=float,
        default=DEFAULT_EYE_HARD_FACTOR,
        help="Hard damping factor [0..1] for eye-sensitive indices.",
    )
    parser.add_argument(
        "--eye-hard-dy-min",
        type=float,
        default=DEFAULT_EYE_HARD_DY_MIN,
        help="Minimum allowed eye vertical delta for hard indices.",
    )
    parser.add_argument(
        "--eye-hard-dy-max",
        type=float,
        default=DEFAULT_EYE_HARD_DY_MAX,
        help="Maximum allowed eye vertical delta for hard indices.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--auto-start-container", action="store_true", default=True)
    parser.add_argument("--no-auto-start-container", dest="auto_start_container", action="store_false")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """
    Resolve path string relative to project root when needed.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    """
    Convert absolute path to project-relative POSIX representation.
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
    Run command and raise with full command if it fails.
    """
    print(f"[cmd] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def run_command_capture(command: list[str]) -> str:
    """
    Run command and return captured stdout.
    """
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} stderr={stderr}")
    return (result.stdout or "").strip()


def is_container_running(container_name: str) -> bool:
    """
    Return True when Docker container is running.
    """
    try:
        stdout = run_command_capture(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
        )
    except RuntimeError:
        return False
    return stdout.strip().lower() == "true"


def ensure_container_running(container_name: str, service_name: str, auto_start: bool) -> None:
    """
    Ensure target Docker container is running before build.
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
    Read a JSON object from disk.
    """
    with path_value.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_value}")
    return payload


def parse_vowel_open_gain_map(raw_value: str) -> dict[str, float]:
    """
    Parse optional per-vowel gain overrides.
    """
    gain_map = dict(DEFAULT_VOWEL_OPEN_GAIN_BY_VISEME)
    if not str(raw_value).strip():
        return gain_map
    payload = json.loads(str(raw_value))
    if not isinstance(payload, dict):
        raise ValueError("vowel-open-gains-json must be a JSON object.")
    valid_keys = set(VOWEL_VISEME_KEYS)
    for key, value in payload.items():
        viseme = str(key).strip()
        if viseme not in valid_keys:
            raise ValueError(f"Invalid vowel key in gains map: {viseme}")
        gain_map[viseme] = float(value)
    return gain_map


def load_viseme_entries(audio_manifest_path: Path) -> tuple[str, list[dict[str, str]]]:
    """
    Load and validate viseme entries from audio manifest.
    """
    payload = read_json(audio_manifest_path)
    base_image = str(payload.get("baseImage", "")).strip()
    visemes = payload.get("visemes")
    if not isinstance(visemes, list) or not visemes:
        raise ValueError(f"Manifest has no visemes: {audio_manifest_path}")

    entries: list[dict[str, str]] = []
    for item in visemes:
        if not isinstance(item, dict):
            raise ValueError("Invalid viseme entry type in manifest.")
        viseme = str(item.get("viseme", "")).strip()
        audio_rel = str(item.get("audio", "")).strip()
        from_viseme = str(item.get("fromViseme", "")).strip()
        to_viseme = str(item.get("toViseme", "")).strip()
        if not viseme:
            raise ValueError("Viseme entry missing 'viseme'.")
        if not audio_rel:
            raise ValueError(f"Viseme {viseme} missing 'audio'.")
        entries.append(
            {
                "viseme": viseme,
                "audio": audio_rel,
                "fromViseme": from_viseme,
                "toViseme": to_viseme,
            }
        )
    return base_image, entries


def extract_template_stats(pkl_path: Path) -> dict[str, Any]:
    """
    Extract lightweight metadata from generated pkl.
    """
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return {"nFrames": 0, "fps": 0, "hasMotion": False}

    motion = payload.get("motion")
    c_lip = payload.get("c_lip_lst")
    c_eyes = payload.get("c_eyes_lst")
    return {
        "nFrames": int(payload.get("n_frames", 0) or 0),
        "fps": float(payload.get("output_fps", 0) or 0),
        "hasMotion": isinstance(motion, list) and len(motion) > 0,
        "lipFrames": len(c_lip) if isinstance(c_lip, list) else 0,
        "eyeFrames": len(c_eyes) if isinstance(c_eyes, list) else 0,
    }


def interpolate_ndarray(value_a: Any, value_b: Any, alpha: float) -> np.ndarray:
    """
    Linearly interpolate two tensor-like values and return float32 ndarray.
    """
    array_a = np.asarray(value_a, dtype=np.float32)
    array_b = np.asarray(value_b, dtype=np.float32)
    return ((1.0 - alpha) * array_a + alpha * array_b).astype(np.float32)


def clone_motion_frame(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Deep-clone one motion frame into float32 numpy arrays.
    """
    cloned: dict[str, np.ndarray] = {}
    for key, value in frame.items():
        cloned[key] = np.asarray(value, dtype=np.float32).copy()
    return cloned


def interpolate_motion_frame(frame_a: dict[str, Any], frame_b: dict[str, Any], alpha: float) -> dict[str, np.ndarray]:
    """
    Interpolate two motion frames key by key.
    """
    all_keys = set(frame_a.keys()) | set(frame_b.keys())
    interpolated: dict[str, np.ndarray] = {}
    for key in all_keys:
        value_a = frame_a.get(key)
        value_b = frame_b.get(key)
        if value_a is None and value_b is None:
            continue
        if value_a is None:
            interpolated[key] = np.asarray(value_b, dtype=np.float32).copy()
            continue
        if value_b is None:
            interpolated[key] = np.asarray(value_a, dtype=np.float32).copy()
            continue
        interpolated[key] = interpolate_ndarray(value_a, value_b, alpha)
    return interpolated


def upsample_scalar_series(series: Any, factor: int) -> list[float]:
    """
    Upsample optional scalar time series with linear interpolation.
    """
    if factor <= 1:
        return list(series) if isinstance(series, list) else []
    if not isinstance(series, list) or not series:
        return []
    if len(series) == 1:
        return [float(series[0])]

    values = [float(item) for item in series]
    upsampled: list[float] = []
    for index in range(len(values) - 1):
        start_value = values[index]
        end_value = values[index + 1]
        upsampled.append(start_value)
        for step in range(1, factor):
            alpha = float(step) / float(factor)
            upsampled.append((1.0 - alpha) * start_value + alpha * end_value)
    upsampled.append(values[-1])
    return upsampled


def upsample_motion_payload(payload: dict[str, Any], factor: int) -> dict[str, Any]:
    """
    Upsample motion frames in a pkl payload by the given temporal factor.
    """
    safe_factor = max(1, int(factor))
    if safe_factor == 1:
        return payload

    motion = payload.get("motion")
    if not isinstance(motion, list) or len(motion) < 2:
        return payload

    upsampled_motion: list[dict[str, np.ndarray]] = []
    for index in range(len(motion) - 1):
        frame_a = motion[index]
        frame_b = motion[index + 1]
        if not isinstance(frame_a, dict) or not isinstance(frame_b, dict):
            raise ValueError("Invalid motion frame format for temporal upsampling.")
        upsampled_motion.append(clone_motion_frame(frame_a))
        for step in range(1, safe_factor):
            alpha = float(step) / float(safe_factor)
            upsampled_motion.append(interpolate_motion_frame(frame_a, frame_b, alpha))
    upsampled_motion.append(clone_motion_frame(motion[-1]))

    output_fps = float(payload.get("output_fps", 0) or 0)
    processed = copy.deepcopy(payload)
    processed["motion"] = upsampled_motion
    processed["n_frames"] = len(upsampled_motion)
    if output_fps > 0:
        processed["output_fps"] = output_fps * float(safe_factor)
    processed["c_eyes_lst"] = upsample_scalar_series(payload.get("c_eyes_lst"), safe_factor)
    processed["c_lip_lst"] = upsample_scalar_series(payload.get("c_lip_lst"), safe_factor)
    return processed


def get_lip_energy(exp_array: np.ndarray) -> float:
    """
    Estimate lip opening energy from expression tensor.
    """
    if exp_array.ndim == 3:
        frame_exp = exp_array[0]
    else:
        frame_exp = exp_array
    frame_exp = np.asarray(frame_exp, dtype=np.float32)
    positive = lambda value: float(max(0.0, value))
    energy = (
        positive(frame_exp[19, 1])
        + positive(frame_exp[20, 1])
        + 0.45 * positive(frame_exp[14, 1])
        + 0.20 * positive(frame_exp[17, 1])
    )
    return energy


def apply_vowel_open_boost_to_payload(
    payload: dict[str, Any],
    viseme_key: str,
    base_delta: float,
    gain_map: dict[str, float],
) -> dict[str, Any]:
    """
    Apply lip-opening boost to vowel viseme motion payload.
    """
    viseme = str(viseme_key).strip()
    if viseme not in VOWEL_VISEME_KEYS:
        return payload
    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        return payload
    safe_base_delta = max(0.0, float(base_delta))
    viseme_gain = float(gain_map.get(viseme, 1.0))
    if safe_base_delta <= 0.0 or viseme_gain <= 0.0:
        return payload

    energies = []
    for frame in motion:
        if not isinstance(frame, dict):
            continue
        exp_value = frame.get("exp")
        if exp_value is None:
            continue
        energies.append(get_lip_energy(np.asarray(exp_value, dtype=np.float32)))
    if not energies:
        return payload
    max_energy = max(energies)
    norm_den = max(max_energy, 1e-6)
    target_delta = safe_base_delta * viseme_gain

    processed = copy.deepcopy(payload)
    processed_motion = processed.get("motion")
    if not isinstance(processed_motion, list):
        return payload
    for frame_index, frame in enumerate(processed_motion):
        if not isinstance(frame, dict) or "exp" not in frame:
            continue
        exp_array = np.asarray(frame["exp"], dtype=np.float32).copy()
        frame_energy = get_lip_energy(exp_array)
        dynamic = frame_energy / norm_den
        envelope = 0.35 + 0.65 * dynamic
        boost = target_delta * envelope
        if exp_array.ndim == 3:
            exp_array[0, 19, 1] += boost * 1.00
            exp_array[0, 20, 1] += boost * 0.85
            exp_array[0, 14, 1] += boost * 0.60
            exp_array[0, 17, 1] += boost * 0.35
            exp_array[0, 19, 2] += boost * 0.22
            exp_array[0, 20, 2] += boost * 0.16
        else:
            exp_array[19, 1] += boost * 1.00
            exp_array[20, 1] += boost * 0.85
            exp_array[14, 1] += boost * 0.60
            exp_array[17, 1] += boost * 0.35
            exp_array[19, 2] += boost * 0.22
            exp_array[20, 2] += boost * 0.16
        frame["exp"] = exp_array
    return processed


def resolve_vowel_boost_key(viseme_key: str, from_viseme: str, to_viseme: str) -> str:
    """
    Resolve which vowel key should receive lip-opening boost.
    """
    viseme = str(viseme_key).strip()
    from_key = str(from_viseme).strip()
    to_key = str(to_viseme).strip()

    if viseme in VOWEL_VISEME_KEYS:
        return viseme
    if "_to_" in viseme:
        return ""
    if from_key == to_key and from_key in VOWEL_VISEME_KEYS:
        return from_key
    token = viseme.split("_", 1)[0].strip()
    if token in VOWEL_VISEME_KEYS and not from_key and not to_key:
        return token
    return ""


def apply_eye_tamed_preset_to_payload(
    payload: dict[str, Any],
    soft_factor: float,
    hard_factor: float,
    hard_dy_min: float,
    hard_dy_max: float,
) -> dict[str, Any]:
    """
    Damp upper-face and eye channels relative to frame-0 baseline.
    """
    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        return payload
    first_frame = motion[0]
    if not isinstance(first_frame, dict) or "exp" not in first_frame:
        return payload

    base_exp = np.asarray(first_frame["exp"], dtype=np.float32).reshape(21, 3).copy()
    safe_soft = float(np.clip(soft_factor, 0.0, 1.0))
    safe_hard = float(np.clip(hard_factor, 0.0, 1.0))
    safe_min = float(min(hard_dy_min, hard_dy_max))
    safe_max = float(max(hard_dy_min, hard_dy_max))

    processed = copy.deepcopy(payload)
    processed_motion = processed.get("motion")
    if not isinstance(processed_motion, list):
        return payload

    for frame in processed_motion:
        if not isinstance(frame, dict) or "exp" not in frame:
            continue
        exp_array = np.asarray(frame["exp"], dtype=np.float32).reshape(21, 3).copy()
        for index in EYE_TAMED_SOFT_INDICES:
            exp_array[index, :] = base_exp[index, :] + (exp_array[index, :] - base_exp[index, :]) * safe_soft
        for index in EYE_TAMED_HARD_INDICES:
            exp_array[index, :] = base_exp[index, :] + (exp_array[index, :] - base_exp[index, :]) * safe_hard
            delta_y = exp_array[index, 1] - base_exp[index, 1]
            exp_array[index, 1] = base_exp[index, 1] + float(np.clip(delta_y, safe_min, safe_max))
        frame["exp"] = exp_array.reshape(1, 21, 3)
    return processed


def apply_motion_postprocess(
    pkl_path: Path,
    upsample_factor: int,
    viseme_key: str,
    from_viseme: str,
    to_viseme: str,
    enable_vowel_open_boost: bool,
    vowel_open_base_delta: float,
    vowel_open_gain_map: dict[str, float],
    enable_eye_tamed_preset: bool,
    eye_soft_factor: float,
    eye_hard_factor: float,
    eye_hard_dy_min: float,
    eye_hard_dy_max: float,
) -> None:
    """
    Apply optional postprocess steps to generated motion pkl.
    """
    safe_factor = max(1, int(upsample_factor))
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid pkl payload type: {type(payload)}")
    processed = payload
    if safe_factor > 1:
        processed = upsample_motion_payload(processed, safe_factor)
    if enable_eye_tamed_preset:
        processed = apply_eye_tamed_preset_to_payload(
            payload=processed,
            soft_factor=eye_soft_factor,
            hard_factor=eye_hard_factor,
            hard_dy_min=eye_hard_dy_min,
            hard_dy_max=eye_hard_dy_max,
        )
    if enable_vowel_open_boost:
        boost_viseme_key = resolve_vowel_boost_key(
            viseme_key=viseme_key,
            from_viseme=from_viseme,
            to_viseme=to_viseme,
        )
        processed = apply_vowel_open_boost_to_payload(
            payload=processed,
            viseme_key=boost_viseme_key,
            base_delta=vowel_open_base_delta,
            gain_map=vowel_open_gain_map,
        )
    with pkl_path.open("wb") as handle:
        pickle.dump(processed, handle)


def build_local_command(
    python_executable: str,
    script_path: Path,
    faster_repo_dir: Path,
    cfg_path: Path,
    audio_path: Path,
    output_pkl_path: Path,
    seed: int,
) -> list[str]:
    """
    Build local host command for one viseme pkl.
    """
    return [
        python_executable,
        str(script_path),
        "--faster-repo-dir",
        str(faster_repo_dir),
        "--cfg",
        str(cfg_path),
        "--driving-audio",
        str(audio_path),
        "--output-pkl",
        str(output_pkl_path),
        "--seed",
        str(int(seed)),
    ]


def build_docker_command(
    container_name: str,
    container_python: str,
    script_path: Path,
    faster_repo_dir: Path,
    cfg_path: Path,
    audio_path: Path,
    output_pkl_path: Path,
    seed: int,
) -> list[str]:
    """
    Build docker exec command for one viseme pkl.
    """
    return [
        "docker",
        "exec",
        container_name,
        container_python,
        to_container_path(script_path),
        "--faster-repo-dir",
        to_container_path(faster_repo_dir),
        "--cfg",
        to_container_path(cfg_path),
        "--driving-audio",
        to_container_path(audio_path),
        "--output-pkl",
        to_container_path(output_pkl_path),
        "--seed",
        str(int(seed)),
    ]


def write_output_manifest(
    output_manifest_path: Path,
    audio_manifest_path: Path,
    base_image_rel: str,
    output_dir: Path,
    results: list[BuildResult],
) -> None:
    """
    Write viseme motion manifest with generated pkl metadata.
    """
    payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceAudioManifest": to_project_relative(audio_manifest_path),
        "baseImage": base_image_rel,
        "outputDir": to_project_relative(output_dir),
        "visemeCount": len(results),
        "visemes": [
            {
                "viseme": item.viseme,
                "audio": to_project_relative(item.audio_path),
                "pkl": to_project_relative(item.pkl_path),
                "fromViseme": item.from_viseme,
                "toViseme": item.to_viseme,
                "elapsedSec": round(item.elapsed_seconds, 3),
                "stats": item.stats,
            }
            for item in results
        ],
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_single_entry(
    entry: dict[str, str],
    args: argparse.Namespace,
    script_path: Path,
    faster_repo_dir: Path,
    cfg_path: Path,
    output_dir: Path,
    upsample_factor: int,
    enable_vowel_open_boost: bool,
    vowel_open_base_delta: float,
    vowel_open_gain_map: dict[str, float],
    enable_eye_tamed_preset: bool,
    eye_soft_factor: float,
    eye_hard_factor: float,
    eye_hard_dy_min: float,
    eye_hard_dy_max: float,
) -> BuildResult:
    """
    Build one viseme pkl entry.
    """
    viseme = entry["viseme"]
    audio_path = resolve_path(entry["audio"])
    from_viseme = str(entry.get("fromViseme", "")).strip()
    to_viseme = str(entry.get("toViseme", "")).strip()
    output_pkl_path = output_dir / f"{viseme}.pkl"

    if not audio_path.exists():
        raise FileNotFoundError(f"missing audio for {viseme}: {audio_path}")

    if output_pkl_path.exists() and not args.overwrite:
        stats = extract_template_stats(output_pkl_path)
        print(f"[skip] {viseme} already exists: {output_pkl_path}")
        return BuildResult(
            viseme=viseme,
            audio_path=audio_path,
            pkl_path=output_pkl_path,
            elapsed_seconds=0.0,
            stats=stats,
            from_viseme=from_viseme,
            to_viseme=to_viseme,
        )

    started_at = time.perf_counter()
    if args.runtime == RUNTIME_DOCKER:
        command = build_docker_command(
            container_name=str(args.docker_container),
            container_python=str(args.docker_python),
            script_path=script_path,
            faster_repo_dir=faster_repo_dir,
            cfg_path=cfg_path,
            audio_path=audio_path,
            output_pkl_path=output_pkl_path,
            seed=int(args.seed),
        )
    else:
        command = build_local_command(
            python_executable=str(args.python_executable),
            script_path=script_path,
            faster_repo_dir=faster_repo_dir,
            cfg_path=cfg_path,
            audio_path=audio_path,
            output_pkl_path=output_pkl_path,
            seed=int(args.seed),
        )

    run_command(command)
    apply_motion_postprocess(
        pkl_path=output_pkl_path,
        upsample_factor=upsample_factor,
        viseme_key=viseme,
        from_viseme=from_viseme,
        to_viseme=to_viseme,
        enable_vowel_open_boost=enable_vowel_open_boost,
        vowel_open_base_delta=vowel_open_base_delta,
        vowel_open_gain_map=vowel_open_gain_map,
        enable_eye_tamed_preset=enable_eye_tamed_preset,
        eye_soft_factor=eye_soft_factor,
        eye_hard_factor=eye_hard_factor,
        eye_hard_dy_min=eye_hard_dy_min,
        eye_hard_dy_max=eye_hard_dy_max,
    )
    elapsed = time.perf_counter() - started_at
    stats = extract_template_stats(output_pkl_path)
    stats["motionUpsampleFactor"] = upsample_factor
    print(
        f"[ok] {viseme} -> {output_pkl_path} "
        f"(frames={stats.get('nFrames', 0)} fps={stats.get('fps', 0)} up={upsample_factor} sec={elapsed:.2f})"
    )
    return BuildResult(
        viseme=viseme,
        audio_path=audio_path,
        pkl_path=output_pkl_path,
        elapsed_seconds=elapsed,
        stats=stats,
        from_viseme=from_viseme,
        to_viseme=to_viseme,
    )


def main() -> None:
    """
    Build all viseme motion templates from manifest.
    """
    args = parse_args()
    audio_manifest_path = resolve_path(args.audio_manifest)
    output_dir = resolve_path(args.output_dir)
    output_manifest_path = resolve_path(args.output_manifest)
    faster_repo_dir = resolve_path(args.faster_repo_dir)
    cfg_path = resolve_path(args.cfg)
    script_path = resolve_path(args.audio_to_pkl_script)

    if not audio_manifest_path.exists():
        raise FileNotFoundError(f"Audio manifest not found: {audio_manifest_path}")
    if not faster_repo_dir.exists():
        raise FileNotFoundError(f"Faster repo not found: {faster_repo_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"Audio->PKL script not found: {script_path}")
    upsample_factor = max(1, int(args.motion_upsample_factor))
    parallel_jobs = max(1, int(args.jobs))
    enable_vowel_open_boost = bool(args.enable_vowel_open_boost)
    vowel_open_base_delta = max(0.0, float(args.vowel_open_base_delta))
    vowel_open_gain_map = parse_vowel_open_gain_map(str(args.vowel_open_gains_json))
    enable_eye_tamed_preset = bool(args.enable_eye_tamed_preset)
    eye_soft_factor = float(args.eye_soft_factor)
    eye_hard_factor = float(args.eye_hard_factor)
    eye_hard_dy_min = float(args.eye_hard_dy_min)
    eye_hard_dy_max = float(args.eye_hard_dy_max)

    if args.runtime == RUNTIME_DOCKER:
        ensure_container_running(
            container_name=str(args.docker_container),
            service_name=str(args.docker_service),
            auto_start=bool(args.auto_start_container),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    base_image_rel, entries = load_viseme_entries(audio_manifest_path)
    viseme_order = {entry["viseme"]: index for index, entry in enumerate(entries)}
    print(f"[info] entries={len(entries)} jobs={parallel_jobs}")

    built_results: list[BuildResult] = []
    failures: list[str] = []
    if parallel_jobs == 1:
        for entry in entries:
            viseme = entry["viseme"]
            try:
                result = build_single_entry(
                    entry=entry,
                    args=args,
                    script_path=script_path,
                    faster_repo_dir=faster_repo_dir,
                    cfg_path=cfg_path,
                    output_dir=output_dir,
                    upsample_factor=upsample_factor,
                    enable_vowel_open_boost=enable_vowel_open_boost,
                    vowel_open_base_delta=vowel_open_base_delta,
                    vowel_open_gain_map=vowel_open_gain_map,
                    enable_eye_tamed_preset=enable_eye_tamed_preset,
                    eye_soft_factor=eye_soft_factor,
                    eye_hard_factor=eye_hard_factor,
                    eye_hard_dy_min=eye_hard_dy_min,
                    eye_hard_dy_max=eye_hard_dy_max,
                )
                built_results.append(result)
            except Exception as exc:  # noqa: BLE001
                message = f"[error] {viseme} build failed: {exc}"
                print(message)
                failures.append(message)
                if not args.continue_on_error:
                    raise
    else:
        future_to_viseme: dict[concurrent.futures.Future[BuildResult], str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            for entry in entries:
                future = executor.submit(
                    build_single_entry,
                    entry,
                    args,
                    script_path,
                    faster_repo_dir,
                    cfg_path,
                    output_dir,
                    upsample_factor,
                    enable_vowel_open_boost,
                    vowel_open_base_delta,
                    vowel_open_gain_map,
                    enable_eye_tamed_preset,
                    eye_soft_factor,
                    eye_hard_factor,
                    eye_hard_dy_min,
                    eye_hard_dy_max,
                )
                future_to_viseme[future] = entry["viseme"]

            for future in concurrent.futures.as_completed(future_to_viseme):
                viseme = future_to_viseme[future]
                try:
                    result = future.result()
                    built_results.append(result)
                except Exception as exc:  # noqa: BLE001
                    message = f"[error] {viseme} build failed: {exc}"
                    print(message)
                    failures.append(message)
                    if not args.continue_on_error:
                        for pending in future_to_viseme:
                            pending.cancel()
                        raise

    built_results.sort(key=lambda item: viseme_order.get(item.viseme, 10**9))

    write_output_manifest(
        output_manifest_path=output_manifest_path,
        audio_manifest_path=audio_manifest_path,
        base_image_rel=base_image_rel,
        output_dir=output_dir,
        results=built_results,
    )

    print(f"[ok] motion manifest -> {output_manifest_path}")
    print(f"[ok] generated viseme pkls -> {len(built_results)}")
    if failures:
        print(f"[warn] failures -> {len(failures)}")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
