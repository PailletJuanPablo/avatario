"""
Build grouped viseme/transition clip libraries from per-entry dataset manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = "output_fasterliveportrait/liveportrait_dataset_poc"
DEFAULT_OUTPUT_ROOT = "output_fasterliveportrait/liveportrait_dataset_poc/clip_library"
CONTINUITY_STATE_DECIMALS = 6
CONTINUITY_SCALE_EPSILON = 1e-6
CONTINUITY_PROFILE = "motion_state_v1"
MOTION_EXP_FLAT_DIM = 63
MOTION_POSE_DIM = 7
MOTION_STATE_DIM = MOTION_EXP_FLAT_DIM + MOTION_POSE_DIM


def build_motion_feature_names() -> list[str]:
    feature_names: list[str] = []
    for exp_index in range(21):
        for axis_name in ("x", "y", "z"):
            feature_names.append(f"exp_{exp_index:02d}_{axis_name}")
    feature_names.extend(("pitch", "yaw", "roll", "scale", "t_x", "t_y", "t_z"))
    return feature_names


MOTION_FEATURE_NAMES = build_motion_feature_names()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group generated viseme and transition clips into centralized folders "
            "and emit manifests."
        )
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_project_relative(path_value: Path) -> str:
    resolved = path_value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sanitize_slug(raw_value: str, max_length: int = 80) -> str:
    normalized = unicodedata.normalize("NFKD", str(raw_value)).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    if not cleaned:
        cleaned = "item"
    return cleaned[:max_length]


def list_entry_manifests(source_root: Path) -> list[Path]:
    manifests: list[Path] = []
    for path in sorted(source_root.glob("*/manifest.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        if "datasetIndex" not in payload or "visemes" not in payload or "transitions" not in payload:
            continue
        manifests.append(path)
    return manifests


def copy_clip(source_path: Path, target_path: Path, overwrite: bool) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Source clip not found: {source_path}")
    if target_path.exists() and not overwrite:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def ensure_numpy_pickle_compatibility() -> None:
    """
    Keep backward compatibility with pickles that reference numpy._core modules.
    """
    numpy_core_numeric = sys.modules.get("numpy.core.numeric")
    if numpy_core_numeric is not None and "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = numpy_core_numeric


def safe_float(raw_value: Any, fallback: float = 0.0) -> float:
    """
    Convert unknown scalar into finite float with fallback.
    """
    try:
        value = float(raw_value) if raw_value is not None else float(fallback)
    except (TypeError, ValueError):
        value = float(fallback)
    if math.isfinite(value):
        return value
    return float(fallback)


def load_entry_motion_context(
    entry_payload: dict[str, Any],
    motion_context_by_pkl_path: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """
    Load motion payload once per pkl and cache parsed state for continuity extraction.
    """
    motion_pkl_text = str(entry_payload.get("motionPkl", "")).strip()
    if not motion_pkl_text:
        return None
    motion_pkl_path = resolve_path(motion_pkl_text)
    cache_key = str(motion_pkl_path)
    if cache_key in motion_context_by_pkl_path:
        return motion_context_by_pkl_path[cache_key]
    if not motion_pkl_path.exists():
        motion_context_by_pkl_path[cache_key] = None
        return None
    ensure_numpy_pickle_compatibility()
    with motion_pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        motion_context_by_pkl_path[cache_key] = None
        return None
    motion_frames = payload.get("motion")
    if not isinstance(motion_frames, list) or not motion_frames:
        motion_context_by_pkl_path[cache_key] = None
        return None
    fps_value = safe_float(payload.get("output_fps"), fallback=25.0)
    if fps_value <= 0.0:
        fps_value = 25.0
    motion_context = {
        "path": motion_pkl_path,
        "fps": fps_value,
        "motionFrames": motion_frames,
        "totalFrames": len(motion_frames),
    }
    motion_context_by_pkl_path[cache_key] = motion_context
    return motion_context


def extract_motion_state_vector(frame_payload: dict[str, Any]) -> np.ndarray | None:
    """
    Flatten one motion frame into a numeric continuity state vector.
    """
    if not isinstance(frame_payload, dict):
        return None
    exp_array = np.asarray(frame_payload.get("exp"), dtype=np.float32).reshape(-1)
    if int(exp_array.size) != MOTION_EXP_FLAT_DIM:
        return None
    pitch_array = np.asarray(frame_payload.get("pitch"), dtype=np.float32).reshape(-1)
    yaw_array = np.asarray(frame_payload.get("yaw"), dtype=np.float32).reshape(-1)
    roll_array = np.asarray(frame_payload.get("roll"), dtype=np.float32).reshape(-1)
    scale_array = np.asarray(frame_payload.get("scale"), dtype=np.float32).reshape(-1)
    translation_array = np.asarray(frame_payload.get("t"), dtype=np.float32).reshape(-1)
    if (
        pitch_array.size < 1
        or yaw_array.size < 1
        or roll_array.size < 1
        or scale_array.size < 1
        or translation_array.size < 3
    ):
        return None
    pose_vector = np.asarray(
        [
            float(pitch_array[0]),
            float(yaw_array[0]),
            float(roll_array[0]),
            float(scale_array[0]),
            float(translation_array[0]),
            float(translation_array[1]),
            float(translation_array[2]),
        ],
        dtype=np.float32,
    )
    state_vector = np.concatenate((exp_array.astype(np.float32, copy=False), pose_vector), axis=0)
    if int(state_vector.size) != MOTION_STATE_DIM:
        return None
    if not np.isfinite(state_vector).all():
        return None
    return state_vector.astype(np.float32, copy=False)


def serialize_state_vector(state_vector: np.ndarray) -> list[float]:
    """
    Convert numeric state vector into compact JSON-compatible values.
    """
    return [round(float(value), CONTINUITY_STATE_DECIMALS) for value in state_vector.tolist()]


def resolve_clip_frame_window(
    clip_start_sec: float,
    clip_end_sec: float,
    fps_value: float,
    total_frames: int,
) -> tuple[int, int]:
    """
    Convert clip time window into inclusive frame indices for start and end.
    """
    safe_total_frames = max(1, int(total_frames))
    safe_fps = max(CONTINUITY_SCALE_EPSILON, float(fps_value))
    safe_start_sec = max(0.0, float(clip_start_sec))
    safe_end_sec = max(safe_start_sec + CONTINUITY_SCALE_EPSILON, float(clip_end_sec))
    start_frame = int(math.floor(safe_start_sec * safe_fps + 1e-9))
    end_frame_exclusive = int(math.ceil(safe_end_sec * safe_fps - 1e-9))
    start_frame = max(0, min(start_frame, safe_total_frames - 1))
    end_frame_exclusive = max(start_frame + 1, min(end_frame_exclusive, safe_total_frames))
    end_frame_inclusive = max(start_frame, end_frame_exclusive - 1)
    return start_frame, end_frame_inclusive


def build_clip_continuity_fields(
    segment_payload: dict[str, Any],
    motion_context: dict[str, Any] | None,
    continuity_state_samples: list[np.ndarray],
    continuity_velocity_samples: list[np.ndarray],
) -> dict[str, Any]:
    """
    Build continuity state payload for one clip segment.
    """
    if motion_context is None:
        return {}
    motion_frames = motion_context.get("motionFrames")
    if not isinstance(motion_frames, list) or not motion_frames:
        return {}
    fps_value = safe_float(motion_context.get("fps"), fallback=25.0)
    total_frames = int(motion_context.get("totalFrames", len(motion_frames)))
    clip_start_sec = safe_float(segment_payload.get("clipStartSec"), fallback=0.0)
    clip_end_sec = safe_float(segment_payload.get("clipEndSec"), fallback=clip_start_sec + 0.04)
    start_frame_index, end_frame_index = resolve_clip_frame_window(
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        fps_value=fps_value,
        total_frames=total_frames,
    )
    start_frame_payload = motion_frames[start_frame_index]
    end_frame_payload = motion_frames[end_frame_index]
    start_state = extract_motion_state_vector(start_frame_payload)
    end_state = extract_motion_state_vector(end_frame_payload)
    if start_state is None or end_state is None:
        return {}
    start_forward_index = min(end_frame_index, start_frame_index + 1)
    end_backward_index = max(start_frame_index, end_frame_index - 1)
    start_forward_payload = motion_frames[start_forward_index]
    end_backward_payload = motion_frames[end_backward_index]
    start_forward_state = extract_motion_state_vector(start_forward_payload)
    end_backward_state = extract_motion_state_vector(end_backward_payload)
    if start_forward_state is None or end_backward_state is None:
        return {}
    start_velocity = (start_forward_state - start_state).astype(np.float32, copy=False)
    end_velocity = (end_state - end_backward_state).astype(np.float32, copy=False)
    continuity_state_samples.append(start_state)
    continuity_state_samples.append(end_state)
    continuity_velocity_samples.append(start_velocity)
    continuity_velocity_samples.append(end_velocity)
    return {
        "continuityProfile": CONTINUITY_PROFILE,
        "continuityStartFrame": int(start_frame_index),
        "continuityEndFrame": int(end_frame_index),
        "continuityStartState": serialize_state_vector(start_state),
        "continuityEndState": serialize_state_vector(end_state),
        "continuityStartVelocity": serialize_state_vector(start_velocity),
        "continuityEndVelocity": serialize_state_vector(end_velocity),
    }


def build_continuity_summary(
    continuity_state_samples: list[np.ndarray],
    continuity_velocity_samples: list[np.ndarray],
) -> dict[str, Any] | None:
    """
    Compute global normalization stats for continuity-aware runtime selection.
    """
    if not continuity_state_samples or not continuity_velocity_samples:
        return None
    stacked_samples = np.stack(continuity_state_samples, axis=0).astype(np.float32, copy=False)
    stacked_velocity_samples = np.stack(continuity_velocity_samples, axis=0).astype(np.float32, copy=False)
    state_mean_vector = stacked_samples.mean(axis=0)
    state_std_vector = stacked_samples.std(axis=0)
    state_std_vector = np.maximum(state_std_vector, CONTINUITY_SCALE_EPSILON)
    velocity_mean_vector = stacked_velocity_samples.mean(axis=0)
    velocity_std_vector = stacked_velocity_samples.std(axis=0)
    velocity_std_vector = np.maximum(velocity_std_vector, CONTINUITY_SCALE_EPSILON)
    return {
        "profile": CONTINUITY_PROFILE,
        "featureCount": int(stacked_samples.shape[1]),
        "stateSampleCount": int(stacked_samples.shape[0]),
        "velocitySampleCount": int(stacked_velocity_samples.shape[0]),
        "featureOrder": MOTION_FEATURE_NAMES,
        "stateMean": serialize_state_vector(state_mean_vector),
        "stateStd": serialize_state_vector(state_std_vector),
        "velocityMean": serialize_state_vector(velocity_mean_vector),
        "velocityStd": serialize_state_vector(velocity_std_vector),
        # Backward-compatible aliases.
        "mean": serialize_state_vector(state_mean_vector),
        "std": serialize_state_vector(state_std_vector),
    }


def build_clip_filename(dataset_index: int, dataset_id: str, segment_index: int, phrase: str) -> str:
    return (
        f"{int(dataset_index):04d}_"
        f"{sanitize_slug(dataset_id, max_length=32)}_"
        f"{int(segment_index):04d}_"
        f"{sanitize_slug(phrase, max_length=64)}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_viseme_clips(
    entry_payload: dict[str, Any],
    output_root: Path,
    overwrite: bool,
    viseme_groups: dict[str, list[dict[str, Any]]],
    motion_context_by_pkl_path: dict[str, dict[str, Any] | None],
    continuity_state_samples: list[np.ndarray],
    continuity_velocity_samples: list[np.ndarray],
) -> None:
    dataset_index = int(entry_payload.get("datasetIndex", 0))
    dataset_id = str(entry_payload.get("id", ""))
    phrase = str(entry_payload.get("phrase", ""))
    entry_dir = str(entry_payload.get("entryDir", ""))
    motion_context = load_entry_motion_context(
        entry_payload=entry_payload,
        motion_context_by_pkl_path=motion_context_by_pkl_path,
    )

    for item in ensure_list(entry_payload.get("visemes")):
        if not isinstance(item, dict):
            continue
        viseme_key = str(item.get("viseme", "")).strip()
        if not viseme_key:
            continue
        segment_index = int(item.get("index", 0))
        clip_org_source = resolve_path(str(item.get("clipOrg", "")))
        clip_crop_source = resolve_path(str(item.get("clipCrop", "")))

        basename = build_clip_filename(
            dataset_index=dataset_index,
            dataset_id=dataset_id,
            segment_index=segment_index,
            phrase=phrase,
        )
        group_dir = output_root / "visemes" / sanitize_slug(viseme_key, max_length=40)
        clip_org_target = group_dir / "clips" / "org" / f"{basename}.mp4"
        clip_crop_target = group_dir / "clips" / "crop" / f"{basename}.mp4"

        copy_clip(clip_org_source, clip_org_target, overwrite=overwrite)
        copy_clip(clip_crop_source, clip_crop_target, overwrite=overwrite)

        clip_payload = {
            "datasetIndex": dataset_index,
            "id": dataset_id,
            "phrase": phrase,
            "entryDir": entry_dir,
            "segmentIndex": segment_index,
            "char": str(item.get("char", "")),
            "startSec": float(item.get("startSec", 0.0) or 0.0),
            "endSec": float(item.get("endSec", 0.0) or 0.0),
            "durationSec": float(item.get("durationSec", 0.0) or 0.0),
            "clipStartSec": float(item.get("clipStartSec", 0.0) or 0.0),
            "clipEndSec": float(item.get("clipEndSec", 0.0) or 0.0),
            "clipDurationSec": float(item.get("clipDurationSec", 0.0) or 0.0),
            "sourceClipOrg": to_project_relative(clip_org_source),
            "sourceClipCrop": to_project_relative(clip_crop_source),
            "clipOrg": to_project_relative(clip_org_target),
            "clipCrop": to_project_relative(clip_crop_target),
        }
        clip_payload.update(
            build_clip_continuity_fields(
                segment_payload=clip_payload,
                motion_context=motion_context,
                continuity_state_samples=continuity_state_samples,
                continuity_velocity_samples=continuity_velocity_samples,
            )
        )
        viseme_groups.setdefault(viseme_key, []).append(clip_payload)


def process_transition_clips(
    entry_payload: dict[str, Any],
    output_root: Path,
    overwrite: bool,
    transition_groups: dict[str, list[dict[str, Any]]],
    motion_context_by_pkl_path: dict[str, dict[str, Any] | None],
    continuity_state_samples: list[np.ndarray],
    continuity_velocity_samples: list[np.ndarray],
) -> None:
    dataset_index = int(entry_payload.get("datasetIndex", 0))
    dataset_id = str(entry_payload.get("id", ""))
    phrase = str(entry_payload.get("phrase", ""))
    entry_dir = str(entry_payload.get("entryDir", ""))
    motion_context = load_entry_motion_context(
        entry_payload=entry_payload,
        motion_context_by_pkl_path=motion_context_by_pkl_path,
    )

    for item in ensure_list(entry_payload.get("transitions")):
        if not isinstance(item, dict):
            continue
        transition_key = str(item.get("key", "")).strip()
        if not transition_key:
            from_viseme = str(item.get("fromViseme", "")).strip()
            to_viseme = str(item.get("toViseme", "")).strip()
            transition_key = f"{from_viseme}_to_{to_viseme}".strip("_")
        if not transition_key:
            continue
        segment_index = int(item.get("index", 0))
        clip_org_source = resolve_path(str(item.get("clipOrg", "")))
        clip_crop_source = resolve_path(str(item.get("clipCrop", "")))

        basename = build_clip_filename(
            dataset_index=dataset_index,
            dataset_id=dataset_id,
            segment_index=segment_index,
            phrase=phrase,
        )
        group_dir = output_root / "transitions" / sanitize_slug(transition_key, max_length=72)
        clip_org_target = group_dir / "clips" / "org" / f"{basename}.mp4"
        clip_crop_target = group_dir / "clips" / "crop" / f"{basename}.mp4"

        copy_clip(clip_org_source, clip_org_target, overwrite=overwrite)
        copy_clip(clip_crop_source, clip_crop_target, overwrite=overwrite)

        clip_payload = {
            "datasetIndex": dataset_index,
            "id": dataset_id,
            "phrase": phrase,
            "entryDir": entry_dir,
            "segmentIndex": segment_index,
            "fromViseme": str(item.get("fromViseme", "")),
            "toViseme": str(item.get("toViseme", "")),
            "timeSec": float(item.get("timeSec", 0.0) or 0.0),
            "startSec": float(item.get("startSec", 0.0) or 0.0),
            "endSec": float(item.get("endSec", 0.0) or 0.0),
            "durationSec": float(item.get("durationSec", 0.0) or 0.0),
            "clipStartSec": float(item.get("clipStartSec", 0.0) or 0.0),
            "clipEndSec": float(item.get("clipEndSec", 0.0) or 0.0),
            "clipDurationSec": float(item.get("clipDurationSec", 0.0) or 0.0),
            "sourceClipOrg": to_project_relative(clip_org_source),
            "sourceClipCrop": to_project_relative(clip_crop_source),
            "clipOrg": to_project_relative(clip_org_target),
            "clipCrop": to_project_relative(clip_crop_target),
        }
        clip_payload.update(
            build_clip_continuity_fields(
                segment_payload=clip_payload,
                motion_context=motion_context,
                continuity_state_samples=continuity_state_samples,
                continuity_velocity_samples=continuity_velocity_samples,
            )
        )
        transition_groups.setdefault(transition_key, []).append(clip_payload)


def write_group_manifests(
    output_root: Path,
    viseme_groups: dict[str, list[dict[str, Any]]],
    transition_groups: dict[str, list[dict[str, Any]]],
    source_root: Path,
) -> tuple[list[str], list[str]]:
    generated_at = datetime.now(timezone.utc).isoformat()
    viseme_manifest_paths: list[str] = []
    transition_manifest_paths: list[str] = []

    for viseme_key in sorted(viseme_groups):
        group_dir = output_root / "visemes" / sanitize_slug(viseme_key, max_length=40)
        manifest_path = group_dir / "manifest.json"
        payload = {
            "version": 1,
            "generatedAtUtc": generated_at,
            "type": "viseme",
            "key": viseme_key,
            "sourceRoot": to_project_relative(source_root),
            "groupDir": to_project_relative(group_dir),
            "clipCount": len(viseme_groups[viseme_key]),
            "clips": viseme_groups[viseme_key],
        }
        write_json(manifest_path, payload)
        viseme_manifest_paths.append(to_project_relative(manifest_path))

    for transition_key in sorted(transition_groups):
        group_dir = output_root / "transitions" / sanitize_slug(transition_key, max_length=72)
        manifest_path = group_dir / "manifest.json"
        payload = {
            "version": 1,
            "generatedAtUtc": generated_at,
            "type": "transition",
            "key": transition_key,
            "sourceRoot": to_project_relative(source_root),
            "groupDir": to_project_relative(group_dir),
            "clipCount": len(transition_groups[transition_key]),
            "clips": transition_groups[transition_key],
        }
        write_json(manifest_path, payload)
        transition_manifest_paths.append(to_project_relative(manifest_path))

    return viseme_manifest_paths, transition_manifest_paths


def main() -> None:
    args = parse_args()
    source_root = resolve_path(str(args.source_root))
    output_root = resolve_path(str(args.output_root))

    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    entry_manifests = list_entry_manifests(source_root)
    if not entry_manifests:
        raise ValueError(f"No entry manifests found in {source_root}")

    viseme_groups: dict[str, list[dict[str, Any]]] = {}
    transition_groups: dict[str, list[dict[str, Any]]] = {}
    motion_context_by_pkl_path: dict[str, dict[str, Any] | None] = {}
    continuity_state_samples: list[np.ndarray] = []
    continuity_velocity_samples: list[np.ndarray] = []
    processed_entries = 0

    for manifest_path in entry_manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        process_viseme_clips(
            entry_payload=payload,
            output_root=output_root,
            overwrite=bool(args.overwrite),
            viseme_groups=viseme_groups,
            motion_context_by_pkl_path=motion_context_by_pkl_path,
            continuity_state_samples=continuity_state_samples,
            continuity_velocity_samples=continuity_velocity_samples,
        )
        process_transition_clips(
            entry_payload=payload,
            output_root=output_root,
            overwrite=bool(args.overwrite),
            transition_groups=transition_groups,
            motion_context_by_pkl_path=motion_context_by_pkl_path,
            continuity_state_samples=continuity_state_samples,
            continuity_velocity_samples=continuity_velocity_samples,
        )
        processed_entries += 1

    viseme_manifest_paths, transition_manifest_paths = write_group_manifests(
        output_root=output_root,
        viseme_groups=viseme_groups,
        transition_groups=transition_groups,
        source_root=source_root,
    )

    global_manifest = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceRoot": to_project_relative(source_root),
        "outputRoot": to_project_relative(output_root),
        "processedEntryManifestCount": processed_entries,
        "visemeGroupCount": len(viseme_groups),
        "transitionGroupCount": len(transition_groups),
        "visemeClipCount": sum(len(items) for items in viseme_groups.values()),
        "transitionClipCount": sum(len(items) for items in transition_groups.values()),
        "visemeGroupManifests": viseme_manifest_paths,
        "transitionGroupManifests": transition_manifest_paths,
    }
    continuity_summary = build_continuity_summary(
        continuity_state_samples=continuity_state_samples,
        continuity_velocity_samples=continuity_velocity_samples,
    )
    if continuity_summary is not None:
        global_manifest["continuity"] = continuity_summary
    global_manifest_path = output_root / "manifest.json"
    write_json(global_manifest_path, global_manifest)

    print(f"[ok] processed entry manifests: {processed_entries}")
    print(f"[ok] viseme groups: {len(viseme_groups)} clip_count={global_manifest['visemeClipCount']}")
    print(f"[ok] transition groups: {len(transition_groups)} clip_count={global_manifest['transitionClipCount']}")
    print(f"[ok] library manifest: {global_manifest_path}")


if __name__ == "__main__":
    main()
