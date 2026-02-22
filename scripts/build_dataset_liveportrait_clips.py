"""
Build offline LivePortrait clips from a dataset JSON with base64 audio and viseme timings.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import pickle
import re
import shutil
import subprocess
import unicodedata
import wave
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = "liveportrait_dataset_full_1771690526508.json"
DEFAULT_BASE_IMAGE_PATH = "output/frames/frame_00095.png"
DEFAULT_OUTPUT_DIR = "output_liveportrait_dataset_poc"
DEFAULT_OUTPUT_MANIFEST = "output_liveportrait_dataset_poc/dataset_manifest.json"
DEFAULT_FASTER_REPO_DIR = "third_party/FasterLivePortrait"
DEFAULT_CFG_PATH = "third_party/FasterLivePortrait/configs/trt_infer.yaml"
DEFAULT_AUDIO_TO_PKL_SCRIPT = "faster_liveportrait_audio_to_pkl.py"
DEFAULT_SOURCE_CACHE_DIR = "output_fasterliveportrait/source_preprocess_cache/liveportrait_dataset_poc"
DEFAULT_DOCKER_CONTAINER = "animation_api"
DEFAULT_DOCKER_SERVICE = "animation-api"
DEFAULT_DOCKER_PYTHON = "/root/miniconda3/bin/python"
DEFAULT_CONTAINER_APP_ROOT = "/app"
DEFAULT_CONTAINER_FASTER_REPO = "/app/third_party/FasterLivePortrait"
DEFAULT_RUNTIME = "docker"
DEFAULT_RANDOM_SEED = 1234
DEFAULT_SAMPLE_RATE_HZ = 24000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2
DEFAULT_START_INDEX = 0
DEFAULT_LIMIT = 0
DEFAULT_TRANSITION_MODE = "pair"
DEFAULT_TRANSITION_PADDING_SEC = 0.08
TIMING_MODE_DATASET = "dataset"
TIMING_MODE_AUDIO_FORCED = "audio-forced"
DEFAULT_TIMING_MODE = TIMING_MODE_AUDIO_FORCED
DEFAULT_ALIGNMENT_DEVICE = "auto"
DEFAULT_MIN_VISEME_FRAMES = 1
DEFAULT_MIN_TRANSITION_FRAMES = 1
DEFAULT_VISEME_CONTEXT_SEC = 0.45
DEFAULT_TRANSITION_CONTEXT_SEC = 0.35
DEFAULT_TARGET_VISEME_SEC = 0.0
DEFAULT_TARGET_TRANSITION_SEC = 0.0
DEFAULT_MOTION_STABILIZATION = True
DEFAULT_FORCE_MOTION_STABILIZATION = False
MOTION_STABILIZATION_PROFILE_V1 = "stabilized_v1"
IDLE_MOTION_PROFILE_V1 = "idle_natural_v1"
EYE_TAMED_SOFT_INDICES = (0, 1, 2, 3, 4, 5, 7, 10, 13)
EYE_TAMED_HARD_INDICES = (11, 15)
IDLE_BLINK_ACTIVE_EYE_INDICES = (11, 13, 15)
IDLE_MOUTH_PRIMARY_INDICES = (19, 20)
IDLE_MOUTH_SUPPORT_INDICES = (14, 17)
IDLE_MOUTH_STRICT_FREEZE_INDICES = (14, 16, 17, 18, 19, 20)
DEFAULT_EYE_SOFT_FACTOR = 0.45
DEFAULT_EYE_HARD_FACTOR = 0.18
DEFAULT_EYE_HARD_DY_MIN = -0.0045
DEFAULT_EYE_HARD_DY_MAX = 0.0035
DEFAULT_HEAD_OUTLIER_Z = 2.8
DEFAULT_HEAD_DELTA_Z = 2.6
DEFAULT_HEAD_EMA_ALPHA = 0.62
DEFAULT_TRANSLATION_OUTLIER_Z = 3.0
DEFAULT_TRANSLATION_DELTA_Z = 2.8
DEFAULT_TRANSLATION_EMA_ALPHA = 0.60
DEFAULT_EYE_OUTLIER_Z = 2.6
DEFAULT_EYE_DELTA_Z = 2.1
DEFAULT_EYE_EMA_ALPHA = 0.42
DEFAULT_SCALE_OUTLIER_Z = 3.2
DEFAULT_SCALE_DELTA_Z = 2.8
DEFAULT_SCALE_EMA_ALPHA = 0.78
DEFAULT_IDLE_MOTION_ENHANCEMENT = True
DEFAULT_FORCE_IDLE_MOTION_ENHANCEMENT = False
DEFAULT_IDLE_EDGE_FADE_SEC = 1.2
DEFAULT_IDLE_PRIMARY_FREQ_HZ = 0.085
DEFAULT_IDLE_SECONDARY_FREQ_SCALE = 1.9
DEFAULT_IDLE_YAW_AMPLITUDE_DEG = 0.85
DEFAULT_IDLE_PITCH_AMPLITUDE_DEG = 0.55
DEFAULT_IDLE_ROLL_AMPLITUDE_DEG = 0.45
DEFAULT_IDLE_TRANSLATION_AMPLITUDE = 0.0015
DEFAULT_IDLE_BLINK_MIN_INTERVAL_SEC = 2.8
DEFAULT_IDLE_BLINK_MAX_INTERVAL_SEC = 5.4
DEFAULT_IDLE_BLINK_DURATION_SEC = 0.068
DEFAULT_IDLE_BLINK_AMPLITUDE = 0.0200
DEFAULT_IDLE_BLINK_SIGN = 1.0
DEFAULT_IDLE_BLINK_CLOSE_SEC = 0.052
DEFAULT_IDLE_BLINK_HOLD_SEC = 0.060
DEFAULT_IDLE_BLINK_DOUBLE_PROBABILITY = 0.34
DEFAULT_IDLE_BLINK_IRREGULARITY = 0.42
DEFAULT_IDLE_BLINK_CLOSE_TARGET_SCALE = 1.80
DEFAULT_IDLE_BLINK_SOFT_UPPER_SCALE = 0.00
DEFAULT_IDLE_BLINK_SOFT_LOWER_SCALE = 0.00
DEFAULT_IDLE_BLINK_CENTER_SCALE = 0.05
DEFAULT_IDLE_BLINK_CENTER_FORCE_SCALE = 0.12
DEFAULT_IDLE_MOUTH_NEUTRAL_STRENGTH = 0.58
DEFAULT_IDLE_MOUTH_TARGET_QUANTILE = 0.62
DEFAULT_IDLE_MOUTH_FLOOR_SIGMA = 0.55
DEFAULT_IDLE_MOUTH_OUTLIER_Z = 2.35
DEFAULT_IDLE_MOUTH_DELTA_Z = 1.95
DEFAULT_IDLE_MOUTH_EMA_ALPHA = 0.36
DEFAULT_IDLE_MOUTH_BLINK_LOCK_STRENGTH = 1.00
DEFAULT_IDLE_MOUTH_BLINK_LOCK_THRESHOLD = 0.10
DEFAULT_IDLE_MOUTH_BLINK_HARD_LOCK_THRESHOLD = 0.45
DEFAULT_IDLE_MOUTH_BLINK_REFERENCE_EMA_ALPHA = 0.02
DEFAULT_IDLE_MOUTH_BLINK_FREEZE_THRESHOLD = 0.06
DEFAULT_IDLE_MOUTH_BLINK_FREEZE_PADDING_FRAMES = 2
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE = True
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_THRESHOLD = 0.03
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_PADDING_FRAMES = 1
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_PRE_FRAMES = 8
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_POST_FRAMES = 0
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_X_MIN_RATIO = 0.41
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_X_MAX_RATIO = 0.59
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_Y_MIN_RATIO = 0.44
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_Y_MAX_RATIO = 0.56
DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_FEATHER_PX = 0.0
DEFAULT_SEGMENT_ENCODER_PRESET = "slow"
DEFAULT_SEGMENT_ENCODER_CRF = 14
DEFAULT_SEGMENT_ENCODER_TUNE = "animation"
DEFAULT_SEGMENT_ENCODER_PROFILE = "high"
DEFAULT_SEGMENT_ENCODER_LEVEL = "4.1"
RUNTIME_DOCKER = "docker"
RUNTIME_LOCAL = "local"
TRANSITION_MODE_PAIR = "pair"
TRANSITION_MODE_CENTERED = "centered"


@dataclass(frozen=True)
class VisemeSpec:
    """
    One viseme segment definition.
    """

    index: int
    char: str
    viseme: str
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass(frozen=True)
class TransitionSpec:
    """
    One transition segment definition.
    """

    index: int
    from_viseme: str
    to_viseme: str
    time_sec: float
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass(frozen=True)
class DatasetEntrySpec:
    """
    One dataset phrase item.
    """

    dataset_index: int
    dataset_id: str
    phrase: str
    duration_sec: float
    audio_base64: str
    visemes: list[VisemeSpec]
    transitions: list[TransitionSpec]


@dataclass(frozen=True)
class SegmentClipResult:
    """
    Output clip paths for one trimmed segment.
    """

    index: int
    key: str
    start_sec: float
    end_sec: float
    duration_sec: float
    clip_org_path: Path
    clip_crop_path: Path


@dataclass(frozen=True)
class EntryBuildResult:
    """
    Output summary for one built dataset entry.
    """

    dataset_index: int
    dataset_id: str
    phrase: str
    duration_sec: float
    entry_dir: Path
    audio_path: Path
    pkl_path: Path
    full_org_path: Path
    full_crop_path: Path
    viseme_results: list[SegmentClipResult]
    transition_results: list[SegmentClipResult]
    entry_manifest_path: Path


@dataclass(frozen=True)
class VideoStreamInfo:
    """
    Basic video stream metadata used for frame-accurate cutting.
    """

    fps: float
    total_frames: int
    duration_sec: float


@dataclass
class MmsAlignmentRuntime:
    """
    Loaded MMS forced-alignment components.
    """

    model: Any
    tokenizer: Any
    aligner: Any
    torch_module: Any
    sample_rate_hz: int
    label_set: set[str]
    device_name: str


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate one LivePortrait clip per dataset phrase and cut independent clips "
            "for each viseme and transition."
        )
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--faster-repo-dir", default=DEFAULT_FASTER_REPO_DIR)
    parser.add_argument("--cfg", default=DEFAULT_CFG_PATH)
    parser.add_argument("--audio-to-pkl-script", default=DEFAULT_AUDIO_TO_PKL_SCRIPT)
    parser.add_argument("--source-cache-dir", default=DEFAULT_SOURCE_CACHE_DIR)
    parser.add_argument("--runtime", choices=[RUNTIME_DOCKER, RUNTIME_LOCAL], default=DEFAULT_RUNTIME)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    parser.add_argument("--docker-service", default=DEFAULT_DOCKER_SERVICE)
    parser.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--audio-channels", type=int, default=DEFAULT_AUDIO_CHANNELS)
    parser.add_argument("--sample-width-bytes", type=int, default=DEFAULT_SAMPLE_WIDTH_BYTES)
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="0 means no limit.")
    parser.add_argument(
        "--transition-mode",
        choices=[TRANSITION_MODE_PAIR, TRANSITION_MODE_CENTERED],
        default=DEFAULT_TRANSITION_MODE,
        help="How to derive transition segment windows from viseme timings.",
    )
    parser.add_argument(
        "--transition-padding-sec",
        type=float,
        default=DEFAULT_TRANSITION_PADDING_SEC,
        help="Padding around transition timestamp when --transition-mode=centered.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=[TIMING_MODE_DATASET, TIMING_MODE_AUDIO_FORCED],
        default=DEFAULT_TIMING_MODE,
        help="Source used to derive viseme timestamps.",
    )
    parser.add_argument(
        "--alignment-device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_ALIGNMENT_DEVICE,
        help="Execution device for forced alignment model.",
    )
    parser.add_argument(
        "--alignment-fallback",
        action="store_true",
        default=True,
        help="Fallback to dataset timings if forced alignment fails.",
    )
    parser.add_argument("--no-alignment-fallback", dest="alignment_fallback", action="store_false")
    parser.add_argument(
        "--min-viseme-frames",
        type=int,
        default=DEFAULT_MIN_VISEME_FRAMES,
        help="Minimum frame count per viseme clip.",
    )
    parser.add_argument(
        "--min-transition-frames",
        type=int,
        default=DEFAULT_MIN_TRANSITION_FRAMES,
        help="Minimum frame count per transition clip.",
    )
    parser.add_argument(
        "--viseme-context-sec",
        type=float,
        default=DEFAULT_VISEME_CONTEXT_SEC,
        help="Temporal context around each aligned viseme (half-window in seconds).",
    )
    parser.add_argument(
        "--transition-context-sec",
        type=float,
        default=DEFAULT_TRANSITION_CONTEXT_SEC,
        help="Temporal context around each transition clip (half-window in seconds).",
    )
    parser.add_argument(
        "--target-viseme-sec",
        type=float,
        default=DEFAULT_TARGET_VISEME_SEC,
        help="Optional output duration for each viseme clip (0 disables retime).",
    )
    parser.add_argument(
        "--target-transition-sec",
        type=float,
        default=DEFAULT_TARGET_TRANSITION_SEC,
        help="Optional output duration for each transition clip (0 disables retime).",
    )
    parser.add_argument(
        "--motion-stabilization",
        action="store_true",
        default=DEFAULT_MOTION_STABILIZATION,
        help="Apply robust PKL smoothing to reduce abrupt eye/head jumps before render.",
    )
    parser.add_argument(
        "--no-motion-stabilization",
        dest="motion_stabilization",
        action="store_false",
    )
    parser.add_argument(
        "--force-motion-stabilization",
        action="store_true",
        default=DEFAULT_FORCE_MOTION_STABILIZATION,
        help="Re-apply PKL stabilization even when already tagged as processed.",
    )
    parser.add_argument(
        "--eye-soft-factor",
        type=float,
        default=DEFAULT_EYE_SOFT_FACTOR,
        help="Relative damping factor [0..1] for soft eye-sensitive expression indices.",
    )
    parser.add_argument(
        "--eye-hard-factor",
        type=float,
        default=DEFAULT_EYE_HARD_FACTOR,
        help="Relative damping factor [0..1] for hard eye-sensitive expression indices.",
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
    parser.add_argument(
        "--head-outlier-z",
        type=float,
        default=DEFAULT_HEAD_OUTLIER_Z,
        help="Robust z-threshold used to clip head-angle outliers.",
    )
    parser.add_argument(
        "--head-delta-z",
        type=float,
        default=DEFAULT_HEAD_DELTA_Z,
        help="Robust z-threshold used to limit head-angle frame-to-frame jumps.",
    )
    parser.add_argument(
        "--head-ema-alpha",
        type=float,
        default=DEFAULT_HEAD_EMA_ALPHA,
        help="EMA alpha [0..1] for head-angle temporal smoothing.",
    )
    parser.add_argument(
        "--translation-outlier-z",
        type=float,
        default=DEFAULT_TRANSLATION_OUTLIER_Z,
        help="Robust z-threshold used to clip translation outliers.",
    )
    parser.add_argument(
        "--translation-delta-z",
        type=float,
        default=DEFAULT_TRANSLATION_DELTA_Z,
        help="Robust z-threshold used to limit translation frame-to-frame jumps.",
    )
    parser.add_argument(
        "--translation-ema-alpha",
        type=float,
        default=DEFAULT_TRANSLATION_EMA_ALPHA,
        help="EMA alpha [0..1] for translation temporal smoothing.",
    )
    parser.add_argument(
        "--eye-outlier-z",
        type=float,
        default=DEFAULT_EYE_OUTLIER_Z,
        help="Robust z-threshold used to clip eye-channel outliers.",
    )
    parser.add_argument(
        "--eye-delta-z",
        type=float,
        default=DEFAULT_EYE_DELTA_Z,
        help="Robust z-threshold used to limit eye-channel frame-to-frame jumps.",
    )
    parser.add_argument(
        "--eye-ema-alpha",
        type=float,
        default=DEFAULT_EYE_EMA_ALPHA,
        help="EMA alpha [0..1] for eye-channel temporal smoothing.",
    )
    parser.add_argument(
        "--idle-motion-enhancement",
        action="store_true",
        default=DEFAULT_IDLE_MOTION_ENHANCEMENT,
        help="Apply subtle idle-only eye/head micro-motion synthesis for natural resting clips.",
    )
    parser.add_argument(
        "--no-idle-motion-enhancement",
        dest="idle_motion_enhancement",
        action="store_false",
    )
    parser.add_argument(
        "--force-idle-motion-enhancement",
        action="store_true",
        default=DEFAULT_FORCE_IDLE_MOTION_ENHANCEMENT,
        help="Re-apply idle motion enhancement even if already tagged in PKL metadata.",
    )
    parser.add_argument(
        "--idle-edge-fade-sec",
        type=float,
        default=DEFAULT_IDLE_EDGE_FADE_SEC,
        help="Fade-in/out window to keep idle clip boundaries stable.",
    )
    parser.add_argument(
        "--idle-primary-freq-hz",
        type=float,
        default=DEFAULT_IDLE_PRIMARY_FREQ_HZ,
        help="Primary low-frequency oscillator for idle head drift.",
    )
    parser.add_argument(
        "--idle-secondary-freq-scale",
        type=float,
        default=DEFAULT_IDLE_SECONDARY_FREQ_SCALE,
        help="Secondary oscillator frequency scale relative to primary frequency.",
    )
    parser.add_argument(
        "--idle-yaw-amplitude-deg",
        type=float,
        default=DEFAULT_IDLE_YAW_AMPLITUDE_DEG,
        help="Idle yaw drift amplitude in degrees.",
    )
    parser.add_argument(
        "--idle-pitch-amplitude-deg",
        type=float,
        default=DEFAULT_IDLE_PITCH_AMPLITUDE_DEG,
        help="Idle pitch drift amplitude in degrees.",
    )
    parser.add_argument(
        "--idle-roll-amplitude-deg",
        type=float,
        default=DEFAULT_IDLE_ROLL_AMPLITUDE_DEG,
        help="Idle roll drift amplitude in degrees.",
    )
    parser.add_argument(
        "--idle-translation-amplitude",
        type=float,
        default=DEFAULT_IDLE_TRANSLATION_AMPLITUDE,
        help="Idle translation drift amplitude in model translation units.",
    )
    parser.add_argument(
        "--idle-blink-min-interval-sec",
        type=float,
        default=DEFAULT_IDLE_BLINK_MIN_INTERVAL_SEC,
        help="Minimum interval between synthetic idle blinks.",
    )
    parser.add_argument(
        "--idle-blink-max-interval-sec",
        type=float,
        default=DEFAULT_IDLE_BLINK_MAX_INTERVAL_SEC,
        help="Maximum interval between synthetic idle blinks.",
    )
    parser.add_argument(
        "--idle-blink-duration-sec",
        type=float,
        default=DEFAULT_IDLE_BLINK_DURATION_SEC,
        help="Approximate Gaussian blink sigma duration in seconds.",
    )
    parser.add_argument(
        "--idle-blink-amplitude",
        type=float,
        default=DEFAULT_IDLE_BLINK_AMPLITUDE,
        help="Peak blink expression amplitude for hard eye channels.",
    )
    parser.add_argument(
        "--idle-blink-sign",
        type=float,
        default=DEFAULT_IDLE_BLINK_SIGN,
        help="Direction sign applied to blink offsets (typically positive for this model).",
    )
    parser.add_argument(
        "--idle-blink-close-sec",
        type=float,
        default=DEFAULT_IDLE_BLINK_CLOSE_SEC,
        help="Per-blink close/open ramp duration in seconds.",
    )
    parser.add_argument(
        "--idle-blink-hold-sec",
        type=float,
        default=DEFAULT_IDLE_BLINK_HOLD_SEC,
        help="Per-blink fully-closed hold duration in seconds.",
    )
    parser.add_argument(
        "--idle-blink-double-probability",
        type=float,
        default=DEFAULT_IDLE_BLINK_DOUBLE_PROBABILITY,
        help="Probability [0..1] of generating a second close blink right after a blink.",
    )
    parser.add_argument(
        "--idle-blink-irregularity",
        type=float,
        default=DEFAULT_IDLE_BLINK_IRREGULARITY,
        help="Irregularity strength [0..1] for blink interval jitter.",
    )
    parser.add_argument(
        "--idle-blink-close-target-scale",
        type=float,
        default=DEFAULT_IDLE_BLINK_CLOSE_TARGET_SCALE,
        help="Scale factor applied to blink amplitude to enforce full closure target.",
    )
    parser.add_argument(
        "--idle-blink-soft-upper-scale",
        type=float,
        default=DEFAULT_IDLE_BLINK_SOFT_UPPER_SCALE,
        help="Blink spillover scale applied to upper eye-adjacent soft channels.",
    )
    parser.add_argument(
        "--idle-blink-soft-lower-scale",
        type=float,
        default=DEFAULT_IDLE_BLINK_SOFT_LOWER_SCALE,
        help="Blink spillover scale applied to lower eye-adjacent soft channels.",
    )
    parser.add_argument(
        "--idle-blink-center-scale",
        type=float,
        default=DEFAULT_IDLE_BLINK_CENTER_SCALE,
        help="Direct blink scale applied to the center eye channel (index 13).",
    )
    parser.add_argument(
        "--idle-blink-center-force-scale",
        type=float,
        default=DEFAULT_IDLE_BLINK_CENTER_FORCE_SCALE,
        help="Forced-closure scale factor applied to center eye channel (index 13).",
    )
    parser.add_argument(
        "--idle-mouth-neutral-strength",
        type=float,
        default=DEFAULT_IDLE_MOUTH_NEUTRAL_STRENGTH,
        help="Blend strength [0..1] that pulls idle mouth channels toward a neutral target.",
    )
    parser.add_argument(
        "--idle-mouth-target-quantile",
        type=float,
        default=DEFAULT_IDLE_MOUTH_TARGET_QUANTILE,
        help="Quantile [0..1] used as neutral mouth target for idle stabilization.",
    )
    parser.add_argument(
        "--idle-mouth-floor-sigma",
        type=float,
        default=DEFAULT_IDLE_MOUTH_FLOOR_SIGMA,
        help="Lower-bound guard for main mouth-open channels in units of robust sigma.",
    )
    parser.add_argument(
        "--idle-mouth-outlier-z",
        type=float,
        default=DEFAULT_IDLE_MOUTH_OUTLIER_Z,
        help="Robust z-threshold used to clip idle mouth outliers.",
    )
    parser.add_argument(
        "--idle-mouth-delta-z",
        type=float,
        default=DEFAULT_IDLE_MOUTH_DELTA_Z,
        help="Robust z-threshold used to limit idle mouth frame-to-frame jumps.",
    )
    parser.add_argument(
        "--idle-mouth-ema-alpha",
        type=float,
        default=DEFAULT_IDLE_MOUTH_EMA_ALPHA,
        help="EMA alpha [0..1] for idle mouth-channel temporal smoothing.",
    )
    parser.add_argument(
        "--idle-mouth-blink-lock-strength",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_LOCK_STRENGTH,
        help="Strength [0..1] of mouth neutral lock applied while eyes are blinking.",
    )
    parser.add_argument(
        "--idle-mouth-blink-lock-threshold",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_LOCK_THRESHOLD,
        help="Blink intensity threshold [0..1] after which mouth lock starts to engage.",
    )
    parser.add_argument(
        "--idle-mouth-blink-hard-lock-threshold",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_HARD_LOCK_THRESHOLD,
        help="Blink intensity threshold [0..1] where mouth is fully pinned to neutral.",
    )
    parser.add_argument(
        "--idle-mouth-blink-reference-ema-alpha",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_REFERENCE_EMA_ALPHA,
        help="EMA alpha [0..1] for mouth reference used by blink lock (lower = steadier).",
    )
    parser.add_argument(
        "--idle-mouth-blink-freeze-threshold",
        type=float,
        default=DEFAULT_IDLE_MOUTH_BLINK_FREEZE_THRESHOLD,
        help="Blink intensity threshold [0..1] used to freeze mouth channels during blink windows.",
    )
    parser.add_argument(
        "--idle-mouth-blink-freeze-padding-frames",
        type=int,
        default=DEFAULT_IDLE_MOUTH_BLINK_FREEZE_PADDING_FRAMES,
        help="Frame padding applied around blink-freeze windows for mouth channels.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze",
        action="store_true",
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE,
        help="Freeze mouth-area pixels around blink windows in rendered idle videos.",
    )
    parser.add_argument(
        "--no-idle-pixel-mouth-freeze",
        dest="idle_pixel_mouth_freeze",
        action="store_false",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-threshold",
        type=float,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_THRESHOLD,
        help="Blink intensity threshold [0..1] to activate mouth-pixel freeze.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-padding-frames",
        type=int,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_PADDING_FRAMES,
        help="Symmetric frame padding around blink windows for mouth-pixel freeze.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-pre-frames",
        type=int,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_PRE_FRAMES,
        help="Extra frames frozen before each blink window.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-post-frames",
        type=int,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_POST_FRAMES,
        help="Extra frames frozen after each blink window.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-x-min-ratio",
        type=float,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_X_MIN_RATIO,
        help="Mouth freeze ROI minimum X ratio in frame coordinates.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-x-max-ratio",
        type=float,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_X_MAX_RATIO,
        help="Mouth freeze ROI maximum X ratio in frame coordinates.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-y-min-ratio",
        type=float,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_Y_MIN_RATIO,
        help="Mouth freeze ROI minimum Y ratio in frame coordinates.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-y-max-ratio",
        type=float,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_Y_MAX_RATIO,
        help="Mouth freeze ROI maximum Y ratio in frame coordinates.",
    )
    parser.add_argument(
        "--idle-pixel-mouth-freeze-feather-px",
        type=float,
        default=DEFAULT_IDLE_PIXEL_MOUTH_FREEZE_FEATHER_PX,
        help="Feather radius (px) for mouth-pixel freeze ROI blending.",
    )
    parser.add_argument(
        "--segment-encoder-preset",
        default=DEFAULT_SEGMENT_ENCODER_PRESET,
        help="x264 preset used for viseme/transition segment clips.",
    )
    parser.add_argument(
        "--segment-encoder-crf",
        type=int,
        default=DEFAULT_SEGMENT_ENCODER_CRF,
        help="x264 CRF used for viseme/transition segment clips (lower = better quality).",
    )
    parser.add_argument(
        "--segment-encoder-tune",
        default=DEFAULT_SEGMENT_ENCODER_TUNE,
        help="Optional x264 tune for segment clips (set empty string to disable).",
    )
    parser.add_argument(
        "--segment-encoder-profile",
        default=DEFAULT_SEGMENT_ENCODER_PROFILE,
        help="x264 profile for segment clips.",
    )
    parser.add_argument(
        "--segment-encoder-level",
        default=DEFAULT_SEGMENT_ENCODER_LEVEL,
        help="x264 level for segment clips.",
    )
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
    Convert absolute path into project-relative POSIX representation when possible.
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


def sanitize_slug(raw_value: str, max_length: int = 64) -> str:
    """
    Convert text into stable ASCII slug for filenames.
    """
    normalized = unicodedata.normalize("NFKD", str(raw_value)).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    if not cleaned:
        cleaned = "item"
    return cleaned[:max_length]


def clamp_time(value: float, duration_sec: float) -> float:
    """
    Clamp a timestamp into [0, duration_sec].
    """
    return max(0.0, min(float(value), float(duration_sec)))


def expand_segment_window_with_context(
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    context_sec: float,
) -> tuple[float, float]:
    """
    Expand a timing window around its center using a configurable half-context.
    """
    safe_start = clamp_time(float(start_sec), duration_sec)
    safe_end = clamp_time(float(end_sec), duration_sec)
    if safe_end <= safe_start:
        safe_end = clamp_time(safe_start + 1e-4, duration_sec)
    center_sec = (safe_start + safe_end) * 0.5
    half_window_sec = max((safe_end - safe_start) * 0.5, max(0.0, float(context_sec)))
    desired_duration_sec = min(float(duration_sec), half_window_sec * 2.0)
    expanded_start = center_sec - desired_duration_sec * 0.5
    expanded_end = center_sec + desired_duration_sec * 0.5
    if expanded_start < 0.0:
        expanded_end = min(float(duration_sec), expanded_end - expanded_start)
        expanded_start = 0.0
    if expanded_end > float(duration_sec):
        overflow = expanded_end - float(duration_sec)
        expanded_start = max(0.0, expanded_start - overflow)
        expanded_end = float(duration_sec)
    expanded_start = clamp_time(expanded_start, duration_sec)
    expanded_end = clamp_time(expanded_end, duration_sec)
    if expanded_end <= expanded_start:
        expanded_end = clamp_time(expanded_start + 1e-4, duration_sec)
    return expanded_start, expanded_end


def run_command(command: list[str]) -> None:
    """
    Run command and raise with captured stderr on failure.
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
    Run command and return stdout text.
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
    Ensure ffmpeg is available in PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not found in PATH.")


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


def parse_viseme_specs(raw_visemes: Any, duration_sec: float) -> list[VisemeSpec]:
    """
    Parse and validate viseme segments from dataset entry.
    """
    if not isinstance(raw_visemes, list) or not raw_visemes:
        raise ValueError("Dataset entry has no visemes.")
    visemes: list[VisemeSpec] = []
    for index, item in enumerate(raw_visemes):
        if not isinstance(item, dict):
            raise ValueError("Invalid viseme entry type.")
        viseme_key = str(item.get("viseme", "")).strip()
        char_value = str(item.get("char", "")).strip()
        start_sec = clamp_time(float(item.get("start_time", 0.0) or 0.0), duration_sec)
        end_sec = clamp_time(float(item.get("end_time", 0.0) or 0.0), duration_sec)
        if not viseme_key:
            raise ValueError(f"Viseme entry {index} is missing viseme key.")
        if end_sec <= start_sec:
            raise ValueError(f"Invalid viseme timing at index {index}: {start_sec}..{end_sec}")
        visemes.append(
            VisemeSpec(
                index=index,
                char=char_value,
                viseme=viseme_key,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
            )
        )
    return visemes


def resolve_transition_window(
    left_viseme: VisemeSpec,
    right_viseme: VisemeSpec,
    transition_time_sec: float,
    duration_sec: float,
    mode: str,
    padding_sec: float,
) -> tuple[float, float]:
    """
    Resolve transition clip time window for one adjacent viseme pair.
    """
    pair_start = clamp_time(left_viseme.start_sec, duration_sec)
    pair_end = clamp_time(right_viseme.end_sec, duration_sec)
    if mode == TRANSITION_MODE_PAIR:
        return pair_start, pair_end

    safe_padding = max(0.0, float(padding_sec))
    left_pad = min(safe_padding, left_viseme.duration_sec)
    right_pad = min(safe_padding, right_viseme.duration_sec)
    centered_start = max(left_viseme.start_sec, float(transition_time_sec) - left_pad)
    centered_end = min(right_viseme.end_sec, float(transition_time_sec) + right_pad)
    centered_start = clamp_time(centered_start, duration_sec)
    centered_end = clamp_time(centered_end, duration_sec)
    if centered_end <= centered_start:
        return pair_start, pair_end
    return centered_start, centered_end


def parse_transition_specs(
    raw_transitions: Any,
    visemes: list[VisemeSpec],
    duration_sec: float,
    mode: str,
    padding_sec: float,
) -> list[TransitionSpec]:
    """
    Parse transitions and derive per-transition time windows.
    """
    expected_count = max(0, len(visemes) - 1)
    if not isinstance(raw_transitions, list):
        raw_transitions = []
    if raw_transitions and len(raw_transitions) != expected_count:
        raise ValueError(f"Transition count mismatch: expected {expected_count}, got {len(raw_transitions)}")

    transitions: list[TransitionSpec] = []
    for index in range(expected_count):
        left_viseme = visemes[index]
        right_viseme = visemes[index + 1]
        raw_item = raw_transitions[index] if index < len(raw_transitions) else {}
        if not isinstance(raw_item, dict):
            raw_item = {}
        from_viseme = str(raw_item.get("from", left_viseme.viseme)).strip() or left_viseme.viseme
        to_viseme = str(raw_item.get("to", right_viseme.viseme)).strip() or right_viseme.viseme
        transition_time_sec = clamp_time(float(raw_item.get("time", left_viseme.end_sec) or left_viseme.end_sec), duration_sec)
        if from_viseme != left_viseme.viseme or to_viseme != right_viseme.viseme:
            raise ValueError(
                "Transition viseme mismatch at index "
                f"{index}: expected {left_viseme.viseme}_to_{right_viseme.viseme}, got {from_viseme}_to_{to_viseme}"
            )
        start_sec, end_sec = resolve_transition_window(
            left_viseme=left_viseme,
            right_viseme=right_viseme,
            transition_time_sec=transition_time_sec,
            duration_sec=duration_sec,
            mode=mode,
            padding_sec=padding_sec,
        )
        if end_sec <= start_sec:
            raise ValueError(f"Invalid transition timing at index {index}: {start_sec}..{end_sec}")
        transitions.append(
            TransitionSpec(
                index=index,
                from_viseme=from_viseme,
                to_viseme=to_viseme,
                time_sec=transition_time_sec,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
            )
        )
    return transitions


def load_dataset_entries(
    dataset_path: Path,
    transition_mode: str,
    transition_padding_sec: float,
) -> list[DatasetEntrySpec]:
    """
    Load and validate entries from dataset JSON.
    """
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {dataset_path}")
    raw_entries = payload.get("dataset")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"Dataset list is missing or empty: {dataset_path}")

    entries: list[DatasetEntrySpec] = []
    for dataset_index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError("Invalid dataset entry type.")
        dataset_id = str(raw_entry.get("id", dataset_index)).strip()
        phrase = str(raw_entry.get("phrase", "")).strip()
        duration_sec = float(raw_entry.get("duration_seconds", 0.0) or 0.0)
        audio_base64 = str(raw_entry.get("audio_data_base64", "")).strip()
        if duration_sec <= 0.0:
            raise ValueError(f"Invalid duration_seconds for entry index {dataset_index}")
        if not audio_base64:
            raise ValueError(f"Missing audio_data_base64 for entry index {dataset_index}")
        visemes = parse_viseme_specs(raw_entry.get("visemes"), duration_sec)
        transitions = parse_transition_specs(
            raw_transitions=raw_entry.get("transitions"),
            visemes=visemes,
            duration_sec=duration_sec,
            mode=transition_mode,
            padding_sec=transition_padding_sec,
        )
        entries.append(
            DatasetEntrySpec(
                dataset_index=dataset_index,
                dataset_id=dataset_id,
                phrase=phrase,
                duration_sec=duration_sec,
                audio_base64=audio_base64,
                visemes=visemes,
                transitions=transitions,
            )
        )
    return entries


def decode_audio_bytes(audio_base64: str) -> bytes:
    """
    Decode dataset base64 audio payload.
    """
    try:
        decoded = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 audio payload.") from exc
    if not decoded:
        raise ValueError("Decoded audio payload is empty.")
    return decoded


def write_pcm_wav(
    audio_bytes: bytes,
    output_path: Path,
    channels: int,
    sample_width_bytes: int,
    sample_rate_hz: int,
    overwrite: bool,
) -> None:
    """
    Write PCM bytes into a WAV container.
    """
    if output_path.exists() and not overwrite:
        return
    if channels <= 0:
        raise ValueError("audio-channels must be > 0")
    if sample_width_bytes <= 0:
        raise ValueError("sample-width-bytes must be > 0")
    if sample_rate_hz <= 0:
        raise ValueError("sample-rate-hz must be > 0")
    bytes_per_frame = int(channels * sample_width_bytes)
    if bytes_per_frame <= 0:
        raise ValueError("Invalid audio frame size.")
    if len(audio_bytes) % bytes_per_frame != 0:
        raise ValueError("Audio payload byte length is not aligned with channels/sample-width.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width_bytes)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(audio_bytes)


def normalize_alignment_char(raw_char: str) -> str:
    """
    Normalize one grapheme for MMS character-level alignment.
    """
    raw_text = str(raw_char).strip()
    if not raw_text or raw_text == "_":
        return ""
    decomposed = unicodedata.normalize("NFKD", raw_text.lower())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    letters_only = "".join(char for char in without_marks if char.isalpha())
    if not letters_only:
        return ""
    return letters_only[0]


def build_alignment_words_from_visemes(
    visemes: list[VisemeSpec],
    label_set: set[str],
) -> tuple[list[str], list[list[int]]]:
    """
    Convert viseme char stream into MMS-compatible words and viseme index mapping.
    """
    words: list[str] = []
    viseme_indices_by_word: list[list[int]] = []
    pending_chars: list[str] = []
    pending_indices: list[int] = []
    for viseme in visemes:
        normalized_char = normalize_alignment_char(viseme.char)
        is_valid_char = bool(normalized_char) and normalized_char in label_set
        if is_valid_char:
            pending_chars.append(normalized_char)
            pending_indices.append(viseme.index)
            continue
        if pending_chars:
            words.append("".join(pending_chars))
            viseme_indices_by_word.append(list(pending_indices))
            pending_chars.clear()
            pending_indices.clear()
    if pending_chars:
        words.append("".join(pending_chars))
        viseme_indices_by_word.append(list(pending_indices))
    return words, viseme_indices_by_word


def load_wav_mono_float_tensor(
    wav_path: Path,
    target_sample_rate_hz: int,
    torch_module: Any,
) -> tuple[Any, int]:
    """
    Load PCM16 WAV as mono float tensor in range [-1, 1].
    """
    with wave.open(str(wav_path), "rb") as reader:
        channels = int(reader.getnchannels())
        sample_width_bytes = int(reader.getsampwidth())
        sample_rate_hz = int(reader.getframerate())
        frame_count = int(reader.getnframes())
        raw_bytes = reader.readframes(frame_count)
    if sample_width_bytes != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported for forced alignment: {wav_path}")
    if channels <= 0:
        raise ValueError(f"Invalid channel count in WAV: {wav_path}")
    pcm_samples = array("h")
    pcm_samples.frombytes(raw_bytes)
    if channels == 1:
        mono_samples = [float(sample) / 32768.0 for sample in pcm_samples]
    else:
        mono_samples: list[float] = []
        for offset in range(0, len(pcm_samples), channels):
            channel_slice = pcm_samples[offset : offset + channels]
            channel_mean = float(sum(channel_slice)) / float(channels)
            mono_samples.append(channel_mean / 32768.0)
    if not mono_samples:
        raise ValueError(f"Audio has no samples: {wav_path}")
    waveform = torch_module.tensor(mono_samples, dtype=torch_module.float32).unsqueeze(0)
    if sample_rate_hz != int(target_sample_rate_hz):
        target_sample_count = max(1, int(round(waveform.shape[1] * float(target_sample_rate_hz) / float(sample_rate_hz))))
        waveform = torch_module.nn.functional.interpolate(
            waveform.unsqueeze(0),
            size=target_sample_count,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        sample_rate_hz = int(target_sample_rate_hz)
    return waveform, sample_rate_hz


def load_mms_alignment_runtime(alignment_device: str) -> MmsAlignmentRuntime:
    """
    Load MMS forced-alignment model and helpers.
    """
    import torch
    from torchaudio.pipelines import MMS_FA

    requested_device = str(alignment_device).strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported alignment device: {alignment_device}")
    if requested_device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--alignment-device=cuda requested but CUDA is not available.")
        device_name = "cuda"
    else:
        device_name = "cpu"

    model = MMS_FA.get_model(with_star=False)
    model.to(device_name)
    model.eval()
    label_set = {label for label in MMS_FA.get_labels(star=None) if label and label != "-"}
    return MmsAlignmentRuntime(
        model=model,
        tokenizer=MMS_FA.get_tokenizer(),
        aligner=MMS_FA.get_aligner(),
        torch_module=torch,
        sample_rate_hz=int(MMS_FA.sample_rate),
        label_set=label_set,
        device_name=device_name,
    )


def build_aligned_viseme_specs(
    visemes: list[VisemeSpec],
    aligned_times_by_index: dict[int, tuple[float, float]],
    duration_sec: float,
) -> list[VisemeSpec]:
    """
    Fill missing viseme timings and return monotonic viseme windows.
    """
    total_visemes = len(visemes)
    starts: list[float | None] = [None] * total_visemes
    ends: list[float | None] = [None] * total_visemes
    for list_index, viseme in enumerate(visemes):
        aligned = aligned_times_by_index.get(int(viseme.index))
        if aligned is None:
            continue
        starts[list_index] = clamp_time(float(aligned[0]), duration_sec)
        ends[list_index] = clamp_time(float(aligned[1]), duration_sec)

    list_index = 0
    while list_index < total_visemes:
        if starts[list_index] is not None and ends[list_index] is not None:
            list_index += 1
            continue
        block_start = list_index
        while list_index < total_visemes and (starts[list_index] is None or ends[list_index] is None):
            list_index += 1
        block_end = list_index - 1
        left_end = 0.0
        right_start = duration_sec
        if block_start > 0 and ends[block_start - 1] is not None:
            left_end = float(ends[block_start - 1])
        if list_index < total_visemes and starts[list_index] is not None:
            right_start = float(starts[list_index])
        if right_start <= left_end:
            right_start = clamp_time(left_end + 0.02 * float(block_end - block_start + 1), duration_sec)
        block_size = block_end - block_start + 1
        interval_sec = max(0.0, right_start - left_end)
        step_sec = interval_sec / float(block_size) if block_size > 0 else 0.0
        for offset in range(block_size):
            segment_start = left_end + step_sec * float(offset)
            segment_end = left_end + step_sec * float(offset + 1)
            starts[block_start + offset] = clamp_time(segment_start, duration_sec)
            ends[block_start + offset] = clamp_time(segment_end, duration_sec)

    min_step_sec = 1e-4
    cursor_sec = 0.0
    aligned_visemes: list[VisemeSpec] = []
    for list_index, viseme in enumerate(visemes):
        raw_start = float(starts[list_index] if starts[list_index] is not None else cursor_sec)
        raw_end = float(ends[list_index] if ends[list_index] is not None else raw_start + min_step_sec)
        segment_start = clamp_time(max(cursor_sec, raw_start), duration_sec)
        segment_end = clamp_time(raw_end, duration_sec)
        if segment_end <= segment_start:
            segment_end = clamp_time(segment_start + min_step_sec, duration_sec)
        if segment_end <= segment_start:
            segment_start = clamp_time(duration_sec - min_step_sec, duration_sec)
            segment_end = duration_sec
        cursor_sec = segment_end
        aligned_visemes.append(
            VisemeSpec(
                index=viseme.index,
                char=viseme.char,
                viseme=viseme.viseme,
                start_sec=segment_start,
                end_sec=segment_end,
                duration_sec=segment_end - segment_start,
            )
        )
    return aligned_visemes


def align_visemes_with_mms(
    visemes: list[VisemeSpec],
    duration_sec: float,
    audio_wav_path: Path,
    runtime: MmsAlignmentRuntime,
) -> list[VisemeSpec]:
    """
    Force-align viseme chars to audio and return refreshed viseme timings.
    """
    words, viseme_indices_by_word = build_alignment_words_from_visemes(visemes, runtime.label_set)
    if not words:
        raise ValueError("No alignable viseme characters found for forced alignment.")
    waveform, sample_rate_hz = load_wav_mono_float_tensor(
        wav_path=audio_wav_path,
        target_sample_rate_hz=runtime.sample_rate_hz,
        torch_module=runtime.torch_module,
    )
    if sample_rate_hz != runtime.sample_rate_hz:
        raise RuntimeError(f"Unexpected sample rate after resample: {sample_rate_hz}")

    waveform = waveform.to(runtime.device_name)
    with runtime.torch_module.inference_mode():
        emission, _ = runtime.model(waveform)
    emission = emission[0].detach().cpu()
    if emission.ndim != 2 or emission.shape[0] <= 0:
        raise RuntimeError("Invalid emission output from forced alignment model.")

    time_per_frame_sec = float(waveform.shape[1]) / float(runtime.sample_rate_hz) / float(emission.shape[0])
    tokens = runtime.tokenizer(words)
    spans_by_word = runtime.aligner(emission, tokens)
    if len(spans_by_word) != len(words):
        raise RuntimeError("Forced alignment output size mismatch.")

    aligned_times_by_index: dict[int, tuple[float, float]] = {}
    for word, viseme_indices, spans in zip(words, viseme_indices_by_word, spans_by_word, strict=True):
        if len(word) != len(viseme_indices) or len(word) != len(spans):
            raise RuntimeError("Forced alignment char length mismatch.")
        for viseme_index, span in zip(viseme_indices, spans, strict=True):
            start_sec = float(span.start) * time_per_frame_sec
            end_sec = float(span.end) * time_per_frame_sec
            if end_sec <= start_sec:
                end_sec = start_sec + time_per_frame_sec
            aligned_times_by_index[int(viseme_index)] = (
                clamp_time(start_sec, duration_sec),
                clamp_time(end_sec, duration_sec),
            )
    if not aligned_times_by_index:
        raise RuntimeError("Forced alignment produced no aligned viseme timings.")
    return build_aligned_viseme_specs(
        visemes=visemes,
        aligned_times_by_index=aligned_times_by_index,
        duration_sec=duration_sec,
    )


def build_transitions_from_visemes(
    visemes: list[VisemeSpec],
    duration_sec: float,
    mode: str,
    padding_sec: float,
) -> list[TransitionSpec]:
    """
    Build transitions directly from ordered viseme windows.
    """
    transitions: list[TransitionSpec] = []
    for index in range(max(0, len(visemes) - 1)):
        left_viseme = visemes[index]
        right_viseme = visemes[index + 1]
        transition_time_sec = clamp_time(left_viseme.end_sec, duration_sec)
        start_sec, end_sec = resolve_transition_window(
            left_viseme=left_viseme,
            right_viseme=right_viseme,
            transition_time_sec=transition_time_sec,
            duration_sec=duration_sec,
            mode=mode,
            padding_sec=padding_sec,
        )
        if end_sec <= start_sec:
            raise ValueError(f"Invalid transition timing at index {index}: {start_sec}..{end_sec}")
        transitions.append(
            TransitionSpec(
                index=index,
                from_viseme=left_viseme.viseme,
                to_viseme=right_viseme.viseme,
                time_sec=transition_time_sec,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
            )
        )
    return transitions


def parse_fraction(raw_value: str) -> float:
    """
    Parse ffmpeg fraction values such as '25/1'.
    """
    text = str(raw_value).strip()
    if not text:
        return 0.0
    if "/" not in text:
        return float(text)
    numerator_text, denominator_text = text.split("/", 1)
    numerator = float(numerator_text)
    denominator = float(denominator_text)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def get_video_stream_info(video_path: Path) -> VideoStreamInfo:
    """
    Read FPS and frame count from a rendered clip.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    stdout = run_command_capture(command)
    payload = json.loads(stdout or "{}")
    streams = payload.get("streams")
    stream = streams[0] if isinstance(streams, list) and streams else {}
    fps = parse_fraction(str(stream.get("avg_frame_rate", "")))
    if fps <= 0.0:
        raise ValueError(f"Unable to resolve FPS for video: {video_path}")
    nb_frames_text = str(stream.get("nb_frames", "")).strip()
    total_frames = int(nb_frames_text) if nb_frames_text.isdigit() else 0
    format_payload = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    duration_sec = float(format_payload.get("duration", 0.0) or 0.0)
    if total_frames <= 0 and duration_sec > 0.0:
        total_frames = int(max(1, round(duration_sec * fps)))
    if total_frames <= 0:
        raise ValueError(f"Unable to resolve frame count for video: {video_path}")
    if duration_sec <= 0.0:
        duration_sec = float(total_frames) / float(fps)
    return VideoStreamInfo(
        fps=float(fps),
        total_frames=int(total_frames),
        duration_sec=float(duration_sec),
    )


def quantize_window_to_frames(
    start_sec: float,
    end_sec: float,
    stream_info: VideoStreamInfo,
    min_frames: int = 1,
) -> tuple[int, int, float, float]:
    """
    Convert second-based times into an inclusive-exclusive frame window.
    """
    safe_start = clamp_time(float(start_sec), stream_info.duration_sec)
    safe_end = clamp_time(float(end_sec), stream_info.duration_sec)
    start_frame = int(math.floor(safe_start * stream_info.fps + 1e-9))
    end_frame = int(math.ceil(safe_end * stream_info.fps - 1e-9))
    start_frame = max(0, min(start_frame, stream_info.total_frames - 1))
    end_frame = max(0, min(end_frame, stream_info.total_frames))
    if end_frame <= start_frame:
        end_frame = min(stream_info.total_frames, start_frame + 1)
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    required_frames = max(1, int(min_frames))
    current_frames = end_frame - start_frame
    if current_frames < required_frames and stream_info.total_frames > 1:
        missing_frames = required_frames - current_frames
        grow_left = missing_frames // 2
        grow_right = missing_frames - grow_left
        expanded_start = start_frame - grow_left
        expanded_end = end_frame + grow_right
        if expanded_start < 0:
            expanded_end += -expanded_start
            expanded_start = 0
        if expanded_end > stream_info.total_frames:
            shift_left = expanded_end - stream_info.total_frames
            expanded_start = max(0, expanded_start - shift_left)
            expanded_end = stream_info.total_frames
        if expanded_end - expanded_start < required_frames:
            if expanded_start == 0:
                expanded_end = min(stream_info.total_frames, required_frames)
            elif expanded_end == stream_info.total_frames:
                expanded_start = max(0, stream_info.total_frames - required_frames)
        if expanded_end > expanded_start:
            start_frame = expanded_start
            end_frame = expanded_end
    adjusted_start_sec = float(start_frame) / stream_info.fps
    adjusted_end_sec = float(end_frame) / stream_info.fps
    return start_frame, end_frame, adjusted_start_sec, adjusted_end_sec


def quantize_ordered_segments_to_frames(
    segment_windows_sec: list[tuple[float, float]],
    stream_info: VideoStreamInfo,
    min_frames: int = 1,
) -> list[tuple[int, int, float, float]]:
    """
    Quantize ordered segments into contiguous, non-overlapping frame windows.
    """
    if not segment_windows_sec:
        return []
    required_frames = max(1, int(min_frames))
    total_segments = len(segment_windows_sec)
    if required_frames * total_segments > stream_info.total_frames:
        required_frames = max(1, stream_info.total_frames // total_segments)
    if required_frames <= 0:
        required_frames = 1
    quantized: list[tuple[int, int, float, float]] = []
    first_start_sec = clamp_time(float(segment_windows_sec[0][0]), stream_info.duration_sec)
    current_start_frame = int(math.floor(first_start_sec * stream_info.fps + 1e-9))
    max_start_frame = max(0, stream_info.total_frames - required_frames * total_segments)
    current_start_frame = max(0, min(current_start_frame, max_start_frame))

    for index, (_, end_sec) in enumerate(segment_windows_sec):
        safe_end_sec = clamp_time(float(end_sec), stream_info.duration_sec)
        raw_end_frame = int(math.ceil(safe_end_sec * stream_info.fps - 1e-9))
        raw_end_frame = max(0, min(raw_end_frame, stream_info.total_frames))
        remaining_segments = total_segments - index - 1
        max_end_frame = stream_info.total_frames - remaining_segments * required_frames
        end_frame = max(current_start_frame + required_frames, raw_end_frame)
        end_frame = min(end_frame, max_end_frame)
        if end_frame <= current_start_frame:
            end_frame = min(stream_info.total_frames, current_start_frame + required_frames)
        adjusted_start_sec = float(current_start_frame) / stream_info.fps
        adjusted_end_sec = float(end_frame) / stream_info.fps
        quantized.append((current_start_frame, end_frame, adjusted_start_sec, adjusted_end_sec))
        current_start_frame = end_frame
    return quantized


def extract_scalar_from_frame(frame: dict[str, Any], key: str, fallback_value: float) -> float:
    """
    Read one scalar value from motion frame tensor-like fields.
    """
    raw_value = frame.get(key)
    if raw_value is None:
        return float(fallback_value)
    array_value = np.asarray(raw_value, dtype=np.float32).reshape(-1)
    if array_value.size <= 0:
        return float(fallback_value)
    return float(array_value[0])


def robust_mad_center_and_scale(series: np.ndarray, min_scale: float) -> tuple[float, float]:
    """
    Compute robust center and scale from median and MAD.
    """
    median_value = float(np.median(series))
    mad_value = float(np.median(np.abs(series - median_value)))
    scale_value = max(float(min_scale), 1.4826 * mad_value)
    return median_value, scale_value


def clip_series_outliers_robust(
    series: np.ndarray,
    z_threshold: float,
    min_scale: float,
) -> np.ndarray:
    """
    Clip scalar series to robust z-score bounds.
    """
    if series.size < 3:
        return series.astype(np.float32, copy=True)
    safe_threshold = max(0.1, float(z_threshold))
    center_value, scale_value = robust_mad_center_and_scale(series, min_scale=min_scale)
    half_span = safe_threshold * scale_value
    lower_bound = center_value - half_span
    upper_bound = center_value + half_span
    return np.clip(series, lower_bound, upper_bound).astype(np.float32, copy=False)


def limit_series_delta_robust(
    series: np.ndarray,
    z_threshold: float,
    min_step: float,
) -> np.ndarray:
    """
    Limit frame-to-frame jumps using robust scale on first differences.
    """
    if series.size < 3:
        return series.astype(np.float32, copy=True)
    deltas = np.diff(series)
    safe_threshold = max(0.1, float(z_threshold))
    _, delta_scale = robust_mad_center_and_scale(deltas, min_scale=max(float(min_step) * 0.5, 1e-8))
    max_step = max(float(min_step), safe_threshold * delta_scale)
    stabilized = np.empty_like(series, dtype=np.float32)
    stabilized[0] = float(series[0])
    for index in range(1, int(series.size)):
        proposed_step = float(series[index]) - float(stabilized[index - 1])
        clipped_step = float(np.clip(proposed_step, -max_step, max_step))
        stabilized[index] = float(stabilized[index - 1]) + clipped_step
    return stabilized


def ema_smooth_series(series: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply one-pass exponential moving average smoothing.
    """
    if series.size < 2:
        return series.astype(np.float32, copy=True)
    safe_alpha = float(np.clip(alpha, 0.0, 1.0))
    if safe_alpha >= 0.999:
        return series.astype(np.float32, copy=True)
    smoothed = np.empty_like(series, dtype=np.float32)
    smoothed[0] = float(series[0])
    keep_factor = 1.0 - safe_alpha
    for index in range(1, int(series.size)):
        smoothed[index] = safe_alpha * float(series[index]) + keep_factor * float(smoothed[index - 1])
    return smoothed


def stabilize_scalar_series(
    series: np.ndarray,
    outlier_z: float,
    delta_z: float,
    ema_alpha: float,
    min_scale: float,
    min_step: float,
) -> np.ndarray:
    """
    Apply robust clipping + jump limiter + EMA smoothing to scalar timeline.
    """
    clipped = clip_series_outliers_robust(series, z_threshold=outlier_z, min_scale=min_scale)
    bounded = limit_series_delta_robust(clipped, z_threshold=delta_z, min_step=min_step)
    return ema_smooth_series(bounded, alpha=ema_alpha)


def freeze_series_on_mask(series: np.ndarray, freeze_mask: np.ndarray) -> np.ndarray:
    """
    Freeze contiguous masked regions to the value right before each region.
    """
    if series.size < 2:
        return series.astype(np.float32, copy=True)
    if freeze_mask.size != series.size:
        raise ValueError("freeze_mask must match series length")
    if not bool(np.any(freeze_mask)):
        return series.astype(np.float32, copy=True)

    frozen = series.astype(np.float32, copy=True)
    mask = freeze_mask.astype(bool, copy=False)
    size = int(series.size)
    cursor = 0
    while cursor < size:
        if not bool(mask[cursor]):
            cursor += 1
            continue
        segment_start = cursor
        while cursor + 1 < size and bool(mask[cursor + 1]):
            cursor += 1
        segment_end = cursor
        anchor_index = max(0, segment_start - 1)
        anchor_value = float(frozen[anchor_index])
        frozen[segment_start : segment_end + 1] = anchor_value
        cursor += 1
    return frozen


def compute_idle_blink_intensity_from_payload(payload: dict[str, Any]) -> np.ndarray:
    """
    Compute normalized blink intensity timeline from idle motion payload.
    """
    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        return np.zeros(0, dtype=np.float32)
    if not isinstance(motion[0], dict):
        return np.zeros(0, dtype=np.float32)

    frame_count = len(motion)
    exp_sequence = np.zeros((frame_count, 21, 3), dtype=np.float32)
    for frame_index, frame in enumerate(motion):
        if not isinstance(frame, dict):
            continue
        exp_value = frame.get("exp")
        if exp_value is None:
            continue
        exp_sequence[frame_index] = np.asarray(exp_value, dtype=np.float32).reshape(21, 3)

    base_exp = exp_sequence[0]
    left_delta = exp_sequence[:, 11, 1] - base_exp[11, 1]
    right_delta = exp_sequence[:, 15, 1] - base_exp[15, 1]
    blink_raw = np.maximum(left_delta, right_delta).astype(np.float32, copy=False)
    median_value = float(np.percentile(blink_raw, 50))
    p99_value = float(np.percentile(blink_raw, 99))
    scale_value = max(1e-6, p99_value - median_value)
    blink_intensity = np.clip((blink_raw - median_value) / scale_value, 0.0, 1.0)
    return blink_intensity.astype(np.float32, copy=False)


def build_freeze_mask_from_blink_intensity(
    blink_intensity: np.ndarray,
    threshold: float,
    padding_frames: int,
    pre_frames: int,
    post_frames: int,
) -> np.ndarray:
    """
    Build one boolean frame mask around blink events.
    """
    if blink_intensity.size <= 0:
        return np.zeros(0, dtype=bool)
    safe_threshold = float(np.clip(threshold, 0.0, 1.0))
    safe_padding = max(0, int(padding_frames))
    safe_pre = max(0, int(pre_frames))
    safe_post = max(0, int(post_frames))

    freeze_mask = blink_intensity >= safe_threshold
    if not bool(np.any(freeze_mask)):
        return freeze_mask.astype(bool, copy=False)

    if safe_padding > 0:
        expanded = freeze_mask.copy()
        for shift in range(1, safe_padding + 1):
            expanded[:-shift] = np.logical_or(expanded[:-shift], freeze_mask[shift:])
            expanded[shift:] = np.logical_or(expanded[shift:], freeze_mask[:-shift])
        freeze_mask = expanded

    if safe_pre > 0:
        expanded = freeze_mask.copy()
        for shift in range(1, safe_pre + 1):
            expanded[shift:] = np.logical_or(expanded[shift:], freeze_mask[:-shift])
        freeze_mask = expanded

    if safe_post > 0:
        expanded = freeze_mask.copy()
        for shift in range(1, safe_post + 1):
            expanded[:-shift] = np.logical_or(expanded[:-shift], freeze_mask[shift:])
        freeze_mask = expanded

    return freeze_mask.astype(bool, copy=False)


def build_idle_mouth_freeze_alpha_mask(
    frame_width: int,
    frame_height: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    Build a soft alpha mask for mouth-area pixel freeze blending.
    """
    import cv2  # noqa: PLC0415

    safe_width = max(2, int(frame_width))
    safe_height = max(2, int(frame_height))
    x_min_ratio = float(np.clip(float(args.idle_pixel_mouth_freeze_x_min_ratio), 0.0, 1.0))
    x_max_ratio = float(np.clip(float(args.idle_pixel_mouth_freeze_x_max_ratio), 0.0, 1.0))
    y_min_ratio = float(np.clip(float(args.idle_pixel_mouth_freeze_y_min_ratio), 0.0, 1.0))
    y_max_ratio = float(np.clip(float(args.idle_pixel_mouth_freeze_y_max_ratio), 0.0, 1.0))
    x0_ratio = min(x_min_ratio, x_max_ratio)
    x1_ratio = max(x_min_ratio, x_max_ratio)
    y0_ratio = min(y_min_ratio, y_max_ratio)
    y1_ratio = max(y_min_ratio, y_max_ratio)

    x0 = int(round(x0_ratio * safe_width))
    x1 = int(round(x1_ratio * safe_width))
    y0 = int(round(y0_ratio * safe_height))
    y1 = int(round(y1_ratio * safe_height))

    x0 = max(0, min(x0, safe_width - 2))
    x1 = max(x0 + 1, min(x1, safe_width))
    y0 = max(0, min(y0, safe_height - 2))
    y1 = max(y0 + 1, min(y1, safe_height))

    alpha_mask = np.zeros((safe_height, safe_width), dtype=np.float32)
    alpha_mask[y0:y1, x0:x1] = 1.0

    feather_px = max(0.0, float(args.idle_pixel_mouth_freeze_feather_px))
    if feather_px > 0.0:
        alpha_mask = cv2.GaussianBlur(
            alpha_mask,
            ksize=(0, 0),
            sigmaX=feather_px,
            sigmaY=feather_px,
            borderType=cv2.BORDER_DEFAULT,
        )
        alpha_mask = np.clip(alpha_mask, 0.0, 1.0).astype(np.float32, copy=False)
    return alpha_mask


def apply_idle_mouth_pixel_freeze_to_video(
    args: argparse.Namespace,
    video_path: Path,
    freeze_mask: np.ndarray,
) -> tuple[int, int]:
    """
    Freeze mouth-area pixels in one rendered video for frames in freeze mask.
    Returns (processed_frame_count, frozen_frame_count).
    """
    import cv2  # noqa: PLC0415

    if freeze_mask.size <= 0 or not bool(np.any(freeze_mask)):
        return 0, 0
    if not video_path.exists():
        raise FileNotFoundError(f"Video for idle pixel freeze not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for idle pixel freeze: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0.0:
            fps = 25.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 1 or frame_height <= 1:
            raise RuntimeError(f"Invalid video dimensions for idle pixel freeze: {video_path}")

        alpha_mask = build_idle_mouth_freeze_alpha_mask(
            frame_width=frame_width,
            frame_height=frame_height,
            args=args,
        )
        alpha_3 = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32, copy=False)

        frames_dir = video_path.parent / f"__tmp_idle_mouth_freeze_{video_path.stem}"
        encoded_temp_path = video_path.parent / f"{video_path.stem}.idle_mouth_freeze_tmp.mp4"
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        if encoded_temp_path.exists():
            encoded_temp_path.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_index = 0
        frozen_frame_count = 0
        previous_original: np.ndarray | None = None
        anchor_frame: np.ndarray | None = None
        freeze_active = False
        png_write_flags = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            should_freeze = frame_index < freeze_mask.size and bool(freeze_mask[frame_index])
            if should_freeze and not freeze_active:
                anchor_frame = previous_original.copy() if previous_original is not None else frame.copy()
                freeze_active = True
            elif not should_freeze:
                freeze_active = False
                anchor_frame = None

            output_frame = frame
            if should_freeze and anchor_frame is not None:
                output_float = (
                    frame.astype(np.float32) * (1.0 - alpha_3) + anchor_frame.astype(np.float32) * alpha_3
                )
                output_frame = np.clip(output_float, 0.0, 255.0).astype(np.uint8)
                frozen_frame_count += 1

            frame_output_path = frames_dir / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(frame_output_path), output_frame, png_write_flags):
                raise RuntimeError(f"Failed to write processed frame: {frame_output_path}")
            previous_original = frame
            frame_index += 1

        if frame_index <= 0:
            raise RuntimeError(f"Video contained zero frames for idle pixel freeze: {video_path}")

        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{float(fps):.6f}",
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            str(args.segment_encoder_preset),
            "-crf",
            str(int(max(0, min(10, int(args.segment_encoder_crf))))),
            "-g",
            "1",
            "-keyint_min",
            "1",
            "-sc_threshold",
            "0",
            "-profile:v",
            str(args.segment_encoder_profile),
            "-level:v",
            str(args.segment_encoder_level),
        ]
        encoder_tune = str(args.segment_encoder_tune).strip()
        if encoder_tune:
            command.extend(["-tune", encoder_tune])
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-movflags",
                "+faststart",
                str(encoded_temp_path),
            ]
        )
        run_command(command)
        if not encoded_temp_path.exists():
            raise RuntimeError(f"Idle pixel-freeze output not generated: {encoded_temp_path}")
        shutil.move(str(encoded_temp_path), str(video_path))
        shutil.rmtree(frames_dir, ignore_errors=True)
        return frame_index, frozen_frame_count
    finally:
        capture.release()


def apply_idle_mouth_pixel_freeze_to_clips(
    args: argparse.Namespace,
    entry: DatasetEntrySpec,
    visemes: list[VisemeSpec],
    transitions: list[TransitionSpec],
    pkl_path: Path,
    full_org_path: Path,
    full_crop_path: Path,
) -> None:
    """
    Apply blink-window mouth pixel freeze to rendered idle clips.
    """
    if not bool(args.idle_pixel_mouth_freeze):
        return
    if not is_idle_entry_candidate(visemes=visemes, transitions=transitions):
        return
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found for idle pixel freeze: {pkl_path}")
    if not full_org_path.exists() or not full_crop_path.exists():
        raise FileNotFoundError(
            f"Rendered clips missing for idle pixel freeze: {full_org_path} / {full_crop_path}"
        )

    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported PKL payload for idle pixel freeze: {type(payload)}")

    blink_intensity = compute_idle_blink_intensity_from_payload(payload)
    freeze_mask = build_freeze_mask_from_blink_intensity(
        blink_intensity=blink_intensity,
        threshold=float(args.idle_pixel_mouth_freeze_threshold),
        padding_frames=int(args.idle_pixel_mouth_freeze_padding_frames),
        pre_frames=int(args.idle_pixel_mouth_freeze_pre_frames),
        post_frames=int(args.idle_pixel_mouth_freeze_post_frames),
    )
    if freeze_mask.size <= 0 or not bool(np.any(freeze_mask)):
        print(f"[warn] idle pixel freeze skipped (empty blink mask): dataset_index={entry.dataset_index}")
        return

    org_frames, org_frozen_frames = apply_idle_mouth_pixel_freeze_to_video(
        args=args,
        video_path=full_org_path,
        freeze_mask=freeze_mask,
    )
    crop_frames, crop_frozen_frames = apply_idle_mouth_pixel_freeze_to_video(
        args=args,
        video_path=full_crop_path,
        freeze_mask=freeze_mask,
    )

    metadata_payload = payload.get("motionIdleEnhancement")
    metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}
    metadata.update(
        {
            "idlePixelMouthFreeze": True,
            "idlePixelMouthFreezeAtUtc": datetime.now(timezone.utc).isoformat(),
            "idlePixelMouthFreezeThreshold": float(args.idle_pixel_mouth_freeze_threshold),
            "idlePixelMouthFreezePaddingFrames": int(args.idle_pixel_mouth_freeze_padding_frames),
            "idlePixelMouthFreezePreFrames": int(args.idle_pixel_mouth_freeze_pre_frames),
            "idlePixelMouthFreezePostFrames": int(args.idle_pixel_mouth_freeze_post_frames),
            "idlePixelMouthFreezeRoi": {
                "xMinRatio": float(args.idle_pixel_mouth_freeze_x_min_ratio),
                "xMaxRatio": float(args.idle_pixel_mouth_freeze_x_max_ratio),
                "yMinRatio": float(args.idle_pixel_mouth_freeze_y_min_ratio),
                "yMaxRatio": float(args.idle_pixel_mouth_freeze_y_max_ratio),
                "featherPx": float(args.idle_pixel_mouth_freeze_feather_px),
            },
            "idlePixelMouthFreezeMaskFrames": int(np.count_nonzero(freeze_mask)),
            "idlePixelMouthFreezeOrgFrames": int(org_frames),
            "idlePixelMouthFreezeOrgFrozenFrames": int(org_frozen_frames),
            "idlePixelMouthFreezeCropFrames": int(crop_frames),
            "idlePixelMouthFreezeCropFrozenFrames": int(crop_frozen_frames),
        }
    )
    payload["motionIdleEnhancement"] = metadata
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[ok] idle pixel mouth freeze -> dataset_index={entry.dataset_index} "
        f"mask_frames={int(np.count_nonzero(freeze_mask))} org={org_frozen_frames}/{org_frames} "
        f"crop={crop_frozen_frames}/{crop_frames}"
    )


def build_rotation_matrix_from_degrees(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build rotation matrix with FasterLivePortrait-compatible convention.
    """
    pitch = np.asarray([[float(pitch_deg)]], dtype=np.float32) / 180.0 * np.pi
    yaw = np.asarray([[float(yaw_deg)]], dtype=np.float32) / 180.0 * np.pi
    roll = np.asarray([[float(roll_deg)]], dtype=np.float32) / 180.0 * np.pi
    ones = np.ones((1, 1), dtype=np.float32)
    zeros = np.zeros((1, 1), dtype=np.float32)

    rot_x = np.concatenate(
        [
            ones,
            zeros,
            zeros,
            zeros,
            np.cos(pitch),
            -np.sin(pitch),
            zeros,
            np.sin(pitch),
            np.cos(pitch),
        ],
        axis=1,
    ).reshape(1, 3, 3)
    rot_y = np.concatenate(
        [
            np.cos(yaw),
            zeros,
            np.sin(yaw),
            zeros,
            ones,
            zeros,
            -np.sin(yaw),
            zeros,
            np.cos(yaw),
        ],
        axis=1,
    ).reshape(1, 3, 3)
    rot_z = np.concatenate(
        [
            np.cos(roll),
            -np.sin(roll),
            zeros,
            np.sin(roll),
            np.cos(roll),
            zeros,
            zeros,
            zeros,
            ones,
        ],
        axis=1,
    ).reshape(1, 3, 3)
    rotation = np.matmul(rot_z, np.matmul(rot_y, rot_x))
    return np.transpose(rotation, (0, 2, 1)).astype(np.float32)


def apply_motion_stabilization_to_payload(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """
    Stabilize eye and head motion channels to reduce abrupt visual jumps.
    """
    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        return payload
    if len(motion) < 3:
        return payload
    if not isinstance(motion[0], dict):
        return payload

    processed = copy.deepcopy(payload)
    processed_motion = processed.get("motion")
    if not isinstance(processed_motion, list) or not processed_motion:
        return payload

    frame_count = len(processed_motion)
    base_exp = np.asarray(processed_motion[0].get("exp"), dtype=np.float32).reshape(21, 3).copy()
    exp_sequence = np.repeat(base_exp[np.newaxis, :, :], frame_count, axis=0)
    for frame_index, frame in enumerate(processed_motion):
        if not isinstance(frame, dict):
            continue
        if "exp" in frame:
            exp_sequence[frame_index] = np.asarray(frame["exp"], dtype=np.float32).reshape(21, 3)

    safe_eye_soft = float(np.clip(float(args.eye_soft_factor), 0.0, 1.0))
    safe_eye_hard = float(np.clip(float(args.eye_hard_factor), 0.0, 1.0))
    safe_eye_min = float(min(float(args.eye_hard_dy_min), float(args.eye_hard_dy_max)))
    safe_eye_max = float(max(float(args.eye_hard_dy_min), float(args.eye_hard_dy_max)))

    for index in EYE_TAMED_SOFT_INDICES:
        exp_sequence[:, index, :] = base_exp[index, :] + (exp_sequence[:, index, :] - base_exp[index, :]) * safe_eye_soft
    for index in EYE_TAMED_HARD_INDICES:
        exp_sequence[:, index, :] = base_exp[index, :] + (exp_sequence[:, index, :] - base_exp[index, :]) * safe_eye_hard
        delta_y = exp_sequence[:, index, 1] - base_exp[index, 1]
        exp_sequence[:, index, 1] = base_exp[index, 1] + np.clip(delta_y, safe_eye_min, safe_eye_max)

    temporal_eye_indices = tuple(sorted(set(EYE_TAMED_SOFT_INDICES) | set(EYE_TAMED_HARD_INDICES)))
    for eye_index in temporal_eye_indices:
        eye_series = exp_sequence[:, eye_index, 1].astype(np.float32, copy=False)
        exp_sequence[:, eye_index, 1] = stabilize_scalar_series(
            eye_series,
            outlier_z=float(args.eye_outlier_z),
            delta_z=float(args.eye_delta_z),
            ema_alpha=float(args.eye_ema_alpha),
            min_scale=1e-4,
            min_step=2e-4,
        )

    first_frame = processed_motion[0]
    base_pitch = extract_scalar_from_frame(first_frame, "pitch", 0.0)
    base_yaw = extract_scalar_from_frame(first_frame, "yaw", 0.0)
    base_roll = extract_scalar_from_frame(first_frame, "roll", 0.0)
    base_scale = extract_scalar_from_frame(first_frame, "scale", 1.0)
    base_t = np.asarray(first_frame.get("t", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if base_t.size < 3:
        base_t = np.pad(base_t, (0, max(0, 3 - base_t.size)), mode="constant")

    pitch_series = np.zeros(frame_count, dtype=np.float32)
    yaw_series = np.zeros(frame_count, dtype=np.float32)
    roll_series = np.zeros(frame_count, dtype=np.float32)
    scale_series = np.zeros(frame_count, dtype=np.float32)
    translation_series = np.zeros((frame_count, 3), dtype=np.float32)
    for frame_index, frame in enumerate(processed_motion):
        if not isinstance(frame, dict):
            frame = {}
        previous_index = max(0, frame_index - 1)
        pitch_fallback = pitch_series[previous_index] if frame_index > 0 else base_pitch
        yaw_fallback = yaw_series[previous_index] if frame_index > 0 else base_yaw
        roll_fallback = roll_series[previous_index] if frame_index > 0 else base_roll
        scale_fallback = scale_series[previous_index] if frame_index > 0 else base_scale
        pitch_series[frame_index] = extract_scalar_from_frame(frame, "pitch", pitch_fallback)
        yaw_series[frame_index] = extract_scalar_from_frame(frame, "yaw", yaw_fallback)
        roll_series[frame_index] = extract_scalar_from_frame(frame, "roll", roll_fallback)
        scale_series[frame_index] = extract_scalar_from_frame(frame, "scale", scale_fallback)

        raw_translation = np.asarray(frame.get("t", base_t), dtype=np.float32).reshape(-1)
        if raw_translation.size < 3:
            raw_translation = np.pad(raw_translation, (0, max(0, 3 - raw_translation.size)), mode="constant")
        translation_series[frame_index, :] = raw_translation[:3]

    pitch_series = stabilize_scalar_series(
        pitch_series,
        outlier_z=float(args.head_outlier_z),
        delta_z=float(args.head_delta_z),
        ema_alpha=float(args.head_ema_alpha),
        min_scale=0.05,
        min_step=0.08,
    )
    yaw_series = stabilize_scalar_series(
        yaw_series,
        outlier_z=float(args.head_outlier_z),
        delta_z=float(args.head_delta_z),
        ema_alpha=float(args.head_ema_alpha),
        min_scale=0.05,
        min_step=0.08,
    )
    roll_series = stabilize_scalar_series(
        roll_series,
        outlier_z=float(args.head_outlier_z),
        delta_z=float(args.head_delta_z),
        ema_alpha=float(args.head_ema_alpha),
        min_scale=0.05,
        min_step=0.08,
    )
    scale_series = stabilize_scalar_series(
        scale_series,
        outlier_z=DEFAULT_SCALE_OUTLIER_Z,
        delta_z=DEFAULT_SCALE_DELTA_Z,
        ema_alpha=DEFAULT_SCALE_EMA_ALPHA,
        min_scale=5e-4,
        min_step=1e-3,
    )
    for axis_index in range(3):
        translation_series[:, axis_index] = stabilize_scalar_series(
            translation_series[:, axis_index],
            outlier_z=float(args.translation_outlier_z),
            delta_z=float(args.translation_delta_z),
            ema_alpha=float(args.translation_ema_alpha),
            min_scale=1e-4,
            min_step=3.5e-4,
        )

    for frame_index, frame in enumerate(processed_motion):
        if not isinstance(frame, dict):
            continue
        frame["exp"] = exp_sequence[frame_index].reshape(1, 21, 3).astype(np.float32)
        frame["pitch"] = np.asarray([[pitch_series[frame_index]]], dtype=np.float32)
        frame["yaw"] = np.asarray([[yaw_series[frame_index]]], dtype=np.float32)
        frame["roll"] = np.asarray([[roll_series[frame_index]]], dtype=np.float32)
        frame["scale"] = np.asarray([[scale_series[frame_index]]], dtype=np.float32)
        frame["t"] = translation_series[frame_index].reshape(1, 3).astype(np.float32)
        frame["R"] = build_rotation_matrix_from_degrees(
            pitch_deg=float(pitch_series[frame_index]),
            yaw_deg=float(yaw_series[frame_index]),
            roll_deg=float(roll_series[frame_index]),
        )

    metadata_payload = processed.get("motionPostprocess")
    metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}
    metadata.update(
        {
            "profile": MOTION_STABILIZATION_PROFILE_V1,
            "stabilizedAtUtc": datetime.now(timezone.utc).isoformat(),
            "eyeSoftFactor": safe_eye_soft,
            "eyeHardFactor": safe_eye_hard,
            "eyeHardDyMin": safe_eye_min,
            "eyeHardDyMax": safe_eye_max,
            "headOutlierZ": float(args.head_outlier_z),
            "headDeltaZ": float(args.head_delta_z),
            "headEmaAlpha": float(args.head_ema_alpha),
            "translationOutlierZ": float(args.translation_outlier_z),
            "translationDeltaZ": float(args.translation_delta_z),
            "translationEmaAlpha": float(args.translation_ema_alpha),
            "eyeOutlierZ": float(args.eye_outlier_z),
            "eyeDeltaZ": float(args.eye_delta_z),
            "eyeEmaAlpha": float(args.eye_ema_alpha),
        }
    )
    processed["motionPostprocess"] = metadata
    return processed


def apply_motion_stabilization_to_pkl(pkl_path: Path, args: argparse.Namespace) -> None:
    """
    Load one PKL motion file, stabilize it, and write back in-place.
    """
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported PKL payload type for stabilization: {type(payload)}")
    if not bool(args.force_motion_stabilization):
        metadata_payload = payload.get("motionPostprocess")
        if isinstance(metadata_payload, dict) and metadata_payload.get("profile") == MOTION_STABILIZATION_PROFILE_V1:
            print(f"[skip] motion stabilization already applied: {pkl_path}")
            return
    processed_payload = apply_motion_stabilization_to_payload(payload=payload, args=args)
    with pkl_path.open("wb") as handle:
        pickle.dump(processed_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[ok] motion stabilization -> {pkl_path}")


def is_idle_entry_candidate(visemes: list[VisemeSpec], transitions: list[TransitionSpec]) -> bool:
    """
    Return True when the entry represents a dedicated idle state segment.
    """
    if len(visemes) != 1:
        return False
    if transitions:
        return False
    return str(visemes[0].viseme).strip().upper() == "IDLE"


def build_temporal_edge_envelope(frame_count: int, fps: float, fade_sec: float) -> np.ndarray:
    """
    Build smooth edge envelope to keep boundary frames close to original pose.
    """
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)
    envelope = np.ones(frame_count, dtype=np.float32)
    safe_fps = max(1e-6, float(fps))
    edge_frames = int(max(0, round(max(0.0, float(fade_sec)) * safe_fps)))
    edge_frames = min(edge_frames, max(0, frame_count // 2))
    if edge_frames <= 0:
        return envelope
    ramp_positions = np.linspace(0.0, 1.0, edge_frames, endpoint=False, dtype=np.float32)
    ramp = ramp_positions * ramp_positions * (3.0 - 2.0 * ramp_positions)
    envelope[:edge_frames] = np.minimum(envelope[:edge_frames], ramp)
    envelope[-edge_frames:] = np.minimum(envelope[-edge_frames:], ramp[::-1])
    return envelope


def build_idle_dual_wave(
    time_sec: np.ndarray,
    primary_hz: float,
    secondary_scale: float,
    phase_primary: float,
    phase_secondary: float,
) -> np.ndarray:
    """
    Create dual-frequency smooth oscillator used for subtle idle drift.
    """
    safe_primary_hz = max(0.005, float(primary_hz))
    safe_secondary_scale = max(1.05, float(secondary_scale))
    primary_wave = np.sin((2.0 * np.pi * safe_primary_hz * time_sec) + float(phase_primary))
    secondary_wave = np.sin((2.0 * np.pi * safe_primary_hz * safe_secondary_scale * time_sec) + float(phase_secondary))
    return (primary_wave + 0.45 * secondary_wave).astype(np.float32, copy=False)


def smoothstep01(values: np.ndarray) -> np.ndarray:
    """
    Smooth step interpolation on [0, 1].
    """
    clipped = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32, copy=False)


def build_idle_blink_pulse(
    time_sec: np.ndarray,
    center_sec: float,
    close_sec: float,
    hold_sec: float,
) -> np.ndarray:
    """
    Build one blink pulse with close-ramp, closed-hold, and open-ramp.
    """
    safe_center = float(center_sec)
    safe_close = max(0.016, float(close_sec))
    safe_hold = max(0.0, float(hold_sec))
    close_ramp_sec = max(0.016, safe_close * 0.55)
    open_ramp_sec = max(0.024, safe_close * 0.78)
    hold_start_sec = safe_center - (safe_hold * 0.5)
    hold_end_sec = safe_center + (safe_hold * 0.5)
    close_start_sec = hold_start_sec - close_ramp_sec

    rise_curve = smoothstep01((time_sec - close_start_sec) / max(close_ramp_sec, 1e-6))
    fall_curve = 1.0 - smoothstep01((time_sec - hold_end_sec) / max(open_ramp_sec, 1e-6))
    pulse = np.minimum(rise_curve, fall_curve).astype(np.float32, copy=False)
    hold_mask = np.logical_and(time_sec >= hold_start_sec, time_sec <= hold_end_sec)
    pulse[hold_mask] = 1.0
    return np.clip(pulse, 0.0, 1.0).astype(np.float32, copy=False)


def build_idle_blink_curves(
    time_sec: np.ndarray,
    envelope: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Synthesize asymmetric left/right blink curves with full-closure hold and irregular timing.
    """
    frame_count = int(time_sec.size)
    left_curve = np.zeros(frame_count, dtype=np.float32)
    right_curve = np.zeros(frame_count, dtype=np.float32)
    if frame_count <= 0:
        return left_curve, right_curve, 0

    clip_duration_sec = float(time_sec[-1]) if frame_count > 1 else 0.0
    fade_buffer_sec = max(0.2, float(args.idle_edge_fade_sec) + 0.35)
    min_center_sec = fade_buffer_sec
    max_center_sec = clip_duration_sec - fade_buffer_sec
    if max_center_sec <= min_center_sec:
        return left_curve, right_curve, 0

    min_interval_sec = max(0.6, float(args.idle_blink_min_interval_sec))
    max_interval_sec = max(min_interval_sec, float(args.idle_blink_max_interval_sec))
    base_amplitude = max(0.0, float(args.idle_blink_amplitude))
    legacy_duration_sec = max(0.02, float(args.idle_blink_duration_sec))
    base_close_sec = max(0.018, float(args.idle_blink_close_sec), legacy_duration_sec * 0.82)
    base_hold_sec = max(0.0, float(args.idle_blink_hold_sec))
    double_probability = float(np.clip(float(args.idle_blink_double_probability), 0.0, 0.95))
    interval_irregularity = float(np.clip(float(args.idle_blink_irregularity), 0.0, 1.0))
    if base_amplitude <= 0.0:
        return left_curve, right_curve, 0

    blink_count = 0
    cursor_sec = min_center_sec + float(rng.uniform(0.25, 0.95))
    interval_mean_sec = max(0.35, (min_interval_sec + max_interval_sec) * 0.5)
    interval_sigma = max(0.05, 0.15 + 0.55 * interval_irregularity)
    while cursor_sec < max_center_sec:
        event_centers = [cursor_sec]
        if float(rng.uniform(0.0, 1.0)) < double_probability:
            double_gap_sec = float(np.clip(rng.normal(0.17, 0.04), 0.09, 0.30))
            next_center = cursor_sec + double_gap_sec
            if next_center < max_center_sec:
                event_centers.append(next_center)

        last_center_sec = cursor_sec
        for event_index, event_center in enumerate(event_centers):
            blink_count += 1
            event_amplitude = base_amplitude * float(rng.uniform(0.92, 1.36))
            if event_index > 0:
                event_amplitude *= float(rng.uniform(0.66, 0.90))
            event_close_sec = max(0.016, base_close_sec * float(rng.uniform(0.78, 1.34)))
            event_hold_sec = max(0.0, base_hold_sec * float(rng.uniform(0.72, 1.42)))
            asym_sec = float(rng.uniform(-0.018, 0.018))
            left_center = event_center + asym_sec
            right_center = event_center - asym_sec
            left_amp = event_amplitude * float(rng.uniform(0.90, 1.12))
            right_amp = event_amplitude * float(rng.uniform(0.90, 1.12))
            left_pulse = build_idle_blink_pulse(
                time_sec=time_sec,
                center_sec=left_center,
                close_sec=event_close_sec,
                hold_sec=event_hold_sec,
            )
            right_pulse = build_idle_blink_pulse(
                time_sec=time_sec,
                center_sec=right_center,
                close_sec=event_close_sec,
                hold_sec=event_hold_sec,
            )
            left_curve += left_amp * left_pulse
            right_curve += right_amp * right_pulse
            last_center_sec = event_center

        sampled_interval_sec = float(np.exp(rng.normal(np.log(interval_mean_sec), interval_sigma)))
        sampled_interval_sec = float(np.clip(sampled_interval_sec, min_interval_sec, max_interval_sec * 1.25))
        cursor_sec = last_center_sec + sampled_interval_sec

    amplitude_cap = max(1e-6, base_amplitude * 2.65)
    left_curve = np.clip(left_curve, 0.0, amplitude_cap)
    right_curve = np.clip(right_curve, 0.0, amplitude_cap)
    left_curve = (left_curve * envelope).astype(np.float32, copy=False)
    right_curve = (right_curve * envelope).astype(np.float32, copy=False)
    return left_curve, right_curve, blink_count


def apply_idle_motion_enhancement_to_payload(
    payload: dict[str, Any],
    args: argparse.Namespace,
    entry: DatasetEntrySpec,
    visemes: list[VisemeSpec],
    transitions: list[TransitionSpec],
) -> dict[str, Any]:
    """
    Add subtle idle blink and micro-motion for dedicated IDLE clips.
    """
    if not bool(args.idle_motion_enhancement):
        return payload
    if not is_idle_entry_candidate(visemes=visemes, transitions=transitions):
        return payload
    motion = payload.get("motion")
    if not isinstance(motion, list) or len(motion) < 3:
        return payload
    if not isinstance(motion[0], dict):
        return payload

    processed = copy.deepcopy(payload)
    processed_motion = processed.get("motion")
    if not isinstance(processed_motion, list) or len(processed_motion) < 3:
        return payload

    frame_count = len(processed_motion)
    safe_fps = max(1.0, float(payload.get("output_fps", 25.0) or 25.0))
    duration_sec = float(frame_count - 1) / safe_fps
    time_sec = np.linspace(0.0, duration_sec, frame_count, dtype=np.float32)
    envelope = build_temporal_edge_envelope(
        frame_count=frame_count,
        fps=safe_fps,
        fade_sec=float(args.idle_edge_fade_sec),
    )

    first_frame = processed_motion[0]
    base_exp = np.asarray(first_frame.get("exp"), dtype=np.float32).reshape(21, 3).copy()
    exp_sequence = np.repeat(base_exp[np.newaxis, :, :], frame_count, axis=0)
    pitch_series = np.zeros(frame_count, dtype=np.float32)
    yaw_series = np.zeros(frame_count, dtype=np.float32)
    roll_series = np.zeros(frame_count, dtype=np.float32)
    scale_series = np.zeros(frame_count, dtype=np.float32)
    translation_series = np.zeros((frame_count, 3), dtype=np.float32)

    base_pitch = extract_scalar_from_frame(first_frame, "pitch", 0.0)
    base_yaw = extract_scalar_from_frame(first_frame, "yaw", 0.0)
    base_roll = extract_scalar_from_frame(first_frame, "roll", 0.0)
    base_scale = extract_scalar_from_frame(first_frame, "scale", 1.0)
    base_t = np.asarray(first_frame.get("t", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if base_t.size < 3:
        base_t = np.pad(base_t, (0, max(0, 3 - base_t.size)), mode="constant")

    for frame_index, frame in enumerate(processed_motion):
        if not isinstance(frame, dict):
            frame = {}
        if "exp" in frame:
            exp_sequence[frame_index] = np.asarray(frame["exp"], dtype=np.float32).reshape(21, 3)
        previous_index = max(0, frame_index - 1)
        pitch_fallback = pitch_series[previous_index] if frame_index > 0 else base_pitch
        yaw_fallback = yaw_series[previous_index] if frame_index > 0 else base_yaw
        roll_fallback = roll_series[previous_index] if frame_index > 0 else base_roll
        scale_fallback = scale_series[previous_index] if frame_index > 0 else base_scale
        pitch_series[frame_index] = extract_scalar_from_frame(frame, "pitch", pitch_fallback)
        yaw_series[frame_index] = extract_scalar_from_frame(frame, "yaw", yaw_fallback)
        roll_series[frame_index] = extract_scalar_from_frame(frame, "roll", roll_fallback)
        scale_series[frame_index] = extract_scalar_from_frame(frame, "scale", scale_fallback)
        raw_translation = np.asarray(frame.get("t", base_t), dtype=np.float32).reshape(-1)
        if raw_translation.size < 3:
            raw_translation = np.pad(raw_translation, (0, max(0, 3 - raw_translation.size)), mode="constant")
        translation_series[frame_index, :] = raw_translation[:3]

    entry_seed = int(sum(ord(char) for char in str(entry.dataset_id)) + int(entry.dataset_index) * 9973)
    rng = np.random.default_rng(int(args.seed) + entry_seed)
    base_primary_hz = max(0.01, float(args.idle_primary_freq_hz))
    secondary_scale = max(1.05, float(args.idle_secondary_freq_scale))
    yaw_wave = build_idle_dual_wave(
        time_sec=time_sec,
        primary_hz=base_primary_hz * float(rng.uniform(0.9, 1.1)),
        secondary_scale=secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    pitch_wave = build_idle_dual_wave(
        time_sec=time_sec,
        primary_hz=base_primary_hz * float(rng.uniform(0.85, 1.05)),
        secondary_scale=secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    roll_wave = build_idle_dual_wave(
        time_sec=time_sec,
        primary_hz=base_primary_hz * float(rng.uniform(0.95, 1.2)),
        secondary_scale=secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    tx_wave = build_idle_dual_wave(
        time_sec=time_sec,
        primary_hz=base_primary_hz * float(rng.uniform(0.82, 1.08)),
        secondary_scale=secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    ty_wave = build_idle_dual_wave(
        time_sec=time_sec,
        primary_hz=base_primary_hz * float(rng.uniform(0.88, 1.16)),
        secondary_scale=secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    tz_wave = build_idle_dual_wave(
        time_sec=time_sec,
        primary_hz=base_primary_hz * float(rng.uniform(0.78, 1.04)),
        secondary_scale=secondary_scale,
        phase_primary=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_secondary=float(rng.uniform(0.0, 2.0 * np.pi)),
    )

    pitch_series += envelope * float(max(0.0, args.idle_pitch_amplitude_deg)) * pitch_wave
    yaw_series += envelope * float(max(0.0, args.idle_yaw_amplitude_deg)) * yaw_wave
    roll_series += envelope * float(max(0.0, args.idle_roll_amplitude_deg)) * roll_wave
    translation_amplitude = float(max(0.0, args.idle_translation_amplitude))
    translation_series[:, 0] += envelope * translation_amplitude * tx_wave
    translation_series[:, 1] += envelope * translation_amplitude * 0.85 * ty_wave
    translation_series[:, 2] += envelope * translation_amplitude * 0.65 * tz_wave

    left_blink_curve, right_blink_curve, blink_count = build_idle_blink_curves(
        time_sec=time_sec,
        envelope=envelope,
        rng=rng,
        args=args,
    )
    blink_sign = float(np.sign(float(args.idle_blink_sign) or 1.0))
    if blink_sign == 0.0:
        blink_sign = 1.0
    exp_sequence[:, 11, 1] += blink_sign * left_blink_curve
    exp_sequence[:, 15, 1] += blink_sign * right_blink_curve
    shared_blink_curve = (left_blink_curve + right_blink_curve) * 0.5
    blink_peak_delta = float(max(float(np.max(left_blink_curve)), float(np.max(right_blink_curve))))
    shared_peak = max(1e-6, float(np.max(shared_blink_curve)))
    blink_intensity = np.clip(shared_blink_curve / shared_peak, 0.0, 1.0).astype(np.float32, copy=False)
    blink_soft_upper_scale = float(np.clip(float(args.idle_blink_soft_upper_scale), 0.0, 0.35))
    blink_soft_lower_scale = float(np.clip(float(args.idle_blink_soft_lower_scale), 0.0, 0.25))
    blink_center_scale = float(np.clip(float(args.idle_blink_center_scale), 0.0, 0.4))
    if blink_soft_upper_scale > 0.0:
        for soft_index in (1, 2, 4, 5):
            exp_sequence[:, soft_index, 1] += blink_sign * shared_blink_curve * blink_soft_upper_scale
    if blink_soft_lower_scale > 0.0:
        for soft_index in (0, 3, 7, 10):
            exp_sequence[:, soft_index, 1] += blink_sign * shared_blink_curve * blink_soft_lower_scale
    if blink_center_scale > 0.0:
        exp_sequence[:, 13, 1] += blink_sign * shared_blink_curve * blink_center_scale

    blink_close_target_scale = max(1.0, float(args.idle_blink_close_target_scale))
    blink_close_target_delta = max(
        float(args.idle_blink_amplitude) * blink_close_target_scale,
        blink_peak_delta * 1.05 + 1e-4,
    )
    blink_center_force_scale = float(np.clip(float(args.idle_blink_center_force_scale), 0.0, 1.0))
    for blink_index, scale_factor in ((11, 1.0), (13, blink_center_force_scale), (15, 1.0)):
        forced_delta = blink_close_target_delta * scale_factor * blink_intensity
        forced_series = base_exp[blink_index, 1] + blink_sign * forced_delta
        if blink_sign >= 0.0:
            exp_sequence[:, blink_index, 1] = np.maximum(exp_sequence[:, blink_index, 1], forced_series)
        else:
            exp_sequence[:, blink_index, 1] = np.minimum(exp_sequence[:, blink_index, 1], forced_series)

    hard_dy_min = min(float(args.eye_hard_dy_min), float(args.eye_hard_dy_max)) * 1.25
    base_hard_dy_max = max(float(args.eye_hard_dy_min), float(args.eye_hard_dy_max))
    hard_dy_max = max(base_hard_dy_max * 3.95, blink_close_target_delta * 1.78 + 1.6e-3)
    for hard_index in EYE_TAMED_HARD_INDICES:
        delta_hard_y = exp_sequence[:, hard_index, 1] - base_exp[hard_index, 1]
        exp_sequence[:, hard_index, 1] = base_exp[hard_index, 1] + np.clip(delta_hard_y, hard_dy_min, hard_dy_max)

    mouth_neutral_strength = float(np.clip(float(args.idle_mouth_neutral_strength), 0.0, 1.0))
    mouth_target_quantile = float(np.clip(float(args.idle_mouth_target_quantile), 0.0, 1.0))
    mouth_floor_sigma = max(0.0, float(args.idle_mouth_floor_sigma))
    mouth_blink_lock_strength = float(np.clip(float(args.idle_mouth_blink_lock_strength), 0.0, 1.0))
    mouth_blink_lock_threshold = float(np.clip(float(args.idle_mouth_blink_lock_threshold), 0.0, 0.98))
    mouth_blink_hard_lock_threshold = float(
        np.clip(float(args.idle_mouth_blink_hard_lock_threshold), mouth_blink_lock_threshold, 1.0)
    )
    mouth_blink_reference_ema_alpha = float(np.clip(float(args.idle_mouth_blink_reference_ema_alpha), 0.0, 1.0))
    mouth_blink_freeze_threshold = float(np.clip(float(args.idle_mouth_blink_freeze_threshold), 0.0, 1.0))
    mouth_blink_freeze_padding_frames = max(0, int(args.idle_mouth_blink_freeze_padding_frames))
    mouth_blink_lock_weight: np.ndarray | None = None
    mouth_blink_hard_lock_mask: np.ndarray | None = None
    mouth_blink_freeze_mask: np.ndarray | None = None
    if mouth_blink_lock_strength > 0.0:
        lock_base = (blink_intensity - mouth_blink_lock_threshold) / max(1e-6, 1.0 - mouth_blink_lock_threshold)
        mouth_blink_lock_weight = mouth_blink_lock_strength * smoothstep01(lock_base)
        mouth_blink_hard_lock_mask = blink_intensity >= mouth_blink_hard_lock_threshold
    if frame_count > 0:
        base_freeze_mask = blink_intensity >= mouth_blink_freeze_threshold
        if mouth_blink_freeze_padding_frames > 0 and bool(np.any(base_freeze_mask)):
            expanded_mask = base_freeze_mask.copy()
            for shift in range(1, mouth_blink_freeze_padding_frames + 1):
                expanded_mask[:-shift] = np.logical_or(expanded_mask[:-shift], base_freeze_mask[shift:])
                expanded_mask[shift:] = np.logical_or(expanded_mask[shift:], base_freeze_mask[:-shift])
            base_freeze_mask = expanded_mask
        if bool(np.any(base_freeze_mask)):
            mouth_blink_freeze_mask = base_freeze_mask
    if mouth_neutral_strength > 0.0:
        mouth_indices = tuple(sorted(set(IDLE_MOUTH_PRIMARY_INDICES) | set(IDLE_MOUTH_SUPPORT_INDICES)))
        for mouth_index in mouth_indices:
            for axis_index in (0, 1, 2):
                mouth_series = exp_sequence[:, mouth_index, axis_index].astype(np.float32, copy=False)
                stabilized_series = stabilize_scalar_series(
                    mouth_series,
                    outlier_z=float(args.idle_mouth_outlier_z),
                    delta_z=float(args.idle_mouth_delta_z),
                    ema_alpha=float(args.idle_mouth_ema_alpha),
                    min_scale=1e-4,
                    min_step=1.5e-4,
                )
                target_value = float(np.quantile(stabilized_series, mouth_target_quantile))
                axis_strength = mouth_neutral_strength
                if axis_index == 2 and mouth_index in IDLE_MOUTH_PRIMARY_INDICES:
                    axis_strength = float(np.clip(mouth_neutral_strength * 1.2, 0.0, 1.0))
                blended_series = target_value + (stabilized_series - target_value) * (1.0 - axis_strength)
                if axis_index == 1 and mouth_index in IDLE_MOUTH_PRIMARY_INDICES and mouth_floor_sigma > 0.0:
                    _, scale_value = robust_mad_center_and_scale(blended_series, min_scale=1e-4)
                    lower_bound = target_value - mouth_floor_sigma * scale_value
                    blended_series = np.maximum(blended_series, lower_bound)
                if mouth_blink_lock_weight is not None:
                    mouth_blink_reference_series = ema_smooth_series(
                        blended_series.astype(np.float32, copy=False),
                        alpha=mouth_blink_reference_ema_alpha,
                    )
                    blended_series = blended_series + (mouth_blink_reference_series - blended_series) * mouth_blink_lock_weight
                if mouth_blink_hard_lock_mask is not None and bool(np.any(mouth_blink_hard_lock_mask)):
                    blended_series = np.where(mouth_blink_hard_lock_mask, mouth_blink_reference_series, blended_series)
                exp_sequence[:, mouth_index, axis_index] = blended_series.astype(np.float32, copy=False)

    if mouth_blink_freeze_mask is not None:
        for mouth_index in IDLE_MOUTH_STRICT_FREEZE_INDICES:
            for axis_index in (0, 1, 2):
                exp_sequence[:, mouth_index, axis_index] = freeze_series_on_mask(
                    exp_sequence[:, mouth_index, axis_index].astype(np.float32, copy=False),
                    mouth_blink_freeze_mask,
                )
        for exp_index in range(exp_sequence.shape[1]):
            for axis_index in range(exp_sequence.shape[2]):
                if exp_index in IDLE_BLINK_ACTIVE_EYE_INDICES and axis_index == 1:
                    continue
                exp_sequence[:, exp_index, axis_index] = freeze_series_on_mask(
                    exp_sequence[:, exp_index, axis_index].astype(np.float32, copy=False),
                    mouth_blink_freeze_mask,
                )
        pitch_series = freeze_series_on_mask(pitch_series.astype(np.float32, copy=False), mouth_blink_freeze_mask)
        yaw_series = freeze_series_on_mask(yaw_series.astype(np.float32, copy=False), mouth_blink_freeze_mask)
        roll_series = freeze_series_on_mask(roll_series.astype(np.float32, copy=False), mouth_blink_freeze_mask)
        scale_series = freeze_series_on_mask(scale_series.astype(np.float32, copy=False), mouth_blink_freeze_mask)
        for axis_index in range(3):
            translation_series[:, axis_index] = freeze_series_on_mask(
                translation_series[:, axis_index].astype(np.float32, copy=False),
                mouth_blink_freeze_mask,
            )

    for frame_index, frame in enumerate(processed_motion):
        if not isinstance(frame, dict):
            continue
        frame["exp"] = exp_sequence[frame_index].reshape(1, 21, 3).astype(np.float32)
        frame["pitch"] = np.asarray([[pitch_series[frame_index]]], dtype=np.float32)
        frame["yaw"] = np.asarray([[yaw_series[frame_index]]], dtype=np.float32)
        frame["roll"] = np.asarray([[roll_series[frame_index]]], dtype=np.float32)
        frame["scale"] = np.asarray([[scale_series[frame_index]]], dtype=np.float32)
        frame["t"] = translation_series[frame_index].reshape(1, 3).astype(np.float32)
        frame["R"] = build_rotation_matrix_from_degrees(
            pitch_deg=float(pitch_series[frame_index]),
            yaw_deg=float(yaw_series[frame_index]),
            roll_deg=float(roll_series[frame_index]),
        )

    metadata_payload = processed.get("motionIdleEnhancement")
    metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}
    metadata.update(
        {
            "profile": IDLE_MOTION_PROFILE_V1,
            "enhancedAtUtc": datetime.now(timezone.utc).isoformat(),
            "datasetId": entry.dataset_id,
            "datasetIndex": int(entry.dataset_index),
            "frameCount": int(frame_count),
            "fps": float(safe_fps),
            "edgeFadeSec": float(args.idle_edge_fade_sec),
            "primaryFrequencyHz": float(args.idle_primary_freq_hz),
            "secondaryFrequencyScale": float(args.idle_secondary_freq_scale),
            "yawAmplitudeDeg": float(args.idle_yaw_amplitude_deg),
            "pitchAmplitudeDeg": float(args.idle_pitch_amplitude_deg),
            "rollAmplitudeDeg": float(args.idle_roll_amplitude_deg),
            "translationAmplitude": float(args.idle_translation_amplitude),
            "blinkMinIntervalSec": float(args.idle_blink_min_interval_sec),
            "blinkMaxIntervalSec": float(args.idle_blink_max_interval_sec),
            "blinkDurationSec": float(args.idle_blink_duration_sec),
            "blinkAmplitude": float(args.idle_blink_amplitude),
            "blinkSign": float(blink_sign),
            "blinkCount": int(blink_count),
            "blinkPeakDelta": float(blink_peak_delta),
            "blinkCloseSec": float(args.idle_blink_close_sec),
            "blinkHoldSec": float(args.idle_blink_hold_sec),
            "blinkDoubleProbability": float(args.idle_blink_double_probability),
            "blinkIrregularity": float(args.idle_blink_irregularity),
            "blinkCloseTargetScale": float(args.idle_blink_close_target_scale),
            "blinkCloseTargetDelta": float(blink_close_target_delta),
            "blinkHardDyMaxUsed": float(hard_dy_max),
            "blinkSoftUpperScale": float(blink_soft_upper_scale),
            "blinkSoftLowerScale": float(blink_soft_lower_scale),
            "blinkCenterScale": float(blink_center_scale),
            "blinkCenterForceScale": float(blink_center_force_scale),
            "mouthNeutralStrength": float(mouth_neutral_strength),
            "mouthTargetQuantile": float(mouth_target_quantile),
            "mouthFloorSigma": float(mouth_floor_sigma),
            "mouthOutlierZ": float(args.idle_mouth_outlier_z),
            "mouthDeltaZ": float(args.idle_mouth_delta_z),
            "mouthEmaAlpha": float(args.idle_mouth_ema_alpha),
            "mouthBlinkLockStrength": float(mouth_blink_lock_strength),
            "mouthBlinkLockThreshold": float(mouth_blink_lock_threshold),
            "mouthBlinkHardLockThreshold": float(mouth_blink_hard_lock_threshold),
            "mouthBlinkReferenceEmaAlpha": float(mouth_blink_reference_ema_alpha),
            "mouthBlinkFreezeThreshold": float(mouth_blink_freeze_threshold),
            "mouthBlinkFreezePaddingFrames": int(mouth_blink_freeze_padding_frames),
            "mouthBlinkFreezeActiveFrames": int(np.count_nonzero(mouth_blink_freeze_mask))
            if mouth_blink_freeze_mask is not None
            else 0,
            "mouthBlinkFreezeIndices": [int(index) for index in IDLE_MOUTH_STRICT_FREEZE_INDICES],
            "mouthBlinkFreezePose": bool(mouth_blink_freeze_mask is not None),
            "blinkFreezeExceptEyeY": bool(mouth_blink_freeze_mask is not None),
            "blinkActiveEyeIndices": [int(index) for index in IDLE_BLINK_ACTIVE_EYE_INDICES],
        }
    )
    processed["motionIdleEnhancement"] = metadata
    return processed


def apply_idle_motion_enhancement_to_pkl(
    pkl_path: Path,
    args: argparse.Namespace,
    entry: DatasetEntrySpec,
    visemes: list[VisemeSpec],
    transitions: list[TransitionSpec],
) -> None:
    """
    Load one PKL and synthesize subtle idle behavior for dedicated IDLE clips.
    """
    if not bool(args.idle_motion_enhancement):
        return
    if not is_idle_entry_candidate(visemes=visemes, transitions=transitions):
        return
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported PKL payload type for idle enhancement: {type(payload)}")
    if not bool(args.force_idle_motion_enhancement):
        metadata_payload = payload.get("motionIdleEnhancement")
        if isinstance(metadata_payload, dict) and metadata_payload.get("profile") == IDLE_MOTION_PROFILE_V1:
            print(f"[skip] idle motion enhancement already applied: {pkl_path}")
            return
    processed_payload = apply_idle_motion_enhancement_to_payload(
        payload=payload,
        args=args,
        entry=entry,
        visemes=visemes,
        transitions=transitions,
    )
    with pkl_path.open("wb") as handle:
        pickle.dump(processed_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[ok] idle motion enhancement -> {pkl_path}")


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


def build_entry_pkl(
    args: argparse.Namespace,
    audio_wav_path: Path,
    output_pkl_path: Path,
    faster_repo_dir: Path,
    cfg_path: Path,
    audio_to_pkl_script: Path,
) -> None:
    """
    Generate one motion pkl from one phrase audio WAV.
    """
    if output_pkl_path.exists() and not bool(args.overwrite):
        print(f"[skip] pkl already exists: {output_pkl_path}")
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
            to_container_path(audio_wav_path),
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
            str(audio_wav_path),
            "--output-pkl",
            str(output_pkl_path),
            "--seed",
            str(int(args.seed)),
        ]
    run_command(command)
    print(f"[ok] pkl -> {output_pkl_path}")


def render_entry_clip(
    args: argparse.Namespace,
    base_image_path: Path,
    pkl_path: Path,
    cfg_path: Path,
    source_cache_dir: Path,
    render_raw_dir: Path,
    output_org_path: Path,
    output_crop_path: Path,
) -> None:
    """
    Render one full phrase org/crop clip from one pkl.
    """
    if output_org_path.exists() and output_crop_path.exists() and not bool(args.overwrite):
        print(f"[skip] full clip already exists: {output_org_path}")
        return
    render_raw_dir.mkdir(parents=True, exist_ok=True)
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.overwrite):
        for stale in render_raw_dir.glob("*"):
            if stale.is_file():
                stale.unlink()

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
            to_container_path(pkl_path),
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
            str(pkl_path),
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
    output_org_path.parent.mkdir(parents=True, exist_ok=True)
    output_crop_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_org, output_org_path)
    shutil.copy2(generated_crop, output_crop_path)
    print(f"[ok] full org clip -> {output_org_path}")
    print(f"[ok] full crop clip -> {output_crop_path}")


def trim_video_segment(
    source_clip_path: Path,
    output_clip_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    target_duration_sec: float,
    overwrite: bool,
    encoder_preset: str,
    encoder_crf: int,
    encoder_tune: str,
    encoder_profile: str,
    encoder_level: str,
) -> None:
    """
    Cut one video segment with frame-accurate trimming and no audio track.
    """
    if output_clip_path.exists() and not overwrite:
        return
    if int(end_frame) <= int(start_frame):
        raise ValueError(f"Invalid frame window: {start_frame}..{end_frame}")
    if float(fps) <= 0.0:
        raise ValueError("fps must be > 0.")
    base_duration_sec = float(int(end_frame) - int(start_frame)) / float(fps)
    safe_target_duration_sec = max(0.0, float(target_duration_sec))
    speed_scale = 1.0
    if safe_target_duration_sec > 0.0 and base_duration_sec > 0.0:
        speed_scale = safe_target_duration_sec / base_duration_sec
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = max(1e-6, float(fps))
    filter_graph = (
        f"trim=start_frame={int(start_frame)}:end_frame={int(end_frame)},"
        f"setpts=(PTS-STARTPTS)*{speed_scale:.6f},fps={output_fps:.6f}"
    )
    safe_crf = int(max(0, min(51, int(encoder_crf))))
    safe_preset = str(encoder_preset).strip() or DEFAULT_SEGMENT_ENCODER_PRESET
    safe_tune = str(encoder_tune).strip()
    safe_profile = str(encoder_profile).strip() or DEFAULT_SEGMENT_ENCODER_PROFILE
    safe_level = str(encoder_level).strip() or DEFAULT_SEGMENT_ENCODER_LEVEL
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_clip_path),
        "-vf",
        filter_graph,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        safe_preset,
        "-crf",
        str(safe_crf),
        "-profile:v",
        safe_profile,
        "-level:v",
        safe_level,
    ]
    if safe_tune:
        command.extend(["-tune", safe_tune])
    command.extend(
        [
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        str(output_clip_path),
        ]
    )
    run_command(command)


def build_segment_clip(
    args: argparse.Namespace,
    segment_index: int,
    segment_key: str,
    start_sec: float,
    end_sec: float,
    combined_org_path: Path,
    combined_crop_path: Path,
    org_stream_info: VideoStreamInfo,
    crop_stream_info: VideoStreamInfo,
    segment_output_dir: Path,
    overwrite: bool,
    min_frames: int,
    target_duration_sec: float,
    org_frame_window: tuple[int, int, float, float] | None = None,
    crop_frame_window: tuple[int, int, float, float] | None = None,
) -> SegmentClipResult:
    """
    Build org/crop clips for one segment timing window.
    """
    safe_start = float(start_sec)
    safe_end = float(end_sec)
    if safe_end <= safe_start:
        raise ValueError(f"Invalid segment window for {segment_key}: {safe_start}..{safe_end}")
    if org_frame_window is None:
        org_start_frame, org_end_frame, adjusted_start_sec, adjusted_end_sec = quantize_window_to_frames(
            start_sec=safe_start,
            end_sec=safe_end,
            stream_info=org_stream_info,
            min_frames=min_frames,
        )
    else:
        org_start_frame, org_end_frame, adjusted_start_sec, adjusted_end_sec = org_frame_window
    if crop_frame_window is None:
        crop_start_frame, crop_end_frame, _, _ = quantize_window_to_frames(
            start_sec=safe_start,
            end_sec=safe_end,
            stream_info=crop_stream_info,
            min_frames=min_frames,
        )
    else:
        crop_start_frame, crop_end_frame, _, _ = crop_frame_window
    org_clip_path = segment_output_dir / "result_org.mp4"
    crop_clip_path = segment_output_dir / "result_crop.mp4"
    trim_video_segment(
        source_clip_path=combined_org_path,
        output_clip_path=org_clip_path,
        start_frame=org_start_frame,
        end_frame=org_end_frame,
        fps=org_stream_info.fps,
        target_duration_sec=target_duration_sec,
        overwrite=overwrite,
        encoder_preset=str(args.segment_encoder_preset),
        encoder_crf=int(args.segment_encoder_crf),
        encoder_tune=str(args.segment_encoder_tune),
        encoder_profile=str(args.segment_encoder_profile),
        encoder_level=str(args.segment_encoder_level),
    )
    trim_video_segment(
        source_clip_path=combined_crop_path,
        output_clip_path=crop_clip_path,
        start_frame=crop_start_frame,
        end_frame=crop_end_frame,
        fps=crop_stream_info.fps,
        target_duration_sec=target_duration_sec,
        overwrite=overwrite,
        encoder_preset=str(args.segment_encoder_preset),
        encoder_crf=int(args.segment_encoder_crf),
        encoder_tune=str(args.segment_encoder_tune),
        encoder_profile=str(args.segment_encoder_profile),
        encoder_level=str(args.segment_encoder_level),
    )
    output_duration_sec = adjusted_end_sec - adjusted_start_sec
    if float(target_duration_sec) > 0.0:
        output_duration_sec = float(target_duration_sec)
    return SegmentClipResult(
        index=segment_index,
        key=segment_key,
        start_sec=adjusted_start_sec,
        end_sec=adjusted_end_sec,
        duration_sec=output_duration_sec,
        clip_org_path=org_clip_path,
        clip_crop_path=crop_clip_path,
    )


def write_entry_manifest(
    entry: DatasetEntrySpec,
    visemes: list[VisemeSpec],
    transitions: list[TransitionSpec],
    entry_dir: Path,
    dataset_path: Path,
    audio_path: Path,
    pkl_path: Path,
    full_org_path: Path,
    full_crop_path: Path,
    viseme_results: list[SegmentClipResult],
    transition_results: list[SegmentClipResult],
    transition_mode: str,
    transition_padding_sec: float,
) -> Path:
    """
    Write one phrase-level output manifest.
    """
    viseme_payload: list[dict[str, Any]] = []
    for segment, viseme in zip(viseme_results, visemes, strict=True):
        viseme_payload.append(
            {
                "index": int(viseme.index),
                "char": viseme.char,
                "viseme": viseme.viseme,
                "startSec": round(viseme.start_sec, 6),
                "endSec": round(viseme.end_sec, 6),
                "durationSec": round(viseme.duration_sec, 6),
                "clipStartSec": round(segment.start_sec, 6),
                "clipEndSec": round(segment.end_sec, 6),
                "clipDurationSec": round(segment.duration_sec, 6),
                "clipOrg": to_project_relative(segment.clip_org_path),
                "clipCrop": to_project_relative(segment.clip_crop_path),
            }
        )

    transition_payload: list[dict[str, Any]] = []
    for segment, transition in zip(transition_results, transitions, strict=True):
        transition_payload.append(
            {
                "index": int(transition.index),
                "fromViseme": transition.from_viseme,
                "toViseme": transition.to_viseme,
                "key": f"{transition.from_viseme}_to_{transition.to_viseme}",
                "timeSec": round(transition.time_sec, 6),
                "startSec": round(transition.start_sec, 6),
                "endSec": round(transition.end_sec, 6),
                "durationSec": round(transition.duration_sec, 6),
                "clipStartSec": round(segment.start_sec, 6),
                "clipEndSec": round(segment.end_sec, 6),
                "clipDurationSec": round(segment.duration_sec, 6),
                "clipOrg": to_project_relative(segment.clip_org_path),
                "clipCrop": to_project_relative(segment.clip_crop_path),
            }
        )

    payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceDataset": to_project_relative(dataset_path),
        "datasetIndex": int(entry.dataset_index),
        "id": entry.dataset_id,
        "phrase": entry.phrase,
        "durationSec": round(entry.duration_sec, 6),
        "entryDir": to_project_relative(entry_dir),
        "audioWav": to_project_relative(audio_path),
        "motionPkl": to_project_relative(pkl_path),
        "fullOrgClip": to_project_relative(full_org_path),
        "fullCropClip": to_project_relative(full_crop_path),
        "transitionMode": transition_mode,
        "transitionPaddingSec": round(float(max(0.0, transition_padding_sec)), 6),
        "visemeCount": len(viseme_payload),
        "transitionCount": len(transition_payload),
        "visemes": viseme_payload,
        "transitions": transition_payload,
    }
    manifest_path = entry_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def build_entry(
    args: argparse.Namespace,
    entry: DatasetEntrySpec,
    dataset_path: Path,
    base_image_path: Path,
    output_dir: Path,
    faster_repo_dir: Path,
    cfg_path: Path,
    audio_to_pkl_script: Path,
    root_source_cache_dir: Path,
    alignment_runtime: MmsAlignmentRuntime | None,
) -> EntryBuildResult:
    """
    Build full clip and segment clips for one dataset entry.
    """
    phrase_slug = sanitize_slug(entry.phrase, max_length=48)
    entry_dir_name = f"{entry.dataset_index:04d}_{sanitize_slug(entry.dataset_id, max_length=24)}_{phrase_slug}"
    entry_dir = output_dir / entry_dir_name
    entry_dir.mkdir(parents=True, exist_ok=True)

    audio_path = entry_dir / "audio" / "phrase.wav"
    pkl_path = entry_dir / "work" / "motion.pkl"
    render_raw_dir = entry_dir / "work" / "render_raw"
    full_org_path = entry_dir / "full" / "result_org.mp4"
    full_crop_path = entry_dir / "full" / "result_crop.mp4"
    source_cache_dir = root_source_cache_dir / entry_dir_name

    audio_bytes = decode_audio_bytes(entry.audio_base64)
    write_pcm_wav(
        audio_bytes=audio_bytes,
        output_path=audio_path,
        channels=int(args.audio_channels),
        sample_width_bytes=int(args.sample_width_bytes),
        sample_rate_hz=int(args.sample_rate_hz),
        overwrite=bool(args.overwrite),
    )

    active_visemes = list(entry.visemes)
    active_transitions = list(entry.transitions)
    if str(args.timing_mode) == TIMING_MODE_AUDIO_FORCED:
        if alignment_runtime is None:
            raise RuntimeError("Forced alignment runtime was not initialized.")
        try:
            active_visemes = align_visemes_with_mms(
                visemes=entry.visemes,
                duration_sec=entry.duration_sec,
                audio_wav_path=audio_path,
                runtime=alignment_runtime,
            )
            active_transitions = build_transitions_from_visemes(
                visemes=active_visemes,
                duration_sec=entry.duration_sec,
                mode=str(args.transition_mode),
                padding_sec=float(args.transition_padding_sec),
            )
            print(
                f"[info] forced alignment applied dataset_index={entry.dataset_index} "
                f"visemes={len(active_visemes)} transitions={len(active_transitions)}"
            )
        except Exception as exc:  # noqa: BLE001
            if bool(args.alignment_fallback):
                print(
                    f"[warn] forced alignment failed dataset_index={entry.dataset_index}; "
                    f"using dataset timings. reason={exc}"
                )
            else:
                raise

    if not bool(args.skip_pkl_build):
        build_entry_pkl(
            args=args,
            audio_wav_path=audio_path,
            output_pkl_path=pkl_path,
            faster_repo_dir=faster_repo_dir,
            cfg_path=cfg_path,
            audio_to_pkl_script=audio_to_pkl_script,
        )
    elif not pkl_path.exists():
        raise FileNotFoundError(f"--skip-pkl-build set but pkl is missing: {pkl_path}")

    if bool(args.motion_stabilization):
        apply_motion_stabilization_to_pkl(
            pkl_path=pkl_path,
            args=args,
        )
    apply_idle_motion_enhancement_to_pkl(
        pkl_path=pkl_path,
        args=args,
        entry=entry,
        visemes=active_visemes,
        transitions=active_transitions,
    )

    if not bool(args.skip_render):
        render_entry_clip(
            args=args,
            base_image_path=base_image_path,
            pkl_path=pkl_path,
            cfg_path=cfg_path,
            source_cache_dir=source_cache_dir,
            render_raw_dir=render_raw_dir,
            output_org_path=full_org_path,
            output_crop_path=full_crop_path,
        )
    elif not full_org_path.exists() or not full_crop_path.exists():
        raise FileNotFoundError(
            f"--skip-render set but full clips are missing: {full_org_path} / {full_crop_path}"
        )

    apply_idle_mouth_pixel_freeze_to_clips(
        args=args,
        entry=entry,
        visemes=active_visemes,
        transitions=active_transitions,
        pkl_path=pkl_path,
        full_org_path=full_org_path,
        full_crop_path=full_crop_path,
    )

    org_stream_info = get_video_stream_info(full_org_path)
    crop_stream_info = get_video_stream_info(full_crop_path)

    min_viseme_frames = max(1, int(args.min_viseme_frames))
    min_transition_frames = max(1, int(args.min_transition_frames))
    viseme_context_sec = max(0.0, float(args.viseme_context_sec))
    transition_context_sec = max(0.0, float(args.transition_context_sec))
    target_viseme_sec = max(0.0, float(args.target_viseme_sec))
    target_transition_sec = max(0.0, float(args.target_transition_sec))

    viseme_results: list[SegmentClipResult] = []
    for viseme in active_visemes:
        clip_start_sec, clip_end_sec = expand_segment_window_with_context(
            start_sec=viseme.start_sec,
            end_sec=viseme.end_sec,
            duration_sec=entry.duration_sec,
            context_sec=viseme_context_sec,
        )
        viseme_dir = entry_dir / "visemes" / f"{viseme.index:04d}_{sanitize_slug(viseme.viseme, max_length=24)}"
        viseme_results.append(
            build_segment_clip(
                args=args,
                segment_index=viseme.index,
                segment_key=viseme.viseme,
                start_sec=clip_start_sec,
                end_sec=clip_end_sec,
                combined_org_path=full_org_path,
                combined_crop_path=full_crop_path,
                org_stream_info=org_stream_info,
                crop_stream_info=crop_stream_info,
                segment_output_dir=viseme_dir,
                overwrite=bool(args.overwrite),
                min_frames=min_viseme_frames,
                target_duration_sec=target_viseme_sec,
            )
        )

    transition_results: list[SegmentClipResult] = []
    for transition in active_transitions:
        clip_start_sec, clip_end_sec = expand_segment_window_with_context(
            start_sec=transition.start_sec,
            end_sec=transition.end_sec,
            duration_sec=entry.duration_sec,
            context_sec=transition_context_sec,
        )
        transition_key = f"{transition.from_viseme}_to_{transition.to_viseme}"
        transition_dir = entry_dir / "transitions" / f"{transition.index:04d}_{sanitize_slug(transition_key, max_length=40)}"
        transition_results.append(
            build_segment_clip(
                args=args,
                segment_index=transition.index,
                segment_key=transition_key,
                start_sec=clip_start_sec,
                end_sec=clip_end_sec,
                combined_org_path=full_org_path,
                combined_crop_path=full_crop_path,
                org_stream_info=org_stream_info,
                crop_stream_info=crop_stream_info,
                segment_output_dir=transition_dir,
                overwrite=bool(args.overwrite),
                min_frames=min_transition_frames,
                target_duration_sec=target_transition_sec,
            )
        )

    entry_manifest_path = write_entry_manifest(
        entry=entry,
        visemes=active_visemes,
        transitions=active_transitions,
        entry_dir=entry_dir,
        dataset_path=dataset_path,
        audio_path=audio_path,
        pkl_path=pkl_path,
        full_org_path=full_org_path,
        full_crop_path=full_crop_path,
        viseme_results=viseme_results,
        transition_results=transition_results,
        transition_mode=str(args.transition_mode),
        transition_padding_sec=float(args.transition_padding_sec),
    )
    print(
        f"[ok] dataset_index={entry.dataset_index} id={entry.dataset_id} "
        f"visemes={len(viseme_results)} transitions={len(transition_results)}"
    )
    return EntryBuildResult(
        dataset_index=entry.dataset_index,
        dataset_id=entry.dataset_id,
        phrase=entry.phrase,
        duration_sec=entry.duration_sec,
        entry_dir=entry_dir,
        audio_path=audio_path,
        pkl_path=pkl_path,
        full_org_path=full_org_path,
        full_crop_path=full_crop_path,
        viseme_results=viseme_results,
        transition_results=transition_results,
        entry_manifest_path=entry_manifest_path,
    )


def write_output_manifest(
    output_manifest_path: Path,
    dataset_path: Path,
    base_image_path: Path,
    output_dir: Path,
    results: list[EntryBuildResult],
    failures: list[str],
    selected_count: int,
    transition_mode: str,
    transition_padding_sec: float,
    motion_stabilization_enabled: bool,
    motion_stabilization_profile: str,
    idle_motion_enhancement_enabled: bool,
    idle_motion_enhancement_profile: str,
    segment_encoder_preset: str,
    segment_encoder_crf: int,
    segment_encoder_tune: str,
    segment_encoder_profile: str,
    segment_encoder_level: str,
) -> None:
    """
    Write global POC output manifest.
    """
    payload = {
        "version": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceDataset": to_project_relative(dataset_path),
        "baseImage": to_project_relative(base_image_path),
        "outputDir": to_project_relative(output_dir),
        "selectedEntryCount": int(selected_count),
        "builtEntryCount": len(results),
        "failureCount": len(failures),
        "transitionMode": transition_mode,
        "transitionPaddingSec": round(float(max(0.0, transition_padding_sec)), 6),
        "motionStabilization": {
            "enabled": bool(motion_stabilization_enabled),
            "profile": str(motion_stabilization_profile),
        },
        "idleMotionEnhancement": {
            "enabled": bool(idle_motion_enhancement_enabled),
            "profile": str(idle_motion_enhancement_profile),
        },
        "segmentEncoding": {
            "codec": "libx264",
            "preset": str(segment_encoder_preset),
            "crf": int(segment_encoder_crf),
            "tune": str(segment_encoder_tune),
            "profile": str(segment_encoder_profile),
            "level": str(segment_encoder_level),
            "pixelFormat": "yuv420p",
            "colorSpace": "bt709",
        },
        "entries": [
            {
                "datasetIndex": int(item.dataset_index),
                "id": item.dataset_id,
                "phrase": item.phrase,
                "durationSec": round(item.duration_sec, 6),
                "entryDir": to_project_relative(item.entry_dir),
                "audioWav": to_project_relative(item.audio_path),
                "motionPkl": to_project_relative(item.pkl_path),
                "fullOrgClip": to_project_relative(item.full_org_path),
                "fullCropClip": to_project_relative(item.full_crop_path),
                "visemeCount": len(item.viseme_results),
                "transitionCount": len(item.transition_results),
                "entryManifest": to_project_relative(item.entry_manifest_path),
            }
            for item in results
        ],
        "failures": failures,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def select_entries(entries: list[DatasetEntrySpec], start_index: int, limit: int) -> list[DatasetEntrySpec]:
    """
    Select dataset entries according to start/limit options.
    """
    safe_start = max(0, int(start_index))
    if safe_start >= len(entries):
        return []
    if int(limit) <= 0:
        return entries[safe_start:]
    return entries[safe_start : safe_start + int(limit)]


def main() -> None:
    """
    Program entry point.
    """
    args = parse_args()
    dataset_path = resolve_path(str(args.dataset_path))
    base_image_path = resolve_path(str(args.base_image))
    output_dir = resolve_path(str(args.output_dir))
    output_manifest_path = resolve_path(str(args.output_manifest))
    faster_repo_dir = resolve_path(str(args.faster_repo_dir))
    cfg_path = resolve_path(str(args.cfg))
    audio_to_pkl_script = resolve_path(str(args.audio_to_pkl_script))
    source_cache_dir = resolve_path(str(args.source_cache_dir))

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    if not faster_repo_dir.exists():
        raise FileNotFoundError(f"Faster repo not found: {faster_repo_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not audio_to_pkl_script.exists():
        raise FileNotFoundError(f"Audio->PKL script not found: {audio_to_pkl_script}")

    ensure_ffmpeg_available()
    if args.runtime == RUNTIME_DOCKER:
        ensure_container_running(
            container_name=str(args.docker_container),
            service_name=str(args.docker_service),
            auto_start=bool(args.auto_start_container),
        )

    entries = load_dataset_entries(
        dataset_path=dataset_path,
        transition_mode=str(args.transition_mode),
        transition_padding_sec=float(args.transition_padding_sec),
    )
    selected_entries = select_entries(entries, start_index=int(args.start_index), limit=int(args.limit))
    if not selected_entries:
        raise ValueError(
            f"No dataset entries selected. total={len(entries)} start-index={args.start_index} limit={args.limit}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[info] selected_entries={len(selected_entries)} "
        f"transition_mode={args.transition_mode} transition_padding_sec={float(args.transition_padding_sec):.3f}"
    )
    print(
        f"[info] motion_stabilization={bool(args.motion_stabilization)} "
        f"profile={MOTION_STABILIZATION_PROFILE_V1} force={bool(args.force_motion_stabilization)}"
    )
    print(
        f"[info] idle_motion_enhancement={bool(args.idle_motion_enhancement)} "
        f"profile={IDLE_MOTION_PROFILE_V1} force={bool(args.force_idle_motion_enhancement)}"
    )
    print(
        "[info] segment_encoding="
        f"libx264/{str(args.segment_encoder_preset)} crf={int(args.segment_encoder_crf)} "
        f"tune={str(args.segment_encoder_tune) or '-'} "
        f"profile={str(args.segment_encoder_profile)} level={str(args.segment_encoder_level)}"
    )

    alignment_runtime: MmsAlignmentRuntime | None = None
    if str(args.timing_mode) == TIMING_MODE_AUDIO_FORCED:
        alignment_runtime = load_mms_alignment_runtime(str(args.alignment_device))
        print(f"[info] timing_mode={args.timing_mode} alignment_device={alignment_runtime.device_name}")
    else:
        print(f"[info] timing_mode={args.timing_mode}")

    results: list[EntryBuildResult] = []
    failures: list[str] = []
    for entry in selected_entries:
        try:
            result = build_entry(
                args=args,
                entry=entry,
                dataset_path=dataset_path,
                base_image_path=base_image_path,
                output_dir=output_dir,
                faster_repo_dir=faster_repo_dir,
                cfg_path=cfg_path,
                audio_to_pkl_script=audio_to_pkl_script,
                root_source_cache_dir=source_cache_dir,
                alignment_runtime=alignment_runtime,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            message = (
                f"[error] build failed for dataset index {entry.dataset_index} "
                f"id={entry.dataset_id} phrase={entry.phrase!r}: {exc}"
            )
            print(message)
            failures.append(message)
            if not bool(args.continue_on_error):
                write_output_manifest(
                    output_manifest_path=output_manifest_path,
                    dataset_path=dataset_path,
                    base_image_path=base_image_path,
                    output_dir=output_dir,
                    results=results,
                    failures=failures,
                    selected_count=len(selected_entries),
                    transition_mode=str(args.transition_mode),
                    transition_padding_sec=float(args.transition_padding_sec),
                    motion_stabilization_enabled=bool(args.motion_stabilization),
                    motion_stabilization_profile=MOTION_STABILIZATION_PROFILE_V1,
                    idle_motion_enhancement_enabled=bool(args.idle_motion_enhancement),
                    idle_motion_enhancement_profile=IDLE_MOTION_PROFILE_V1,
                    segment_encoder_preset=str(args.segment_encoder_preset),
                    segment_encoder_crf=int(args.segment_encoder_crf),
                    segment_encoder_tune=str(args.segment_encoder_tune),
                    segment_encoder_profile=str(args.segment_encoder_profile),
                    segment_encoder_level=str(args.segment_encoder_level),
                )
                raise

    write_output_manifest(
        output_manifest_path=output_manifest_path,
        dataset_path=dataset_path,
        base_image_path=base_image_path,
        output_dir=output_dir,
        results=results,
        failures=failures,
        selected_count=len(selected_entries),
        transition_mode=str(args.transition_mode),
        transition_padding_sec=float(args.transition_padding_sec),
        motion_stabilization_enabled=bool(args.motion_stabilization),
        motion_stabilization_profile=MOTION_STABILIZATION_PROFILE_V1,
        idle_motion_enhancement_enabled=bool(args.idle_motion_enhancement),
        idle_motion_enhancement_profile=IDLE_MOTION_PROFILE_V1,
        segment_encoder_preset=str(args.segment_encoder_preset),
        segment_encoder_crf=int(args.segment_encoder_crf),
        segment_encoder_tune=str(args.segment_encoder_tune),
        segment_encoder_profile=str(args.segment_encoder_profile),
        segment_encoder_level=str(args.segment_encoder_level),
    )
    print(f"[ok] output manifest -> {output_manifest_path}")
    print(
        f"[ok] built entries -> {len(results)} "
        f"(failures={len(failures)} selected={len(selected_entries)} total={len(entries)})"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
