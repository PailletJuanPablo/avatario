"""
Tune JoyVASA/FasterLivePortrait motion PKL payloads before rendering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


DEFAULT_MOUTH_INDICES = (6, 12, 14, 17, 19, 20)
DEFAULT_MOUTH_AXES = (1,)
DEFAULT_EXP_JUMP_THRESHOLD = 0.025
DEFAULT_MOUTH_JUMP_THRESHOLD = 0.05
DEFAULT_LIP_SYNC_MIN_RATIO = 0.03
DEFAULT_LIP_SYNC_MAX_RATIO = 0.32


@dataclass(frozen=True)
class MotionPklTuningConfig:
    """
    Lightweight tuning profile for JoyVASA motion payloads.
    """

    enabled: bool = True
    reanchor_first_n: int = 5
    mouth_open_factor: float = 1.18
    pose_smooth_window: int = 5
    translation_smooth_window: int = 3
    exp_smooth_window: int = 3
    pose_jump_threshold: float = 8.0
    translation_jump_threshold: float = 0.03
    exp_jump_threshold: float = DEFAULT_EXP_JUMP_THRESHOLD
    mouth_jump_threshold: float = DEFAULT_MOUTH_JUMP_THRESHOLD
    mouth_indices: tuple[int, ...] = DEFAULT_MOUTH_INDICES
    mouth_axes: tuple[int, ...] = DEFAULT_MOUTH_AXES
    lip_sync_enabled: bool = False
    lip_sync_min_ratio: float = DEFAULT_LIP_SYNC_MIN_RATIO
    lip_sync_max_ratio: float = DEFAULT_LIP_SYNC_MAX_RATIO
    lip_sync_smooth_window: int = 5
    lip_sync_strength: float = 1.15
    lip_sync_power: float = 0.85
    lip_sync_attack: float = 1.0
    lip_sync_release: float = 1.0
    lip_sync_offset_ms: int = 0
    mouth_floor_strength: float = 0.26
    mouth_peak_clamp: float = 0.0


def get_default_motion_pkl_tuning_config() -> MotionPklTuningConfig:
    """
    Return the default tuning profile.
    """

    return MotionPklTuningConfig()


def build_lip_ratio_sequence_from_audio(
    audio_path: str,
    frame_total: int,
    fps: float,
    config: MotionPklTuningConfig | None = None,
) -> list[list[float]]:
    """
    Build one lip-ratio control sequence aligned to the audio timeline.
    """

    safe_config = config or get_default_motion_pkl_tuning_config()
    if frame_total <= 0 or float(fps) <= 0:
        return []

    import torchaudio

    audio_tensor, sample_rate = torchaudio.load(audio_path)
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=False)
    else:
        audio_tensor = audio_tensor.reshape(-1)

    audio = audio_tensor.detach().cpu().numpy().astype(np.float32)
    if audio.size == 0:
        closed_ratio = float(np.clip(safe_config.lip_sync_min_ratio, 0.0, 1.0))
        return [[closed_ratio] for _ in range(int(frame_total))]

    window_sample_count = max(1, int(round(float(sample_rate) * 0.04)))
    half_window = max(1, window_sample_count // 2)
    offset_seconds = float(safe_config.lip_sync_offset_ms) / 1000.0
    rms_values: list[float] = []

    for frame_index in range(int(frame_total)):
        audio_time_seconds = (float(frame_index) / float(fps)) - offset_seconds
        center_sample = int(round(audio_time_seconds * float(sample_rate)))
        start_sample = max(0, center_sample - half_window)
        end_sample = min(audio.shape[0], center_sample + half_window)
        if end_sample <= start_sample:
            rms_values.append(0.0)
            continue
        frame_window = audio[start_sample:end_sample]
        rms_values.append(float(np.sqrt(np.mean(np.square(frame_window), dtype=np.float64))))

    envelope = np.asarray(rms_values, dtype=np.float32)
    envelope = _normalize_audio_envelope(envelope)
    if safe_config.lip_sync_smooth_window > 1:
        envelope = _moving_average(envelope, safe_config.lip_sync_smooth_window)

    safe_strength = max(0.0, float(safe_config.lip_sync_strength))
    safe_power = max(1e-3, float(safe_config.lip_sync_power))
    envelope = np.power(np.clip(envelope, 0.0, 1.0), safe_power)
    envelope = np.clip(envelope * safe_strength, 0.0, 1.0)
    envelope = _apply_attack_release_filter(
        envelope,
        attack=float(safe_config.lip_sync_attack),
        release=float(safe_config.lip_sync_release),
    )

    min_ratio = float(np.clip(safe_config.lip_sync_min_ratio, 0.0, 1.0))
    max_ratio = float(np.clip(safe_config.lip_sync_max_ratio, min_ratio, 1.0))
    ratios = min_ratio + envelope * (max_ratio - min_ratio)
    return [[float(value)] for value in ratios.tolist()]


def apply_audio_lip_sync_to_payload(
    payload: dict,
    audio_path: str,
    config: MotionPklTuningConfig | None = None,
) -> dict:
    """
    Use the audio envelope to reinforce mouth expression inside the motion PKL.
    """

    safe_config = config or get_default_motion_pkl_tuning_config()
    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        raise ValueError("Invalid motion payload: missing non-empty motion list.")

    mouth_indices = _sanitize_indices(safe_config.mouth_indices, 21)
    mouth_axes = _sanitize_indices(safe_config.mouth_axes, 3)
    if not mouth_indices or not mouth_axes:
        return dict(payload)

    fps = float(payload.get("output_fps", 25) or 25)
    lip_ratio_sequence = build_lip_ratio_sequence_from_audio(
        audio_path=audio_path,
        frame_total=len(motion),
        fps=fps,
        config=safe_config,
    )
    if not lip_ratio_sequence:
        return dict(payload)

    min_ratio = float(np.clip(safe_config.lip_sync_min_ratio, 0.0, 1.0))
    max_ratio = float(np.clip(safe_config.lip_sync_max_ratio, min_ratio, 1.0))
    ratio_values = np.asarray([float(values[0]) for values in lip_ratio_sequence], dtype=np.float32)
    if max_ratio > min_ratio:
        audio_activity = np.clip((ratio_values - min_ratio) / (max_ratio - min_ratio), 0.0, 1.0)
    else:
        audio_activity = np.zeros_like(ratio_values, dtype=np.float32)

    exp, scale, t, pitch, yaw, roll = _extract_motion_arrays(motion)
    exp = _apply_audio_envelope_to_mouth(exp, audio_activity, safe_config, mouth_indices, mouth_axes)
    if safe_config.mouth_jump_threshold > 0:
        exp = _clamp_expression_subset(exp, safe_config.mouth_jump_threshold, mouth_indices)

    tuned_payload = _rebuild_payload(payload, motion, exp, scale, t, pitch, yaw, roll)
    tuned_payload["c_lip_lst"] = lip_ratio_sequence
    return tuned_payload


def tune_motion_pkl_payload(payload: dict, config: MotionPklTuningConfig | None = None) -> dict:
    """
    Apply deterministic motion cleanup to one JoyVASA/FasterLivePortrait PKL payload.
    """

    safe_config = config or get_default_motion_pkl_tuning_config()
    if not safe_config.enabled:
        return dict(payload)

    motion = payload.get("motion")
    if not isinstance(motion, list) or not motion:
        raise ValueError("Invalid motion payload: missing non-empty motion list.")

    exp, scale, t, pitch, yaw, roll = _extract_motion_arrays(motion)
    original_pitch = pitch.copy()
    original_yaw = yaw.copy()
    original_roll = roll.copy()
    mouth_indices = _sanitize_indices(safe_config.mouth_indices, 21)
    mouth_axes = _sanitize_indices(safe_config.mouth_axes, 3)
    non_mouth_indices = tuple(index for index in range(21) if index not in mouth_indices)

    if safe_config.pose_smooth_window > 1:
        pitch = _median_smooth(pitch, safe_config.pose_smooth_window)
        yaw = _median_smooth(yaw, safe_config.pose_smooth_window)
        roll = _median_smooth(roll, safe_config.pose_smooth_window)

    if safe_config.translation_smooth_window > 1:
        t = _median_smooth(t, safe_config.translation_smooth_window)

    if safe_config.exp_smooth_window > 1 and non_mouth_indices:
        exp = _smooth_expression_subset(exp, safe_config.exp_smooth_window, non_mouth_indices)

    if safe_config.pose_jump_threshold > 0:
        pitch = _clamp_consecutive_delta(pitch, safe_config.pose_jump_threshold)
        yaw = _clamp_consecutive_delta(yaw, safe_config.pose_jump_threshold)
        roll = _clamp_consecutive_delta(roll, safe_config.pose_jump_threshold)

    if safe_config.translation_jump_threshold > 0:
        t = _clamp_consecutive_delta(t, safe_config.translation_jump_threshold)

    if safe_config.exp_jump_threshold > 0 and non_mouth_indices:
        exp = _clamp_expression_subset(exp, safe_config.exp_jump_threshold, non_mouth_indices)

    if safe_config.reanchor_first_n > 1:
        anchor_span = min(len(motion), max(1, int(safe_config.reanchor_first_n)))
        exp[0] = np.median(exp[:anchor_span], axis=0)
        scale[0] = np.median(scale[:anchor_span], axis=0)
        t[0] = np.median(t[:anchor_span], axis=0)
        pitch[0] = np.median(pitch[:anchor_span], axis=0)
        yaw[0] = np.median(yaw[:anchor_span], axis=0)
        roll[0] = np.median(roll[:anchor_span], axis=0)

    if safe_config.mouth_open_factor > 0 and safe_config.mouth_open_factor != 1.0 and mouth_indices and mouth_axes:
        exp = _boost_mouth_opening(exp, safe_config.mouth_open_factor, mouth_indices, mouth_axes)

    if safe_config.mouth_jump_threshold > 0 and mouth_indices:
        exp = _clamp_expression_subset(exp, safe_config.mouth_jump_threshold, mouth_indices)

    pose_changed = _pose_arrays_changed(
        original_pitch,
        original_yaw,
        original_roll,
        pitch,
        yaw,
        roll,
    )
    return _rebuild_payload(
        payload,
        motion,
        exp,
        scale,
        t,
        pitch,
        yaw,
        roll,
        recompute_rotation=pose_changed,
    )


def iter_streaming_tuned_motion_infos(
    motion_info_iter: Iterable[list | tuple],
    *,
    frame_total: int,
    fps: float,
    audio_path: str,
    config: MotionPklTuningConfig | None = None,
) -> Iterator[list | tuple]:
    """
    Apply causal motion tuning and lip-sync reinforcement on the streaming path.
    """

    safe_config = config or get_default_motion_pkl_tuning_config()
    if not safe_config.enabled and not safe_config.lip_sync_enabled:
        yield from motion_info_iter
        return

    processor = _StreamingAudioMotionProcessor(
        config=safe_config,
        frame_total=frame_total,
        fps=fps,
        audio_path=audio_path,
    )
    for motion_info in motion_info_iter:
        yield from processor.push(motion_info)
    yield from processor.flush()


class _StreamingAudioMotionProcessor:
    """
    Stateful causal post-processor for the direct JoyVASA stream.
    """

    def __init__(
        self,
        *,
        config: MotionPklTuningConfig,
        frame_total: int,
        fps: float,
        audio_path: str,
    ) -> None:
        self.config = config
        self.frame_total = max(0, int(frame_total))
        self.fps = max(1.0, float(fps or 25.0))
        self.audio_path = str(audio_path or "").strip()
        self.mouth_indices = _sanitize_indices(config.mouth_indices, 21)
        self.mouth_axes = _sanitize_indices(config.mouth_axes, 3)
        self.non_mouth_indices = tuple(index for index in range(21) if index not in self.mouth_indices)
        self.anchor_span = max(1, int(config.reanchor_first_n or 1))
        self.pose_window = _normalize_window(config.pose_smooth_window)
        self.translation_window = _normalize_window(config.translation_smooth_window)
        self.exp_window = _normalize_window(config.exp_smooth_window)

        self.pending_motion_infos: list[list | tuple] = []
        self.anchor_ready = False
        self.processed_frame_count = 0

        self.anchor_exp: np.ndarray | None = None
        self.anchor_scale: np.ndarray | None = None
        self.anchor_t: np.ndarray | None = None
        self.anchor_pitch: np.ndarray | None = None
        self.anchor_yaw: np.ndarray | None = None
        self.anchor_roll: np.ndarray | None = None

        self.pitch_history: deque[np.ndarray] = deque(maxlen=max(1, self.pose_window))
        self.yaw_history: deque[np.ndarray] = deque(maxlen=max(1, self.pose_window))
        self.roll_history: deque[np.ndarray] = deque(maxlen=max(1, self.pose_window))
        self.translation_history: deque[np.ndarray] = deque(maxlen=max(1, self.translation_window))
        self.exp_history: deque[np.ndarray] = deque(maxlen=max(1, self.exp_window))

        self.previous_pitch: np.ndarray | None = None
        self.previous_yaw: np.ndarray | None = None
        self.previous_roll: np.ndarray | None = None
        self.previous_t: np.ndarray | None = None
        self.previous_exp: np.ndarray | None = None

        self.reference_delta: np.ndarray | None = None
        self.reference_sign: np.ndarray | None = None
        self.lip_ratio_sequence = (
            build_lip_ratio_sequence_from_audio(
                audio_path=self.audio_path,
                frame_total=self.frame_total,
                fps=self.fps,
                config=self.config,
            )
            if self.config.lip_sync_enabled and self.audio_path and self.frame_total > 0
            else []
        )
        self.audio_activity_sequence = self._build_audio_activity_sequence(self.lip_ratio_sequence)

    def push(self, motion_info: list | tuple) -> list[list | tuple]:
        self.pending_motion_infos.append(motion_info)
        if not self.anchor_ready and len(self.pending_motion_infos) < self.anchor_span:
            return []
        if not self.anchor_ready:
            self._initialize_anchor_from_pending()
        return self._flush_pending()

    def flush(self) -> list[list | tuple]:
        if not self.pending_motion_infos:
            return []
        if not self.anchor_ready:
            self._initialize_anchor_from_pending()
        return self._flush_pending()

    def _initialize_anchor_from_pending(self) -> None:
        if self.anchor_ready or not self.pending_motion_infos:
            return
        motion_frames = [
            item[0]
            for item in self.pending_motion_infos[: max(1, min(len(self.pending_motion_infos), self.anchor_span))]
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], dict)
        ]
        if not motion_frames:
            self.anchor_ready = True
            return
        exp, scale, t, pitch, yaw, roll = _extract_motion_arrays(motion_frames)
        self.anchor_exp = np.median(exp, axis=0)
        self.anchor_scale = np.median(scale, axis=0)
        self.anchor_t = np.median(t, axis=0)
        self.anchor_pitch = np.median(pitch, axis=0)
        self.anchor_yaw = np.median(yaw, axis=0)
        self.anchor_roll = np.median(roll, axis=0)
        self.anchor_ready = True

    def _flush_pending(self) -> list[list | tuple]:
        outputs: list[list | tuple] = []
        while self.pending_motion_infos:
            outputs.append(self._process_one(self.pending_motion_infos.pop(0)))
        return outputs

    def _process_one(self, motion_info: list | tuple) -> list | tuple:
        if not isinstance(motion_info, (list, tuple)) or not motion_info or not isinstance(motion_info[0], dict):
            self.processed_frame_count += 1
            return motion_info

        motion_frame = dict(motion_info[0])
        raw_exp = _reshape_frame_array(motion_frame, "exp", (1, 21, 3))
        raw_scale = _reshape_frame_array(motion_frame, "scale", (1, 1))
        raw_t = _reshape_frame_array(motion_frame, "t", (1, 3))
        raw_pitch = _reshape_frame_array(motion_frame, "pitch", (1, 1))
        raw_yaw = _reshape_frame_array(motion_frame, "yaw", (1, 1))
        raw_roll = _reshape_frame_array(motion_frame, "roll", (1, 1))

        tuned_exp = raw_exp.copy()
        tuned_scale = raw_scale.copy()
        tuned_t = raw_t.copy()
        tuned_pitch = raw_pitch.copy()
        tuned_yaw = raw_yaw.copy()
        tuned_roll = raw_roll.copy()

        if self.config.enabled:
            tuned_pitch = self._smooth_pose_value(self.pitch_history, raw_pitch, self.pose_window)
            tuned_yaw = self._smooth_pose_value(self.yaw_history, raw_yaw, self.pose_window)
            tuned_roll = self._smooth_pose_value(self.roll_history, raw_roll, self.pose_window)
            tuned_t = self._smooth_pose_value(self.translation_history, raw_t, self.translation_window)
            tuned_exp = self._smooth_expression_value(raw_exp)

            tuned_pitch = _clamp_delta_against_previous(
                tuned_pitch,
                self.previous_pitch,
                self.config.pose_jump_threshold,
            )
            tuned_yaw = _clamp_delta_against_previous(
                tuned_yaw,
                self.previous_yaw,
                self.config.pose_jump_threshold,
            )
            tuned_roll = _clamp_delta_against_previous(
                tuned_roll,
                self.previous_roll,
                self.config.pose_jump_threshold,
            )
            tuned_t = _clamp_delta_against_previous(
                tuned_t,
                self.previous_t,
                self.config.translation_jump_threshold,
            )
            tuned_exp = self._clamp_expression_subset_online(
                tuned_exp,
                self.previous_exp,
                self.config.exp_jump_threshold,
                self.non_mouth_indices,
            )
            if self.processed_frame_count == 0 and self.anchor_span > 1:
                tuned_exp = np.asarray(self.anchor_exp, dtype=np.float32).reshape(1, 21, 3).copy()
                tuned_scale = np.asarray(self.anchor_scale, dtype=np.float32).reshape(1, 1).copy()
                tuned_t = np.asarray(self.anchor_t, dtype=np.float32).reshape(1, 3).copy()
                tuned_pitch = np.asarray(self.anchor_pitch, dtype=np.float32).reshape(1, 1).copy()
                tuned_yaw = np.asarray(self.anchor_yaw, dtype=np.float32).reshape(1, 1).copy()
                tuned_roll = np.asarray(self.anchor_roll, dtype=np.float32).reshape(1, 1).copy()

        if (
            self.config.enabled
            and self.config.mouth_open_factor > 0
            and self.config.mouth_open_factor != 1.0
            and self.mouth_indices
            and self.mouth_axes
            and self.anchor_exp is not None
        ):
            tuned_exp = self._boost_mouth_frame(tuned_exp)

        if self.config.lip_sync_enabled and self.mouth_indices and self.mouth_axes and self.anchor_exp is not None:
            tuned_exp = self._apply_lip_sync_frame(tuned_exp)

        if self.config.mouth_jump_threshold > 0 and self.mouth_indices:
            tuned_exp = self._clamp_expression_subset_online(
                tuned_exp,
                self.previous_exp,
                self.config.mouth_jump_threshold,
                self.mouth_indices,
            )

        motion_frame["exp"] = tuned_exp.astype(np.float32)
        motion_frame["scale"] = tuned_scale.astype(np.float32)
        motion_frame["t"] = tuned_t.astype(np.float32)
        motion_frame["pitch"] = tuned_pitch.astype(np.float32)
        motion_frame["yaw"] = tuned_yaw.astype(np.float32)
        motion_frame["roll"] = tuned_roll.astype(np.float32)

        pose_changed = _pose_arrays_changed(
            raw_pitch,
            raw_yaw,
            raw_roll,
            tuned_pitch,
            tuned_yaw,
            tuned_roll,
        )
        if pose_changed and ("R" in motion_frame or "R_d" in motion_frame):
            rotation = _get_rotation_matrix(
                motion_frame["pitch"].reshape(-1),
                motion_frame["yaw"].reshape(-1),
                motion_frame["roll"].reshape(-1),
            ).reshape(1, 3, 3).astype(np.float32)
            if "R" in motion_frame:
                motion_frame["R"] = rotation
            if "R_d" in motion_frame:
                motion_frame["R_d"] = rotation

        motion_info_items = list(motion_info)
        motion_info_items[0] = motion_frame
        if self.config.lip_sync_enabled and self.processed_frame_count < len(self.lip_ratio_sequence):
            lip_ratio = self.lip_ratio_sequence[self.processed_frame_count]
            if len(motion_info_items) >= 3:
                motion_info_items[2] = lip_ratio
            else:
                while len(motion_info_items) < 2:
                    motion_info_items.append(None)
                motion_info_items.append(lip_ratio)

        self.previous_pitch = tuned_pitch.copy()
        self.previous_yaw = tuned_yaw.copy()
        self.previous_roll = tuned_roll.copy()
        self.previous_t = tuned_t.copy()
        self.previous_exp = tuned_exp.copy()
        self.processed_frame_count += 1
        return tuple(motion_info_items) if isinstance(motion_info, tuple) else motion_info_items

    def _smooth_pose_value(
        self,
        history: deque[np.ndarray],
        current_value: np.ndarray,
        window_value: int,
    ) -> np.ndarray:
        history.append(current_value.copy())
        if window_value <= 1 or len(history) <= 1:
            return current_value.copy()
        return np.median(np.stack(list(history), axis=0), axis=0).astype(np.float32)

    def _smooth_expression_value(self, current_exp: np.ndarray) -> np.ndarray:
        if self.exp_window <= 1 or not self.non_mouth_indices:
            return current_exp.copy()
        current_subset = current_exp[:, self.non_mouth_indices, :].copy()
        self.exp_history.append(current_subset)
        smoothed_exp = current_exp.copy()
        smoothed_exp[:, self.non_mouth_indices, :] = np.median(
            np.stack(list(self.exp_history), axis=0),
            axis=0,
        ).astype(np.float32)
        return smoothed_exp

    def _clamp_expression_subset_online(
        self,
        current_exp: np.ndarray,
        previous_exp: np.ndarray | None,
        max_delta: float,
        indices: tuple[int, ...],
    ) -> np.ndarray:
        if previous_exp is None or max_delta <= 0 or not indices:
            return current_exp.copy()
        clamped_exp = current_exp.copy()
        clamped_exp[:, indices, :] = _clamp_delta_against_previous(
            current_exp[:, indices, :],
            previous_exp[:, indices, :],
            max_delta,
        )
        return clamped_exp

    def _boost_mouth_frame(self, current_exp: np.ndarray) -> np.ndarray:
        boosted_exp = current_exp.copy()
        baseline_exp = np.asarray(self.anchor_exp, dtype=np.float32)
        safe_factor = max(0.0, float(self.config.mouth_open_factor))
        for mouth_index in self.mouth_indices:
            for axis_index in self.mouth_axes:
                boosted_exp[0, mouth_index, axis_index] = baseline_exp[mouth_index, axis_index] + (
                    boosted_exp[0, mouth_index, axis_index] - baseline_exp[mouth_index, axis_index]
                ) * safe_factor
        return boosted_exp

    def _apply_lip_sync_frame(self, current_exp: np.ndarray) -> np.ndarray:
        if not self.lip_ratio_sequence or self.processed_frame_count >= len(self.audio_activity_sequence):
            return current_exp.copy()
        baseline_exp = np.asarray(self.anchor_exp, dtype=np.float32)
        baseline_subset = baseline_exp[self.mouth_indices, :]
        mouth_subset = current_exp[0, self.mouth_indices, :].copy()
        mouth_delta = mouth_subset[:, self.mouth_axes] - baseline_subset[:, self.mouth_axes]
        abs_delta = np.abs(mouth_delta).astype(np.float32)
        if self.reference_delta is None:
            self.reference_delta = abs_delta.copy()
        else:
            self.reference_delta = np.maximum(self.reference_delta * 0.96, abs_delta).astype(np.float32)

        safe_strength = max(0.0, float(self.config.lip_sync_strength))
        safe_power = max(1e-3, float(self.config.lip_sync_power))
        audio_activity = float(np.clip(self.audio_activity_sequence[self.processed_frame_count], 0.0, 1.0))
        frame_gain = np.power(audio_activity, max(1e-3, 1.0 / max(safe_strength, 1e-3)))
        frame_gain = float(np.clip(frame_gain * safe_strength, 0.0, 1.0 + safe_strength * 0.6))

        floor_scale = max(0.0, float(self.config.mouth_floor_strength))
        target_floor = audio_activity * self.reference_delta * floor_scale

        mouth_delta_sign = np.sign(mouth_delta).astype(np.float32)
        dominant_sign = np.sign(np.sum(mouth_delta, axis=0)).astype(np.float32)
        if self.reference_sign is None:
            self.reference_sign = np.where(dominant_sign == 0, 1.0, dominant_sign).astype(np.float32)
        else:
            self.reference_sign = np.where(dominant_sign == 0, self.reference_sign, dominant_sign).astype(np.float32)
        mouth_delta_sign = np.where(mouth_delta_sign == 0, self.reference_sign[None, :], mouth_delta_sign)

        synced_delta = np.maximum(abs_delta * frame_gain, target_floor).astype(np.float32)
        peak_clamp = max(0.0, float(self.config.mouth_peak_clamp))
        if peak_clamp > 0 and self.reference_delta is not None:
            peak_limit = np.maximum(self.reference_delta * peak_clamp, 1e-4).astype(np.float32)
            synced_delta = np.minimum(synced_delta, peak_limit)

        synced_subset = mouth_subset.copy()
        synced_subset[:, self.mouth_axes] = baseline_subset[:, self.mouth_axes] + (mouth_delta_sign * synced_delta)
        synced_exp = current_exp.copy()
        synced_exp[0, self.mouth_indices, :] = synced_subset
        return synced_exp

    def _build_audio_activity_sequence(self, lip_ratio_sequence: list[list[float]]) -> np.ndarray:
        if not lip_ratio_sequence:
            return np.zeros((0,), dtype=np.float32)
        min_ratio = float(np.clip(self.config.lip_sync_min_ratio, 0.0, 1.0))
        max_ratio = float(np.clip(self.config.lip_sync_max_ratio, min_ratio, 1.0))
        ratio_values = np.asarray([float(item[0]) for item in lip_ratio_sequence], dtype=np.float32)
        if max_ratio <= min_ratio:
            return np.zeros_like(ratio_values, dtype=np.float32)
        return np.clip((ratio_values - min_ratio) / (max_ratio - min_ratio), 0.0, 1.0).astype(np.float32)


def _extract_motion_arrays(
    motion: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exp = np.concatenate([_reshape_frame_array(frame, "exp", (1, 21, 3)) for frame in motion], axis=0)
    scale = np.concatenate([_reshape_frame_array(frame, "scale", (1, 1)) for frame in motion], axis=0)
    t = np.concatenate([_reshape_frame_array(frame, "t", (1, 3)) for frame in motion], axis=0)
    pitch = np.concatenate([_reshape_frame_array(frame, "pitch", (1, 1)) for frame in motion], axis=0)
    yaw = np.concatenate([_reshape_frame_array(frame, "yaw", (1, 1)) for frame in motion], axis=0)
    roll = np.concatenate([_reshape_frame_array(frame, "roll", (1, 1)) for frame in motion], axis=0)
    return exp, scale, t, pitch, yaw, roll


def _reshape_frame_array(frame: dict, key: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    if key not in frame:
        raise ValueError(f"Invalid motion frame: missing '{key}'.")
    array = np.asarray(frame[key], dtype=np.float32)
    if array.shape != expected_shape:
        try:
            array = array.reshape(expected_shape)
        except ValueError as exc:
            raise ValueError(f"Unexpected shape for '{key}': {array.shape}") from exc
    return array


def _sanitize_indices(values: tuple[int, ...], upper_bound: int) -> tuple[int, ...]:
    sanitized: list[int] = []
    for value in values:
        safe_value = int(value)
        if 0 <= safe_value < upper_bound and safe_value not in sanitized:
            sanitized.append(safe_value)
    return tuple(sanitized)


def _normalize_window(window_value: int) -> int:
    safe_window = max(0, int(window_value))
    if safe_window <= 1:
        return 0
    if safe_window % 2 == 0:
        safe_window += 1
    return safe_window


def _clamp_delta_against_previous(
    current_value: np.ndarray,
    previous_value: np.ndarray | None,
    max_delta: float,
) -> np.ndarray:
    safe_delta = float(max_delta)
    if previous_value is None or safe_delta <= 0:
        return current_value.copy()
    return previous_value + np.clip(current_value - previous_value, -safe_delta, safe_delta)


def _moving_average(values: np.ndarray, window_value: int) -> np.ndarray:
    window = _normalize_window(window_value)
    if window == 0 or values.shape[0] <= 1:
        return values.copy()

    radius = window // 2
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _normalize_audio_envelope(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)

    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 95))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _median_smooth(values: np.ndarray, window_value: int) -> np.ndarray:
    window = _normalize_window(window_value)
    if window == 0 or values.shape[0] <= 1:
        return values.copy()

    radius = window // 2
    smoothed = values.copy()
    for index in range(values.shape[0]):
        start_index = max(0, index - radius)
        end_index = min(values.shape[0], index + radius + 1)
        smoothed[index] = np.median(values[start_index:end_index], axis=0)
    return smoothed


def _smooth_expression_subset(exp: np.ndarray, window_value: int, indices: tuple[int, ...]) -> np.ndarray:
    if not indices:
        return exp.copy()
    smoothed = exp.copy()
    smoothed_subset = _median_smooth(exp[:, indices, :], window_value)
    smoothed[:, indices, :] = smoothed_subset
    return smoothed


def _clamp_consecutive_delta(values: np.ndarray, max_delta: float) -> np.ndarray:
    safe_delta = float(max_delta)
    if safe_delta <= 0 or values.shape[0] <= 1:
        return values.copy()

    clamped = values.copy()
    for index in range(1, clamped.shape[0]):
        delta = clamped[index] - clamped[index - 1]
        clamped[index] = clamped[index - 1] + np.clip(delta, -safe_delta, safe_delta)
    return clamped


def _clamp_expression_subset(exp: np.ndarray, max_delta: float, indices: tuple[int, ...]) -> np.ndarray:
    if not indices:
        return exp.copy()
    clamped = exp.copy()
    subset = _clamp_consecutive_delta(exp[:, indices, :], max_delta)
    clamped[:, indices, :] = subset
    return clamped


def _boost_mouth_opening(
    exp: np.ndarray,
    mouth_open_factor: float,
    mouth_indices: tuple[int, ...],
    mouth_axes: tuple[int, ...],
) -> np.ndarray:
    boosted = exp.copy()
    baseline = boosted[0].copy()
    safe_factor = max(0.0, float(mouth_open_factor))

    for mouth_index in mouth_indices:
        for axis_index in mouth_axes:
            boosted[1:, mouth_index, axis_index] = baseline[mouth_index, axis_index] + (
                boosted[1:, mouth_index, axis_index] - baseline[mouth_index, axis_index]
            ) * safe_factor
    return boosted


def _apply_audio_envelope_to_mouth(
    exp: np.ndarray,
    audio_activity: np.ndarray,
    config: MotionPklTuningConfig,
    mouth_indices: tuple[int, ...],
    mouth_axes: tuple[int, ...],
) -> np.ndarray:
    synced = exp.copy()
    if synced.shape[0] <= 1:
        return synced

    anchor_span = min(synced.shape[0], max(1, int(config.reanchor_first_n)))
    baseline = np.median(synced[:anchor_span], axis=0)
    mouth_subset = synced[:, mouth_indices, :].copy()
    baseline_subset = baseline[mouth_indices, :]
    mouth_delta = mouth_subset[:, :, mouth_axes] - baseline_subset[:, mouth_axes][None, :, :]

    safe_strength = max(0.0, float(config.lip_sync_strength))
    frame_gain = np.power(np.clip(audio_activity, 0.0, 1.0), max(1e-3, 1.0 / max(safe_strength, 1e-3)))
    frame_gain = np.clip(frame_gain * safe_strength, 0.0, 1.0 + safe_strength * 0.6).astype(np.float32)

    reference_delta = np.percentile(np.abs(mouth_delta), 90, axis=0)
    floor_scale = max(0.0, float(config.mouth_floor_strength))
    target_floor = audio_activity[:, None, None] * reference_delta[None, :, :] * floor_scale

    delta_sign = np.sign(mouth_delta)
    dominant_sign = np.sign(np.sum(mouth_delta, axis=0, keepdims=True))
    dominant_sign = np.where(dominant_sign == 0, 1.0, dominant_sign)
    delta_sign = np.where(delta_sign == 0, dominant_sign, delta_sign)

    synced_delta = np.maximum(np.abs(mouth_delta) * frame_gain[:, None, None], target_floor)
    peak_clamp = max(0.0, float(config.mouth_peak_clamp))
    if peak_clamp > 0:
        peak_limit = np.maximum(reference_delta[None, :, :] * peak_clamp, 1e-4)
        synced_delta = np.minimum(synced_delta, peak_limit.astype(np.float32))
    mouth_subset[:, :, mouth_axes] = baseline_subset[:, mouth_axes][None, :, :] + (delta_sign * synced_delta)
    synced[:, mouth_indices, :] = mouth_subset
    return synced


def _apply_attack_release_filter(values: np.ndarray, attack: float, release: float) -> np.ndarray:
    filtered = values.astype(np.float32).copy()
    if filtered.shape[0] <= 1:
        return filtered

    attack_alpha = float(np.clip(attack, 0.0, 1.0))
    release_alpha = float(np.clip(release, 0.0, 1.0))
    if attack_alpha >= 1.0 and release_alpha >= 1.0:
        return filtered

    previous_value = float(filtered[0])
    for index in range(1, filtered.shape[0]):
        target_value = float(filtered[index])
        alpha = attack_alpha if target_value >= previous_value else release_alpha
        filtered[index] = previous_value + alpha * (target_value - previous_value)
        previous_value = float(filtered[index])
    return filtered


def _pose_arrays_changed(
    original_pitch: np.ndarray,
    original_yaw: np.ndarray,
    original_roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
) -> bool:
    return not (
        np.allclose(original_pitch, pitch)
        and np.allclose(original_yaw, yaw)
        and np.allclose(original_roll, roll)
    )


def _rebuild_payload(
    payload: dict,
    original_motion: list[dict],
    exp: np.ndarray,
    scale: np.ndarray,
    t: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
    recompute_rotation: bool = False,
) -> dict:
    tuned_payload = dict(payload)
    tuned_motion: list[dict] = []

    for index, original_frame in enumerate(original_motion):
        tuned_frame = dict(original_frame)
        tuned_frame["exp"] = exp[index:index + 1].astype(np.float32)
        tuned_frame["scale"] = scale[index:index + 1].astype(np.float32)
        tuned_frame["t"] = t[index:index + 1].astype(np.float32)
        tuned_frame["pitch"] = pitch[index:index + 1].astype(np.float32)
        tuned_frame["yaw"] = yaw[index:index + 1].astype(np.float32)
        tuned_frame["roll"] = roll[index:index + 1].astype(np.float32)

        if recompute_rotation and ("R" in tuned_frame or "R_d" in tuned_frame):
            rotation = _get_rotation_matrix(
                tuned_frame["pitch"].reshape(-1),
                tuned_frame["yaw"].reshape(-1),
                tuned_frame["roll"].reshape(-1),
            ).reshape(1, 3, 3).astype(np.float32)
            if "R" in tuned_frame:
                tuned_frame["R"] = rotation
            if "R_d" in tuned_frame:
                tuned_frame["R_d"] = rotation
        tuned_motion.append(tuned_frame)

    tuned_payload["motion"] = tuned_motion
    tuned_payload["n_frames"] = len(tuned_motion)
    return tuned_payload


def _get_rotation_matrix(pitch_values: np.ndarray, yaw_values: np.ndarray, roll_values: np.ndarray) -> np.ndarray:
    pitch = np.asarray(pitch_values, dtype=np.float32).reshape(-1, 1) / 180.0 * np.pi
    yaw = np.asarray(yaw_values, dtype=np.float32).reshape(-1, 1) / 180.0 * np.pi
    roll = np.asarray(roll_values, dtype=np.float32).reshape(-1, 1) / 180.0 * np.pi

    batch_size = pitch.shape[0]
    ones = np.ones((batch_size, 1), dtype=np.float32)
    zeros = np.zeros((batch_size, 1), dtype=np.float32)

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
    ).reshape(batch_size, 3, 3)

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
    ).reshape(batch_size, 3, 3)

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
    ).reshape(batch_size, 3, 3)

    rotation = np.matmul(rot_z, np.matmul(rot_y, rot_x))
    return np.transpose(rotation, (0, 2, 1)).astype(np.float32)
