from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import realtime_stream_api as api


TEST_API_TOKEN = "test-token"
TEST_SESSION_ID = "integration-session"
TEST_TEMPLATE_ID = "integration_template.pkl"
TEST_AUDIO_DURATION_SEC = 1.0
TEST_FRAME_TOTAL = 20
TEST_PLAYBACK_FPS = 20.0
TEST_READY_PROGRESS = 0.35


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate_socket:
        candidate_socket.bind(("127.0.0.1", 0))
        return int(candidate_socket.getsockname()[1])


def poll_until(predicate, timeout_sec: float, interval_sec: float = 0.02):
    deadline = time.time() + timeout_sec
    last_value = None
    while time.time() < deadline:
        last_value = predicate()
        if last_value:
            return last_value
        time.sleep(interval_sec)
    return last_value


class FakeProcess:
    def __init__(self, job_id: str, status_provider):
        self._job_id = job_id
        self._status_provider = status_provider

    def poll(self):
        status = self._status_provider(self._job_id)
        if str(status.get("state") or "") == "done":
            return 0
        return None


class AvatarApiHttpIntegrationTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._patchers = []
        cls._worker_stop_event = threading.Event()
        cls._worker_thread = None
        cls._job_created_at_perf = {}
        cls._job_generation_fps_overrides = {}
        cls._next_job_generation_fps_override = None
        cls._stream_lines = []
        cls._stream_stop_event = threading.Event()
        cls._temp_dir = Path(tempfile.mkdtemp(prefix="avatar_api_http_test_", dir=str(api.PROJECT_ROOT)))
        cls._template_path = cls._temp_dir / TEST_TEMPLATE_ID
        cls._template_path.write_bytes(b"template-pack")
        cls._template_path.with_suffix(".png").write_bytes(b"png")
        cls._template_path_meta = Path(f"{cls._template_path}{api.SOURCE_TEMPLATE_PACK_META_SUFFIX}")
        cls._template_path_meta.write_text(
            json.dumps(
                {
                    "created_at": "2026-04-05T18:00:00Z",
                    "frame_total": TEST_FRAME_TOTAL,
                    "is_source_video": False,
                    "source_fps": TEST_PLAYBACK_FPS,
                    "source_path": "output/source.png",
                }
            ),
            encoding="utf-8",
        )

        cls._apply_patch("DEFAULT_API_TOKEN", TEST_API_TOKEN)
        cls._apply_patch("API_TOKEN_ENABLED", True)
        cls._apply_patch("AVATAR_IDLE_MIN_HOLD_SEC", 0.0)
        cls._apply_patch("AVATAR_STATE_POLL_SLEEP_SEC", 0.01)
        cls._apply_patch("AVATAR_READY_DYNAMIC_MARGIN_SEC", 0.0)
        cls._apply_patch("AVATAR_READY_BUFFER_MAX_PROGRESS", TEST_READY_PROGRESS)

        cls._apply_function_patch("ensure_avatar_worker_started", cls._ensure_test_avatar_worker_started)
        cls._apply_function_patch("create_and_enqueue_audio_job", cls._fake_create_and_enqueue_audio_job)
        cls._apply_function_patch("read_job_stream_status", cls._fake_read_job_stream_status)
        cls._apply_function_patch("stream_continuous_avatar_video", cls._fake_stream_continuous_avatar_video)

        cls._reset_backend_state()

        cls._app = api.create_app()
        cls._port = get_free_port()
        cls._server = uvicorn.Server(
            uvicorn.Config(
                app=cls._app,
                host="127.0.0.1",
                port=cls._port,
                log_level="error",
                access_log=False,
                lifespan="off",
            )
        )
        cls._server_thread = threading.Thread(target=cls._server.run, daemon=True)
        cls._server_thread.start()
        started = poll_until(lambda: bool(cls._server.started), timeout_sec=5.0)
        if not started:
            raise RuntimeError("Integration API server did not start in time")
        cls._base_url = f"http://127.0.0.1:{cls._port}"
        cls._auth_headers = {
            "Authorization": f"Bearer {TEST_API_TOKEN}",
            "X-Avatar-Session-Id": TEST_SESSION_ID,
        }

    @classmethod
    def tearDownClass(cls):
        cls._stream_stop_event.set()
        if cls._server is not None:
            cls._server.should_exit = True
        if cls._server_thread is not None:
            cls._server_thread.join(timeout=5.0)
        cls._worker_stop_event.set()
        if cls._worker_thread is not None:
            cls._worker_thread.join(timeout=5.0)
        for patcher in reversed(cls._patchers):
            patcher.stop()
        cls._reset_backend_state()
        with contextlib_suppress():
            if cls._template_path.exists():
                cls._template_path.unlink()
            if cls._template_path_meta.exists():
                cls._template_path_meta.unlink()
            preview_path = cls._template_path.with_suffix(api.SOURCE_TEMPLATE_PACK_PREVIEW_SUFFIX)
            if preview_path.exists():
                preview_path.unlink()
            cls._temp_dir.rmdir()

    @classmethod
    def _apply_patch(cls, attribute_name: str, replacement):
        patcher = patch.object(api, attribute_name, replacement)
        patcher.start()
        cls._patchers.append(patcher)

    @classmethod
    def _apply_function_patch(cls, attribute_name: str, replacement):
        patcher = patch.object(api, attribute_name, replacement)
        patcher.start()
        cls._patchers.append(patcher)

    @classmethod
    def _reset_backend_state(cls):
        cls._stream_stop_event.set()
        cls._worker_stop_event.set()
        if cls._worker_thread is not None:
            cls._worker_thread.join(timeout=2.0)
        cls._worker_thread = None
        cls._worker_stop_event = threading.Event()
        with api.JOBS_LOCK:
            api.JOBS.clear()
        with api.AVATAR_SESSION_STATES_LOCK:
            api.AVATAR_SESSION_STATES.clear()
        api.AVATAR_WORKER_THREAD = None
        cls._job_created_at_perf.clear()
        cls._job_generation_fps_overrides.clear()
        cls._next_job_generation_fps_override = None
        cls._stream_lines.clear()
        cls._stream_stop_event.clear()

    def setUp(self):
        self._reset_backend_state()

    @classmethod
    def _status_for_job_id(cls, job_id: str) -> dict[str, object]:
        created_at_perf = cls._job_created_at_perf[job_id]
        elapsed_sec = max(0.0, time.perf_counter() - created_at_perf)
        generation_fps = float(cls._job_generation_fps_overrides.get(job_id, TEST_PLAYBACK_FPS))
        frame_index = min(TEST_FRAME_TOTAL, int(elapsed_sec * generation_fps))
        progress = min(1.0, frame_index / TEST_FRAME_TOTAL)
        state = "done" if frame_index >= TEST_FRAME_TOTAL else "running"
        return {
            "elapsedSec": elapsed_sec,
            "fps": TEST_PLAYBACK_FPS,
            "frameIndex": frame_index,
            "frameTotal": TEST_FRAME_TOTAL,
            "message": "completed" if state == "done" else "rendering",
            "progress": progress,
            "state": state,
        }

    @classmethod
    def _fake_read_job_stream_status(cls, job):
        return dict(cls._status_for_job_id(job.job_id))

    @classmethod
    def _ensure_test_avatar_worker_started(cls):
        if cls._worker_thread is not None and cls._worker_thread.is_alive():
            return

        cls._worker_stop_event.clear()

        def worker_loop():
            while not cls._worker_stop_event.is_set():
                with api.AVATAR_SESSION_STATES_LOCK:
                    avatar_session_ids = list(api.AVATAR_SESSION_STATES.keys())
                for avatar_session_id in avatar_session_ids:
                    api.advance_avatar_state_machine(avatar_session_id)
                time.sleep(api.AVATAR_STATE_POLL_SLEEP_SEC)

        cls._worker_thread = threading.Thread(target=worker_loop, daemon=True)
        cls._worker_thread.start()

    @classmethod
    async def _fake_create_and_enqueue_audio_job(cls, **kwargs):
        avatar_session_id = str(kwargs["avatar_session_id"])
        current_job_count = len(api.JOBS) + 1
        job_id = f"job_{current_job_count}"
        output_rel = Path("output_fasterliveportrait") / "integration_api" / job_id
        output_abs = (api.PROJECT_ROOT / output_rel).resolve()
        stream_rel = output_rel / "stream"
        stream_abs = (api.PROJECT_ROOT / stream_rel).resolve()
        inputs_rel = output_rel / "inputs"
        audio_rel = inputs_rel / "audio.wav"
        audio_abs = (api.PROJECT_ROOT / audio_rel).resolve()
        audio_abs.parent.mkdir(parents=True, exist_ok=True)
        audio_abs.write_bytes(b"fake-audio")
        log_abs = output_abs / api.RUN_LOG_FILE_NAME
        log_abs.parent.mkdir(parents=True, exist_ok=True)
        log_abs.write_text("integration log", encoding="utf-8")
        process = FakeProcess(job_id, cls._status_for_job_id)
        job = api.JobRecord(
            job_id=job_id,
            avatar_session_id=avatar_session_id,
            created_at_ms=api.now_ms(),
            mode=str(kwargs.get("mode") or api.DEFAULT_MODE),
            source_frame_arg=str(cls._template_path),
            source_frame_abs=cls._template_path,
            source_template_pack_id=cls._template_path.name,
            source_template_pack_abs=cls._template_path,
            output_rel=output_rel,
            output_abs=output_abs,
            stream_rel=stream_rel,
            stream_abs=stream_abs,
            stream_shm_prefix=f"itest_{job_id}",
            audio_input_rel=audio_rel,
            audio_input_abs=audio_abs,
            stream_audio_input_abs=audio_abs,
            audio_original_name="voice.wav",
            audio_duration_sec=TEST_AUDIO_DURATION_SEC,
            audio_motion_stride=int(kwargs.get("motion_stride") or api.DEFAULT_AUDIO_MOTION_STRIDE),
            generation_frame_count=kwargs.get("generation_frame_count"),
            audio_eye_tamed_preset=bool(kwargs.get("audio_eye_tamed_preset", api.DEFAULT_AUDIO_EYE_TAMED_PRESET)),
            audio_eye_soft_factor=float(kwargs.get("audio_eye_soft_factor", api.DEFAULT_AUDIO_EYE_SOFT_FACTOR)),
            audio_eye_hard_factor=float(kwargs.get("audio_eye_hard_factor", api.DEFAULT_AUDIO_EYE_HARD_FACTOR)),
            audio_eye_hard_dy_min=float(kwargs.get("audio_eye_hard_dy_min", api.DEFAULT_AUDIO_EYE_HARD_DY_MIN)),
            audio_eye_hard_dy_max=float(kwargs.get("audio_eye_hard_dy_max", api.DEFAULT_AUDIO_EYE_HARD_DY_MAX)),
            audio_motion_tuning_enabled=bool(
                kwargs.get("audio_motion_tuning_enabled", api.DEFAULT_AUDIO_MOTION_TUNING_ENABLED)
            ),
            audio_reanchor_first_n=int(kwargs.get("audio_reanchor_first_n", api.DEFAULT_AUDIO_REANCHOR_FIRST_N)),
            audio_mouth_open_factor=float(kwargs.get("audio_mouth_open_factor", api.DEFAULT_AUDIO_MOUTH_OPEN_FACTOR)),
            audio_pose_smooth_window=int(kwargs.get("audio_pose_smooth_window", api.DEFAULT_AUDIO_POSE_SMOOTH_WINDOW)),
            audio_exp_smooth_window=int(kwargs.get("audio_exp_smooth_window", api.DEFAULT_AUDIO_EXP_SMOOTH_WINDOW)),
            audio_pose_jump_threshold=float(
                kwargs.get("audio_pose_jump_threshold", api.DEFAULT_AUDIO_POSE_JUMP_THRESHOLD)
            ),
            audio_translation_jump_threshold=float(
                kwargs.get("audio_translation_jump_threshold", api.DEFAULT_AUDIO_TRANSLATION_JUMP_THRESHOLD)
            ),
            audio_lip_sync_assist=bool(kwargs.get("audio_lip_sync_assist", api.DEFAULT_AUDIO_LIP_SYNC_ASSIST)),
            audio_lip_sync_min_ratio=float(kwargs.get("audio_lip_sync_min_ratio", api.DEFAULT_AUDIO_LIP_SYNC_MIN_RATIO)),
            audio_lip_sync_max_ratio=float(kwargs.get("audio_lip_sync_max_ratio", api.DEFAULT_AUDIO_LIP_SYNC_MAX_RATIO)),
            audio_lip_sync_smooth_window=int(
                kwargs.get("audio_lip_sync_smooth_window", api.DEFAULT_AUDIO_LIP_SYNC_SMOOTH_WINDOW)
            ),
            audio_lip_sync_strength=float(kwargs.get("audio_lip_sync_strength", api.DEFAULT_AUDIO_LIP_SYNC_STRENGTH)),
            audio_lip_sync_power=float(kwargs.get("audio_lip_sync_power", api.DEFAULT_AUDIO_LIP_SYNC_POWER)),
            audio_lip_sync_attack=float(kwargs.get("audio_lip_sync_attack", api.DEFAULT_AUDIO_LIP_SYNC_ATTACK)),
            audio_lip_sync_release=float(kwargs.get("audio_lip_sync_release", api.DEFAULT_AUDIO_LIP_SYNC_RELEASE)),
            audio_lip_sync_offset_ms=int(kwargs.get("audio_lip_sync_offset_ms", api.DEFAULT_AUDIO_LIP_SYNC_OFFSET_MS)),
            audio_mouth_floor_strength=float(
                kwargs.get("audio_mouth_floor_strength", api.DEFAULT_AUDIO_MOUTH_FLOOR_STRENGTH)
            ),
            audio_mouth_peak_clamp=float(kwargs.get("audio_mouth_peak_clamp", api.DEFAULT_AUDIO_MOUTH_PEAK_CLAMP)),
            driving_multiplier=float(kwargs.get("driving_multiplier", api.DEFAULT_DRIVING_MULTIPLIER)),
            cfg_scale=float(kwargs.get("cfg_scale", api.DEFAULT_CFG_SCALE)),
            joyvasa_inference_steps=int(kwargs.get("joyvasa_inference_steps", api.DEFAULT_JOYVASA_INFERENCE_STEPS)),
            animation_region=str(kwargs.get("animation_region") or api.DEFAULT_ANIMATION_REGION),
            stitching_enabled=bool(kwargs.get("stitching", api.DEFAULT_STITCHING_ENABLED)),
            relative_motion_enabled=bool(kwargs.get("relative_motion", api.DEFAULT_RELATIVE_MOTION_ENABLED)),
            paste_back_enabled=bool(kwargs.get("paste_back", api.DEFAULT_PASTE_BACK_ENABLED)),
            defer_paste_back_enabled=False,
            log_rel=output_rel / api.RUN_LOG_FILE_NAME,
            log_abs=log_abs,
            process=process,
        )
        with api.JOBS_LOCK:
            api.JOBS[job.job_id] = job
        api.ensure_avatar_session_state(avatar_session_id)
        cls._job_created_at_perf[job_id] = time.perf_counter()
        cls._job_generation_fps_overrides[job_id] = float(
            cls._next_job_generation_fps_override
            if cls._next_job_generation_fps_override is not None
            else TEST_PLAYBACK_FPS
        )
        cls._next_job_generation_fps_override = None
        return api.build_job_payload(job)

    @classmethod
    async def _fake_stream_continuous_avatar_video(cls, avatar_session_id, chunk_sender, should_stop=None):
        has_seen_talking = False
        emitted_line_count = 0
        while emitted_line_count < 60 and not cls._stream_stop_event.is_set():
            if should_stop is not None and await should_stop():
                break
            snapshot = api.get_avatar_state_snapshot(avatar_session_id)
            mode = str(snapshot.get("mode") or "")
            current_job_id = str(snapshot.get("currentJobId") or "")
            line = f"{mode}:{current_job_id}:{snapshot.get('bufferedStartProgress', 0):.2f}\n".encode("utf-8")
            await chunk_sender(line)
            cls._stream_lines.append(line.decode("utf-8").strip())
            if mode == api.AVATAR_MODE_TALKING:
                has_seen_talking = True
            elif has_seen_talking and mode == api.AVATAR_MODE_IDLE:
                break
            emitted_line_count += 1
            await asyncio.sleep(0.03)

    def test_enqueue_processing_switches_avatar_stream_after_ready_threshold(self):
        health_response = requests.get(f"{self._base_url}/api/health", headers=self._auth_headers, timeout=5)
        self.assertEqual(health_response.status_code, 200)
        health_payload = health_response.json()
        self.assertAlmostEqual(float(health_payload["defaultAvatarBufferedStartProgress"]), TEST_READY_PROGRESS)

        initial_status_response = requests.get(
            f"{self._base_url}/api/avatar/status",
            headers=self._auth_headers,
            timeout=5,
        )
        self.assertEqual(initial_status_response.status_code, 200)
        initial_status_payload = initial_status_response.json()
        self.assertEqual(initial_status_payload["mode"], api.AVATAR_MODE_IDLE)
        self.assertEqual(initial_status_payload["currentJobId"], "")

        stream_lines = []
        stream_started_event = threading.Event()

        def consume_stream():
            with requests.get(
                f"{self._base_url}/api/avatar/video.mp4",
                headers=self._auth_headers,
                stream=True,
                timeout=5,
            ) as stream_response:
                self.assertEqual(stream_response.status_code, 200)
                stream_started_event.set()
                for line in stream_response.iter_lines(chunk_size=1, decode_unicode=True):
                    if line:
                        stream_lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                    if self._stream_stop_event.is_set():
                        break

        stream_thread = threading.Thread(target=consume_stream, daemon=True)
        stream_thread.start()
        self.assertTrue(stream_started_event.wait(timeout=2.0))

        self.assertTrue(
            poll_until(lambda: len(stream_lines) >= 2, timeout_sec=1.0),
            "stream should emit idle lines before enqueue",
        )
        self.assertTrue(all(line.startswith(f"{api.AVATAR_MODE_IDLE}:") for line in stream_lines[:2]))

        enqueue_response = requests.post(
            f"{self._base_url}/api/avatar/enqueue",
            headers=self._auth_headers,
            data={
                "source_template_pack": str(self._template_path),
            },
            files={
                "audio": ("voice.wav", b"fake-audio", "audio/wav"),
            },
            timeout=5,
        )
        self.assertEqual(enqueue_response.status_code, 200)
        enqueue_payload = enqueue_response.json()
        self.assertEqual(enqueue_payload["jobId"], "job_1")

        below_threshold_status = poll_until(
            lambda: requests.get(
                f"{self._base_url}/api/jobs/job_1/status",
                headers=self._auth_headers,
                timeout=5,
            ).json(),
            timeout_sec=0.2,
        )
        self.assertIsNotNone(below_threshold_status)
        self.assertLess(float(below_threshold_status["status"]["progress"]), TEST_READY_PROGRESS)

        avatar_before_ready_response = requests.get(
            f"{self._base_url}/api/avatar/status",
            headers=self._auth_headers,
            timeout=5,
        )
        avatar_before_ready_payload = avatar_before_ready_response.json()
        self.assertEqual(avatar_before_ready_payload["mode"], api.AVATAR_MODE_IDLE)
        self.assertEqual(avatar_before_ready_payload["currentJobId"], "")

        ready_job_status = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/jobs/job_1/status")
            if float(self._request_json(f"{self._base_url}/api/jobs/job_1/status")["status"]["progress"]) >= TEST_READY_PROGRESS
            else None,
            timeout_sec=2.0,
        )
        self.assertIsNotNone(ready_job_status, "job should reach ready threshold")
        self.assertGreaterEqual(float(ready_job_status["status"]["progress"]), TEST_READY_PROGRESS)

        avatar_ready_payload = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/avatar/status")
            if self._request_json(f"{self._base_url}/api/avatar/status")["mode"] == api.AVATAR_MODE_TALKING
            else None,
            timeout_sec=2.0,
        )
        self.assertIsNotNone(avatar_ready_payload, "avatar should switch to talking once ready")
        self.assertEqual(avatar_ready_payload["currentJobId"], "job_1")

        talking_stream_line = poll_until(
            lambda: next((line for line in stream_lines if line.startswith(f"{api.AVATAR_MODE_TALKING}:job_1")), None),
            timeout_sec=2.0,
        )
        self.assertIsNotNone(talking_stream_line, "stream should switch from idle to talking")

        final_avatar_payload = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/avatar/status")
            if self._request_json(f"{self._base_url}/api/avatar/status")["mode"] == api.AVATAR_MODE_IDLE
            and self._request_json(f"{self._base_url}/api/avatar/status")["currentJobId"] == ""
            else None,
            timeout_sec=3.0,
        )
        self.assertIsNotNone(final_avatar_payload, "avatar should return to idle after audio duration")

        self._stream_stop_event.set()
        stream_thread.join(timeout=3.0)
        self.assertTrue(
            any(line.startswith(f"{api.AVATAR_MODE_IDLE}:") for line in stream_lines),
            "stream must include idle chunks",
        )
        self.assertTrue(
            any(line.startswith(f"{api.AVATAR_MODE_TALKING}:job_1") for line in stream_lines),
            "stream must include talking chunks",
        )

    def test_avatar_status_is_talking_from_the_first_talking_stream_chunk(self):
        requests.get(f"{self._base_url}/api/health", headers=self._auth_headers, timeout=5)

        stream_lines = []
        stream_started_event = threading.Event()

        def consume_stream():
            with requests.get(
                f"{self._base_url}/api/avatar/video.mp4",
                headers=self._auth_headers,
                stream=True,
                timeout=5,
            ) as stream_response:
                self.assertEqual(stream_response.status_code, 200)
                stream_started_event.set()
                for line in stream_response.iter_lines(chunk_size=1, decode_unicode=True):
                    if line:
                        stream_lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                    if self._stream_stop_event.is_set():
                        break

        stream_thread = threading.Thread(target=consume_stream, daemon=True)
        stream_thread.start()
        self.assertTrue(stream_started_event.wait(timeout=2.0))

        enqueue_response = requests.post(
            f"{self._base_url}/api/avatar/enqueue",
            headers=self._auth_headers,
            data={
                "source_template_pack": str(self._template_path),
            },
            files={
                "audio": ("voice.wav", b"fake-audio", "audio/wav"),
            },
            timeout=5,
        )
        self.assertEqual(enqueue_response.status_code, 200)

        talking_stream_line = poll_until(
            lambda: next((line for line in stream_lines if line.startswith(f"{api.AVATAR_MODE_TALKING}:job_1")), None),
            timeout_sec=2.0,
        )
        self.assertIsNotNone(talking_stream_line, "stream should start emitting talking chunks")

        avatar_status_at_stream_start = self._request_json(f"{self._base_url}/api/avatar/status")
        self.assertEqual(
            avatar_status_at_stream_start["mode"],
            api.AVATAR_MODE_TALKING,
            "avatar status must already be talking when the first talking stream chunk is visible",
        )
        self.assertEqual(avatar_status_at_stream_start["currentJobId"], "job_1")

        self._stream_stop_event.set()
        stream_thread.join(timeout=3.0)

    def test_avatar_returns_to_idle_when_audio_ends_while_job_progress_remains_visible(self):
        self.__class__._next_job_generation_fps_override = 10.0

        enqueue_response = requests.post(
            f"{self._base_url}/api/avatar/enqueue",
            headers=self._auth_headers,
            data={
                "source_template_pack": str(self._template_path),
            },
            files={
                "audio": ("voice.wav", b"fake-audio", "audio/wav"),
            },
            timeout=5,
        )
        self.assertEqual(enqueue_response.status_code, 200)

        avatar_ready_payload = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/avatar/status")
            if self._request_json(f"{self._base_url}/api/avatar/status")["mode"] == api.AVATAR_MODE_TALKING
            else None,
            timeout_sec=2.0,
        )
        self.assertIsNotNone(avatar_ready_payload, "avatar should switch to talking before audio ends")
        self.assertEqual(avatar_ready_payload["currentJobId"], "job_1")
        self.assertGreater(float(avatar_ready_payload["currentJobProgress"]), 0.0)

        avatar_idle_payload = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/avatar/status")
            if self._request_json(f"{self._base_url}/api/avatar/status")["mode"] == api.AVATAR_MODE_IDLE
            and self._request_json(f"{self._base_url}/api/avatar/status")["currentJobId"] == ""
            else None,
            timeout_sec=2.5,
        )
        self.assertIsNotNone(avatar_idle_payload, "avatar should return to idle immediately after audio duration")
        self.assertEqual(avatar_idle_payload["runningJobId"], "job_1")
        self.assertEqual(avatar_idle_payload["runningJobState"], "running")
        self.assertGreater(float(avatar_idle_payload["runningJobProgress"]), 0.0)
        self.assertLess(float(avatar_idle_payload["runningJobProgress"]), 1.0)
        self.assertEqual(int(avatar_idle_payload["runningJobFrameTotal"]), TEST_FRAME_TOTAL)
        self.assertEqual(int(avatar_idle_payload["currentJobFrameIndex"]), 0)
        self.assertEqual(float(avatar_idle_payload["currentJobProgress"]), 0.0)

        final_job_payload = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/jobs/job_1/status")
            if self._request_json(f"{self._base_url}/api/jobs/job_1/status")["state"] == "done"
            else None,
            timeout_sec=2.0,
        )
        self.assertIsNotNone(final_job_payload, "job should still finish after avatar returns to idle")
        self.assertEqual(float(final_job_payload["status"]["progress"]), 1.0)

    def test_slow_generation_waits_beyond_static_progress_before_talking(self):
        self.__class__._next_job_generation_fps_override = 10.0

        enqueue_response = requests.post(
            f"{self._base_url}/api/avatar/enqueue",
            headers=self._auth_headers,
            data={
                "source_template_pack": str(self._template_path),
            },
            files={
                "audio": ("voice.wav", b"fake-audio", "audio/wav"),
            },
            timeout=5,
        )
        self.assertEqual(enqueue_response.status_code, 200)

        progress_threshold_status = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/jobs/job_1/status")
            if float(self._request_json(f"{self._base_url}/api/jobs/job_1/status")["status"]["progress"]) >= TEST_READY_PROGRESS
            else None,
            timeout_sec=1.0,
        )
        self.assertIsNotNone(progress_threshold_status, "job should cross the static progress threshold")
        self.assertLess(float(progress_threshold_status["status"]["progress"]), 0.5)

        avatar_before_dynamic_ready = self._request_json(f"{self._base_url}/api/avatar/status")
        self.assertEqual(
            avatar_before_dynamic_ready["mode"],
            api.AVATAR_MODE_IDLE,
            "avatar must stay idle when generation throughput is too slow for the static threshold",
        )
        self.assertEqual(avatar_before_dynamic_ready["currentJobId"], "")

        avatar_ready_payload = poll_until(
            lambda: self._request_json(f"{self._base_url}/api/avatar/status")
            if self._request_json(f"{self._base_url}/api/avatar/status")["mode"] == api.AVATAR_MODE_TALKING
            else None,
            timeout_sec=2.5,
        )
        self.assertIsNotNone(avatar_ready_payload, "avatar should eventually switch to talking once enough real buffer exists")
        self.assertEqual(avatar_ready_payload["currentJobId"], "job_1")
        self.assertGreaterEqual(float(avatar_ready_payload["currentJobProgress"]), 0.5)

    def _request_json(self, url: str):
        response = requests.get(url, headers=self._auth_headers, timeout=5)
        self.assertEqual(response.status_code, 200)
        return response.json()


class contextlib_suppress:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return True


if __name__ == "__main__":
    unittest.main()
