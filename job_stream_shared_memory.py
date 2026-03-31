from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any


STREAM_SHM_MAGIC = b"ASTRM001"
STREAM_SHM_VERSION = 1
STREAM_SHM_SLOT_BYTES = max(
    65536,
    int(os.getenv("ANIMATION_STREAM_SHM_SLOT_BYTES", "196608").strip() or "196608"),
)
STREAM_SHM_STATUS_CAPACITY_BYTES = max(
    4096,
    int(os.getenv("ANIMATION_STREAM_SHM_STATUS_BYTES", "65536").strip() or "65536"),
)
STREAM_SHM_HEADER_FORMAT = "<8sIIIIQQII"
STREAM_SHM_HEADER_SIZE = struct.calcsize(STREAM_SHM_HEADER_FORMAT)
STREAM_SHM_SEQUENCE_RETRIES = 6


@dataclass(frozen=True)
class JobStreamSharedMemoryNames:
    prefix: str
    meta_name: str
    frames_name: str


@dataclass(frozen=True)
class JobStreamSharedMemoryHeader:
    frame_capacity: int
    slot_size: int
    latest_frame_index: int
    updated_at_ms: int
    sequence: int
    status_length: int


def now_ms() -> int:
    return int(time.time() * 1000)


def build_job_stream_shm_prefix(stream_id: str) -> str:
    normalized = str(stream_id or "").strip()
    if not normalized:
        raise ValueError("stream_id is required")
    return normalized


def build_job_stream_shm_names(prefix: str) -> JobStreamSharedMemoryNames:
    safe_prefix = build_job_stream_shm_prefix(prefix)
    digest = hashlib.sha1(safe_prefix.encode("utf-8")).hexdigest()[:20]
    return JobStreamSharedMemoryNames(
        prefix=safe_prefix,
        meta_name=f"animation_stream_meta_{digest}",
        frames_name=f"animation_stream_frames_{digest}",
    )


def build_job_stream_metadata_size(frame_capacity: int, status_capacity: int) -> int:
    safe_frame_capacity = max(1, int(frame_capacity))
    safe_status_capacity = max(1024, int(status_capacity))
    return STREAM_SHM_HEADER_SIZE + (safe_frame_capacity * 4) + safe_status_capacity


def cleanup_existing_job_stream_shared_memory(prefix: str) -> None:
    names = build_job_stream_shm_names(prefix)
    for memory_name in (names.meta_name, names.frames_name):
        try:
            existing = shared_memory.SharedMemory(name=memory_name, create=False)
        except FileNotFoundError:
            continue
        try:
            existing.close()
        finally:
            try:
                existing.unlink()
            except FileNotFoundError:
                pass


class JobStreamSharedMemoryWriter:
    def __init__(
        self,
        prefix: str,
        frame_capacity: int,
        slot_size: int = STREAM_SHM_SLOT_BYTES,
        status_capacity: int = STREAM_SHM_STATUS_CAPACITY_BYTES,
        cleanup_existing: bool = True,
    ) -> None:
        self.names = build_job_stream_shm_names(prefix)
        self.frame_capacity = max(1, int(frame_capacity))
        self.slot_size = max(1024, int(slot_size))
        self.status_capacity = max(1024, int(status_capacity))
        self.latest_frame_index = 0
        self.updated_at_ms = 0
        self.sequence = 0
        self.status_length = 0
        if cleanup_existing:
            cleanup_existing_job_stream_shared_memory(self.names.prefix)
        metadata_size = build_job_stream_metadata_size(self.frame_capacity, self.status_capacity)
        frames_size = self.frame_capacity * self.slot_size
        self.meta = shared_memory.SharedMemory(name=self.names.meta_name, create=True, size=metadata_size)
        self.frames = shared_memory.SharedMemory(name=self.names.frames_name, create=True, size=frames_size)
        self.meta.buf[:] = b"\x00" * metadata_size
        self.frames.buf[:] = b"\x00" * frames_size
        self._write_header(sequence=0)

    def _lengths_offset(self) -> int:
        return STREAM_SHM_HEADER_SIZE

    def _status_offset(self) -> int:
        return STREAM_SHM_HEADER_SIZE + (self.frame_capacity * 4)

    def _frame_offset(self, frame_index: int) -> int:
        return (max(1, int(frame_index)) - 1) * self.slot_size

    def _write_header(self, sequence: int) -> None:
        struct.pack_into(
            STREAM_SHM_HEADER_FORMAT,
            self.meta.buf,
            0,
            STREAM_SHM_MAGIC,
            STREAM_SHM_VERSION,
            int(self.frame_capacity),
            int(self.slot_size),
            int(self.latest_frame_index),
            int(self.updated_at_ms),
            int(sequence),
            int(self.status_length),
            int(self.status_capacity),
        )
        self.sequence = int(sequence)

    def _begin_write(self) -> int:
        next_sequence = int(self.sequence) + 1
        if next_sequence % 2 == 0:
            next_sequence += 1
        self._write_header(sequence=next_sequence)
        return next_sequence

    def _finish_write(self, working_sequence: int) -> None:
        self.updated_at_ms = now_ms()
        self._write_header(sequence=working_sequence + 1)

    def write_status_payload(self, payload: dict[str, Any]) -> None:
        payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(payload_bytes) > self.status_capacity:
            raise ValueError(
                f"status payload exceeds shared-memory capacity ({len(payload_bytes)} > {self.status_capacity})"
            )
        working_sequence = self._begin_write()
        status_offset = self._status_offset()
        self.meta.buf[status_offset:status_offset + len(payload_bytes)] = payload_bytes
        self.status_length = len(payload_bytes)
        self._finish_write(working_sequence)

    def publish_frame(self, frame_index: int, frame_bytes: bytes) -> None:
        safe_frame_index = max(1, int(frame_index))
        if safe_frame_index > self.frame_capacity:
            raise ValueError(
                f"frame index exceeds shared-memory capacity ({safe_frame_index} > {self.frame_capacity})"
            )
        if not frame_bytes:
            raise ValueError("frame bytes are empty")
        if len(frame_bytes) > self.slot_size:
            raise ValueError(
                f"frame bytes exceed shared-memory slot size ({len(frame_bytes)} > {self.slot_size})"
            )
        working_sequence = self._begin_write()
        frame_offset = self._frame_offset(safe_frame_index)
        self.frames.buf[frame_offset:frame_offset + len(frame_bytes)] = frame_bytes
        struct.pack_into(
            "<I",
            self.meta.buf,
            self._lengths_offset() + ((safe_frame_index - 1) * 4),
            len(frame_bytes),
        )
        self.latest_frame_index = safe_frame_index
        self._finish_write(working_sequence)

    def close(self) -> None:
        self.meta.close()
        self.frames.close()

    def unlink(self) -> None:
        try:
            self.meta.unlink()
        except FileNotFoundError:
            pass
        try:
            self.frames.unlink()
        except FileNotFoundError:
            pass


class JobStreamSharedMemoryReader:
    def __init__(self, prefix: str) -> None:
        self.names = build_job_stream_shm_names(prefix)
        self.meta: shared_memory.SharedMemory | None = None
        self.frames: shared_memory.SharedMemory | None = None
        self._attach()

    def _attach(self) -> None:
        self.meta = shared_memory.SharedMemory(name=self.names.meta_name, create=False)
        self.frames = shared_memory.SharedMemory(name=self.names.frames_name, create=False)

    def _read_header_once(self) -> JobStreamSharedMemoryHeader:
        if self.meta is None:
            raise RuntimeError("shared memory reader is closed")
        (
            magic,
            version,
            frame_capacity,
            slot_size,
            latest_frame_index,
            updated_at_ms,
            sequence,
            status_length,
            status_capacity,
        ) = struct.unpack_from(STREAM_SHM_HEADER_FORMAT, self.meta.buf, 0)
        if magic != STREAM_SHM_MAGIC:
            raise RuntimeError("invalid stream shared-memory magic")
        if int(version) != STREAM_SHM_VERSION:
            raise RuntimeError("unsupported stream shared-memory version")
        if int(status_length) < 0 or int(status_length) > int(status_capacity):
            raise RuntimeError("invalid stream shared-memory status length")
        return JobStreamSharedMemoryHeader(
            frame_capacity=max(1, int(frame_capacity)),
            slot_size=max(1024, int(slot_size)),
            latest_frame_index=max(0, int(latest_frame_index)),
            updated_at_ms=max(0, int(updated_at_ms)),
            sequence=max(0, int(sequence)),
            status_length=max(0, int(status_length)),
        )

    def _status_offset(self, header: JobStreamSharedMemoryHeader) -> int:
        return STREAM_SHM_HEADER_SIZE + (header.frame_capacity * 4)

    def _frame_length_offset(self, frame_index: int) -> int:
        return STREAM_SHM_HEADER_SIZE + ((max(1, int(frame_index)) - 1) * 4)

    def _frame_offset(self, header: JobStreamSharedMemoryHeader, frame_index: int) -> int:
        return (max(1, int(frame_index)) - 1) * header.slot_size

    def read_status_payload(self) -> dict[str, Any] | None:
        if self.meta is None:
            return None
        for _ in range(STREAM_SHM_SEQUENCE_RETRIES):
            header_before = self._read_header_once()
            if header_before.sequence % 2 != 0:
                time.sleep(0)
                continue
            status_bytes = b""
            if header_before.status_length > 0:
                status_offset = self._status_offset(header_before)
                status_bytes = bytes(
                    self.meta.buf[status_offset:status_offset + header_before.status_length]
                )
            header_after = self._read_header_once()
            if header_before.sequence != header_after.sequence or header_after.sequence % 2 != 0:
                time.sleep(0)
                continue
            if not status_bytes:
                return None
            payload = json.loads(status_bytes.decode("utf-8"))
            return payload if isinstance(payload, dict) else None
        return None

    def read_frame(self, frame_index: int) -> bytes | None:
        if self.meta is None or self.frames is None:
            return None
        safe_frame_index = max(1, int(frame_index))
        for _ in range(STREAM_SHM_SEQUENCE_RETRIES):
            header_before = self._read_header_once()
            if header_before.sequence % 2 != 0:
                time.sleep(0)
                continue
            if safe_frame_index > header_before.frame_capacity:
                return None
            frame_length = struct.unpack_from(
                "<I",
                self.meta.buf,
                self._frame_length_offset(safe_frame_index),
            )[0]
            if frame_length <= 0 or frame_length > header_before.slot_size:
                return None
            frame_offset = self._frame_offset(header_before, safe_frame_index)
            frame_bytes = bytes(self.frames.buf[frame_offset:frame_offset + frame_length])
            header_after = self._read_header_once()
            if header_before.sequence != header_after.sequence or header_after.sequence % 2 != 0:
                time.sleep(0)
                continue
            return frame_bytes
        return None

    def read_latest_frame(self) -> tuple[int, bytes] | None:
        if self.meta is None:
            return None
        header = self._read_header_once()
        if header.latest_frame_index <= 0:
            return None
        frame_bytes = self.read_frame(header.latest_frame_index)
        if not frame_bytes:
            return None
        return header.latest_frame_index, frame_bytes

    def close(self) -> None:
        if self.meta is not None:
            self.meta.close()
            self.meta = None
        if self.frames is not None:
            self.frames.close()
            self.frames = None

