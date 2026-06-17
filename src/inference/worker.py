"""
src/inference/worker.py
=======================
Threaded inference with a 1-slot (latest-wins) buffer.

Frame capture must never block on inference. The capture loop calls
``submit(frame)`` which is non-blocking and overwrites the single buffer slot;
if a frame is still waiting when a newer one arrives, the older one is dropped.
A background worker thread always processes the most recent frame and publishes
the latest :class:`InferenceResult`.

This mirrors how the live system will behave on the robot: a fast camera feed,
a slower edge detector, and a guarantee that we are always working on the
freshest available frame.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.inference.base import InferenceEngine, RawDetection
from src.sources.base import Frame


@dataclass
class InferenceResult:
    """The detector's output for one processed frame, plus timing."""

    frame: Frame
    detections: list[RawDetection]
    inference_ms: float
    fps: float


class ThreadedInferenceWorker:
    """Runs an :class:`InferenceEngine` on a background thread (latest-wins)."""

    def __init__(self, engine: InferenceEngine, name: str = "inference-worker") -> None:
        self._engine = engine
        self._name = name
        self._cond = threading.Condition()
        self._pending: Optional[Frame] = None
        self._latest: Optional[InferenceResult] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None

        # Stats (read after a run for the summary).
        self.frames_submitted = 0
        self.frames_processed = 0
        self.frames_dropped = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        with self._cond:
            self._stop = False
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def submit(self, frame: Frame) -> None:
        """Non-blocking: overwrite the buffer slot with the newest frame."""
        with self._cond:
            if self._pending is not None:
                self.frames_dropped += 1  # previous unprocessed frame is discarded
            self._pending = frame
            self.frames_submitted += 1
            self._cond.notify()

    def get_latest(self) -> Optional[InferenceResult]:
        """Return the most recent result (or None) without blocking."""
        with self._cond:
            return self._latest

    def wait_for(self, frame_id: int, timeout: float = 10.0) -> Optional[InferenceResult]:
        """Block until a result for ``frame_id`` is available (or timeout)."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while True:
                if self._latest is not None and self._latest.frame.frame_id == frame_id:
                    return self._latest
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return self._latest
                self._cond.wait(timeout=remaining)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop and self._pending is None:
                    return
                frame = self._pending
                self._pending = None
            # Heavy work happens OUTSIDE the lock so submit() never blocks.
            t0 = time.perf_counter()
            detections = self._engine.infer(frame)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
            with self._cond:
                self._latest = InferenceResult(frame, detections, dt_ms, fps)
                self.frames_processed += 1
                self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "ThreadedInferenceWorker":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
