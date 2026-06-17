"""
src/inference/base.py
=====================
Abstract inference engine -- swappable seam #2.

PoC implementation (later step): Ultralytics YOLO running on the laptop CPU
(e.g. ``src/inference/yolo_engine.py``).
Hardware implementation (Phase 2): a quantised INT8 TensorFlow-Lite model on a
Google Coral Edge TPU. Both return the *same* ``list[RawDetection]``, so the
prescriptive / localisation / messaging layers are untouched.

NOTE: the Coral/TFLite path is intentionally NOT implemented in this build --
this interface only reserves a clean seam for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with sources.base
    from src.sources.base import Frame


@dataclass
class RawDetection:
    """A raw model output, *before* the prescriptive engine enriches it.

    Deliberately smaller than :class:`src.schema.Detection`: the engine only
    knows what it sees (class, confidence, box, tracker id). Severity, urgency,
    recommended action and the geo-tag are added later in the pipeline.
    """

    defect_class: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    track_id: int = -1  # -1 until a tracker assigns an id


class InferenceEngine(ABC):
    """Detects (and optionally tracks) defects in a single frame."""

    @abstractmethod
    def load(self) -> None:
        """Load weights / allocate the interpreter. Call once before infer()."""

    @abstractmethod
    def infer(self, frame: "Frame") -> list[RawDetection]:
        """Run detection on one frame and return zero or more raw detections."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier written into Detection.model (e.g. 'yolov8n', 'tflite-int8')."""
