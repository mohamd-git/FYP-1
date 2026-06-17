"""
src/inference/yolo_engine.py
============================
Concrete InferenceEngine using Ultralytics YOLO on the CPU.

PoC implementation of swappable seam #2. In Phase 2 a TFLite/Coral engine with
the same interface replaces it with zero downstream changes.

Behaviour:
  * If custom-trained rail-defect weights exist at the configured path, use them.
  * Otherwise fall back to a pretrained ``yolov8n.pt`` (auto-downloaded by
    Ultralytics on first use) and print a clear PLACEHOLDER warning -- that
    model knows COCO objects, not rail defects, so labels will be COCO classes
    until real weights are trained.
  * Each call measures inference latency (ms) and the implied FPS.

Heavy imports (ultralytics, torch, numpy) are done lazily inside methods so this
module can be imported even before those packages are installed.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Sequence

from src.inference.base import InferenceEngine, RawDetection
from src.sources.base import Frame

logger = logging.getLogger(__name__)


class YoloEngine(InferenceEngine):
    """Ultralytics YOLO detector (+ optional tracking) running on CPU."""

    def __init__(
        self,
        *,
        model_path: str | Path = "models/yolo_rail.pt",
        fallback_model: str = "yolov8n.pt",
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        class_names: Sequence[str] | None = None,
        track: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = Path(model_path)
        self.fallback_model = fallback_model
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_names = list(class_names) if class_names else None
        self.track = bool(track)
        self.tracker = tracker

        self._model = None
        self._names: dict[int, str] = {}
        self._model_name = "uninitialised"
        self.is_placeholder = False

        # Per-frame metrics, updated by infer().
        self.last_inference_ms: float = 0.0
        self.last_fps: float = 0.0

    @classmethod
    def from_config(cls, config: dict) -> "YoloEngine":
        """Build a YoloEngine from a parsed config dict."""
        from src.config import resolve_path

        inf = config.get("inference", {})
        return cls(
            model_path=resolve_path(inf.get("model_path", "models/yolo_rail.pt")),
            fallback_model=inf.get("fallback_model", "yolov8n.pt"),
            device=inf.get("device", "cpu"),
            imgsz=int(inf.get("image_size", 640)),
            conf=float(inf.get("confidence_threshold", 0.35)),
            iou=float(inf.get("iou_threshold", 0.45)),
            class_names=inf.get("class_names"),
            track=bool(inf.get("track", True)),
            tracker=inf.get("tracker", "bytetrack.yaml"),
        )

    # ---- InferenceEngine interface -------------------------------------- #
    def load(self) -> None:
        from ultralytics import YOLO  # heavy import kept local

        if self.model_path.is_file():
            self._model = YOLO(str(self.model_path))
            self._model_name = self.model_path.stem
            self.is_placeholder = False
            logger.info("Loaded custom rail-defect weights: %s", self.model_path)
        else:
            self._print_placeholder_warning()
            self._model = YOLO(self.fallback_model)  # auto-downloads if missing
            self._model_name = f"{Path(self.fallback_model).stem} (placeholder/COCO)"
            self.is_placeholder = True

        self._names = dict(getattr(self._model, "names", {}) or {})

        # Keep CPU latency predictable across machines.
        try:
            import torch

            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass

    def infer(self, frame: Frame) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("YoloEngine.load() must be called before infer().")

        t0 = time.perf_counter()
        if self.track:
            results = self._model.track(
                frame.image, persist=True, conf=self.conf, iou=self.iou,
                imgsz=self.imgsz, device=self.device, tracker=self.tracker, verbose=False,
            )
        else:
            results = self._model.predict(
                frame.image, conf=self.conf, iou=self.iou,
                imgsz=self.imgsz, device=self.device, verbose=False,
            )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_fps = 1000.0 / self.last_inference_ms if self.last_inference_ms > 0 else 0.0
        return self._parse(results)

    @property
    def model_name(self) -> str:
        return self._model_name

    # ---- helpers --------------------------------------------------------- #
    def warmup(self, size: int | None = None) -> None:
        """Run one dummy inference so the first real frame is timed fairly."""
        if self._model is None:
            return
        try:
            import numpy as np

            s = size or self.imgsz
            dummy = Frame(frame_id=-1, image=np.zeros((s, s, 3), dtype="uint8"),
                          timestamp=time.time())
            self.infer(dummy)
        except Exception:
            pass
        finally:
            self.last_inference_ms = 0.0
            self.last_fps = 0.0

    def _print_placeholder_warning(self) -> None:
        logger.warning(
            "PLACEHOLDER MODEL IN USE -- no custom weights at %s; falling back to "
            "pretrained '%s' (COCO). Detects generic objects, NOT rail defects; labels "
            "will be COCO classes until you train real weights (see train.py). Pipeline "
            "mechanics (detect -> track -> metrics) are exercised regardless.",
            self.model_path, self.fallback_model,
        )

    def _parse(self, results) -> list[RawDetection]:
        detections: list[RawDetection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        # .tolist() avoids importing numpy here and copies off any tensor device.
        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        clss = boxes.cls.tolist()
        ids = boxes.id.tolist() if boxes.id is not None else None

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x = max(0, int(round(x1)))
            y = max(0, int(round(y1)))
            w = max(1, int(round(x2 - x1)))
            h = max(1, int(round(y2 - y1)))
            cls_idx = int(clss[i])
            name = self._names.get(cls_idx, str(cls_idx))
            track_id = int(ids[i]) if ids is not None else -1
            detections.append(
                RawDetection(
                    defect_class=name,
                    confidence=float(confs[i]),
                    bbox_xywh=(x, y, w, h),
                    track_id=track_id,
                )
            )
        return detections
