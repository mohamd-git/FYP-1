"""
src/assembler.py
================
Combines the outputs of the separate subsystems into the contract messages:

    vision (RawDetection)  +  prescriptive (Prescription)  +  localisation
    (GeoPosition)  +  timestamp  +  a saved image_ref
        ->  src.schema.Detection   (validated, ready for MQTT/SQLite)

    localisation (GeoPosition)  +  fps / inference_ms  +  battery
        ->  src.schema.Telemetry

``build_detection`` / ``build_telemetry`` are pure (no I/O) so they unit-test
easily; ``save_crop`` is the only function that touches disk (and imports cv2
lazily). Construction goes through the Pydantic models, so an invalid field
raises immediately -- the assembler can never emit an off-contract message.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.inference.base import RawDetection
from src.localisation.base import GeoPosition
from src.schema import Detection, Location, Telemetry


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_detection(
    *,
    raw: RawDetection,
    prescription: Any,  # src.prescriptive.engine.Prescription (duck-typed to avoid a hard import)
    location: GeoPosition,
    frame_id: int,
    image_ref: str,
    model: str,
    timestamp: datetime | None = None,
) -> Detection:
    """Assemble a schema-valid :class:`Detection` from the subsystem outputs.

    Args:
        raw: the detector output (defect_class, confidence, bbox_xywh, track_id).
        prescription: the prescriptive result (severity, urgency_score,
            recommended_action).
        location: geo-position from the localisation source.
        frame_id: the source frame index.
        image_ref: path/URL of the saved crop or annotated frame.
        model: identifier of the model that produced the detection.
        timestamp: detection time (defaults to now, UTC).
    """
    return Detection(
        timestamp=timestamp or _now(),
        frame_id=frame_id,
        track_id=raw.track_id,
        defect_class=raw.defect_class,            # validated against DefectClass
        confidence=raw.confidence,
        bbox_xywh=list(raw.bbox_xywh),
        severity=prescription.severity,           # validated against Severity
        urgency_score=prescription.urgency_score,
        recommended_action=prescription.recommended_action,
        location=Location(
            lat=location.lat, lng=location.lng, chainage_m=location.chainage_m
        ),
        image_ref=image_ref,
        model=model,
    )


def build_telemetry(
    *,
    location: GeoPosition,
    fps: float,
    inference_ms: float,
    battery_pct: float,
    timestamp: datetime | None = None,
) -> Telemetry:
    """Assemble a schema-valid :class:`Telemetry` sample."""
    return Telemetry(
        timestamp=timestamp or _now(),
        lat=location.lat,
        lng=location.lng,
        chainage_m=location.chainage_m,
        speed_mps=location.speed_mps,
        battery_pct=battery_pct,
        fps=fps,
        inference_ms=inference_ms,
    )


def simulate_battery(
    elapsed_s: float, start_pct: float = 100.0, drain_per_min: float = 0.5
) -> float:
    """Simple linear battery model, clamped to [0, 100]."""
    pct = start_pct - drain_per_min * (elapsed_s / 60.0)
    return max(0.0, min(100.0, pct))


def save_image(image: Any, out_dir: str | Path, name: str) -> str:
    """Write a BGR image array to ``out_dir/name.jpg``.

    Returns an ``image_ref`` string: a path relative to the project root (posix
    style) when possible, else the absolute path.
    """
    import cv2  # lazy: keeps the pure builders importable without OpenCV

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jpg"
    cv2.imwrite(str(out_path), image)
    try:
        from src.config import PROJECT_ROOT

        return out_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return str(out_path)


def crop_bbox(image: Any, bbox_xywh: tuple[int, int, int, int],
              context: float = 3.0, min_side: int = 300) -> Any:
    """Return a padded, context-rich square region around the bbox (clamped to
    the frame). Captures the defect *plus* the surrounding track so the saved
    image stays legible for operator verification, rather than a tiny tight box
    that pixelates when shown larger.
    """
    x, y, w, h = (int(v) for v in bbox_xywh)
    height, width = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    cx, cy = x + w / 2.0, y + h / 2.0
    side = min(max(max(w, h) * context, float(min_side)), float(min(width, height)))
    x0 = int(round(min(max(0.0, cx - side / 2.0), width - side)))
    y0 = int(round(min(max(0.0, cy - side / 2.0), height - side)))
    x1, y1 = int(round(x0 + side)), int(round(y0 + side))
    crop = image[y0:y1, x0:x1]
    return crop if getattr(crop, "size", 0) else image


def save_crop(
    image: Any,                # numpy BGR frame (H, W, 3)
    bbox_xywh: tuple[int, int, int, int],
    out_dir: str | Path,
    name: str,
) -> str:
    """Crop the bbox region of ``image`` and save it; return the image_ref."""
    return save_image(crop_bbox(image, bbox_xywh), out_dir, name)
