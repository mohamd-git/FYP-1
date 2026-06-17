"""
src/schema.py
=============
Single source of truth for every message that travels over MQTT and is
persisted to SQLite. Every module imports these models, so the wire format can
never drift between the publisher (the AGV pipeline) and the consumers (the
dashboard, the database layer).

Implements the two contract messages defined in the project brief:

* ``Detection`` -> published to topic ``agv/detections``
* ``Telemetry``  -> published to topic ``agv/telemetry``

A small ``Status`` model is also provided for the ``agv/status`` heartbeat. The
brief names that topic but does not fix its fields, so ``Status`` is a minimal,
sensible scaffold -- a documented design choice, not a brief requirement.

These are Pydantic v2 models: construction validates the data, and
``model_dump_json()`` / ``model_validate_json()`` produce/parse the exact JSON
used on the wire.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #
class DefectClass(str, Enum):
    """The seven defect classes the detector is trained to recognise.

    Two families:
      * rail-surface defects : crack, spalling, corrugation, squat
      * fastener defects     : missing_fastener, broken_fastener, loose_fastener
    """

    CRACK = "crack"
    SPALLING = "spalling"
    CORRUGATION = "corrugation"
    SQUAT = "squat"
    MISSING_FASTENER = "missing_fastener"
    BROKEN_FASTENER = "broken_fastener"
    LOOSE_FASTENER = "loose_fastener"


class Severity(str, Enum):
    """Severity band assigned by the prescriptive engine."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SystemState(str, Enum):
    """Coarse AGV state reported on the heartbeat topic (agv/status)."""

    ONLINE = "online"
    INSPECTING = "inspecting"
    OFFLINE = "offline"
    ERROR = "error"


def _utc_now() -> datetime:
    """Timezone-aware UTC timestamp (serialises to ISO-8601 with offset)."""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Nested object
# --------------------------------------------------------------------------- #
class Location(BaseModel):
    """Geo-tag for a detection: WGS-84 coordinates plus track chainage.

    ``chainage_m`` is the distance travelled along the rail from the start of
    the run, in metres -- the railway-standard way of referencing a position
    along a track.
    """

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude, WGS-84 degrees.")
    lng: float = Field(..., ge=-180.0, le=180.0, description="Longitude, WGS-84 degrees.")
    chainage_m: float = Field(..., ge=0.0, description="Distance along track from start, metres.")


# --------------------------------------------------------------------------- #
# Contract message: Detection  ->  topic agv/detections
# --------------------------------------------------------------------------- #
class Detection(BaseModel):
    """A single enriched defect detection.

    This model is the heart of the project's novelty: it couples *detecting* a
    defect (defect_class, confidence, bbox) with *acting* on it (severity,
    urgency_score, recommended_action) and *locating* it (location) in one
    record streamed live to the operator.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    detection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique id for this detection (uuid4).",
    )
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="When the detection was produced (ISO-8601, UTC).",
    )
    frame_id: int = Field(..., ge=0, description="Index of the source frame.")
    track_id: int = Field(
        ...,
        ge=-1,
        description="Multi-object-tracker id; -1 means not (yet) tracked.",
    )
    defect_class: DefectClass = Field(..., description="Predicted defect class.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence, 0..1.")
    bbox_xywh: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x, y, w, h] in pixels (top-left origin).",
    )
    severity: Severity = Field(..., description="Severity band (prescriptive engine).")
    urgency_score: int = Field(
        ..., ge=0, le=100, description="Prioritisation score 0..100 (higher = sooner)."
    )
    recommended_action: str = Field(
        ..., min_length=1, description="Human-readable maintenance action."
    )
    location: Location = Field(..., description="Geo-tag for this detection.")
    image_ref: str = Field(
        ..., min_length=1, description="Path/URL to the saved annotated crop or frame."
    )
    model: str = Field(
        ..., min_length=1, description="Identifier of the model that produced this detection."
    )

    @field_validator("bbox_xywh")
    @classmethod
    def _validate_bbox(cls, value: list[int]) -> list[int]:
        """Enforce a sane [x, y, w, h]: origin non-negative, size positive."""
        x, y, w, h = value
        if x < 0 or y < 0:
            raise ValueError("bbox x and y must be >= 0")
        if w <= 0 or h <= 0:
            raise ValueError("bbox width and height must be > 0")
        return value

    def to_json(self) -> str:
        """Serialise to the exact JSON string published on MQTT."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> "Detection":
        """Parse and validate a JSON payload received from MQTT."""
        return cls.model_validate_json(payload)


# --------------------------------------------------------------------------- #
# Contract message: Telemetry  ->  topic agv/telemetry
# --------------------------------------------------------------------------- #
class Telemetry(BaseModel):
    """Periodic health/position sample for the live dashboard gauges."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=_utc_now, description="Sample time (ISO-8601, UTC).")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Current latitude.")
    lng: float = Field(..., ge=-180.0, le=180.0, description="Current longitude.")
    chainage_m: float = Field(..., ge=0.0, description="Current chainage, metres.")
    speed_mps: float = Field(..., ge=0.0, description="Ground speed, metres/second.")
    battery_pct: float = Field(..., ge=0.0, le=100.0, description="Battery remaining, percent.")
    fps: float = Field(..., ge=0.0, description="Pipeline throughput, frames/second.")
    inference_ms: float = Field(..., ge=0.0, description="Per-frame inference latency, ms.")

    def to_json(self) -> str:
        """Serialise to the exact JSON string published on MQTT."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> "Telemetry":
        """Parse and validate a JSON payload received from MQTT."""
        return cls.model_validate_json(payload)


# --------------------------------------------------------------------------- #
# Heartbeat message: Status  ->  topic agv/status
# --------------------------------------------------------------------------- #
class Status(BaseModel):
    """Liveness heartbeat. The brief fixes the topic but not the fields; this field set is a documented design choice."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    timestamp: datetime = Field(default_factory=_utc_now, description="Heartbeat time (ISO-8601).")
    state: SystemState = Field(..., description="Coarse AGV state.")
    detail: str = Field(default="", description="Optional free-text detail.")

    def to_json(self) -> str:
        """Serialise to the exact JSON string published on MQTT."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> "Status":
        """Parse and validate a JSON payload received from MQTT."""
        return cls.model_validate_json(payload)


__all__ = [
    "DefectClass",
    "Severity",
    "SystemState",
    "Location",
    "Detection",
    "Telemetry",
    "Status",
]
