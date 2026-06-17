"""
src/dedup.py
============
Track-based deduplication.

One physical defect spans many consecutive frames; ByteTrack gives all those
detections the SAME ``track_id``. This module consolidates them into ONE record
per track, keeping the highest-confidence frame as the representative (its bbox,
confidence and image crop), plus first/last-seen timestamps and a frame count.

The aggregator is pure (no disk I/O): it only slices the representative crop out
of the frame (a NumPy view-copy). The pipeline saves that crop and persists the
consolidated record at the end of the run.

Untracked detections (``track_id == -1``) are never merged -- each becomes its
own record (keyed by a running counter), so we never collapse two genuinely
different unidentified defects into one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from src.inference.base import RawDetection
from src.localisation.base import GeoPosition

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Observation:
    """A single (representative) sighting of a defect on one frame."""

    raw: RawDetection
    prescription: Any            # src.prescriptive.engine.Prescription
    position: GeoPosition
    frame_id: int
    timestamp: datetime
    crop: "np.ndarray"           # cropped BGR image of this sighting


@dataclass
class ConsolidatedDefect:
    """One physical defect, consolidated across all the frames it appeared in."""

    defect_key: str              # stable primary key ("track-7" / "untracked-3")
    best: Observation            # the highest-confidence sighting (representative)
    first_seen: datetime
    last_seen: datetime
    first_frame_id: int
    last_frame_id: int
    frame_count: int
    image_ref: str = ""          # set by the pipeline once the representative crop is saved

    @property
    def track_id(self) -> int:
        return self.best.raw.track_id


class TrackAggregator:
    """Consolidates per-frame detections into one record per track_id."""

    def __init__(self) -> None:
        self._records: dict[str, ConsolidatedDefect] = {}
        self._untracked = 0

    def update(
        self,
        *,
        raw: RawDetection,
        prescription: Any,
        position: GeoPosition,
        frame_id: int,
        timestamp: datetime,
        frame_image: "np.ndarray",
    ) -> Optional[ConsolidatedDefect]:
        """Fold one detection into its track's record.

        Returns the consolidated record when it is newly created OR its
        representative improved (something worth publishing / re-persisting);
        otherwise returns None.
        """
        if raw.track_id >= 0:
            key = f"track-{raw.track_id}"
            existing = self._records.get(key)
        else:
            key = f"untracked-{self._untracked}"   # each untracked sighting is its own record
            self._untracked += 1
            existing = None

        is_best = existing is None or raw.confidence > existing.best.raw.confidence
        if is_best:
            from src.assembler import crop_bbox  # local import avoids a cycle

            obs = Observation(
                raw=raw,
                prescription=prescription,
                position=position,
                frame_id=frame_id,
                timestamp=timestamp,
                crop=crop_bbox(frame_image, tuple(raw.bbox_xywh)).copy(),
            )

        if existing is None:
            record = ConsolidatedDefect(
                defect_key=key,
                best=obs,
                first_seen=timestamp,
                last_seen=timestamp,
                first_frame_id=frame_id,
                last_frame_id=frame_id,
                frame_count=1,
            )
            self._records[key] = record
            return record  # new defect -> publish + persist

        existing.frame_count += 1
        existing.first_seen = min(existing.first_seen, timestamp)
        existing.last_seen = max(existing.last_seen, timestamp)
        existing.first_frame_id = min(existing.first_frame_id, frame_id)
        existing.last_frame_id = max(existing.last_frame_id, frame_id)
        if is_best:
            existing.best = obs
            return existing  # improved representative -> publish + persist
        return None

    def consolidated(self) -> list[ConsolidatedDefect]:
        """Return all consolidated defect records (one per track)."""
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)
