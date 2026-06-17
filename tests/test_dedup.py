"""Tests for track-based de-duplication (one record per defect, best representative)."""
from datetime import datetime, timedelta, timezone

import numpy as np

from src.dedup import TrackAggregator
from src.inference.base import RawDetection
from src.localisation.base import GeoPosition

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _pos() -> GeoPosition:
    return GeoPosition(lat=3.1, lng=101.6, chainage_m=5.0, speed_mps=0.4)


def _img() -> "np.ndarray":
    return np.zeros((64, 64, 3), dtype="uint8")


def _raw(conf: float, track_id: int = 1, cls: str = "crack") -> RawDetection:
    return RawDetection(defect_class=cls, confidence=conf, bbox_xywh=(5, 5, 20, 20), track_id=track_id)


def _update(agg, conf, track_id=1, frame_id=0, t=None):
    return agg.update(raw=_raw(conf, track_id), prescription=None, position=_pos(),
                      frame_id=frame_id, timestamp=t or T0, frame_image=_img())


def test_one_track_many_frames_collapses_to_one_record():
    agg = TrackAggregator()
    confs = [0.4, 0.6, 0.9, 0.7, 0.5]
    for i, c in enumerate(confs):
        _update(agg, c, track_id=1, frame_id=i, t=T0 + timedelta(seconds=i))
    recs = agg.consolidated()
    assert len(recs) == 1
    rec = recs[0]
    assert rec.frame_count == 5
    assert rec.best.raw.confidence == 0.9          # highest-confidence representative kept
    assert rec.first_frame_id == 0 and rec.last_frame_id == 4


def test_untracked_detections_are_never_merged():
    agg = TrackAggregator()
    for i in range(3):
        _update(agg, 0.8, track_id=-1, frame_id=i, t=T0 + timedelta(seconds=i))
    assert len(agg.consolidated()) == 3            # each untracked sighting is its own record


def test_distinct_tracks_stay_separate():
    agg = TrackAggregator()
    _update(agg, 0.8, track_id=1, frame_id=0)
    _update(agg, 0.8, track_id=2, frame_id=1)
    assert len(agg.consolidated()) == 2


def test_update_signals_only_on_new_or_improved():
    agg = TrackAggregator()
    assert _update(agg, 0.5, frame_id=0) is not None   # new track -> publishable
    assert _update(agg, 0.4, frame_id=1) is None       # worse -> no event
    assert _update(agg, 0.9, frame_id=2) is not None    # improved -> event
