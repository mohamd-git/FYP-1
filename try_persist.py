"""
try_persist.py
==============
Step 5 demo: tracking -> de-duplication -> SQLite persistence.

One physical defect spans many frames (same track_id). This runs the real
``Pipeline`` with a SCRIPTED detector that emits stable track_ids across many
frames with frame-varying confidence -- standing in for the trained model +
ByteTrack (the placeholder COCO model has no rail classes). It proves that a
defect seen in N frames becomes exactly ONE persisted row, keeping the
highest-confidence frame as the representative.

Run:
    python try_persist.py
"""

from __future__ import annotations

import time
from typing import Iterator

from src.config import load_config, resolve_path
from src.inference.base import InferenceEngine, RawDetection
from src.localisation.sim_track import SimTrack
from src.pipeline import Pipeline
from src.prescriptive.engine import Prescriber
from src.sources.base import Frame, FrameSource
from src.sources.file_source import FileSource
from src.storage.db import Database

N_FRAMES = 120

# Three physical defects, each spanning many frames with a stable track_id.
# (track_id, defect_class, start_frame, end_frame, peak_frame, peak_conf, bbox)
DEFECT_SCRIPT = [
    (1, "crack", 0, 49, 25, 0.93, (200, 180, 70, 50)),            # 50 frames
    (2, "missing_fastener", 20, 35, 28, 0.91, (420, 300, 60, 55)),  # 16 frames
    (3, "corrugation", 60, 95, 78, 0.62, (150, 260, 120, 45)),    # 36 frames
]
EXPECTED_DEFECTS = len(DEFECT_SCRIPT)


def _conf_at(frame_id: int, start: int, end: int, peak: int, top: float, base: float = 0.40) -> float:
    """Triangular confidence curve: `base` at the ends, `top` at `peak`."""
    if frame_id <= peak:
        f = (frame_id - start) / max(1, peak - start)
    else:
        f = (end - frame_id) / max(1, end - peak)
    return round(base + (top - base) * max(0.0, min(1.0, f)), 2)


def build_script() -> dict[int, list[RawDetection]]:
    """frame_id -> list of RawDetections that frame should produce."""
    script: dict[int, list[RawDetection]] = {i: [] for i in range(N_FRAMES)}
    for tid, cls, start, end, peak, top, bbox in DEFECT_SCRIPT:
        for fid in range(start, end + 1):
            script[fid].append(
                RawDetection(defect_class=cls, confidence=_conf_at(fid, start, end, peak, top),
                             bbox_xywh=bbox, track_id=tid)
            )
    return script


class ScriptedEngine(InferenceEngine):
    """Stand-in detector: returns predetermined detections per frame_id.

    Simulates a trained rail model + ByteTrack (stable track_ids across frames).
    """

    def __init__(self, script: dict[int, list[RawDetection]]) -> None:
        self._script = script

    def load(self) -> None:  # nothing to load
        pass

    def infer(self, frame: Frame) -> list[RawDetection]:
        return self._script.get(frame.frame_id, [])

    @property
    def model_name(self) -> str:
        return "scripted (stand-in for trained yolo_rail + ByteTrack)"


class SyntheticSource(FrameSource):
    """Fallback frame source if the sample clip is absent (gray frames)."""

    def __init__(self, n: int = N_FRAMES, w: int = 640, h: int = 480, fps: float = 15.0) -> None:
        self._n, self._w, self._h, self._fps, self._t0 = n, w, h, fps, 0.0

    def open(self) -> None:
        self._t0 = time.time()

    def frames(self) -> Iterator[Frame]:
        import numpy as np

        for i in range(self._n):
            yield Frame(i, np.full((self._h, self._w, 3), 45, "uint8"), self._t0 + i / self._fps)

    def close(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return self._fps


def main() -> None:
    config = load_config()

    video = resolve_path((config.get("paths", {}) or {}).get("video_input", "data/sample_run.mp4"))
    if video.is_file():
        source: FrameSource = FileSource.from_config(config)
        source.target_fps = 0.0  # process fast; positions come from video-timeline timestamps
        src_kind = f"sample clip ({video.name})"
    else:
        source = SyntheticSource()
        src_kind = "synthetic frames (sample clip not found)"

    db = Database.from_config(config).connect()
    pipeline = Pipeline(
        source=source,
        engine=ScriptedEngine(build_script()),
        localisation=SimTrack.from_config(config),
        prescriber=Prescriber.from_yaml(),
        db=db,
        crops_dir=resolve_path((config.get("paths", {}) or {}).get("crops_dir", "data/output/crops")),
    )

    print(f"Running pipeline over {src_kind} ...")
    summary = pipeline.run(reset=True)
    print(f"  frames processed       : {summary['frames']}")
    print(f"  raw detections (frames): {summary['raw_detections']}")
    print(f"  persisted defects (rows): {summary['persisted_defects']}")
    print(f"  -> de-dup ratio: {summary['raw_detections']} detections collapsed into "
          f"{summary['persisted_defects']} defect records\n")

    print("=" * 100)
    print(" Consolidated defect register (SQLite)")
    print("=" * 100)
    print(f"  {'track':>5}  {'class':<16} {'sev':<6} {'urg':>3} {'conf':>5} {'frames':>6} "
          f"{'chainage':>9}  image_ref")
    for r in db.all_defects():
        print(f"  {r['track_id']:>5}  {r['defect_class']:<16} {r['severity']:<6} "
              f"{r['urgency_score']:>3} {r['confidence']:>5.2f} {r['frame_count']:>6} "
              f"{r['chainage_m']:>8.1f}m  {r['image_ref']}")
    print(f"\n  severity counts: {db.severity_counts()}")

    csv_path = db.export_csv(resolve_path("data/output/defect_register.csv"))
    json_path = db.export_json(resolve_path("data/output/defect_register.json"))
    print(f"  exported: {csv_path.name}, {json_path.name} (under data/output/)")

    # ---- acceptance assertions ---------------------------------------- #
    print("\n" + "-" * 100)
    print(" Acceptance checks")
    print("-" * 100)
    assert db.count() == EXPECTED_DEFECTS, f"expected {EXPECTED_DEFECTS} rows, got {db.count()}"
    print(f"  [PASS] one row per physical defect ({db.count()} rows for {EXPECTED_DEFECTS} defects)")

    crack = db.get_by_track(1)
    assert crack is not None
    assert crack["frame_count"] == 50, f"expected 50 frames, got {crack['frame_count']}"
    print(f"  [PASS] track 1 (crack) seen in {crack['frame_count']} frames -> 1 row")
    assert abs(crack["confidence"] - 0.93) < 1e-9, f"representative conf {crack['confidence']}"
    print(f"  [PASS] representative confidence = {crack['confidence']:.2f} (the per-track maximum)")
    rep_img = resolve_path(crack["image_ref"])
    assert rep_img.is_file(), f"representative image missing: {rep_img}"
    print(f"  [PASS] representative image saved: {crack['image_ref']}")

    db.close()
    print("\nAll acceptance checks passed. OK")


if __name__ == "__main__":
    main()
