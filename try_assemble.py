"""
try_assemble.py
===============
Step 4 demo: geo-tag detections and assemble the full contract messages.

It runs the localisation source along the bundled corridor (data/track.csv),
takes a set of representative detections (stand-ins for the trained model -- the
placeholder COCO model has no rail classes), prescribes a maintenance decision
for each, saves an image crop from the sample clip, and assembles a schema-valid
``Detection``. It also emits ``Telemetry`` samples as the AGV advances.

Run:
    python try_assemble.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from src.assembler import build_detection, build_telemetry, save_crop, simulate_battery
from src.config import load_config, resolve_path
from src.inference.base import RawDetection
from src.localisation.sim_track import SimTrack, generate_sample_track
from src.prescriptive.engine import Prescriber
from src.sources.base import Frame
from src.sources.file_source import FileSource

# Representative detections (defect_class, confidence, bbox_xywh) -- what the
# TRAINED model will emit. Used here because the placeholder COCO model knows no
# rail classes. Bboxes fit a 640x480 frame.
SAMPLE_DETECTIONS: list[tuple[str, float, tuple[int, int, int, int]]] = [
    ("missing_fastener", 0.91, (120, 300, 70, 55)),
    ("crack", 0.82, (240, 180, 60, 40)),
    ("broken_fastener", 0.67, (400, 320, 65, 50)),
    ("spalling", 0.63, (300, 150, 110, 80)),
    ("corrugation", 0.55, (150, 250, 120, 45)),
    ("loose_fastener", 0.72, (500, 270, 60, 50)),
    ("squat", 0.49, (220, 360, 70, 60)),
    ("crack", 0.40, (350, 200, 45, 30)),
]

SECONDS_BETWEEN_SAMPLES = 20.0   # so the AGV visibly advances along the corridor
MODEL_TAG = "sample (stand-in for trained yolo_rail)"


def _get_frames(source: FileSource, n: int) -> list[Frame]:
    """Read up to n frames from the sample clip; fall back to synthetic frames."""
    try:
        source.open()
        gen = source.frames()
        frames = []
        for _ in range(n):
            frame = next(gen, None)
            if frame is None:
                break
            frames.append(frame)
        if frames:
            return frames
    except Exception as exc:
        print(f"(sample clip unavailable -- using synthetic frames: {exc})")
    import numpy as np

    return [Frame(frame_id=i, image=np.full((480, 640, 3), 45, "uint8"), timestamp=time.time())
            for i in range(n)]


def main() -> None:
    config = load_config()

    # Localisation -- ensure the corridor exists, then open it.
    track = SimTrack.from_config(config)
    if not track.csv_path.is_file():
        generate_sample_track(track.csv_path)
    track.open()
    p0 = track.position_at_chainage(0.0)
    print(f"Corridor: {track.num_points} points, {track.length_m:.0f} m, "
          f"start ({p0.lat:.5f}, {p0.lng:.5f}) @ {track.inspection_speed_mps} m/s\n")

    prescriber = Prescriber.from_yaml()
    source = FileSource.from_config(config)
    frames = _get_frames(source, len(SAMPLE_DETECTIONS))
    crops_dir = resolve_path((config.get("paths", {}) or {}).get("crops_dir", "data/output/crops"))

    base_time = datetime.now(timezone.utc)
    tel_cfg = config.get("telemetry", {}) or {}
    batt_start = float(tel_cfg.get("battery_start_pct", 100.0))
    batt_drain = float(tel_cfg.get("battery_drain_per_min", 0.5))

    detections = []
    telemetries = []

    print("=" * 96)
    print(" Assembled detections (geo-tagged, prescribed)")
    print("=" * 96)
    for i, (cls, conf, bbox) in enumerate(SAMPLE_DETECTIONS):
        frame = frames[i % len(frames)]
        h, w = frame.image.shape[:2]
        elapsed = i * SECONDS_BETWEEN_SAMPLES
        ts = base_time + timedelta(seconds=elapsed)

        raw = RawDetection(defect_class=cls, confidence=conf, bbox_xywh=bbox, track_id=i + 1)
        prescription = prescriber.prescribe_bbox(cls, conf, bbox, (w, h))
        position = track.position_at_elapsed(elapsed)
        image_ref = save_crop(frame.image, bbox, crops_dir, f"det_{i:03d}_{cls}")

        det = build_detection(
            raw=raw, prescription=prescription, location=position,
            frame_id=frame.frame_id, image_ref=image_ref, model=MODEL_TAG, timestamp=ts,
        )
        detections.append(det)

        # Telemetry once per detection step.
        telemetries.append(build_telemetry(
            location=position, fps=15.0, inference_ms=57.5,
            battery_pct=simulate_battery(elapsed, batt_start, batt_drain), timestamp=ts,
        ))

        print(f"  #{i} {cls:<16} conf={conf:.2f} -> {prescription.severity:<6} "
              f"urg={det.urgency_score:3d} [{prescription.band}]  @ chainage "
              f"{position.chainage_m:5.1f} m ({position.lat:.5f}, {position.lng:.5f})")

    # Show two full, schema-valid Detection messages.
    print("\n" + "=" * 96)
    print(" Sample Detection JSON (full contract message)")
    print("=" * 96)
    for det in detections[:2]:
        print(det.model_dump_json(indent=2))
        print("-" * 60)

    # Show two full Telemetry messages.
    print("=" * 96)
    print(" Sample Telemetry JSON (full contract message)")
    print("=" * 96)
    for tel in telemetries[:2]:
        print(tel.model_dump_json(indent=2))
        print("-" * 60)

    # Prove every message round-trips through the schema validator.
    ok_det = all(Detection_roundtrips(d) for d in detections)
    ok_tel = all(Telemetry_roundtrips(t) for t in telemetries)

    print("=" * 96)
    print(f" Assembled {len(detections)} Detection + {len(telemetries)} Telemetry messages.")
    print(f" Schema round-trip valid: detections={ok_det}  telemetry={ok_tel}")
    print(f" Crops saved under: {crops_dir}")
    print("=" * 96)
    assert ok_det and ok_tel, "Some assembled messages failed schema validation."


def Detection_roundtrips(det) -> bool:
    from src.schema import Detection

    return Detection.model_validate_json(det.model_dump_json()) == det


def Telemetry_roundtrips(tel) -> bool:
    from src.schema import Telemetry

    return Telemetry.model_validate_json(tel.model_dump_json()) == tel


if __name__ == "__main__":
    main()
