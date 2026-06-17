"""
src/pipeline.py
===============
End-to-end inspection pipeline -- the single orchestrator:

    frame -> detect+track -> prescribe -> localise -> de-dup -> persist -> publish

Per frame:
  1. read a frame                         (FrameSource)
  2. detect + track defects (ByteTrack)   (InferenceEngine -> stable track_id)
  3. resolve position                     (LocalisationSource)
  4. enrich: severity/urgency/action      (Prescriber + rules.yaml)
  5. consolidate by track_id              (dedup.TrackAggregator)
  6. persist + publish on a new/improved defect:
        - SQLite upsert (storage.Database)         -> agv asset register
        - MQTT publish Detection (messaging)       -> agv/detections
  7. periodically publish Telemetry + a heartbeat  -> agv/telemetry, agv/status

Components are dependency-injected (a scripted detector can drive the same path)
and also buildable from config via :meth:`Pipeline.from_config`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.assembler import build_detection, build_telemetry, save_image, simulate_battery
from src.dedup import TrackAggregator
from src.schema import DefectClass, Status, SystemState

if TYPE_CHECKING:
    from src.inference.base import InferenceEngine
    from src.localisation.base import LocalisationSource
    from src.messaging.publisher import MqttPublisher
    from src.prescriptive.engine import Prescriber
    from src.sources.base import FrameSource
    from src.storage.db import Database

# Valid contract defect-class labels (anything else is skipped or remapped).
_VALID_CLASSES = {c.value for c in DefectClass}
logger = logging.getLogger(__name__)


class Pipeline:
    """Assembles the swappable components and runs the inspection loop."""

    def __init__(
        self,
        *,
        source: "FrameSource",
        engine: "InferenceEngine",
        localisation: "LocalisationSource",
        prescriber: "Prescriber",
        db: "Database",
        crops_dir: str | Path,
        publisher: Optional["MqttPublisher"] = None,
        class_map: Optional[dict[str, str]] = None,
        telemetry_interval_s: float = 1.0,
        battery_start_pct: float = 100.0,
        battery_drain_per_min: float = 0.5,
    ) -> None:
        self.source = source
        self.engine = engine
        self.localisation = localisation
        self.prescriber = prescriber
        self.db = db
        self.crops_dir = Path(crops_dir)
        self.publisher = publisher
        self.class_map = class_map or {}
        self.telemetry_interval_s = float(telemetry_interval_s)
        self.battery_start_pct = float(battery_start_pct)
        self.battery_drain_per_min = float(battery_drain_per_min)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Pipeline":
        """Build the real PoC pipeline (incl. MQTT publisher) from config."""
        from src.config import resolve_path
        from src.inference.yolo_engine import YoloEngine
        from src.localisation.sim_track import SimTrack
        from src.messaging.publisher import MqttPublisher
        from src.prescriptive.engine import Prescriber
        from src.sources.file_source import FileSource
        from src.storage.db import Database

        tele = config.get("telemetry", {}) or {}
        return cls(
            source=FileSource.from_config(config),
            engine=YoloEngine.from_config(config),
            localisation=SimTrack.from_config(config),
            prescriber=Prescriber.from_yaml(),
            db=Database.from_config(config).connect(),
            crops_dir=resolve_path((config.get("paths", {}) or {}).get("crops_dir", "data/output/crops")),
            publisher=MqttPublisher.from_config(config),
            class_map=(config.get("inference", {}) or {}).get("class_map"),
            telemetry_interval_s=tele.get("publish_interval_s", 1.0),
            battery_start_pct=tele.get("battery_start_pct", 100.0),
            battery_drain_per_min=tele.get("battery_drain_per_min", 0.5),
        )

    def _effective_class(self, name: str) -> str:
        return self.class_map.get(name, name)

    # ------------------------------------------------------------------ #
    def run(self, *, reset: bool = True) -> dict[str, int]:
        """Run the inspection: detect -> ... -> persist + publish live.

        Returns a summary dict of counts. Always cleans up the source,
        localisation and MQTT connection, even on error / Ctrl-C.
        """
        self.source.open()
        # Open localisation right after the source (both fast) so its time origin
        # lines up with the frame timestamps BEFORE the slow model load -- otherwise
        # early frames get negative elapsed time and clamp to chainage 0.
        self.localisation.open()
        self.engine.load()
        warmup = getattr(self.engine, "warmup", None)
        if callable(warmup):
            warmup()  # first-inference init, so reported fps/latency is representative
        if reset:
            self.db.reset()

        # Connect MQTT; degrade gracefully to persist-only if no broker.
        publisher = self.publisher
        if publisher is not None:
            if publisher.connect():
                logger.info("MQTT connected to %s; publishing to %s",
                            publisher.active_broker, list(publisher.topics.values()))
                publisher.publish_status(Status(state=SystemState.ONLINE, detail="inspection started"))
            else:
                logger.warning("No MQTT broker reachable (start a local Mosquitto or check the "
                               "HiveMQ fallback network). Continuing WITHOUT publishing; SQLite "
                               "persistence stays active.")
                publisher = None

        aggregator = TrackAggregator()
        counts = {"frames": 0, "raw_detections": 0, "skipped_classes": 0,
                  "published_detections": 0, "telemetry_sent": 0}
        t_wall_start = time.time()
        mission_t0: Optional[float] = None
        last_tel = 0.0

        try:
            for frame in self.source.frames():
                image = frame.image
                if image is None or getattr(image, "size", 0) == 0:
                    continue  # robustness: skip empty/garbled frames
                counts["frames"] += 1
                if mission_t0 is None:
                    mission_t0 = frame.timestamp

                height, width = image.shape[:2]
                position = self.localisation.position_at(frame.timestamp)
                ts = datetime.fromtimestamp(frame.timestamp, tz=timezone.utc)

                for raw in self.engine.infer(frame):
                    counts["raw_detections"] += 1
                    name = self._effective_class(raw.defect_class)
                    if name not in _VALID_CLASSES:
                        counts["skipped_classes"] += 1
                        continue
                    eff = raw if name == raw.defect_class else replace(raw, defect_class=name)
                    prescription = self.prescriber.prescribe_bbox(
                        name, raw.confidence, tuple(raw.bbox_xywh), (width, height)
                    )
                    event = aggregator.update(
                        raw=eff, prescription=prescription, position=position,
                        frame_id=frame.frame_id, timestamp=ts, frame_image=image,
                    )
                    if event is not None:  # new defect or improved representative
                        event.image_ref = save_image(event.best.crop, self.crops_dir, event.defect_key)
                        detection = build_detection(
                            raw=event.best.raw, prescription=event.best.prescription,
                            location=event.best.position, frame_id=event.best.frame_id,
                            image_ref=event.image_ref, model=self.engine.model_name,
                            timestamp=event.best.timestamp,
                        )
                        self.db.upsert_defect(
                            detection, defect_key=event.defect_key,
                            first_seen=event.first_seen, last_seen=event.last_seen,
                            frame_count=event.frame_count,
                            first_frame_id=event.first_frame_id, last_frame_id=event.last_frame_id,
                        )
                        if publisher and publisher.publish_detection(detection):
                            counts["published_detections"] += 1

                # Periodic telemetry + heartbeat (wall-clock paced).
                now = time.time()
                if publisher and (now - last_tel) >= self.telemetry_interval_s:
                    self._send_telemetry(publisher, position, counts["frames"], t_wall_start,
                                         frame.timestamp - mission_t0, ts)
                    counts["telemetry_sent"] += 1
                    last_tel = now
        finally:
            # Correct the final frame_count / last_seen for every defect.
            for cd in aggregator.consolidated():
                image_ref = cd.image_ref or save_image(cd.best.crop, self.crops_dir, cd.defect_key)
                detection = build_detection(
                    raw=cd.best.raw, prescription=cd.best.prescription, location=cd.best.position,
                    frame_id=cd.best.frame_id, image_ref=image_ref, model=self.engine.model_name,
                    timestamp=cd.best.timestamp,
                )
                self.db.upsert_defect(
                    detection, defect_key=cd.defect_key, first_seen=cd.first_seen,
                    last_seen=cd.last_seen, frame_count=cd.frame_count,
                    first_frame_id=cd.first_frame_id, last_frame_id=cd.last_frame_id,
                )
            self.source.close()
            self.localisation.close()
            if publisher:
                publisher.publish_status(Status(state=SystemState.OFFLINE, detail="inspection complete"))
                publisher.disconnect()

        counts["persisted_defects"] = len(aggregator)
        return counts

    def _send_telemetry(self, publisher, position, frames, t_wall_start, mission_elapsed, ts) -> None:
        fps = float(getattr(self.engine, "last_fps", 0.0) or 0.0)
        if fps <= 0:
            fps = frames / max(0.1, time.time() - t_wall_start)
        inference_ms = float(getattr(self.engine, "last_inference_ms", 0.0) or 0.0)
        battery = simulate_battery(mission_elapsed, self.battery_start_pct, self.battery_drain_per_min)
        telemetry = build_telemetry(
            location=position, fps=round(fps, 1), inference_ms=round(inference_ms, 1),
            battery_pct=round(battery, 1), timestamp=ts,
        )
        publisher.publish_telemetry(telemetry)
        publisher.publish_status(Status(
            state=SystemState.INSPECTING, detail=f"chainage {position.chainage_m:.1f} m"
        ))
