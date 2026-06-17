"""
try_vision.py
=============
Step 2 demo of the vision core.

Reads a clip (or image folder), runs detection + tracking on a BACKGROUND
thread with a 1-slot latest-wins buffer, draws boxes/labels, saves annotated
frames to ``captures/`` and prints per-frame class / confidence / latency / FPS.

Run:
    python try_vision.py

Inputs and thresholds come from config.yaml. With no custom rail-defect weights
present, a pretrained COCO YOLO is used as a clearly-labelled placeholder so the
full pipeline still runs end to end.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.config import PROJECT_ROOT, load_config
from src.inference.base import RawDetection
from src.inference.worker import InferenceResult, ThreadedInferenceWorker
from src.inference.yolo_engine import YoloEngine
from src.sources.file_source import FileSource

CAPTURES_DIR: Path = PROJECT_ROOT / "captures"

# Fixed colours (BGR) for the contract defect classes; any other label (e.g. a
# placeholder COCO class) gets a deterministic colour derived from its name.
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "crack": (0, 0, 255),
    "spalling": (0, 128, 255),
    "corrugation": (0, 215, 255),
    "squat": (0, 255, 255),
    "missing_fastener": (255, 0, 0),
    "broken_fastener": (255, 0, 255),
    "loose_fastener": (255, 128, 0),
}


def color_for(label: str) -> tuple[int, int, int]:
    if label in _CLASS_COLORS:
        return _CLASS_COLORS[label]
    h = abs(hash(label))
    return (60 + h % 180, 60 + (h // 180) % 180, 60 + (h // 32400) % 180)


def draw_detections(image, detections: list[RawDetection]):
    """Draw each detection's box and a readable label onto the image."""
    for d in detections:
        x, y, w, h = d.bbox_xywh
        c = color_for(d.defect_class)
        cv2.rectangle(image, (x, y), (x + w, y + h), c, 2)
        tag = f"{d.defect_class} {d.confidence:.2f}"
        if d.track_id >= 0:
            tag += f" #{d.track_id}"
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytop = max(0, y - th - bl - 4)
        cv2.rectangle(image, (x, ytop), (x + tw + 4, ytop + th + bl + 4), c, -1)
        cv2.putText(image, tag, (x + 2, ytop + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def clear_captures() -> int:
    """Remove annotated frames from a previous run; return how many."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for old in CAPTURES_DIR.glob("frame_*.jpg"):
        old.unlink()
        removed += 1
    return removed


def main() -> None:
    config = load_config()
    source = FileSource.from_config(config)
    engine = YoloEngine.from_config(config)

    print("Loading model ...")
    engine.load()
    engine.warmup()
    print(f"Model ready: {engine.model_name}\n")

    removed = clear_captures()
    if removed:
        print(f"(cleared {removed} old frame(s) from captures/)\n")

    worker = ThreadedInferenceWorker(engine)

    saved = 0
    last_saved_id = -1
    last_submitted_id = -1
    infer_ms_sum = 0.0

    def handle(result: "InferenceResult | None") -> None:
        nonlocal saved, last_saved_id, infer_ms_sum
        if result is None or result.frame.frame_id == last_saved_id:
            return
        last_saved_id = result.frame.frame_id
        annotated = draw_detections(result.frame.image.copy(), result.detections)
        cv2.imwrite(str(CAPTURES_DIR / f"frame_{result.frame.frame_id:05d}.jpg"), annotated)
        saved += 1
        infer_ms_sum += result.inference_ms
        if result.detections:
            shown = ", ".join(
                f"{d.defect_class}:{d.confidence:.2f}" for d in result.detections[:4]
            )
            if len(result.detections) > 4:
                shown += " ..."
        else:
            shown = "(none)"
        print(f"[frame {result.frame.frame_id:05d}] dets={len(result.detections):<2} "
              f"{shown}  | infer={result.inference_ms:6.1f} ms  fps={result.fps:5.1f}")

    source.open()  # populate fps / frame_count before we print them
    total = source.frame_count
    print(f"Source: {source.source_type}  fps~{source.fps:.1f}  "
          f"frames={'?' if total < 0 else total}\n")

    wall_start = time.perf_counter()
    with source, worker:
        for frame in source.frames():
            worker.submit(frame)
            last_submitted_id = frame.frame_id
            handle(worker.get_latest())
        # Drain: ensure the final submitted frame is processed and saved.
        if last_submitted_id >= 0:
            handle(worker.wait_for(last_submitted_id, timeout=15.0))
    wall = time.perf_counter() - wall_start

    print("\n" + "=" * 60)
    print(" try_vision summary")
    print("=" * 60)
    print(f"  frames submitted              : {worker.frames_submitted}")
    print(f"  frames processed              : {worker.frames_processed}")
    print(f"  frames dropped (1-slot buffer): {worker.frames_dropped}")
    print(f"  annotated frames saved        : {saved}  ->  {CAPTURES_DIR}")
    if saved:
        print(f"  average inference latency     : {infer_ms_sum / saved:6.1f} ms")
    if wall > 0:
        print(f"  end-to-end throughput         : {worker.frames_processed / wall:5.1f} fps (wall)")
    print(f"  model                         : {engine.model_name}")
    if engine.is_placeholder:
        print("  NOTE: placeholder COCO model -- labels are NOT rail defects yet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
