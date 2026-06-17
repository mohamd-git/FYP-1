"""
src/sources/file_source.py
==========================
Concrete FrameSource that reads frames from a video file OR a folder of images.

This is the PoC implementation of swappable seam #1. In Phase 2 a live
CameraSource with the same interface replaces it with zero downstream changes.

Which input is used is driven by config:
    source.type          -> "video" | "images"
    paths.video_input    -> the video file
    paths.image_folder   -> the image folder
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2

from src.sources.base import Frame, FrameSource

# Image extensions recognised when reading an image folder.
_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class FileSource(FrameSource):
    """Read frames from a video file or an image folder.

    Args:
        source_type: "video" reads ``video_path``; "images" reads every image in
            ``image_folder`` (sorted by name).
        video_path: video file (used when source_type == "video").
        image_folder: folder of images (used when source_type == "images").
        loop: if True, restart from the beginning when exhausted (endless demo).
        target_fps: if > 0, throttle iteration to ~this many frames/second
            (simulate real-time); 0 = yield as fast as possible.
    """

    def __init__(
        self,
        *,
        source_type: str = "video",
        video_path: str | Path = "data/sample_run.mp4",
        image_folder: str | Path = "data/frames",
        loop: bool = False,
        target_fps: float = 0.0,
    ) -> None:
        self.source_type = source_type.lower().strip()
        self.video_path = Path(video_path)
        self.image_folder = Path(image_folder)
        self.loop = bool(loop)
        self.target_fps = float(target_fps)

        self._cap: "cv2.VideoCapture | None" = None
        self._image_files: list[Path] = []
        self._nominal_fps: float = 0.0
        self._frame_count: int = -1  # best-effort total (-1 = unknown)
        self._t_start: float = 0.0   # wall-clock anchor; frame timestamps follow the video timeline
        self._opened = False

    # ---- factory --------------------------------------------------------- #
    @classmethod
    def from_config(cls, config: dict) -> "FileSource":
        """Build a FileSource from a parsed config dict."""
        from src.config import resolve_path  # local import avoids an import cycle

        src = config.get("source", {})
        paths = config.get("paths", {})
        return cls(
            source_type=src.get("type", "video"),
            video_path=resolve_path(paths.get("video_input", "data/sample_run.mp4")),
            image_folder=resolve_path(paths.get("image_folder", "data/frames")),
            loop=bool(src.get("loop", False)),
            target_fps=float(src.get("target_fps", 0) or 0),
        )

    # ---- FrameSource interface ------------------------------------------ #
    def open(self) -> None:
        if self._opened:
            return  # idempotent: safe to call before iterating and via __enter__
        if self.source_type == "video":
            self._open_video()
        elif self.source_type == "images":
            self._open_images()
        else:
            raise ValueError(
                f"Unknown source.type {self.source_type!r}; expected 'video' or 'images'."
            )
        self._t_start = time.time()
        self._opened = True

    def _open_video(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}\n"
                f"Put a clip there or set paths.video_input in config.yaml."
            )
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.video_path}")
        self._cap = cap
        self._nominal_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self._nominal_fps <= 0:
            self._nominal_fps = 30.0  # sensible default if the container lacks fps
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_count = count if count > 0 else -1

    def _open_images(self) -> None:
        if not self.image_folder.is_dir():
            raise FileNotFoundError(
                f"Image folder not found: {self.image_folder}\n"
                f"Create it (add images) or set paths.image_folder in config.yaml."
            )
        files = sorted(
            p for p in self.image_folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No images found in: {self.image_folder}")
        self._image_files = files
        self._frame_count = len(files)
        # Images have no intrinsic fps; use target_fps or a default for telemetry.
        self._nominal_fps = self.target_fps if self.target_fps > 0 else 30.0

    def frames(self) -> Iterator[Frame]:
        if not self._opened:
            self.open()
        if self.source_type == "video":
            yield from self._iter_video()
        else:
            yield from self._iter_images()

    def _throttle(self, frame_index: int, start_time: float) -> None:
        """Sleep so iteration approximates target_fps (no-op if target_fps<=0)."""
        if self.target_fps <= 0:
            return
        due = start_time + (frame_index + 1) / self.target_fps
        delay = due - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def _iter_video(self) -> Iterator[Frame]:
        assert self._cap is not None
        frame_id = 0
        start = time.perf_counter()
        while True:
            ok, image = self._cap.read()
            if not ok:
                if self.loop and frame_id > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield Frame(frame_id=frame_id, image=image,
                        timestamp=self._t_start + frame_id / self._nominal_fps)
            self._throttle(frame_id, start)
            frame_id += 1

    def _iter_images(self) -> Iterator[Frame]:
        frame_id = 0
        start = time.perf_counter()
        idx = 0
        while idx < len(self._image_files):
            image = cv2.imread(str(self._image_files[idx]))
            idx += 1
            if image is None:
                continue  # skip unreadable file
            yield Frame(frame_id=frame_id, image=image,
                        timestamp=self._t_start + frame_id / self._nominal_fps)
            self._throttle(frame_id, start)
            frame_id += 1
            if idx >= len(self._image_files) and self.loop:
                idx = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    @property
    def fps(self) -> float:
        return self._nominal_fps

    @property
    def frame_count(self) -> int:
        """Best-effort total frame count (-1 if unknown)."""
        return self._frame_count
