"""
src/sources/base.py
===================
Abstract frame source -- swappable seam #1.

PoC implementation (later step): reads frames from a video file or a folder of
images (e.g. ``src/sources/video_source.py``).
Hardware implementation (Phase 2): reads frames from a live USB/CSI camera, with
the *exact same* interface, so nothing downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # only needed for type checking; not required to import this module
    import numpy as np


@dataclass
class Frame:
    """One captured frame handed to the inference engine."""

    frame_id: int
    image: "np.ndarray"  # BGR image, shape (H, W, 3), dtype uint8 (OpenCV convention)
    timestamp: float  # capture time, epoch seconds


class FrameSource(ABC):
    """Produces a stream of :class:`Frame` objects.

    Concrete subclasses decide *where* frames come from; everything downstream
    only iterates :meth:`frames`. Implementations should be usable as a context
    manager::

        with VideoSource(path) as src:
            for frame in src.frames():
                ...
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource (open the file / camera)."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames in order until the source is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames-per-second of the source (used for timing/telemetry)."""

    # Structural context-manager sugar (not business logic).
    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
