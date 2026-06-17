"""Localisation sources: the swappable seam for AGV positioning.

PoC: ``SimTrack`` replays a CSV corridor at the configured inspection speed.
Phase 2: real GPS + wheel odometry + IMU behind the same LocalisationSource API.
"""

from src.localisation.base import GeoPosition, LocalisationSource
from src.localisation.sim_track import SimTrack, TrackPoint, generate_sample_track, load_track

__all__ = [
    "GeoPosition",
    "LocalisationSource",
    "SimTrack",
    "TrackPoint",
    "generate_sample_track",
    "load_track",
]
