"""Tests for the contract schema: Detection / Telemetry validation + round-trip."""
import pytest
from pydantic import ValidationError

from src.schema import DefectClass, Detection, Location, Severity, Telemetry


def _det(**overrides) -> Detection:
    base = dict(
        frame_id=1, track_id=1, defect_class="crack", confidence=0.9,
        bbox_xywh=[10, 10, 40, 30], severity="High", urgency_score=80,
        recommended_action="inspect", location=Location(lat=3.1, lng=101.6, chainage_m=5.0),
        image_ref="crops/x.jpg", model="yolov8n",
    )
    base.update(overrides)
    return Detection(**base)


def test_detection_valid():
    d = _det()
    assert d.defect_class is DefectClass.CRACK
    assert d.severity is Severity.HIGH


def test_detection_json_roundtrip():
    d = _det()
    assert Detection.from_json(d.to_json()) == d


def test_detection_rejects_unknown_defect_class():
    with pytest.raises(ValidationError):
        _det(defect_class="rust")


def test_detection_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        _det(confidence=1.5)


def test_detection_rejects_degenerate_bbox():
    with pytest.raises(ValidationError):
        _det(bbox_xywh=[0, 0, 0, 10])  # zero width


def test_detection_rejects_urgency_out_of_range():
    with pytest.raises(ValidationError):
        _det(urgency_score=150)


def test_detection_forbids_extra_fields():
    with pytest.raises(ValidationError):
        _det(unexpected="nope")


def test_location_rejects_out_of_range_lat():
    with pytest.raises(ValidationError):
        Location(lat=200.0, lng=0.0, chainage_m=0.0)


def test_telemetry_valid_and_roundtrip():
    t = Telemetry(lat=3.1, lng=101.6, chainage_m=10.0, speed_mps=0.4,
                  battery_pct=99.0, fps=15.0, inference_ms=57.0)
    assert Telemetry.from_json(t.to_json()) == t


def test_telemetry_rejects_battery_out_of_range():
    with pytest.raises(ValidationError):
        Telemetry(lat=0, lng=0, chainage_m=0, speed_mps=0,
                  battery_pct=150.0, fps=0, inference_ms=0)
