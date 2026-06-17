"""Tests for the prescriptive engine: each action band + the severity rubric."""
import pytest

from src.prescriptive.engine import Prescriber

FRAME = 1920 * 1080  # reference frame area for size-normalisation


@pytest.fixture(scope="module")
def pres() -> Prescriber:
    return Prescriber.from_yaml()


# ---- one test per action band ----------------------------------------- #
def test_band_immediate(pres):
    p = pres.prescribe("missing_fastener", 0.90, 90 * 70, FRAME)
    assert p.band == "Immediate (within 24 h)"
    assert p.urgency_score >= 75


def test_band_schedule(pres):
    p = pres.prescribe("spalling", 0.65, 200 * 150, FRAME)
    assert p.band == "Schedule within 7 days"
    assert 50 <= p.urgency_score < 75


def test_band_routine(pres):
    p = pres.prescribe("corrugation", 0.55, 120 * 60, FRAME)
    assert p.band == "Routine (next maintenance cycle)"
    assert 25 <= p.urgency_score < 50


def test_band_monitor(pres):
    p = pres.prescribe("corrugation", 0.18, 30 * 20, FRAME)
    assert p.band == "Monitor"
    assert p.urgency_score < 25


# ---- severity rubric --------------------------------------------------- #
def test_fixed_severities(pres):
    assert pres.prescribe("missing_fastener", 0.9, 100, FRAME).severity == "High"
    assert pres.prescribe("broken_fastener", 0.6, 100, FRAME).severity == "High"
    assert pres.prescribe("spalling", 0.6, 100, FRAME).severity == "Medium"
    assert pres.prescribe("corrugation", 0.6, 100, FRAME).severity == "Low"


def test_crack_escalates_on_high_confidence(pres):
    assert pres.prescribe("crack", 0.82, 60 * 40, FRAME).severity == "High"


def test_crack_escalates_on_large_bbox(pres):
    # >=2% of the frame, even at lower confidence, should escalate to High
    assert pres.prescribe("crack", 0.50, int(0.03 * FRAME), FRAME).severity == "High"


def test_crack_stays_medium_when_small_and_uncertain(pres):
    assert pres.prescribe("crack", 0.48, 50 * 30, FRAME).severity == "Medium"


# ---- output contract --------------------------------------------------- #
def test_recommended_action_contains_band(pres):
    p = pres.prescribe("missing_fastener", 0.9, 90 * 70, FRAME)
    assert p.band in p.recommended_action


def test_urgency_is_clamped_0_100(pres):
    hi = pres.prescribe("missing_fastener", 1.0, FRAME, FRAME)  # max everything
    lo = pres.prescribe("corrugation", 0.0, 0, FRAME)
    assert 0 <= lo.urgency_score <= 100
    assert 0 <= hi.urgency_score <= 100
    assert isinstance(hi.urgency_score, int)
