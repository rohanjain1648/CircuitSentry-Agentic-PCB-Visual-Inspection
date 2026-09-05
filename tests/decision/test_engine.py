from circuitsentry.vision.score import Region
from circuitsentry.decision.engine import decide, RegionAction, HIGH_CONFIDENCE, LOW_CONFIDENCE, MAX_RECAPTURE_ATTEMPTS


def _region(confidence):
    return Region(bbox=(10, 10, 20, 20), confidence=confidence, defect_type_guess="missing_or_extra_component")


def test_high_confidence_flags_for_approval():
    [result] = decide([_region(HIGH_CONFIDENCE + 0.01)], attempt_counts={})
    assert isinstance(result, RegionAction)
    assert result.action == "flag_for_approval"


def test_low_confidence_passes():
    [result] = decide([_region(LOW_CONFIDENCE - 0.01)], attempt_counts={})
    assert result.action == "pass"


def test_ambiguous_confidence_requests_recapture():
    mid = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2
    [result] = decide([_region(mid)], attempt_counts={})
    assert result.action == "recapture"


def test_ambiguous_confidence_flags_after_max_attempts():
    mid = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2
    region = _region(mid)
    attempt_counts = {region.bbox: MAX_RECAPTURE_ATTEMPTS}
    [result] = decide([region], attempt_counts=attempt_counts)
    assert result.action == "flag_for_approval"
