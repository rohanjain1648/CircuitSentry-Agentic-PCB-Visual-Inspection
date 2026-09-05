from dataclasses import dataclass
from circuitsentry.vision.score import Region

HIGH_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.3
MAX_RECAPTURE_ATTEMPTS = 2  # spec §4/§8: bounded loop


@dataclass(frozen=True)
class RegionAction:
    region: Region
    action: str  # "pass" | "recapture" | "flag_for_approval"


def decide(regions: list[Region], attempt_counts: dict[tuple, int]) -> list[RegionAction]:
    """Map each Region to its next action given how many recapture attempts it has already had."""
    results = []
    for region in regions:
        attempts = attempt_counts.get(region.bbox, 0)
        if region.confidence >= HIGH_CONFIDENCE:
            action = "flag_for_approval"
        elif region.confidence < LOW_CONFIDENCE:
            action = "pass"
        elif attempts >= MAX_RECAPTURE_ATTEMPTS:
            action = "flag_for_approval"
        else:
            action = "recapture"
        results.append(RegionAction(region=region, action=action))
    return results
