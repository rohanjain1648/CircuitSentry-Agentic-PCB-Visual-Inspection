from dataclasses import dataclass
from circuitsentry.vision.score import Region

HIGH_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.3
MAX_RECAPTURE_ATTEMPTS = 2  # spec §4/§8: bounded loop


@dataclass(frozen=True)
class RegionAction:
    region: Region
    action: str  # "pass" | "recapture" | "flag_for_approval"


def decide(
    regions: list[Region],
    attempt_counts: dict[str, int],
    region_ids: list[str] | None = None,
) -> list[RegionAction]:
    """Map each Region to its next action given how many recapture attempts it has already had.

    ``attempt_counts`` is keyed by *region id*, not by bbox. A real recapture
    uploads a cropped+zoomed image, so the same physical defect lands at a
    completely different bbox on the next iteration; keying attempts by bbox
    would silently reset the count and defeat the ≤2-attempt bound from
    spec §4/§8. Region ids are assigned by detection order within a run
    (see ``lambda_perception.handler``).

    Simplification: ordering is a stable identity only while the region list
    stays comparable across iterations (the single/few-region case this
    project targets). Multi-region recapture across wildly different frame
    compositions is not perfectly identity-tracked — a full solution would
    remap coordinates through the recapture homography.

    ``region_ids`` defaults to the positional index of each region, stringified.
    """
    ids = region_ids if region_ids is not None else [str(i) for i in range(len(regions))]
    results = []
    for region, region_id in zip(regions, ids):
        attempts = attempt_counts.get(region_id, 0)
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
