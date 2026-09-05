from dataclasses import dataclass
import cv2
import numpy as np

# Confidence saturates once a candidate region reaches this pixel area.
AREA_SATURATION_PX = 2000


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float                # 0.0-1.0
    defect_type_guess: str


def _guess_defect_type(w: int, h: int) -> str:
    aspect = max(w, h) / max(1, min(w, h))
    if aspect > 4:
        return "solder_bridge"
    if aspect > 1.8:
        return "scratch_or_burn"
    return "missing_or_extra_component"


def score_regions(mask: np.ndarray) -> list[Region]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        confidence = min(1.0, area / AREA_SATURATION_PX)
        regions.append(Region(
            bbox=(x, y, w, h),
            confidence=round(confidence, 3),
            defect_type_guess=_guess_defect_type(w, h),
        ))
    return regions
