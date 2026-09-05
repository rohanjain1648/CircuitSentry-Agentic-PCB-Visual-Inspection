import numpy as np
import cv2
from eval.run_eval import evaluate


def _clean_pair():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (20, 20), (60, 60), (200, 200, 200), -1)
    cv2.circle(golden, (150, 150), 30, (50, 50, 50), -1)
    cv2.rectangle(golden, (20, 120), (70, 170), (0, 0, 0), -1)
    return golden.copy(), golden


def _defective_pair():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (20, 20), (60, 60), (200, 200, 200), -1)
    cv2.circle(golden, (150, 150), 30, (50, 50, 50), -1)
    cv2.rectangle(golden, (20, 120), (70, 170), (0, 0, 0), -1)
    frame = golden.copy()
    cv2.rectangle(frame, (120, 20), (170, 70), (255, 255, 255), -1)  # big, unambiguous defect
    return frame, golden


def test_evaluate_returns_metrics_dict():
    clean_frame, clean_golden = _clean_pair()
    defect_frame, defect_golden = _defective_pair()
    dataset = [
        {"frame": clean_frame, "golden_ref": clean_golden, "has_defect": False},
        {"frame": defect_frame, "golden_ref": defect_golden, "has_defect": True},
    ]
    result = evaluate(dataset)
    assert set(result.keys()) == {"precision", "recall", "f1", "escalation_rate"}
    assert result["recall"] == 1.0  # the injected defect must be detected
