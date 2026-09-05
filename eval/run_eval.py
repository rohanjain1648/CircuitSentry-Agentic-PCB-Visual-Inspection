from circuitsentry.vision.pipeline import inspect
from circuitsentry.decision.engine import HIGH_CONFIDENCE, LOW_CONFIDENCE


def evaluate(dataset: list[dict]) -> dict:
    """Run inspect() over a labeled dataset and compute detection metrics.

    Each dataset entry: {"frame": np.ndarray, "golden_ref": np.ndarray, "has_defect": bool}
    """
    true_positives = false_positives = false_negatives = 0
    escalated_count = 0

    for entry in dataset:
        regions = inspect(entry["frame"], entry["golden_ref"])
        detected = any(r.confidence >= HIGH_CONFIDENCE for r in regions)
        escalated = any(LOW_CONFIDENCE <= r.confidence < HIGH_CONFIDENCE for r in regions)

        if escalated:
            escalated_count += 1
        if detected and entry["has_defect"]:
            true_positives += 1
        elif detected and not entry["has_defect"]:
            false_positives += 1
        elif not detected and entry["has_defect"]:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    escalation_rate = escalated_count / len(dataset) if dataset else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "escalation_rate": round(escalation_rate, 3),
    }
