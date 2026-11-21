"""Metrics use a fixed label order, including classes absent from a test fold."""

import math


def prediction_metrics(rows, names, target_key="target_index", prediction_key="predicted_index"):
    """Report true infinite NLL for a hard routing error, as JSON-safe null."""
    result = classification_metrics([r[target_key] for r in rows], [r[prediction_key] for r in rows], names)
    probabilities = [r["probabilities"][r[target_key]] for r in rows]
    zeros = sum(p == 0 for p in probabilities)
    result["nll"] = None if zeros else -sum(math.log(p) for p in probabilities) / len(rows)
    result["nll_status"] = "infinite" if zeros else "finite"
    result["zero_true_probability_count"] = zeros
    return result


def classification_metrics(labels, predictions, names):
    """Fixed-label metrics; absent classes remain visible with zero support."""
    labels, predictions = list(labels), list(predictions)
    k = len(names)
    if not labels or len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have equal nonzero length")
    matrix = [[0] * k for _ in range(k)]
    for truth, prediction in zip(labels, predictions):
        if (
            type(truth) is not int
            or type(prediction) is not int
            or not 0 <= truth < k
            or not 0 <= prediction < k
        ):
            raise ValueError("Class index outside the fixed taxonomy")
        matrix[truth][prediction] += 1
    per_class = []
    for c, name in enumerate(names):
        tp = matrix[c][c]
        support = sum(matrix[c])
        predicted = sum(row[c] for row in matrix)
        per_class.append(
            {
                "class_index": c,
                "class_name": name,
                "support": support,
                "predicted_count": predicted,
                "correct": tp,
                "precision": tp / predicted if predicted else 0.0,
                "recall": tp / support if support else 0.0,
                "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
            }
        )
    present = [row for row in per_class if row["support"]]
    correct = sum(matrix[i][i] for i in range(k))
    return {
        "num_patients": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels),
        "balanced_accuracy": sum(row["recall"] for row in present) / len(present),
        "macro_f1": sum(row["f1"] for row in per_class) / k,
        "macro_f1_present_classes": sum(row["f1"] for row in present) / len(present),
        "num_supported_classes": len(present),
        "macro_f1_definition": "mean over all fixed classes; absent classes have F1=0",
        "balanced_accuracy_definition": "mean recall over classes with true support",
        "confusion_matrix": matrix,
        "confusion_matrix_order": list(names),
        "per_class": per_class,
    }
