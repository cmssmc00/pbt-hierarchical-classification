"""Three-group taxonomy and decoders for the joint B/C/D experiments."""

import math
from pbt.taxonomy import CLASS_NAMES

GROUPS = ((0, 1, 2), (3, 5, 7), (4, 6, 8, 9))
CLASS_TO_GROUP = tuple(next(g for g, m in enumerate(GROUPS) if c in m) for c in range(10))
CLASS_TO_LOCAL = tuple(GROUPS[CLASS_TO_GROUP[c]].index(c) for c in range(10))


def _probabilities(values, size):
    values = [float(v) for v in values]
    if len(values) != size or any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
        raise ValueError("Invalid probability vector")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Probability vector must sum to one")
    return values


def decode_hard_predictions(coarse_probabilities, conditional_probabilities):
    """Root argmax, then argmax INSIDE that group. Other groups stay zero.

    conditional_probabilities contains three batches with widths 3, 3, and 4.
    Ties consistently select the first class in the fixed taxonomy order.
    Root probabilities never multiply child scores for cross-group selection.
    """
    n = len(coarse_probabilities)
    if len(conditional_probabilities) != 3 or any(len(batch) != n for batch in conditional_probabilities):
        raise ValueError("Expected three conditional batches matching the root batch")
    results = []
    for i, row in enumerate(coarse_probabilities):
        coarse = _probabilities(row, 3)
        children = [_probabilities(conditional_probabilities[g][i], len(GROUPS[g])) for g in range(3)]
        group = max(range(3), key=lambda g: coarse[g])
        local = max(range(len(GROUPS[group])), key=lambda j: children[group][j])
        predicted = GROUPS[group][local]
        routed = [0.0] * 10
        for cls, probability in zip(GROUPS[group], children[group]):
            routed[cls] = probability
        results.append(
            {
                "predicted_group_index": group,
                "predicted_class_index": predicted,
                "predicted_local_index": local,
                "coarse_probabilities": coarse,
                "conditional_probabilities": children,
                "routed_probabilities": routed,
            }
        )
    return results


def decode_predictions(coarse_probabilities, conditional_probabilities, global_probabilities):
    """Final prediction is ONLY global argmax; auxiliary routing is diagnostic.

    All ten global probabilities are preserved, including groups that the
    coarse classifier did not select. This function never sees true labels.
    """
    hard = decode_hard_predictions(coarse_probabilities, conditional_probabilities)
    if len(global_probabilities) != len(hard):
        raise ValueError("Global probability batch must match auxiliary batches")
    results = []
    for probabilities, old in zip(global_probabilities, hard):
        probabilities = _probabilities(probabilities, len(CLASS_NAMES))
        predicted = max(range(len(CLASS_NAMES)), key=lambda c: probabilities[c])
        results.append(
            {
                "predicted_class_index": predicted,
                "predicted_group_index": CLASS_TO_GROUP[predicted],
                "predicted_local_index": CLASS_TO_LOCAL[predicted],
                "global_probabilities": probabilities,
                "coarse_probabilities": old["coarse_probabilities"],
                "conditional_probabilities": old["conditional_probabilities"],
                "coarse_predicted_group_index": old["predicted_group_index"],
                "hard_predicted_class_index": old["predicted_class_index"],
                "hard_predicted_local_index": old["predicted_local_index"],
                "hard_routed_probabilities": old["routed_probabilities"],
                "crossed_group": CLASS_TO_GROUP[predicted] != old["predicted_group_index"],
            }
        )
    return results


def balanced_path_weights(labels):
    """Training-only global class weights, root weights and within-group weights."""
    counts = [0] * 10
    for label in labels:
        if type(label) is not int or not 0 <= label < 10:
            raise ValueError("Invalid ten-class training target")
        counts[label] += 1
    if any(n == 0 for n in counts):
        raise ValueError("All ten leaf classes must occur in the training partition")
    group_counts = [sum(counts[c] for c in members) for members in GROUPS]
    coarse = [sum(counts) / (3 * n) for n in group_counts]
    fine = [group_counts[CLASS_TO_GROUP[c]] / (len(GROUPS[CLASS_TO_GROUP[c]]) * counts[c]) for c in range(10)]
    global_weights = [sum(counts) / (len(CLASS_NAMES) * count) for count in counts]
    return {
        "global": global_weights,
        "coarse": coarse,
        "fine": fine,
        "class_counts": counts,
        "group_counts": group_counts,
    }
