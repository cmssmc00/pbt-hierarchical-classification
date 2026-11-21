"""Pure inference: train-only local IHC evidence and irreversible six-to-ten routing."""

import math

from pbt.taxonomy import GROUPS
from .root import softmax

MODES = ("wsi", "clinical", "ihc", "both")


def uncertain(p, config):
    ordered = sorted(p, reverse=True)
    rule = config["uncertainty"]
    return (
        ordered[0] < rule["max_probability_below"] or ordered[0] - ordered[1] < rule["top_two_margin_below"]
    )


def fit_local_ihc(train_ids, labels, sidecar, members, shrinkage):
    """Same shrinkage estimator as stage 1, pooled only within this parent."""
    n = [[0] * 5 for _ in members]
    positive = [[0] * 5 for _ in members]
    for cid in train_ids:
        local = members.index(labels[cid])
        for j, value in enumerate(sidecar[cid]["ihc"]):
            if value is not None:
                if value not in (0, 1):
                    raise ValueError("IHC must be explicit 0/1 or missing")
                n[local][j] += 1
                positive[local][j] += value
    pooled = [(1 + sum(row[j] for row in positive)) / (2 + sum(row[j] for row in n)) for j in range(5)]
    rates = [
        [(positive[k][j] + shrinkage * pooled[j]) / (n[k][j] + shrinkage) for j in range(5)]
        for k in range(len(members))
    ]
    return {
        "train_ids": list(train_ids),
        "members": list(members),
        "observed_counts": n,
        "positive_counts": positive,
        "global_rate": pooled,
        "rates": rates,
        "shrinkage": shrinkage,
    }


def local_ihc_scores(record, fitted, config):
    observed = [j for j, value in enumerate(record["ihc"]) if value is not None]
    if not observed:
        return [0.0] * len(fitted["members"]), []
    cap, total_cap = config["ihc_marker_log_ratio_cap"], config["ihc_total_log_ratio_cap"]
    scores = []
    for row in fitted["rates"]:
        total = 0.0
        for j in observed:
            positive = record["ihc"][j] == 1
            rate, pooled = row[j], fitted["global_rate"][j]
            value = math.log((rate if positive else 1 - rate) / (pooled if positive else 1 - pooled))
            total += max(-cap, min(cap, value))
        scores.append(max(-total_cap, min(total_cap, total / math.sqrt(len(observed)))))
    return scores, observed


def local_predict(logits, record, fitted, config, temperature, beta=0.0):
    """No diagnosis/true label enters this function. All three classes stay available."""
    calibrated = [float(x) / temperature for x in logits]
    image = softmax(calibrated)
    gate = uncertain(image, config)
    scores, observed = local_ihc_scores(record, fitted, config) if gate else ([0.0] * len(logits), [])
    adjustment = [beta * x for x in scores]
    probabilities = softmax([x + d for x, d in zip(calibrated, adjustment)]) if any(adjustment) else image
    return {
        "probabilities": probabilities,
        "predicted_local_index": max(range(len(logits)), key=lambda k: probabilities[k]),
        "image_probabilities": image,
        "uncertain": gate,
        "ihc_eligible": bool(observed),
        "ihc_applied": bool(beta and observed),
        "ihc_markers_used": observed if beta else [],
        "logit_adjustment": adjustment,
    }


def transform_local(raw, sidecar, fitted, config, temperature, beta=0.0):
    return [
        {**row, **local_predict(row["logits"], sidecar[row["case_id"]], fitted, config, temperature, beta)}
        for row in raw
    ]


def validation_score(rows):
    if not rows or any(r["local_label"] is None for r in rows):
        raise ValueError("Local selection requires only true-parent validation patients")
    return {
        "correct": sum(r["predicted_local_index"] == r["local_label"] for r in rows),
        "nll": -sum(math.log(max(r["probabilities"][r["local_label"]], 1e-30)) for r in rows) / len(rows),
    }


def select_local(validation, sidecar, fitted, config):
    temps = []
    for temperature in config["temperature_grid"]:
        score = validation_score(transform_local(validation, sidecar, fitted, config, temperature))
        temps.append({"temperature": temperature, "nll": score["nll"]})
    temperature = min(temps, key=lambda x: (x["nll"], abs(x["temperature"] - 1)))["temperature"]
    candidates = []
    for beta in config["ihc_strengths"]:
        score = validation_score(transform_local(validation, sidecar, fitted, config, temperature, beta))
        candidates.append({"beta": beta, **score})
    chosen = min(candidates, key=lambda x: (-x["correct"], x["nll"], x["beta"]))
    return {
        "partition": "inner_validation_true_parent_subset",
        "validation_ids": [r["case_id"] for r in validation],
        "temperature": temperature,
        "temperature_candidates": temps,
        "ihc_candidates": candidates,
        "chosen": chosen,
        "outer_test_used": False,
        "gate": config["uncertainty"],
    }


def hard_route(group_probabilities, child_predictions):
    """Choose the parent first; a child cannot override it, even if very confident."""
    if len(group_probabilities) != 6 or not math.isclose(sum(group_probabilities), 1.0, abs_tol=1e-6):
        raise ValueError("Require normalized six-group probabilities")
    parent = max(range(6), key=lambda g: group_probabilities[g])
    members = GROUPS[parent]
    child = child_predictions[parent] if len(members) > 1 else None
    conditional = child["probabilities"] if child else [1.0]
    if len(conditional) != len(members) or not math.isclose(sum(conditional), 1.0, abs_tol=1e-6):
        raise ValueError("Invalid conditional probabilities")
    local = max(range(len(members)), key=lambda k: conditional[k])
    hard = [0.0] * 10
    for member, p in zip(members, conditional):
        hard[member] = p
    return {
        "predicted_group_index": parent,
        "predicted_class_index": members[local],
        "hard_probabilities": hard,
        "conditional_probabilities": conditional,
        "stage2_used": child is not None,
        "stage2_uncertain": child["uncertain"] if child else False,
        "stage2_ihc_applied": child["ihc_applied"] if child else False,
        "stage2_ihc_markers_used": child["ihc_markers_used"] if child else [],
    }
