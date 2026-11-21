"""Training-only categorical evidence, used after frozen child image inference."""

import math
from itertools import product


def age_bin(age, config):
    return sum(float(age) > boundary for boundary in config["age_upper_bounds_years"])


def clip(x, cap):
    return max(-cap, min(cap, x))


def softmax(values):
    top = max(values)
    v = [math.exp(x - top) for x in values]
    return [x / sum(v) for x in v]


def fit_distribution(ids, labels, records, members, feature, config):
    k = 3 if feature == "age" else 10
    counts = [[0] * k for _ in members]
    for cid in ids:
        rec = records[cid]
        if not rec[feature + "_mask"]:
            continue
        value = age_bin(rec["age"], config) if feature == "age" else rec["location_index"]
        if not 0 <= value < k:
            raise ValueError("Invalid observed clinical category")
        counts[members.index(labels[cid])][value] += 1
    total = sum(map(sum, counts))
    pooled_counts = [sum(row[j] for row in counts) for j in range(k)]
    pooled = [(n + 1) / (total + k) for n in pooled_counts]
    shrinkage = config["shrinkage_pseudocount"]
    rates = [
        [(n + shrinkage * pooled[j]) / (sum(row) + shrinkage) for j, n in enumerate(row)] for row in counts
    ]
    return {
        "counts": counts,
        "observed_per_class": list(map(sum, counts)),
        "pooled_counts": pooled_counts,
        "pooled_rates": pooled,
        "rates": rates,
    }


def fit(ids, labels, records, members, parent, config):
    if set(ids) != set(labels) or set(ids) != set(records):
        raise ValueError("Fitting must receive exactly the training cases, no validation/test")
    return {
        "train_ids": list(ids),
        "members": list(members),
        "parent": parent,
        "age": fit_distribution(ids, labels, records, members, "age", config),
        "location": fit_distribution(ids, labels, records, members, "location", config),
    }


def evidence(record, fitted, config):
    scores = {"age": [0.0] * 3, "location": [0.0] * 3}
    used = {"age": False, "location": False}
    reasons = {}
    for feature in ["age", "location"]:
        if not record[feature + "_mask"]:
            reasons[feature] = "missing"
            continue
        value = age_bin(record["age"], config) if feature == "age" else record["location_index"]
        if feature == "location" and value not in config["location_indices"][f"G{fitted['parent'] + 1}"]:
            reasons[feature] = "outside_prespecified_sites"
            continue
        f = fitted[feature]
        minimum = (
            config["age_min_training_observations_in_bin"]
            if feature == "age"
            else config["location_min_training_observations_at_site"]
        )
        if f["pooled_counts"][value] < minimum:
            reasons[feature] = "insufficient_training_observations"
            continue
        raw = [math.log(row[value] / f["pooled_rates"][value]) for row in f["rates"]]
        if max(raw) - min(raw) < math.log(config["minimum_training_likelihood_contrast"]):
            reasons[feature] = "weak_training_contrast"
            continue
        scores[feature] = [clip(x, config[feature + "_score_cap"]) for x in raw]
        used[feature] = True
        reasons[feature] = "age_bin_" + str(value) if feature == "age" else "location_" + str(value)
    return scores, used, reasons


def predict(image_probabilities, baseline_probabilities, record, fitted, config, age_weight, location_weight):
    """Inputs intentionally contain no true diagnosis or patient identifier."""
    ordered = sorted(image_probabilities, reverse=True)
    gate = (
        ordered[0] < config["uncertainty"]["max_probability_below"]
        or ordered[0] - ordered[1] < config["uncertainty"]["top_two_margin_below"]
    )
    if gate:
        score, eligible, reasons = evidence(record, fitted, config)
    else:
        score = {"age": [0.0] * 3, "location": [0.0] * 3}
        eligible = {"age": False, "location": False}
        reasons = {"age": "confident_image", "location": "confident_image"}
    adjustment = [
        clip(age_weight * a + location_weight * b, config["combined_clinical_score_cap"])
        for a, b in zip(score["age"], score["location"])
    ]
    p = (
        softmax([math.log(max(x, 1e-300)) + d for x, d in zip(baseline_probabilities, adjustment)])
        if any(adjustment)
        else list(baseline_probabilities)
    )
    return {
        "probabilities": p,
        "predicted_local_index": max(range(3), key=lambda j: p[j]),
        "clinical_gate": gate,
        "age_eligible": eligible["age"],
        "location_eligible": eligible["location"],
        "age_applied": bool(age_weight and eligible["age"]),
        "location_applied": bool(location_weight and eligible["location"]),
        "age_scores": score["age"],
        "location_scores": score["location"],
        "clinical_adjustment": adjustment,
        "eligibility_reasons": reasons,
    }


def transform(rows, records, fitted, config, age_weight, location_weight):
    out = []
    for r in rows:
        decision = predict(
            r["image_probabilities"],
            r["probabilities"],
            records[r["case_id"]],
            fitted,
            config,
            age_weight,
            location_weight,
        )
        out.append(
            {
                **r,
                "baseline_probabilities": r["probabilities"],
                "baseline_predicted_local_index": r["predicted_local_index"],
                **decision,
            }
        )
    return out


def score(rows):
    if not rows or any(r["local_label"] is None for r in rows):
        raise ValueError("Selection needs true-parent validation cases only")
    return {
        "correct": sum(r["predicted_local_index"] == r["local_label"] for r in rows),
        "nll": -sum(math.log(max(r["probabilities"][r["local_label"]], 1e-300)) for r in rows) / len(rows),
    }


def select(validation, records, fitted, config, variant):
    ages = config["age_weight_grid"] if variant in ("age", "age_location") else [0.0]
    locations = config["location_weight_grid"] if variant in ("location", "age_location") else [0.0]
    candidates = []
    for a, l in product(ages, locations):
        candidates.append(
            {
                "age_weight": a,
                "location_weight": l,
                **score(transform(validation, records, fitted, config, a, l)),
            }
        )
    best = min(
        candidates,
        key=lambda x: (
            -x["correct"],
            x["nll"],
            x["age_weight"] + x["location_weight"],
            x["location_weight"],
            x["age_weight"],
        ),
    )
    return {
        "variant": variant,
        "validation_ids": [r["case_id"] for r in validation],
        "chosen": best,
        "candidates": candidates,
        "outer_test_used": False,
    }
