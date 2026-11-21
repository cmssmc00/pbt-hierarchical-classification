"""Training-only marker likelihoods and gated, literature-motivated clinical cues.

Inference functions accept logits and an evidence record, never a true label.
Missing marker calls contribute exactly zero. No modality trains a second Teacher.
"""

from __future__ import annotations
import math
from itertools import product

from pbt.taxonomy import GROUP_NAMES
from pbt.metrics import classification_metrics


def softmax(logits):
    top = max(logits)
    values = [math.exp(x - top) for x in logits]
    return [x / sum(values) for x in values]


def uncertainty(probabilities, config):
    ranked = sorted(range(6), key=lambda g: (-probabilities[g], g))
    return (
        probabilities[ranked[0]] < config["max_probability_below"]
        or probabilities[ranked[0]] - probabilities[ranked[1]] < config["top_two_margin_below"]
    )


def clinical_cues(record, probabilities):
    """Positive hints only, restricted to the image's two leading candidates.

    The age <=3 ventricular hint is an experimental cutoff motivated by the
    young age distribution in CPT-SIOP-2000, not a diagnostic decision rule.
    """
    boosts = [0.0] * 6
    rules = []
    if not record["location_mask"]:
        return boosts, rules
    location = record["location_index"]
    if location in (3, 4):
        boosts[0] = 1.0
        rules.append("brainstem_thalamus_glial_family")
    elif location == 7:
        boosts[1] = boosts[4] = 1.0
        rules.append("pineal_embryonal_or_germ_cell_candidates")
    elif location == 8:
        boosts[0] = boosts[4] = 1.0
        rules.append("sellar_glial_or_non_glial_candidates")
    elif location == 2:
        boosts[2] = 0.5
        rules.append("ventricular_ependymal_candidate")
        if record["age_mask"] and record["age"] <= 3.0:
            boosts[3] = 1.0
            rules.append("young_ventricular_choroid_plexus_candidate")
    shortlist = set(sorted(range(6), key=lambda g: (-probabilities[g], g))[:2])
    boosts = [b if g in shortlist else 0.0 for g, b in enumerate(boosts)]
    if not any(boosts):
        rules = []
    return boosts, rules


def fit_ihc(train_ids, labels, sidecar, shrinkage):
    """Smoothed Bernoulli likelihoods from explicit training POS/NEG only."""
    n = [[0] * 5 for _ in range(6)]
    positive = [[0] * 5 for _ in range(6)]
    for cid in train_ids:
        g = labels[cid]
        for j, value in enumerate(sidecar[cid]["ihc"]):
            if value is not None:
                if value not in (0, 1):
                    raise ValueError("Marker calls must be 0, 1 or missing")
                n[g][j] += 1
                positive[g][j] += value
    global_rate = [
        (1 + sum(positive[g][j] for g in range(6))) / (2 + sum(n[g][j] for g in range(6))) for j in range(5)
    ]
    rate = [
        [(positive[g][j] + shrinkage * global_rate[j]) / (n[g][j] + shrinkage) for j in range(5)]
        for g in range(6)
    ]
    return {
        "train_ids": list(train_ids),
        "observed_counts": n,
        "positive_counts": positive,
        "global_rate": global_rate,
        "rates": rate,
        "shrinkage": shrinkage,
    }


def ihc_cues(record, fitted, config):
    observed = [j for j, value in enumerate(record["ihc"]) if value is not None]
    if not observed:
        return [0.0] * 6, []
    marker_cap = config["ihc_marker_log_ratio_cap"]
    total_cap = config["ihc_total_log_ratio_cap"]
    values = []
    for g in range(6):
        total = 0.0
        for j in observed:
            positive = record["ihc"][j] == 1
            rate, global_rate = fitted["rates"][g][j], fitted["global_rate"][j]
            value = math.log(
                (rate if positive else 1 - rate) / (global_rate if positive else 1 - global_rate)
            )
            total += max(-marker_cap, min(marker_cap, value))
        # Temper correlated markers rather than assuming five independent tests.
        values.append(max(-total_cap, min(total_cap, total / math.sqrt(len(observed)))))
    return values, observed


def apply(logits, record, fitted, config, temperature, alpha=0.0, beta=0.0):
    base_logits = [x / temperature for x in logits]
    base = softmax(base_logits)
    gate = uncertainty(base, config["uncertainty"])
    clinical, rules = clinical_cues(record, base) if gate else ([0.0] * 6, [])
    ihc, markers = ihc_cues(record, fitted, config) if gate else ([0.0] * 6, [])
    delta = [alpha * a + beta * b for a, b in zip(clinical, ihc)]
    # Exact identity is useful for auditing disabled or unavailable evidence.
    probs = softmax([x + d for x, d in zip(base_logits, delta)]) if any(delta) else base
    pred = max(range(6), key=lambda g: probs[g])
    return {
        "probabilities": probs,
        "predicted_group_index": pred,
        "image_probabilities": base,
        "image_predicted_group_index": max(range(6), key=lambda g: base[g]),
        "uncertain": gate,
        "clinical_eligible": bool(rules),
        "ihc_eligible": bool(markers),
        "clinical_applied": bool(alpha and rules),
        "ihc_applied": bool(beta and markers),
        "clinical_rules": rules if alpha else [],
        "ihc_markers_used": markers if beta else [],
        "logit_adjustment": delta,
    }


def evaluate(rows):
    labels = [r["true_group_index"] for r in rows]
    pred = [r["predicted_group_index"] for r in rows]
    return {
        "metrics": classification_metrics(labels, pred, GROUP_NAMES),
        "nll": -sum(math.log(max(r["probabilities"][r["true_group_index"]], 1e-30)) for r in rows)
        / len(rows),
    }


def transform(raw, sidecar, fitted, config, temperature, alpha=0.0, beta=0.0):
    rows = []
    for r in raw:
        prediction = apply(r["logits"], sidecar[r["case_id"]], fitted, config, temperature, alpha, beta)
        rows.append(
            {**r, **prediction, "correct": prediction["predicted_group_index"] == r["true_group_index"]}
        )
    return rows


def select_settings(validation, sidecar, fitted, config):
    if not validation:
        raise ValueError("Inner-validation cases are required for selection")
    temperatures = []
    for t in config["temperature_grid"]:
        rows = transform(validation, sidecar, fitted, config, t)
        score = evaluate(rows)
        temperatures.append({"temperature": t, "nll": score["nll"]})
    temperature = min(temperatures, key=lambda x: (x["nll"], abs(x["temperature"] - 1)))["temperature"]
    settings = {}
    for mode in config["modalities"]:
        alphas = config["clinical_strengths"] if mode in ("clinical", "both") else [0.0]
        betas = config["ihc_strengths"] if mode in ("ihc", "both") else [0.0]
        candidates = []
        for alpha, beta in product(alphas, betas):
            score = evaluate(transform(validation, sidecar, fitted, config, temperature, alpha, beta))
            candidates.append(
                {"alpha": alpha, "beta": beta, "correct": score["metrics"]["correct"], "nll": score["nll"]}
            )
        winner = min(
            candidates, key=lambda x: (-x["correct"], x["nll"], x["alpha"] + x["beta"], x["alpha"], x["beta"])
        )
        settings[mode] = {"chosen": winner, "candidates": candidates}
    return {
        "partition": "inner_validation",
        "validation_ids": [r["case_id"] for r in validation],
        "temperature": temperature,
        "temperature_candidates": temperatures,
        "modes": settings,
        "outer_test_used": False,
        "gate": config["uncertainty"],
    }
