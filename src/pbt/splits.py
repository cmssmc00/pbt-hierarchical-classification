"""Patient-disjoint outer folds with separate inner validation."""

from collections import defaultdict
import numpy as np


def validate_splits(records, payload):
    ids = {r.patient_id for r in records}
    folds = payload.get("folds", [])
    if len(folds) < 2:
        raise ValueError("At least two outer folds are required")
    tests = []
    for i, fold in enumerate(folds):
        if fold.get("fold") != i:
            raise ValueError("Fold indices must be consecutive starting at zero")
        parts = [fold.get(k, []) for k in ("train", "validation", "test")]
        if any(not p or not isinstance(p, list) for p in parts):
            raise ValueError("Each fold needs nonempty train, validation, and test lists")
        combined = sum(parts, [])
        if len(combined) != len(ids) or set(combined) != ids:
            raise ValueError("Each fold must partition all patients exactly once")
        tests.extend(parts[2])
    if len(tests) != len(ids) or set(tests) != ids:
        raise ValueError("Outer test folds must cover every patient exactly once")
    return folds


def make_splits(records, n_folds=5, validation_fraction=0.125, seed=101):
    if n_folds < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Invalid number of folds or validation fraction")
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r.label].append(r.patient_id)
    if None in by_class or set(by_class) != set(range(10)):
        raise ValueError("Split generation requires labeled patients from all ten source classes")
    outer = [[] for _ in range(n_folds)]
    for label, ids in sorted(by_class.items()):
        shuffled = list(rng.permutation(sorted(ids)))
        for j, pid in enumerate(shuffled):
            outer[(j + label) % n_folds].append(str(pid))
    folds = []
    for i, test in enumerate(outer):
        train, valid = [], []
        for label, ids in sorted(by_class.items()):
            remaining = sorted(set(ids) - set(test))
            if len(remaining) < 2:
                raise ValueError("Too few patients to retain each class in both train and validation")
            remaining = list(rng.permutation(remaining))
            n = min(len(remaining) - 1, max(1, round(len(remaining) * validation_fraction)))
            valid.extend(str(x) for x in remaining[:n])
            train.extend(str(x) for x in remaining[n:])
        folds.append({"fold": i, "train": sorted(train), "validation": sorted(valid), "test": sorted(test)})
    payload = {"schema_version": 1, "seed": seed, "folds": folds}
    validate_splits(records, payload)
    return payload
