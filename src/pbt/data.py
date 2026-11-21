"""Strict manifest loading and lazy HDF5 patient bags.

Patients, not slides, are the independent unit. Feature files for a patient are
pooled before patch sampling. Clinical evidence stays outside neural training
unless a historical configuration explicitly enables the direct clinical prior.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .io import read_json
from .taxonomy import IHC_NAMES, LOCATION_NAMES, class_index


@dataclass(frozen=True)
class Patient:
    patient_id: str
    label: int | None
    features: tuple[Path, ...]
    evidence: dict


def load_manifest(path, feature_root=None, require_labels=True):
    path = Path(path).resolve()
    root = Path(feature_root).resolve() if feature_root else path.parent
    payload = read_json(path)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("patients"), list):
        raise ValueError("Expected schema_version=1 and a patients list")
    records, identifiers, files = [], set(), set()
    for row in payload["patients"]:
        pid = row.get("patient_id")
        if not isinstance(pid, str) or not pid.strip() or pid in identifiers:
            raise ValueError("Each patient_id must be a unique nonempty string")
        identifiers.add(pid)
        label = class_index(row["label"]) if row.get("label") is not None else None
        if require_labels and label is None:
            raise ValueError("Training/evaluation manifests require labels")
        relative = row.get("features")
        if not isinstance(relative, list) or not relative:
            raise ValueError("Every patient requires at least one feature path")
        paths = []
        for item in relative:
            if not isinstance(item, str) or Path(item).is_absolute():
                raise ValueError("Feature paths must be relative to the feature root")
            p = (root / item).resolve()
            if not p.is_relative_to(root) or not p.is_file() or p.suffix.lower() not in (".h5", ".hdf5"):
                raise ValueError("Feature file missing, outside feature root, or not HDF5")
            if p in files:
                raise ValueError("A feature file cannot be repeated or shared between patients")
            files.add(p)
            paths.append(p)
        age = row.get("age_years")
        if age is not None and (
            isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0
        ):
            raise ValueError("Age must be a nonnegative finite number or null")
        location = row.get("location")
        if location is not None and location not in LOCATION_NAMES:
            raise ValueError("Location must match the documented vocabulary or be null")
        location_index = LOCATION_NAMES.index(location) if location is not None else 10
        ihc = row.get("ihc", {})
        if not isinstance(ihc, dict) or set(ihc) - set(IHC_NAMES):
            raise ValueError("Unknown IHC field; use the documented five-marker vocabulary")
        calls = []
        for marker in IHC_NAMES:
            value = ihc.get(marker)
            if value not in (None, "POS", "NEG"):
                raise ValueError(
                    "IHC values must be explicit POS, NEG, or null; inferred calls are not accepted"
                )
            calls.append(None if value is None else int(value == "POS"))
        evidence = {
            "age": 0.0 if age is None else float(age),
            "age_mask": age is not None,
            "location_index": location_index,
            "location_mask": location_index != 10,
            "ihc": calls,
        }
        records.append(Patient(pid, label, tuple(paths), evidence))
    if not records:
        raise ValueError("The manifest contains no patients")
    return records


def load_bag(record, feature_dim, feature_key="features", max_patches=None, rng=None):
    import h5py

    with ExitStack() as stack:
        datasets = []
        for p in record.features:
            f = stack.enter_context(h5py.File(p, "r"))
            if feature_key not in f:
                raise ValueError("HDF5 file is missing the configured feature dataset")
            d = f[feature_key]
            if len(d.shape) != 2 or d.shape[1] != feature_dim:
                raise ValueError(f"Expected an [N, {feature_dim}] feature matrix")
            datasets.append(d)
        total = sum(len(d) for d in datasets)
        if total == 0:
            raise ValueError("Empty patient bag")
        selected = None
        if max_patches is not None and total > max_patches:
            if rng is None or max_patches <= 0:
                raise ValueError("Patch sampling requires a positive cap and an explicit RNG")
            selected = np.sort(rng.choice(total, max_patches, replace=False))
        arrays, offset = [], 0
        for d in datasets:
            if selected is None:
                a = np.asarray(d[...], dtype=np.float32)
            else:
                indices = selected[(selected >= offset) & (selected < offset + len(d))] - offset
                a = np.asarray(d[indices], dtype=np.float32) if len(indices) else None
            offset += len(d)
            if a is not None and len(a):
                if not np.isfinite(a).all():
                    raise ValueError("Nonfinite patch features")
                arrays.append(a)
        return np.concatenate(arrays, axis=0)


def make_dataset(records, config, training, label_map, include_clinical=False):
    import tensorflow as tf

    records = list(records)
    feature_dim = config["data"]["feature_dim"]
    rng = np.random.default_rng(config["training"]["seed"])

    def generator():
        order = np.arange(len(records))
        if training:
            rng.shuffle(order)
        for i in order:
            rec = records[int(i)]
            patches = load_bag(
                rec,
                feature_dim,
                config["data"]["feature_key"],
                config["data"]["max_train_patches"] if training else None,
                rng,
            )
            inputs = {"patches": patches, "patch_mask": np.ones(len(patches), dtype=bool)}
            if include_clinical:
                for key, dtype in (
                    ("age", np.float32),
                    ("age_mask", np.float32),
                    ("location_mask", np.float32),
                ):
                    inputs[key] = np.array([rec.evidence[key]], dtype=dtype)
                inputs["location"] = np.array([rec.evidence["location_index"]], dtype=np.int32)
            yield inputs, np.int32(label_map.get(rec.label, 0))

    signature = {
        "patches": tf.TensorSpec((None, feature_dim), tf.float32),
        "patch_mask": tf.TensorSpec((None,), tf.bool),
    }
    if include_clinical:
        for key in ("age", "age_mask", "location_mask"):
            signature[key] = tf.TensorSpec((1,), tf.float32)
        signature["location"] = tf.TensorSpec((1,), tf.int32)
    ds = tf.data.Dataset.from_generator(generator, output_signature=(signature, tf.TensorSpec((), tf.int32)))
    batch = config["training"]["batch_size"] if training else config["training"]["test_batch_size"]
    shapes = {k: v.shape for k, v in signature.items()}
    ds = ds.padded_batch(batch, padded_shapes=(shapes, ()))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    return ds.with_options(options).prefetch(1)
