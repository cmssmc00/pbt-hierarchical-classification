"""Generate fictional records and random embeddings for software checks only."""

from pathlib import Path
import numpy as np

from .io import write_json
from .taxonomy import IHC_NAMES, LOCATION_NAMES


def generate(destination, feature_dim=8, per_class=6, seed=731):
    import h5py

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    features = destination / "features"
    features.mkdir()
    rng = np.random.default_rng(seed)
    records = []
    centers = rng.normal(size=(10, feature_dim))
    for label in range(10):
        for j in range(per_class):
            index = len(records)
            paths = []
            for slide in range(2 if j == 0 else 1):
                relative = f"features/synthetic_slide_{index:04d}_{slide}.h5"
                patches = centers[label] + rng.normal(size=(int(rng.integers(6, 15)), feature_dim))
                with h5py.File(destination / relative, "w") as f:
                    f.create_dataset("features", data=patches.astype(np.float32))
                paths.append(relative)
            records.append(
                {
                    "patient_id": f"synthetic_patient_{index:04d}",
                    "label": f"C{label + 1}",
                    "features": paths,
                    "age_years": None if j == 0 else float(rng.integers(0, 19)),
                    "location": None if j == 0 else LOCATION_NAMES[int(rng.integers(0, 10))],
                    "ihc": {marker: [None, "POS", "NEG"][int(rng.integers(0, 3))] for marker in IHC_NAMES},
                }
            )
    write_json(
        {
            "schema_version": 1,
            "synthetic": True,
            "description": "Fictional software-test data, not clinical observations",
            "patients": records,
        },
        destination / "manifest.json",
    )
    return destination / "manifest.json"
