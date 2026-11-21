"""Prediction from private saved weights; true labels are optional and unused."""

from pathlib import Path

from .data import load_manifest
from .io import read_json, write_json
from .pipeline import fuse, root_spec, validate_config
from .taxonomy import GROUPS
from .training import build_model, predict


def predict_saved(run_directory, manifest, output, fold=0, feature_root=None):
    run_directory, output = Path(run_directory), Path(output)
    if output.exists():
        raise FileExistsError("Prediction output already exists")
    config = validate_config(read_json(run_directory / "config.json"))
    directory = run_directory / f"fold_{fold}"
    if read_json(directory / "status.json")["status"] != "complete":
        raise ValueError("The requested fold is not complete")
    records = load_manifest(manifest, feature_root, require_labels=False)
    side = {r.patient_id: r.evidence for r in records}
    kind, width, mapping, _ = root_spec(config["task"])
    root = build_model(config, kind, width)
    location = "refit" if config["training"]["selection"] == "epoch_selection" else "root"
    root.load_weights(directory / location / "selected.weights.h5")
    raw = predict(root, records, config, mapping, kind)
    if config["task"] in ("flat6", "hard6to10"):
        children = {}
        if config["task"] == "hard6to10":
            for parent, members in enumerate(GROUPS):
                if len(members) == 1:
                    continue
                child = build_model(config, "flat", len(members))
                child.load_weights(directory / f"child_{parent}" / "selected.weights.h5")
                children[parent] = predict(
                    child, records, config, {label: j for j, label in enumerate(members)}
                )
        modes = fuse(raw, children, side, read_json(directory / "selection.json"), config)
    else:
        modes = {"model": raw}
    result = {
        mode: [
            {
                "patient_id": r["case_id"],
                "predicted_index": r["predicted_index"],
                "probabilities": r["probabilities"],
            }
            for r in rows
        ]
        for mode, rows in modes.items()
    }
    write_json({"task": config["task"], "fold": fold, "modes": result}, output)
    return result
