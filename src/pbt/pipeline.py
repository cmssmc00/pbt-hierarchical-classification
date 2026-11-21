"""Train, select on inner validation, then evaluate held-out patients.

The four evidence settings reuse the same image networks. All folds and all
nodes train serially. Every fitted evidence object is private run output.
"""

from pathlib import Path
import math

from .data import load_manifest
from .evidence import root as root_evidence
from .evidence import hierarchy as local_evidence
from .evidence import clinical
from .io import now, read_json, write_json, sha256
from .metrics import prediction_metrics
from .splits import validate_splits
from .taxonomy import CLASS_NAMES, CLASS_TO_GROUP, GROUPS, GROUP_NAMES
from .training import fit, predict, refit


def validate_config(c):
    tasks = ("binary", "flat6", "flat10", "hard6to10", "joint_hard", "joint_global")
    if c.get("task") not in tasks:
        raise ValueError(f"Task must be one of {tasks}")
    t, m, d = c["training"], c["method"], c["data"]
    for key in ("epochs", "batch_size", "test_batch_size", "schedule_epochs", "patience"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(d["feature_dim"]) is not int or d["feature_dim"] < 1:
        raise ValueError("feature_dim must be positive")
    if d["max_train_patches"] is not None and (
        type(d["max_train_patches"]) is not int or d["max_train_patches"] < 1
    ):
        raise ValueError("max_train_patches must be a positive integer or null")
    if not 0 < t["minimum_learning_rate"] <= t["learning_rate"]:
        raise ValueError("Invalid learning rate range")
    if t["selection"] not in ("fixed", "early_stopping", "epoch_selection"):
        raise ValueError("Unknown training selection")
    if t["selection"] == "epoch_selection":
        if c["task"] != "joint_global" or not t["candidate_epochs"]:
            raise ValueError("Candidate epoch selection is supported for joint_global")
        if any(type(e) is not int or not 1 <= e <= t["epochs"] for e in t["candidate_epochs"]):
            raise ValueError("Candidate epochs must fall within the training budget")
    if c["task"] == "joint_hard" and t["selection"] != "fixed":
        raise ValueError("joint_hard uses fixed epochs")
    if not 0 <= m["label_smoothing"] < 1 or any(
        m[k] < 0 for k in ("weight_decay", "focal_gamma", "weight_power")
    ):
        raise ValueError("Invalid loss/optimizer settings")
    if c["task"].startswith("joint") and (
        m["label_smoothing"] or m["focal_gamma"] or m["weight_decay"] or m["weight_power"] != 1
    ):
        raise ValueError("Joint B/C/D presets retain the original loss/optimizer configuration")
    if c["task"] in ("flat6", "hard6to10") and c["model"]["use_clinical"]:
        raise ValueError("Evidence experiments train on image features only")
    if c["task"] in ("flat6", "hard6to10"):
        e = c["evidence"]
        if e["modalities"] != ["wsi", "clinical", "ihc", "both"]:
            raise ValueError("Evidence runs require all four prespecified modes")
        for key in ("clinical_strengths", "ihc_strengths"):
            if 0 not in e[key] or any(not math.isfinite(v) or v < 0 for v in e[key]):
                raise ValueError("Evidence strength grids must include zero and be nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in e["temperature_grid"]) or e["ihc_shrinkage"] <= 0:
            raise ValueError("Temperature and shrinkage must be positive")
    return c


def root_spec(task):
    if task in ("flat6", "hard6to10"):
        return "flat", 6, dict(enumerate(CLASS_TO_GROUP)), GROUP_NAMES
    if task == "binary":
        return "flat", 2, {i: int(i == 9) for i in range(10)}, ("Tumor", "Non-tumor")
    return (task if task.startswith("joint") else "flat"), 10, {i: i for i in range(10)}, CLASS_NAMES


def select_root(train, validation, side, config):
    labels = {r.patient_id: CLASS_TO_GROUP[r.label] for r in train}
    fitted = root_evidence.fit_ihc(list(labels), labels, side, config["evidence"]["ihc_shrinkage"])
    valid = [{**r, "true_group_index": r["target_index"]} for r in validation]
    selected = root_evidence.select_settings(valid, side, fitted, config["evidence"])
    return {"ihc": fitted, "selection": selected}


def select_child(train, validation, side, parent, config):
    members = GROUPS[parent]
    labels = {r.patient_id: r.label for r in train}
    local = [{**r, "local_label": r["target_index"]} for r in validation]
    evidence_config = config["evidence"]
    fitted = local_evidence.fit_local_ihc(
        list(labels), labels, side, members, evidence_config["ihc_shrinkage"]
    )
    selected = local_evidence.select_local(local, side, fitted, evidence_config)
    state = {"ihc": fitted, "selection": selected}
    if config["stage2_clinical"]["enabled"]:
        clinical_config = config["stage2_clinical"]
        train_side = {cid: side[cid] for cid in labels}
        clinical_fit = clinical.fit(list(labels), labels, train_side, members, parent, clinical_config)
        state["clinical"] = clinical_fit
        state["clinical_selection"] = {}
        for mode in ("clinical", "both"):
            beta = selected["chosen"]["beta"] if mode == "both" else 0.0
            baseline = local_evidence.transform_local(
                local, side, fitted, evidence_config, selected["temperature"], beta
            )
            state["clinical_selection"][mode] = clinical.select(
                baseline, side, clinical_fit, clinical_config, "age_location"
            )
    return state


def fuse(raw_root, raw_children, side, states, config):
    """Apply frozen parameters; no true label is read to make a prediction."""
    modes = {}
    rstate = states["root"]
    root_selection = rstate["selection"]
    child_by_id = {g: {r["case_id"]: r for r in rows} for g, rows in raw_children.items()}
    for mode in config["evidence"]["modalities"]:
        chosen = root_selection["modes"][mode]["chosen"]
        result = []
        for row in raw_root:
            pid = row["case_id"]
            root = root_evidence.apply(
                row["logits"],
                side[pid],
                rstate["ihc"],
                config["evidence"],
                root_selection["temperature"],
                chosen["alpha"],
                chosen["beta"],
            )
            if config["task"] == "flat6":
                result.append({**row, **root, "predicted_index": root["predicted_group_index"]})
                continue
            parent = root["predicted_group_index"]
            child = None
            if len(GROUPS[parent]) > 1:
                state = states["children"][str(parent)]
                selection = state["selection"]
                beta = selection["chosen"]["beta"] if mode in ("ihc", "both") else 0.0
                child = local_evidence.local_predict(
                    child_by_id[parent][pid]["logits"],
                    side[pid],
                    state["ihc"],
                    config["evidence"],
                    selection["temperature"],
                    beta,
                )
                if config["stage2_clinical"]["enabled"] and mode in ("clinical", "both"):
                    weights = state["clinical_selection"][mode]["chosen"]
                    corrected = clinical.predict(
                        child["image_probabilities"],
                        child["probabilities"],
                        side[pid],
                        state["clinical"],
                        config["stage2_clinical"],
                        weights["age_weight"],
                        weights["location_weight"],
                    )
                    child = {**child, **corrected}
            decoded = local_evidence.hard_route(root["probabilities"], {parent: child})
            result.append(
                {
                    "case_id": pid,
                    "target_index": row["source_class_index"],
                    "predicted_index": decoded["predicted_class_index"],
                    "probabilities": decoded["hard_probabilities"],
                    "root": root,
                    "child": child,
                    "predicted_group_index": parent,
                }
            )
        modes[mode] = result
    return modes


def summarize(modes, task):
    names = GROUP_NAMES if task == "flat6" else (("Tumor", "Non-tumor") if task == "binary" else CLASS_NAMES)
    result = {}
    reference = {r["case_id"]: r for r in modes.get("wsi", [])}
    for mode, rows in modes.items():
        metrics = prediction_metrics(rows, names)
        if reference:
            before = {pid: r["predicted_index"] == r["target_index"] for pid, r in reference.items()}
            metrics["paired_vs_wsi"] = {
                "corrected": sum(
                    not before[r["case_id"]] and r["predicted_index"] == r["target_index"] for r in rows
                ),
                "newly_wrong": sum(
                    before[r["case_id"]] and r["predicted_index"] != r["target_index"] for r in rows
                ),
            }
        if task == "hard6to10":
            root_rows = [
                {
                    "target_index": CLASS_TO_GROUP[r["target_index"]],
                    "predicted_index": r["predicted_group_index"],
                    "probabilities": r["root"]["probabilities"],
                }
                for r in rows
            ]
            metrics["six_group_routing"] = prediction_metrics(root_rows, GROUP_NAMES)
        result[mode] = metrics
    return result


def run(config, manifest, splits, output, feature_root=None):
    config = validate_config(config)
    records = load_manifest(manifest, feature_root)
    folds = validate_splits(records, read_json(splits))
    by_id = {r.patient_id: r for r in records}
    side = {r.patient_id: r.evidence for r in records}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(config, output / "config.json")
    write_json(
        {
            "manifest_sha256": sha256(manifest),
            "splits_sha256": sha256(splits),
            "patient_count": len(records),
            "task": config["task"],
            "data_included_in_repository": False,
        },
        output / "provenance.json",
    )
    status = {"status": "running", "started_at": now(), "task": config["task"]}
    write_json(status, output / "status.json")
    kind, n_classes, mapping, _ = root_spec(config["task"])
    collected = {}
    try:
        for fold in folds:
            index = fold["fold"]
            directory = output / f"fold_{index}"
            directory.mkdir()
            state = {"status": "running", "started_at": now(), "fold": index}
            write_json(state, directory / "status.json")
            train, valid, test = ([by_id[pid] for pid in fold[k]] for k in ("train", "validation", "test"))
            root, raw_valid, metadata = fit(
                train, valid, config, directory / "root", mapping, n_classes, kind
            )
            states, children = {"root": {}, "children": {}}, {}
            if config["task"] in ("flat6", "hard6to10"):
                states["root"] = select_root(train, raw_valid, side, config)
            if config["task"] == "hard6to10":
                for parent, members in enumerate(GROUPS):
                    if len(members) == 1:
                        continue
                    child_train = [r for r in train if r.label in members]
                    child_valid = [r for r in valid if r.label in members]
                    local_map = {label: i for i, label in enumerate(members)}
                    child_model, child_validation, _ = fit(
                        child_train,
                        child_valid,
                        config,
                        directory / f"child_{parent}",
                        local_map,
                        len(members),
                    )
                    children[parent] = child_model
                    states["children"][str(parent)] = select_child(
                        child_train, child_validation, side, parent, config
                    )
            write_json(
                {"selected_at": now(), "outer_test_used": False, **states}, directory / "selection.json"
            )
            if config["training"]["selection"] == "epoch_selection":
                root = refit(
                    train + valid,
                    config,
                    directory / "refit",
                    mapping,
                    n_classes,
                    metadata["selected_epoch"],
                    kind,
                )
            # All parameter decisions are frozen before any outer-test feature loading.
            raw_test = predict(root, test, config, mapping, kind)
            if config["task"] in ("flat6", "hard6to10"):
                raw_children = {
                    g: predict(model, test, config, {label: i for i, label in enumerate(GROUPS[g])})
                    for g, model in children.items()
                }
                modes = fuse(raw_test, raw_children, side, states, config)
            else:
                modes = {"model": raw_test}
            for mode, rows in modes.items():
                for r in rows:
                    r["fold"] = index
                write_json({"partition": "outer_test", "cases": rows}, directory / f"{mode}_predictions.json")
                collected.setdefault(mode, []).extend(rows)
            write_json(summarize(modes, config["task"]), directory / "metrics.json")
            state.update(status="complete", finished_at=now())
            write_json(state, directory / "status.json")
        for rows in collected.values():
            if len(rows) != len(records) or {r["case_id"] for r in rows} != set(by_id):
                raise RuntimeError("OOF predictions must cover each patient exactly once")
        summary = {
            "task": config["task"],
            "patients": len(records),
            "folds": len(folds),
            "modes": summarize(collected, config["task"]),
            "validation": "patient-disjoint cross-validation; not external validation",
        }
        write_json(summary, output / "summary.json")
        status.update(status="complete", finished_at=now())
        write_json(status, output / "status.json")
        return summary
    except BaseException as exc:
        status.update(status="failed_or_interrupted", finished_at=now(), error=str(exc))
        write_json(status, output / "status.json")
        raise
