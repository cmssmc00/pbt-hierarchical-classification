"""Serial supervised training with patient-level checkpoint selection.

No test records enter fit(). Fixed epochs, conventional NLL early stopping, and
the historical candidate-epoch selection/refit protocol are separate options.
"""

import math
from pathlib import Path

import numpy as np

from .data import make_dataset
from .early_stopping import EarlyStopping
from .evidence.root import softmax
from .io import write_json, sha256
from .metrics import prediction_metrics


def balanced_weights(labels, n_classes, power=1.0):
    counts = np.bincount(labels, minlength=n_classes)
    if len(counts) != n_classes or np.any(counts == 0):
        raise ValueError("Every modeled class must be represented in the training partition")
    weights = (len(labels) / (n_classes * counts)) ** power
    weights /= np.dot(weights, counts) / len(labels)
    return weights.tolist()


def build_model(config, kind, n_classes):
    if kind == "joint_hard":
        from .models.joint_hard import build_model as build
    elif kind == "joint_global":
        from .models.joint_global import build_model as build
    elif kind == "flat":
        from .models.flat import build_model as build

        return build(config, n_classes)
    else:
        raise ValueError("Unknown model kind")
    return build(config)


def predict(model, records, config, label_map, kind="flat"):
    include_clinical = config["model"]["use_clinical"]
    rows, offset = [], 0
    for inputs, _ in make_dataset(records, config, False, label_map, include_clinical):
        values = {k: v.numpy().tolist() for k, v in model(inputs, training=False).items()}
        width = len(next(iter(values.values())))
        for j in range(width):
            rec = records[offset]
            if kind == "joint_hard":
                from .models.three_group import decode_hard_predictions

                root = [softmax(values["coarse"][j])]
                children = [[softmax(values[f"leaf_{g}"][j])] for g in range(3)]
                decoded = decode_hard_predictions(root, children)[0]
                p, logits = decoded["routed_probabilities"], None
            else:
                logits = values["global" if kind == "joint_global" else "groups"][j]
                p = softmax(logits)
            pred = max(range(len(p)), key=lambda k: p[k])
            rows.append(
                {
                    "case_id": rec.patient_id,
                    "source_class_index": rec.label,
                    "target_index": label_map.get(rec.label),
                    "logits": logits,
                    "probabilities": p,
                    "predicted_index": pred,
                }
            )
            offset += 1
    if offset != len(records):
        raise RuntimeError("Prediction order/coverage mismatch")
    return rows


def fit(train, validation, config, target, label_map, n_classes, kind="flat", selection=None, epochs=None):
    import tensorflow as tf

    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    t, method = config["training"], config["method"]
    selection = selection or t["selection"]
    epochs = epochs or t["epochs"]
    if selection not in ("fixed", "early_stopping", "epoch_selection"):
        raise ValueError("Unknown checkpoint selection method")
    if kind == "joint_hard" and selection != "fixed":
        raise ValueError("Hard joint routing uses fixed epochs; its test NLL can be infinite")
    if selection == "epoch_selection" and kind != "joint_global":
        raise ValueError("Historical epoch selection/refit is provided for the joint global model")
    if not train or not validation:
        raise ValueError("Training and inner validation must be nonempty")
    if {r.patient_id for r in train} & {r.patient_id for r in validation}:
        raise ValueError("Training and validation patients overlap")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(t["seed"])
    tf.config.experimental.enable_op_determinism()
    teacher = build_model(config, kind, n_classes)
    labels = [label_map[r.label] for r in train]
    weights = balanced_weights(labels, n_classes, method["weight_power"])
    if kind.startswith("joint"):
        from .models.three_group import balanced_path_weights

        path_weights = balanced_path_weights(labels)
    steps = math.ceil(len(train) / t["batch_size"])
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        t["learning_rate"],
        steps * t["schedule_epochs"],
        alpha=t["minimum_learning_rate"] / t["learning_rate"],
    )
    if method["weight_decay"]:
        optimizer = tf.keras.optimizers.AdamW(
            schedule, weight_decay=method["weight_decay"], clipnorm=t["gradient_clip_norm"]
        )
    else:
        optimizer = tf.keras.optimizers.Adam(schedule, clipnorm=t["gradient_clip_norm"])

    @tf.function(reduce_retracing=True)
    def step(inputs, y):
        with tf.GradientTape() as tape:
            output = teacher(inputs, training=True)
            if kind == "joint_global":
                from .models.joint_global import supervised_losses

                value = supervised_losses(
                    output, y, path_weights["coarse"], path_weights["fine"], path_weights["global"]
                )["total"]
            elif kind == "joint_hard":
                from .models.joint_hard import supervised_losses

                value = supervised_losses(output, y, path_weights["coarse"], path_weights["fine"])["total"]
            else:
                logits = output["groups"]
                smooth = method["label_smoothing"]
                if smooth:
                    targets = tf.one_hot(y, n_classes) * (1 - smooth) + smooth / n_classes
                    value = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                else:
                    value = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                if method["focal_gamma"]:
                    prob = tf.gather(tf.nn.softmax(logits), y, axis=1, batch_dims=1)
                    value *= tf.pow(1 - prob, method["focal_gamma"])
                value *= tf.gather(tf.constant(weights, tf.float32), y)
            loss = tf.reduce_mean(value)
            tf.debugging.assert_all_finite(loss, "Nonfinite training loss")
        grads = tape.gradient(loss, teacher.trainable_variables)
        optimizer.apply_gradients((g, v) for g, v in zip(grads, teacher.trainable_variables) if g is not None)
        return tf.reduce_sum(value)

    data = make_dataset(train, config, True, label_map, config["model"]["use_clinical"])
    stopper = EarlyStopping(t["patience"], t["min_delta"])
    history, candidates = [], []
    best_key, selected_epoch = None, None
    checkpoint = target / "selected.weights.h5"
    for epoch in range(1, epochs + 1):
        total, seen = 0.0, 0
        for inputs, y in data:
            total += float(step(inputs, y))
            seen += int(tf.shape(y)[0])
        if seen != len(train) or int(optimizer.iterations) != steps * epoch:
            raise RuntimeError("Unexpected patient or optimizer update count")
        row = {
            "epoch": epoch,
            "loss": total / seen,
            "patients": seen,
            "optimizer_updates": int(optimizer.iterations),
        }
        evaluate = (
            selection == "early_stopping"
            or selection == "fixed"
            and epoch == epochs
            or selection == "epoch_selection"
            and epoch in t["candidate_epochs"]
        )
        stop, improved = False, False
        if evaluate:
            valid = predict(teacher, validation, config, label_map, kind)
            metric = prediction_metrics(valid, [str(i) for i in range(n_classes)])
            row["validation"] = {k: metric[k] for k in ("correct", "nll", "nll_status")}
            if selection == "early_stopping":
                if metric["nll"] is None:
                    raise ValueError("Early stopping requires finite validation NLL")
                decision = stopper.observe(epoch, metric["nll"])
                row["early_stopping"] = decision
                improved, stop = decision["improved"], decision["stop"]
            elif selection == "epoch_selection":
                key = (-metric["correct"], metric["nll"], epoch)
                improved = best_key is None or key < best_key
                if improved:
                    best_key = key
                candidates.append({"epoch": epoch, **row["validation"]})
            else:
                improved = True
            if improved:
                selected_epoch = epoch
                teacher.save_weights(checkpoint)
        history.append(row)
        write_json(history, target / "history.json")
        print(f"{target.name}: epoch={epoch}/{epochs} loss={total / seen:.5f}", flush=True)
        if stop:
            break
    if selected_epoch is None:
        raise ValueError("No candidate checkpoint was evaluated")
    teacher.load_weights(checkpoint)
    valid = predict(teacher, validation, config, label_map, kind)
    meta = {
        "selection": selection,
        "selected_epoch": selected_epoch,
        "epochs_ran": len(history),
        "optimizer_updates": int(optimizer.iterations),
        "train_ids": [r.patient_id for r in train],
        "validation_ids": [r.patient_id for r in validation],
        "class_weights": weights,
        "checkpoint_sha256": sha256(checkpoint),
        "tensorflow_version": tf.__version__,
        "outer_test_used": False,
        "candidate_epochs": candidates,
    }
    write_json(meta, target / "fit.json")
    return teacher, valid, meta


def refit(train, config, target, label_map, n_classes, epochs, kind="joint_global"):
    """Fresh initialization on the full outer training pool at the chosen epoch.

    The zero-feature dummy validation workaround is deliberately avoided: refitting
    has its own loop and never evaluates a test/validation patient.
    """
    import tensorflow as tf
    from .models.three_group import balanced_path_weights
    from .models.joint_global import supervised_losses

    if kind != "joint_global":
        raise ValueError("Refit is reserved for the original joint global protocol")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    t = config["training"]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(t["seed"])
    tf.config.experimental.enable_op_determinism()
    model = build_model(config, kind, n_classes)
    weights = balanced_path_weights([label_map[r.label] for r in train])
    steps = math.ceil(len(train) / t["batch_size"])
    lr = tf.keras.optimizers.schedules.CosineDecay(
        t["learning_rate"],
        steps * t["schedule_epochs"],
        alpha=t["minimum_learning_rate"] / t["learning_rate"],
    )
    optimizer = tf.keras.optimizers.Adam(lr, clipnorm=t["gradient_clip_norm"])

    @tf.function(reduce_retracing=True)
    def step(inputs, labels):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                supervised_losses(
                    model(inputs, training=True),
                    labels,
                    weights["coarse"],
                    weights["fine"],
                    weights["global"],
                )["total"]
            )
            tf.debugging.assert_all_finite(loss, "Nonfinite refit loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            (g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None
        )
        return loss

    ds = make_dataset(train, config, True, label_map, config["model"]["use_clinical"])
    history = []
    for epoch in range(1, epochs + 1):
        total, seen = 0.0, 0
        for inputs, labels in ds:
            n = int(tf.shape(labels)[0])
            total += float(step(inputs, labels)) * n
            seen += n
        if seen != len(train) or int(optimizer.iterations) != epoch * steps:
            raise RuntimeError("Refit update count mismatch")
        history.append(
            {
                "epoch": epoch,
                "loss": total / seen,
                "patients": seen,
                "optimizer_updates": int(optimizer.iterations),
            }
        )
    model.save_weights(target / "selected.weights.h5")
    write_json(history, target / "history.json")
    write_json(
        {
            "selected_epoch": epochs,
            "reinitialized": True,
            "outer_test_used": False,
            "train_ids": [r.patient_id for r in train],
            "optimizer_updates": epochs * steps,
        },
        target / "fit.json",
    )
    return model
