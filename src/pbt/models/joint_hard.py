"""Joint three-group hierarchical model retained from the original experiments."""

import math
import tensorflow as tf
from .layers import ValidateBags, MaskedGatedAttention, LinearClinicalPrior, _positive_integer
from .three_group import CLASS_TO_GROUP, CLASS_TO_LOCAL, GROUPS


def build_model(config):
    """Build a TF 2.14/2.15 functional model returning four batches of logits.

    Inputs are a dictionary containing ``patches`` (N,P,D),
    ``patch_mask`` (N,P; bool), and, when clinical input is enabled, ``age``,
    ``age_mask``, ``location_mask`` (each N,1) and ``location`` (N,1; int32).
    Outputs are ``coarse`` (N,3), ``leaf_0`` (N,3), ``leaf_1`` (N,3), and
    ``leaf_2`` (N,4). No prediction route or argmax is used by the model.
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    feature_dim = _positive_integer(data_config.get("feature_dim", 768), "feature_dim")
    projection_dim = _positive_integer(model_config.get("projection_dim", 256), "projection_dim")
    attention_dim = _positive_integer(model_config.get("attention_dim", 128), "attention_dim")
    fusion_dim = _positive_integer(model_config.get("fusion_dim", 128), "fusion_dim")
    patch_dropout = float(model_config.get("patch_dropout", 0.1))
    fusion_dropout = float(model_config.get("fusion_dropout", 0.2))
    if not 0 <= patch_dropout < 1 or not 0 <= fusion_dropout < 1:
        raise ValueError("Dropout rates must be in [0, 1)")
    use_clinical = model_config.get("use_clinical", True)
    if not isinstance(use_clinical, bool):
        raise ValueError("use_clinical must be a boolean")

    features = tf.keras.Input(shape=(None, feature_dim), dtype=tf.float32, name="patches")
    mask = tf.keras.Input(shape=(None,), dtype=tf.bool, name="patch_mask")
    inputs = {"patches": features, "patch_mask": mask}
    encoded = ValidateBags(name="validate_bags")([features, mask])
    encoded = tf.keras.layers.Dense(projection_dim, name="shared_projection")(encoded)
    encoded = tf.keras.layers.LayerNormalization(name="shared_layer_norm")(encoded)
    encoded = tf.keras.layers.Activation("gelu", name="shared_gelu")(encoded)
    encoded = tf.keras.layers.Dropout(patch_dropout, name="shared_patch_dropout")(encoded)

    if use_clinical:
        num_locations = _positive_integer(model_config.get("num_locations", 11), "num_locations")
        age_scale = float(model_config.get("age_scale_years", 25.0))
        prior_strength = float(model_config.get("prior_strength_init", 0.1))
        if not math.isfinite(age_scale) or age_scale <= 0:
            raise ValueError("age_scale_years must be finite and positive")
        if not 0 < prior_strength < 1:
            raise ValueError("prior_strength_init must be strictly between zero and one")
        inputs.update(
            {
                "age": tf.keras.Input(shape=(1,), dtype=tf.float32, name="age"),
                "age_mask": tf.keras.Input(shape=(1,), dtype=tf.float32, name="age_mask"),
                "location": tf.keras.Input(shape=(1,), dtype=tf.int32, name="location"),
                "location_mask": tf.keras.Input(shape=(1,), dtype=tf.float32, name="location_mask"),
            }
        )
        clinical_inputs = [inputs[key] for key in ("age", "age_mask", "location", "location_mask")]

    outputs = {}
    for head, width in [("coarse", len(GROUPS))] + [
        (f"leaf_{g}", len(group)) for g, group in enumerate(GROUPS)
    ]:
        pooled = MaskedGatedAttention(attention_dim, name=f"{head}_attention")([encoded, mask])
        fused = tf.keras.layers.Dense(fusion_dim, activation="gelu", name=f"{head}_fusion")(pooled)
        fused = tf.keras.layers.Dropout(fusion_dropout, name=f"{head}_fusion_dropout")(fused)
        logits = tf.keras.layers.Dense(width, name=f"{head}_wsi_logits" if use_clinical else head)(fused)
        if use_clinical:
            prior = LinearClinicalPrior(
                width, num_locations, age_scale, prior_strength, name=f"{head}_clinical_prior"
            )(clinical_inputs)
            logits = tf.keras.layers.Add(name=head)([logits, prior])
        outputs[head] = logits
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="hierarchy_single_teacher")


def supervised_losses(outputs, labels, coarse_weights, fine_weights):
    """Return unreduced ``coarse``, ``fine``, and ``total`` vectors of length N.

    Labels are global zero-based ten-class integer targets. Coarse weights have
    length three; fine weights have length ten. A patient's fine loss is only
    its TRUE group's cross-entropy. Other heads receive exactly zero fine-loss
    gradient for that patient. Legal dummy local labels permit vectorized CE.
    The caller must apply ``tf.reduce_mean(losses['total'])``; no batch, group,
    or regularization reduction is performed here. Root argmax is never used.
    """
    labels = tf.convert_to_tensor(labels)
    if not labels.dtype.is_integer:
        raise TypeError("Labels must be integer global class indices")
    labels = tf.ensure_shape(labels, (None,))
    coarse_weights = tf.ensure_shape(tf.convert_to_tensor(coarse_weights, dtype=tf.float32), (len(GROUPS),))
    fine_weights = tf.ensure_shape(
        tf.convert_to_tensor(fine_weights, dtype=tf.float32), (len(CLASS_TO_GROUP),)
    )
    checks = [
        tf.debugging.assert_greater_equal(
            labels, tf.cast(0, labels.dtype), message="Negative global class index"
        ),
        tf.debugging.assert_less(
            labels, tf.cast(len(CLASS_TO_GROUP), labels.dtype), message="Global class index out of range"
        ),
        tf.debugging.assert_non_negative(coarse_weights, message="Coarse weights must be nonnegative"),
        tf.debugging.assert_non_negative(fine_weights, message="Fine weights must be nonnegative"),
        tf.debugging.assert_all_finite(coarse_weights, "Nonfinite coarse weights"),
        tf.debugging.assert_all_finite(fine_weights, "Nonfinite fine weights"),
    ]
    with tf.control_dependencies(checks):
        true_group = tf.gather(tf.constant(CLASS_TO_GROUP, dtype=tf.int32), labels)
        true_local = tf.gather(tf.constant(CLASS_TO_LOCAL, dtype=tf.int32), labels)

    def logits_for(key, width):
        logits = tf.ensure_shape(tf.cast(outputs[key], tf.float32), (None, width))
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(
                    tf.shape(logits)[0], tf.shape(labels)[0], message=f"{key} batch size mismatch"
                )
            ]
        ):
            return tf.identity(logits)

    coarse = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=true_group, logits=logits_for("coarse", len(GROUPS))
    ) * tf.gather(coarse_weights, true_group)
    fine = tf.zeros_like(coarse)
    for g, members in enumerate(GROUPS):
        selected = tf.equal(true_group, g)
        dummy_local = tf.where(selected, true_local, tf.zeros_like(true_local))
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=dummy_local, logits=logits_for(f"leaf_{g}", len(members))
        )
        fine += tf.where(selected, ce, tf.zeros_like(ce))
    fine *= tf.gather(fine_weights, labels)
    return {"coarse": coarse, "fine": fine, "total": coarse + fine}
