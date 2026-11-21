"""One gated-attention Teacher for flat classification or a hierarchy node."""

import math
import tensorflow as tf
from .layers import ValidateBags, MaskedGatedAttention, LinearClinicalPrior, _positive_integer


def build_model(config, num_classes=6):
    """Build one patient-level image classifier with a configurable label count."""
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
    use_clinical = model_config.get("use_clinical", False)
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

    pooled = MaskedGatedAttention(attention_dim, name="group_attention")([encoded, mask])
    fused = tf.keras.layers.Dense(fusion_dim, activation="gelu", name="group_fusion")(pooled)
    fused = tf.keras.layers.Dropout(fusion_dropout, name="group_fusion_dropout")(fused)
    logits = tf.keras.layers.Dense(num_classes, name="group_wsi_logits")(fused)
    if use_clinical:
        prior = LinearClinicalPrior(
            num_classes, num_locations, age_scale, prior_strength, name="group_clinical_prior"
        )(clinical_inputs)
        logits = tf.keras.layers.Add(name="groups")([logits, prior])
    return tf.keras.Model(inputs=inputs, outputs={"groups": logits}, name="flat_sixgroup_teacher")
