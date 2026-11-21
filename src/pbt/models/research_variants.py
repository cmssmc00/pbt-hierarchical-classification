"""Isolated single-Teacher variants; the copied baseline remains unchanged."""

from __future__ import annotations

import tensorflow as tf

from .layers import LinearClinicalPrior, MaskedGatedAttention, ValidateBags
from .joint_global import build_model as build_baseline, supervised_losses as supervised_losses
from .three_group import CLASS_TO_GROUP, CLASS_TO_LOCAL


@tf.keras.utils.register_keras_serializable(package="hierarchy_improvements")
class ClassAttention(tf.keras.layers.Layer):
    """Shared attention gates, with one patch distribution per leaf class."""

    def __init__(self, attention_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim, self.num_classes = int(attention_dim), int(num_classes)
        self.tanh_gate = tf.keras.layers.Dense(self.attention_dim, activation="tanh")
        self.sigmoid_gate = tf.keras.layers.Dense(self.attention_dim, activation="sigmoid")
        self.score = tf.keras.layers.Dense(self.num_classes, use_bias=False)

    def call(self, inputs):
        features, mask = inputs
        scores = self.score(self.tanh_gate(features) * self.sigmoid_gate(features))
        scores = tf.where(mask[..., None], scores, tf.constant(float("-inf"), scores.dtype))
        attention = tf.nn.softmax(scores, axis=1)
        attention = tf.where(mask[..., None], attention, tf.zeros_like(attention))
        safe = tf.where(mask[..., None], features, tf.zeros_like(features))
        return tf.einsum("bpc,bpd->bcd", attention, safe)

    def get_config(self):
        return {**super().get_config(), "attention_dim": self.attention_dim, "num_classes": self.num_classes}


@tf.keras.utils.register_keras_serializable(package="hierarchy_improvements")
class ClassLogits(tf.keras.layers.Layer):
    """Score each class's own representation without mixing class axes."""

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=(int(input_shape[1]), int(input_shape[2])), initializer="glorot_uniform"
        )
        self.bias = self.add_weight("bias", shape=(int(input_shape[1]),), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs):
        return tf.reduce_sum(inputs * self.kernel[None, ...], axis=-1) + self.bias


@tf.keras.utils.register_keras_serializable(package="hierarchy_improvements")
class HierarchicalResidual(tf.keras.layers.Layer):
    """Finite log-joint prior plus bounded, penalized learned corrections.

    The calibration variables are never optimized on outer-test patients.
    A correction bound is a regularizer, not a promise to overturn arbitrarily
    extreme prior odds. No class or group is masked out.
    """

    def __init__(self, bound=2.0, penalty=0.01, **kwargs):
        super().__init__(**kwargs)
        self.bound, self.penalty = float(bound), float(penalty)

    def build(self, input_shape):
        self.temperature = self.add_weight("temperature", shape=(), initializer="ones", trainable=False)
        self.correction_scale = self.add_weight(
            "correction_scale", shape=(), initializer="ones", trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        coarse, *others = inputs
        children, raw = others[:3], others[3]
        root_log = tf.nn.log_softmax(coarse / self.temperature)
        leaf_log = [tf.nn.log_softmax(x / self.temperature) for x in children]
        prior = tf.stack(
            [
                root_log[:, CLASS_TO_GROUP[c]] + leaf_log[CLASS_TO_GROUP[c]][:, CLASS_TO_LOCAL[c]]
                for c in range(10)
            ],
            axis=-1,
        )
        delta = self.bound * tf.math.tanh(raw)
        self.add_loss(self.penalty * tf.reduce_mean(tf.square(delta)))
        return prior + self.correction_scale * delta

    def get_config(self):
        return {**super().get_config(), "bound": self.bound, "penalty": self.penalty}


def build_model(config):
    variant = config.get("variant", {})
    architecture = variant.get("architecture", "baseline")
    if architecture == "baseline":
        return build_baseline(config)
    if architecture not in {"class_attention", "residual"}:
        raise ValueError("Unknown architecture")
    m, d = config["model"], config["data"]
    features = tf.keras.Input((None, d["feature_dim"]), dtype=tf.float32, name="patches")
    mask = tf.keras.Input((None,), dtype=tf.bool, name="patch_mask")
    inputs = {"patches": features, "patch_mask": mask}
    encoded = ValidateBags(name="validate_bags")([features, mask])
    encoded = tf.keras.layers.Dense(m["projection_dim"], name="shared_projection")(encoded)
    encoded = tf.keras.layers.LayerNormalization(name="shared_layer_norm")(encoded)
    encoded = tf.keras.layers.Activation("gelu", name="shared_gelu")(encoded)
    encoded = tf.keras.layers.Dropout(m["patch_dropout"], name="shared_patch_dropout")(encoded)
    if m["use_clinical"]:
        for key in ("age", "age_mask", "location", "location_mask"):
            inputs[key] = tf.keras.Input((1,), dtype=tf.int32 if key == "location" else tf.float32, name=key)
        clinical = [inputs[key] for key in ("age", "age_mask", "location", "location_mask")]
    outputs, embeddings, probabilities = {}, [], []
    for head, width in [("coarse", 3), ("leaf_0", 3), ("leaf_1", 3), ("leaf_2", 4)]:
        specific = architecture == "class_attention" and head == "leaf_0"
        attention = (
            ClassAttention(m["attention_dim"], width, name=f"{head}_attention")
            if specific
            else MaskedGatedAttention(m["attention_dim"], name=f"{head}_attention")
        )
        pooled = attention([encoded, mask])
        fused = tf.keras.layers.Dense(m["fusion_dim"], activation="gelu", name=f"{head}_fusion")(pooled)
        fused = tf.keras.layers.Dropout(m["fusion_dropout"], name=f"{head}_fusion_dropout")(fused)
        layer_name = f"{head}_wsi_logits" if m["use_clinical"] else head
        logits = (
            ClassLogits(name=layer_name)(fused)
            if specific
            else tf.keras.layers.Dense(width, name=layer_name)(fused)
        )
        if m["use_clinical"]:
            prior = LinearClinicalPrior(
                width,
                m["num_locations"],
                m["age_scale_years"],
                m["prior_strength_init"],
                name=f"{head}_clinical_prior",
            )(clinical)
            logits = tf.keras.layers.Add(name=head)([logits, prior])
        outputs[head] = logits
        # Preserve the original 525-dimensional global input. The class-specific
        # representations remain separate for leaf logits; mean only for global.
        embeddings.append(tf.reduce_mean(fused, axis=1) if specific else fused)
        probabilities.append(tf.keras.layers.Softmax(name=f"{head}_probabilities")(logits))
    combined = tf.keras.layers.Concatenate(name="global_inputs")(embeddings + probabilities)
    refined = tf.keras.layers.Dense(m["fusion_dim"], activation="gelu", name="global_fusion")(combined)
    refined = tf.keras.layers.Dropout(m["fusion_dropout"], name="global_dropout")(refined)
    if architecture == "residual":
        raw = tf.keras.layers.Dense(
            10, kernel_initializer="zeros", bias_initializer="zeros", name="correction_raw"
        )(refined)
        outputs["global"] = HierarchicalResidual(
            variant["correction_bound"], variant["correction_penalty"], name="global"
        )([outputs[k] for k in ("coarse", "leaf_0", "leaf_1", "leaf_2")] + [raw])
    else:
        outputs["global"] = tf.keras.layers.Dense(10, name="global")(refined)
    return tf.keras.Model(inputs, outputs, name=f"single_teacher_{architecture}")
