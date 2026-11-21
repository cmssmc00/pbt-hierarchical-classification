"""Shared, masked gated-attention layers for patient bags."""

import math
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="pbt")
class ValidateBags(tf.keras.layers.Layer):
    """Reject empty bags and nonfinite valid patches; erase padded patches."""

    def call(self, inputs):
        features, mask = inputs
        mask = tf.cast(mask, tf.bool)
        checks = [
            tf.debugging.assert_equal(
                tf.shape(features)[:2],
                tf.shape(mask),
                message="Patch features and patch mask must have matching bags",
            ),
            tf.debugging.assert_equal(
                tf.reduce_all(tf.reduce_any(mask, axis=1)),
                True,
                message="Every patient bag must contain at least one valid patch",
            ),
        ]
        with tf.control_dependencies(checks):
            features = tf.where(mask[..., None], features, tf.zeros_like(features))
            return tf.debugging.check_numerics(features, "Nonfinite valid patch features")


@tf.keras.utils.register_keras_serializable(package="pbt")
class MaskedGatedAttention(tf.keras.layers.Layer):
    """Independent tanh/sigmoid gated attention, normalized over valid patches."""

    def __init__(self, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = int(attention_dim)
        self.tanh_gate = tf.keras.layers.Dense(self.attention_dim, activation="tanh", name="tanh_gate")
        self.sigmoid_gate = tf.keras.layers.Dense(
            self.attention_dim, activation="sigmoid", name="sigmoid_gate"
        )
        self.score = tf.keras.layers.Dense(1, use_bias=False, name="score")

    def call(self, inputs):
        features, mask = inputs
        mask = tf.cast(mask, tf.bool)
        logits = self.score(self.tanh_gate(features) * self.sigmoid_gate(features))
        logits = tf.cast(tf.squeeze(logits, axis=-1), tf.float32)
        logits = tf.where(mask, logits, tf.constant(float("-inf"), tf.float32))
        weights = tf.nn.softmax(logits, axis=1)
        # Keep padding exactly zero rather than relying on exp(-large_number).
        weights = tf.where(mask, weights, tf.zeros_like(weights))
        weights /= tf.reduce_sum(weights, axis=1, keepdims=True)
        masked_features = tf.where(mask[..., None], features, tf.zeros_like(features))
        return tf.reduce_sum(masked_features * tf.cast(weights[..., None], features.dtype), axis=1)

    def get_config(self):
        return {**super().get_config(), "attention_dim": self.attention_dim}


@tf.keras.utils.register_keras_serializable(package="pbt")
class LinearClinicalPrior(tf.keras.layers.Layer):
    """Centered direct age/location logits with a learned sigmoid strength.

    The age term is ``(age / age_scale_years * kernel + bias) * age_mask``.
    The location term is a categorical lookup multiplied by ``location_mask``.
    Missing values are erased before arithmetic or lookup, so even missing NaN
    ages and -1 location sentinels contribute exactly zero. Observed ages must
    be finite; observed location indices must be in the configured vocabulary.
    """

    def __init__(self, num_classes, num_locations, age_scale_years=25.0, prior_strength_init=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.num_locations = int(num_locations)
        self.age_scale_years = float(age_scale_years)
        self.prior_strength_init = float(prior_strength_init)

    def build(self, input_shape):
        self.age_kernel = self.add_weight("age_kernel", shape=(1, self.num_classes), initializer="zeros")
        self.age_bias = self.add_weight("age_bias", shape=(self.num_classes,), initializer="zeros")
        self.location_logits = self.add_weight(
            "location_logits", shape=(self.num_locations, self.num_classes), initializer="zeros"
        )
        initial_logit = math.log(self.prior_strength_init / (1.0 - self.prior_strength_init))
        self.strength_logit = self.add_weight(
            "strength_logit", shape=(), initializer=tf.keras.initializers.Constant(initial_logit)
        )
        super().build(input_shape)

    def call(self, inputs):
        age, age_mask, location, location_mask = inputs
        location = tf.reshape(location, (-1,))
        age_mask = tf.cast(age_mask, age.dtype)
        location_mask = tf.cast(location_mask, age.dtype)
        masks = tf.concat([age_mask, location_mask], axis=1)
        valid_masks = tf.reduce_all(tf.logical_or(tf.equal(masks, 0), tf.equal(masks, 1)))
        age_observed = tf.equal(age_mask, 1)
        location_observed = tf.reshape(tf.equal(location_mask, 1), (-1,))
        safe_age = tf.where(age_observed, age, tf.zeros_like(age))
        safe_location = tf.where(location_observed, location, tf.zeros_like(location))
        checks = [
            tf.debugging.assert_equal(
                valid_masks, True, message="Clinical masks must be exactly zero or one"
            ),
            tf.debugging.assert_greater_equal(
                safe_location, 0, message="Observed location index cannot be negative"
            ),
            tf.debugging.assert_less(
                safe_location, self.num_locations, message="Observed location index is out of range"
            ),
        ]
        with tf.control_dependencies(checks):
            safe_age = tf.debugging.check_numerics(safe_age, "Nonfinite observed age")
            age_logits = tf.matmul(safe_age / self.age_scale_years, self.age_kernel) + self.age_bias
            age_logits = tf.where(age_observed, age_logits, tf.zeros_like(age_logits))
            location_logits = tf.gather(self.location_logits, safe_location)
            location_logits = tf.where(
                location_observed[:, None], location_logits, tf.zeros_like(location_logits)
            )
            prior = age_logits + location_logits
            prior -= tf.reduce_mean(prior, axis=-1, keepdims=True)
            return tf.math.sigmoid(self.strength_logit) * prior

    def get_config(self):
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "num_locations": self.num_locations,
            "age_scale_years": self.age_scale_years,
            "prior_strength_init": self.prior_strength_init,
        }


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
