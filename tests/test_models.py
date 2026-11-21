from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from pbt.early_stopping import EarlyStopping
from pbt.io import read_json
from pbt.models.flat import build_model
from pbt.models.joint_hard import supervised_losses
from pbt.models.three_group import decode_predictions
from pbt.pipeline import validate_config
from pbt.training import balanced_weights

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_padding_does_not_affect_logits(self):
        config = deepcopy(read_json(ROOT / "configs/flat6_fixed25.json"))
        config["data"]["feature_dim"] = 4
        config["model"].update(projection_dim=8, attention_dim=4, fusion_dim=4)
        model = build_model(config)
        patches = tf.ones((1, 2, 4))
        a = model({"patches": patches, "patch_mask": tf.ones((1, 2), tf.bool)}, training=False)["groups"]
        padded = tf.concat([patches, tf.fill((1, 2, 4), float("nan"))], axis=1)
        b = model(
            {"patches": padded, "patch_mask": tf.constant([[True, True, False, False]])}, training=False
        )["groups"]
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model({"patches": patches, "patch_mask": tf.zeros((1, 2), tf.bool)})

    def test_child_loss_uses_true_parent_even_when_root_is_wrong(self):
        outputs = {
            "coarse": tf.Variable([[10.0, -10.0, -10.0]]),
            "leaf_0": tf.Variable([[0.0, 0.0, 0.0]]),
            "leaf_1": tf.Variable([[0.0, 1.0, 0.0]]),
            "leaf_2": tf.Variable([[0.0, 0.0, 0.0, 0.0]]),
        }
        with tf.GradientTape() as tape:
            loss = supervised_losses(outputs, tf.constant([3]), [1.0] * 3, [1.0] * 10)
            total = tf.reduce_mean(loss["total"])
        gradients = tape.gradient(total, [outputs["leaf_0"], outputs["leaf_1"], outputs["leaf_2"]])
        self.assertTrue(np.isfinite(float(total)))
        self.assertTrue(np.all(gradients[0].numpy() == 0))
        self.assertTrue(np.any(gradients[1].numpy() != 0))
        self.assertTrue(np.all(gradients[2].numpy() == 0))

    def test_global_head_can_cross_parent(self):
        prediction = decode_predictions(
            [[0.9, 0.05, 0.05]],
            [[[1.0, 0, 0]], [[1.0, 0, 0]], [[1.0, 0, 0, 0]]],
            [[0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]],
        )[0]
        self.assertEqual(prediction["predicted_class_index"], 7)
        self.assertTrue(prediction["crossed_group"])

    def test_early_stopping_patience_and_minimum_delta(self):
        s = EarlyStopping(patience=2, min_delta=0.01)
        self.assertTrue(s.observe(1, 1.0)["improved"])
        self.assertFalse(s.observe(2, 0.995)["improved"])
        decision = s.observe(3, 1.1)
        self.assertTrue(decision["stop"])
        self.assertEqual(decision["best_epoch"], 1)

    def test_weights_use_training_distribution(self):
        np.testing.assert_allclose(balanced_weights([0, 0, 0, 1], 2), [2 / 3, 2.0])
        np.testing.assert_allclose(balanced_weights([0, 0, 0, 1], 2, power=0), [1, 1])

    def test_all_presets_validate(self):
        for path in (ROOT / "configs").glob("*.json"):
            with self.subTest(path=path.name):
                validate_config(read_json(path))
