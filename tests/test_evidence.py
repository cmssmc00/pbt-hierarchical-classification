import copy
from pathlib import Path
import unittest

from pbt.evidence import root, hierarchy, clinical
from pbt.io import read_json
from pbt.metrics import prediction_metrics

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/hard6to10_fixed25.json")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "age": 2.0,
            "age_mask": True,
            "location_index": 2,
            "location_mask": True,
            "ihc": [1, 0, None, None, None],
        }
        self.side = {"synthetic_train": self.record, "synthetic_held_out": {**self.record, "ihc": [0] * 5}}
        self.labels = {"synthetic_train": 0, "synthetic_held_out": 5}
        self.fitted = root.fit_ihc(["synthetic_train"], self.labels, self.side, 8)

    def test_held_out_markers_do_not_fit_likelihood(self):
        changed = copy.deepcopy(self.side)
        changed["synthetic_held_out"]["ihc"] = [1] * 5
        self.assertEqual(self.fitted, root.fit_ihc(["synthetic_train"], self.labels, changed, 8))

    def test_confident_prediction_unchanged(self):
        output = root.apply([12, 0, 0, 0, 0, 0], self.record, self.fitted, CONFIG["evidence"], 1, 1, 1)
        self.assertFalse(output["uncertain"])
        self.assertEqual(output["probabilities"], output["image_probabilities"])

    def test_missing_and_disabled_evidence_identity(self):
        missing = {
            "age": 0,
            "age_mask": False,
            "location_index": 10,
            "location_mask": False,
            "ihc": [None] * 5,
        }
        for record, alpha, beta in ((missing, 1, 1), (self.record, 0, 0)):
            output = root.apply([0] * 6, record, self.fitted, CONFIG["evidence"], 1, alpha, beta)
            self.assertEqual(output["probabilities"], output["image_probabilities"])

    def test_root_rules_only_affect_top_two(self):
        boosts, _ = root.clinical_cues(self.record, [0.1, 0.1, 0.4, 0.3, 0.05, 0.05])
        self.assertEqual(boosts, [0, 0, 0.5, 1, 0, 0])
        boosts, _ = root.clinical_cues(self.record, [0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
        self.assertEqual(boosts, [0] * 6)

    def test_hard_routing_cannot_cross_group(self):
        child = {
            "probabilities": [0.01, 0.01, 0.98],
            "uncertain": False,
            "ihc_applied": False,
            "ihc_markers_used": [],
        }
        decoded = hierarchy.hard_route([0.6, 0.4, 0, 0, 0, 0], {0: child, 1: child})
        self.assertEqual(decoded["predicted_class_index"], 2)
        self.assertEqual(decoded["hard_probabilities"][3:], [0] * 7)

    def test_hard_error_nll_is_explicitly_infinite(self):
        metrics = prediction_metrics(
            [{"target_index": 1, "predicted_index": 0, "probabilities": [1.0, 0.0]}], ["a", "b"]
        )
        self.assertIsNone(metrics["nll"])
        self.assertEqual(metrics["nll_status"], "infinite")

    def test_stage2_fit_requires_training_only(self):
        with self.assertRaises(ValueError):
            clinical.fit(["synthetic_train"], self.labels, self.side, (0, 1, 2), 0, CONFIG["stage2_clinical"])
