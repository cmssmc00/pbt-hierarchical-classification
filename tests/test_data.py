import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pbt.data import load_bag, load_manifest
from pbt.io import read_json, write_json
from pbt.splits import make_splits, validate_splits
from pbt.synthetic import generate


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = generate(Path(self.tmp.name) / "synthetic", feature_dim=8, per_class=6)
        self.records = load_manifest(self.manifest)

    def tearDown(self):
        self.tmp.cleanup()

    def test_patient_bags_join_slides_and_sample(self):
        record = self.records[0]
        self.assertEqual(len(record.features), 2)
        full = load_bag(record, 8)
        sampled = load_bag(record, 8, max_patches=4, rng=np.random.default_rng(3))
        self.assertGreater(len(full), 4)
        self.assertEqual(sampled.shape, (4, 8))
        for patch in sampled:
            self.assertTrue(any(np.array_equal(patch, x) for x in full))

    def test_no_patient_or_slide_overlap(self):
        folds = make_splits(self.records)
        validate_splits(self.records, folds)
        for fold in folds["folds"]:
            self.assertFalse(set(fold["train"]) & set(fold["test"]))
            self.assertFalse(set(fold["validation"]) & set(fold["test"]))
        broken = copy.deepcopy(folds)
        broken["folds"][0]["test"][0] = broken["folds"][0]["train"][0]
        with self.assertRaises(ValueError):
            validate_splits(self.records, broken)

    def test_duplicate_feature_rejected(self):
        payload = read_json(self.manifest)
        payload["patients"][1]["features"] = payload["patients"][0]["features"]
        write_json(payload, self.manifest)
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_inferred_or_numeric_marker_not_accepted(self):
        payload = read_json(self.manifest)
        for value in ("inferred POS", 0, False):
            payload["patients"][0]["ihc"]["GFAP"] = value
            write_json(payload, self.manifest)
            with self.assertRaises(ValueError):
                load_manifest(self.manifest)

    def test_missing_is_not_negative(self):
        payload = read_json(self.manifest)
        payload["patients"][0]["ihc"] = {"GFAP": "NEG", "INI1": None}
        write_json(payload, self.manifest)
        record = load_manifest(self.manifest)[0]
        self.assertEqual(record.evidence["ihc"], [0, None, None, None, None])

    def test_feature_escape_and_bad_dimension_rejected(self):
        with self.assertRaises(ValueError):
            load_bag(self.records[0], 11)
        payload = read_json(self.manifest)
        payload["patients"][0]["features"] = ["../outside.h5"]
        write_json(payload, self.manifest)
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_unlabeled_prediction_manifest(self):
        payload = read_json(self.manifest)
        for row in payload["patients"]:
            row.pop("label")
        write_json(payload, self.manifest)
        self.assertTrue(all(r.label is None for r in load_manifest(self.manifest, require_labels=False)))
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)
