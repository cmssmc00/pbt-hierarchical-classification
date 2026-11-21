"""Command-line workflows for private local datasets and synthetic demos."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .data import load_bag, load_manifest
from .io import read_json, write_json
from .pipeline import run, validate_config
from .splits import make_splits, validate_splits


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pbt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "make-splits", "run", "run-suite", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--feature-root", type=Path)
        if name in ("validate", "run"):
            p.add_argument("--config", required=True, type=Path)
        if name in ("validate", "run", "run-suite"):
            p.add_argument("--splits", required=name != "validate", type=Path)
        if name != "validate":
            p.add_argument("--output", required=True, type=Path)
        if name == "validate":
            p.add_argument("--check-features", action="store_true")
        if name == "make-splits":
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--validation-fraction", type=float, default=0.125)
            p.add_argument("--seed", type=int, default=101)
        if name == "run-suite":
            p.add_argument("--configs", nargs="+", required=True, type=Path)
        if name == "predict":
            p.add_argument("--run-dir", required=True, type=Path)
            p.add_argument("--fold", type=int, default=0)
    demo = sub.add_parser("demo")
    demo.add_argument("--config", required=True, type=Path)
    demo.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "demo":
        from .synthetic import generate

        config = deepcopy(read_json(args.config))
        config["data"].update(feature_dim=8, max_train_patches=12)
        config["model"].update(projection_dim=8, attention_dim=4, fusion_dim=8)
        config["training"].update(epochs=2, schedule_epochs=5, candidate_epochs=[1, 2], patience=1)
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = generate(args.output / "synthetic_data")
        records = load_manifest(manifest)
        split_path = args.output / "synthetic_splits.json"
        write_json(make_splits(records, n_folds=3), split_path)
        result = run(config, manifest, split_path, args.output / "run")
        print("SYNTHETIC SOFTWARE CHECK ONLY — these metrics are not clinical results.")
    elif args.command == "run":
        result = run(read_json(args.config), args.manifest, args.splits, args.output, args.feature_root)
    elif args.command == "run-suite":
        # Intentionally synchronous: finish every fold of a variant before the next.
        configs = [validate_config(read_json(p)) for p in args.configs]
        names = [c["name"] for c in configs]
        if len(names) != len(set(names)) or any(Path(n).name != n or n in (".", "..") for n in names):
            raise ValueError("Variant names must be unique simple directory names")
        args.output.mkdir(parents=True, exist_ok=False)
        result = {}
        for config in configs:
            result[config["name"]] = run(
                config, args.manifest, args.splits, args.output / config["name"], args.feature_root
            )
        write_json(result, args.output / "comparison.json")
    elif args.command == "predict":
        from .inference import predict_saved

        result = predict_saved(args.run_dir, args.manifest, args.output, args.fold, args.feature_root)
        print(f"Saved predictions for {len(next(iter(result.values())))} patients.")
        return
    else:
        records = load_manifest(args.manifest, args.feature_root)
        if args.command == "make-splits":
            if args.output.exists():
                raise FileExistsError("Split output already exists")
            result = make_splits(records, args.folds, args.validation_fraction, args.seed)
            write_json(result, args.output)
            print(f"Saved {len(result['folds'])} patient-disjoint folds.")
            return
        config = validate_config(read_json(args.config))
        if args.splits:
            validate_splits(records, read_json(args.splits))
        if args.check_features:
            for record in records:
                load_bag(record, config["data"]["feature_dim"], config["data"]["feature_key"])
        result = {
            "patients": len(records),
            "slides": sum(len(r.features) for r in records),
            "features_checked": args.check_features,
            "training_started": False,
        }
    if "modes" in result:
        print(
            json.dumps(
                {
                    m: {
                        k: s[k]
                        for k in ("correct", "num_patients", "accuracy", "balanced_accuracy", "macro_f1")
                    }
                    for m, s in result["modes"].items()
                },
                indent=2,
            )
        )
    elif args.command == "validate":
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
