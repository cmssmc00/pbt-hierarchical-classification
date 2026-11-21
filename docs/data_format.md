# Local data format

## Manifest

The input is a JSON object with `schema_version: 1` and a `patients` list. A
fictional example is in [manifest.example.json](../examples/manifest.example.json).
That template is documentation; its feature file does not exist. Use `pbt demo`
to generate executable synthetic examples.

| Field | Meaning |
|---|---|
| `patient_id` | Locally unique patient key; never commit real keys |
| `label` | `C1` to `C10`; required for training/evaluation, optional for prediction |
| `features` | Nonempty list of relative `.h5`/`.hdf5` paths, one per slide |
| `age_years` | Observed nonnegative age or `null` |
| `location` | One vocabulary entry below, or `null` |
| `ihc` | Explicit `POS`/`NEG` observations; omitted markers or `null` mean missing |

Feature paths resolve against the manifest's directory, or the directory given
with `--feature-root`. They must remain inside that directory, including after
symlink resolution. A feature file cannot be shared by different patients or
listed twice. Patient bags combine all slides before sampling patches.

Each feature file contains a rank-two `features` dataset, `[N, D]`. Choose `D`
with `data.feature_dim`; do not silently substitute embeddings from a different
encoder with another dimension or patching protocol. Patch coordinates may be
present but are not used by the current attention model. Empty and nonfinite
bags are rejected. Training samples up to the configured patch cap; evaluation
uses all available patches.

## Labels

| Label | Category |
|---|---|
| C1 | Low-grade glial tumors |
| C2 | High-grade glial tumors |
| C3 | Glioneuronal tumors |
| C4 | High-grade embryonal tumors |
| C5 | Ependymal tumors |
| C6 | Tumors of uncertain histogenesis |
| C7 | Choroid plexus tumors |
| C8 | Pineal tumors |
| C9 | Non-glial tumors |
| C10 | Non-tumor |

Public display names normalize spelling; the C-number mapping preserves the
source project's label order. Binary targets use C1–C9 versus C10, never an
unrelated tumor flag. Internal class and group indices start at zero.

## Location vocabulary

The index order is part of the evidence rules:

0. `Cerebral hemispheric/lobar`
1. `Cerebellum`
2. `Ventricular/periventricular`
3. `Brainstem/tectum`
4. `Thalamus/deep gray`
5. `Cerebellopontine angle`
6. `Posterior fossa, NOS`
7. `Pineal region`
8. `Sellar/suprasellar/optic`
9. `Spinal cord`
10. `Unknown`

`Unknown` and `null` both disable location evidence. There is no default site
imputation. Age and location can be missing independently.

## IHC

Marker order is `GFAP`, `Synaptophysin`, `INI1`, `H3K27M`, `ALK1`. The loader
accepts only explicit `POS`, `NEG`, or `null` calls. Missing is not negative.
Do not infer a marker result from the diagnosis, overwrite unknowns with a
class's typical result, or populate this schema with diagnosis-derived calls.
If the source distinguishes measured and inferred calls, retain only measured
calls during local conversion. The code cannot recover observation provenance
once it has been removed upstream.

## Split files

```json
{
  "schema_version": 1,
  "folds": [
    {
      "fold": 0,
      "train": ["synthetic_training_patient"],
      "validation": ["synthetic_validation_patient"],
      "test": ["synthetic_test_patient"]
    }
  ]
}
```

This snippet shows the structure, not a complete runnable split. Real split
files require at least two folds. In each fold, the three partitions must cover
the manifest exactly once; across folds, each patient must occur in outer test
exactly once. The generator requires enough examples to retain every source
class in training and validation. For very rare classes, design and document an
appropriate partition locally instead of weakening leakage checks.

The public schema uses `validation`, rather than the older internal `eval` key.
No automatic conversion reads a private workbook or original study manifest.
