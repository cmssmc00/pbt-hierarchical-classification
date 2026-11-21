# Experiments and provenance

## Runnable presets

| Configuration | Purpose |
|---|---|
| `binary_fixed25.json` | C1–C9 versus C10, image-only MIL |
| `flat10_fixed25.json` | Direct ten-class image baseline |
| `flat6_fixed25.json` | Direct six-group baseline with four evidence modes |
| `hard6to10_fixed25.json` | Six-to-ten hard hierarchy, including child clinical evidence |
| `hard6to10_stage1_clinical_only.json` | Same hierarchy, clinical cues at root only; IHC at both stages |
| `joint_hard_fixed25.json` | Original joint three-group hard model (B) |
| `joint_global_fixed25.json` | Original joint cross-group global head (C) |
| `joint_global_epoch_selection.json` | C with candidate epoch selection and fresh refit (D) |

Ten additional `flat6_01_*` through `flat6_10_*` configurations represent the
serial evidence-suite variants: fixed 25 epochs; early stopping; label smoothing;
AdamW; increased dropout; a smaller network; tempered class weights; focal loss;
a lower learning rate; and more sampled training patches. The latter variants
use the early-stopping protocol. `flat6_fixed25` and `flat6_01_fixed25` describe
the same baseline and are both provided for convenient standalone/suite use.

Run any selection of presets with `pbt run-suite --configs ...`. The CLI keeps
variant and fold execution serial. Configurations are JSON so every effective
hyperparameter is visible and saved with the local run.

## Consolidation of the source project

This release extracts model layers, joint-model losses, three-group decoders,
root evidence cues, child IHC fusion, child clinical fusion, and early-stopping
logic from the corresponding internal experiment families. Shared classes are
defined once. The public data loader, CLI and orchestration replace the original
machine-bound experiment wrappers.

Meaningful changes from the private archive:

- A public C-number manifest replaces team-specific workbooks and JSON adapters.
- All patient counts, training steps and batch counts are computed from inputs.
- Private source hashes, identifiers, exclusions and fixed fold membership are
  not distributed.
- Standalone hard-hierarchy runs train their own root model. The historical
  follow-up instead reused already frozen root checkpoints. Supply the same
  data/configuration to study the same protocol, but do not claim identical
  numerical results across this refactor or different environments.
- The new split generator creates fresh partitions; it does not reconstruct the
  private study's folds.
- Class display names have spelling normalized; C1–C10 ordering is unchanged.
- Child clinical probability calculations use a small numerical floor only for
  finite internal log-score updates. Final hard-routing NLL remains explicitly
  infinite when the true class is masked out.

The original study artifacts remain separate from this release. No real-data
training is needed or launched to validate publication of this repository.

## Historical research modules

`pbt.models.research_variants` retains the residual-correction and class-specific
attention architectures. They are reference components rather than additional
CLI presets; their original search orchestration is not part of this release.

`pbt.legacy.models` and `pbt.legacy.ced` retain the earlier multimodal Teacher,
CREDIT Student and credal-distillation operations. The main CLI uses supervised
Teachers only. See [legacy_credit.py](../examples/legacy_credit.py) for a
synthetic forward/backward example. The older MAMMOTH implementation is not
vendored; see its [upstream repository](https://github.com/mahmoodlab/MAMMOTH)
if studying that separate encoder branch.

Repeated binary tuning wrappers, spreadsheet parsers, internal audit reports,
operational queue monitors and report-generation scripts are not copied into
the public package. Their reusable model/training mechanisms are represented
by the shared components and presets rather than duplicate source trees.

## Reproducibility limits

The default seed is 101. Neural fitting, evidence fitting, and validation
selection have separate roles; the outer test partition only evaluates frozen
choices. Different TensorFlow versions/hardware and different split membership
can change results. Synthetic tests verify implementation behavior, not clinical
effectiveness. The repository does not claim independent external validation.
