# Data privacy and publication boundary

Only reviewed source files, configuration templates, documentation, synthetic
examples and tests belong in the public repository.

Do not commit patient or slide identifiers, source workbooks, clinical/IHC
records, real embeddings, raw images, actual fold memberships, model weights,
fitted evidence statistics, individual predictions, private reports, account
credentials, or infrastructure paths. De-identified patient-level data still
falls outside this repository's release boundary.

Store authorized data under `private/` or outside this checkout. Generated
artifacts belong under `outputs/`. Both are ignored by Git, as are common image,
feature, checkpoint and tabular-data extensions. `.gitignore` is only one layer:
the release also uses a positive file/path allowlist and scans tracked text.

Before publishing any future change, run:

```bash
python tools/check_public_tree.py
git diff --cached --stat
git diff --cached
```

The check rejects non-allowlisted tracked files, binary payloads, sensitive file
types, machine-specific paths, common credential patterns and identifier-like
strings. It scans the Git index as well as tracked working files. It cannot
prove that arbitrary prose contains no sensitive information; manually review
all added content. Do not bypass a failure by renaming a data file.

The first release starts with a fresh Git history built from the reviewed public
directory. No original repository history, private reports or run folders are
copied into that history. Runtime code performs local computation and has no
data-upload or telemetry client.
