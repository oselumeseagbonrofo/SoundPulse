# SoundPulse ML Workspace

This workspace owns the ML-first implementation phases for SoundPulse.

Current scope:

- M1: workspace bootstrap.
- M2: shared-contract freeze and validation.
- M3: dataset manifest validation and split integrity tooling.

Directory layout:

- `config/`: dataset, training, and evaluation config inputs.
- `data/`: local dataset manifests and placeholders. Raw audio is ignored from git.
- `export/`: generated release artifacts and export manifests.
- `evaluation/`: benchmark protocols and generated summaries.
- `notebooks/`: notebook-first data curation, model training, export, and inference workflows.
- `scripts/`: executable entrypoints for validation, data prep, training, and export.
- `src/soundpulse_ml/`: reusable Python modules.

Immediate commands:

- `python -m soundpulse_ml.contracts`: print resolved contract summary.
- `python scripts/validate_contracts.py`: validate required shared contract files exist and are internally consistent.
- `python scripts/validate_dataset_manifest.py ml/config/dataset_manifest.template.json`: validate dataset manifest structure and split integrity.
- Open notebook 1 in `ml/notebooks/` to start dataset curation, then run notebook 2 for MobileNetV4ConvSmall and notebook 3 for Kaggle-hosted YAMNet.

Implementation notes:

- Shared contracts live in `packages/shared_contracts/v1/` and should be treated as canonical.
- Runtime and app work should not invent parallel copies of labels, thresholds, or preprocessing settings.
- Dataset manifests should carry canonical `class_slug` values from the shared contracts and stable `source_group_id` values for leakage checks.
