# SoundPulse ML Workspace

This workspace owns the ML-first implementation phases for SoundPulse.

Current scope:

- M1: workspace bootstrap.
- M2: shared-contract freeze and validation.
- M3: phase-1 dataset ingestion, manifest generation, and split integrity tooling.

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
- `python scripts/build_phase1_manifest.py`: generate the phase-1 curated manifest, split summary CSV, and class-to-source mapping from the local raw datasets.
- `python scripts/validate_dataset_manifest.py ml/data/curated/soundpulse_manifest.v1.json`: validate the generated dataset manifest structure and split integrity.
- Open notebook 1 in `ml/notebooks/` to start dataset curation, then run notebook 2 for the TinyCNN baseline and notebook 3 for Kaggle-hosted YAMNet.

Implementation notes:

- Shared contracts live in `packages/shared_contracts/v1/` and should be treated as canonical.
- Runtime and app work should not invent parallel copies of labels, thresholds, or preprocessing settings.
- Dataset manifests should carry canonical `class_slug` values from the shared contracts and stable `source_group_id` values for leakage checks.
- Phase 1 uses ESC-50 and FSD50K only. AudioSet remains disabled until the local download is complete.
- The phase-1 background class uses the null-background FSD50K classes listed in `sound_data_classes.md`: dishes and pots and pans, frying food, music, and mechanical fan.
