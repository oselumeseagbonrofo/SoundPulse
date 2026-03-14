# SoundPulse ML Workspace

This workspace owns the ML-first implementation phases for SoundPulse.

Current scope:

- M1: workspace bootstrap.
- M2: shared-contract freeze and validation.
- M3: phase-1 dataset ingestion, manifest generation, and split integrity tooling.
- M4: reusable preprocessing/feature pipeline, deterministic checks, and parity drift tests.

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
- `python scripts/generate_preprocessing_features.py --job-config ml/config/preprocessing/feature_generation.default.json`: generate fixed-window log-mel features from manifest rows using the shared preprocessing contract.
- `python scripts/validate_preprocessing_determinism.py --job-config ml/config/preprocessing/determinism.default.json`: verify repeated preprocessing runs are deterministic within tolerance (`atol=1e-7`, `rtol=1e-6`).
- `python scripts/run_preprocessing_parity.py --job-config ml/config/preprocessing/parity.default.json`: compare current preprocessing outputs against committed fixture baselines and fail on drift beyond tolerance (`atol=1e-6`, `rtol=1e-5`).
- `python scripts/build_export_package.py`: build the M8 runtime-ready export package under `ml/export/artifacts/generated/release_v1.0.0/` with manifest, checksums, model metadata, and model cards.
- `python scripts/validate_export_artifacts.py`: validate export package integrity, checksums, schema conformance shape, threshold payload/version consistency, and metadata consistency.
- `python scripts/run_release_gate.py --preprocessing-parity-status not_run`: run M9 release-gate checks and emit `ml/evaluation/release_gate_summary.json` plus `ml/evaluation/release_gate_report.md`.
- Open notebook 1 in `ml/notebooks/` to start dataset curation, then run notebook 2 for the TinyCNN baseline and notebook 3 for Kaggle-hosted YAMNet.

Implementation notes:

- Shared contracts live in `packages/shared_contracts/v1/` and should be treated as canonical.
- Runtime and app work should not invent parallel copies of labels, thresholds, or preprocessing settings.
- Dataset manifests should carry canonical `class_slug` values from the shared contracts and stable `source_group_id` values for leakage checks.
- M4 preprocessing is strictly parameterized by `packages/shared_contracts/v1/preprocessing_config.json`; feature tensors are shape-validated against `(n_mels, time_frames) = (64, 97)` per 1-second window.
- M4 parity fixtures live at `ml/evaluation/fixtures/preprocessing_parity_baseline.v1.json`, with latest run outputs in `ml/evaluation/preprocessing_parity_report.json` and `ml/evaluation/preprocessing_determinism_report.json`.
- Phase 1 uses ESC-50 and FSD50K only. AudioSet remains disabled until the local download is complete.
- The phase-1 background class uses the null-background FSD50K classes listed in `sound_data_classes.md`: dishes and pots and pans, frying food, music, and mechanical fan.
