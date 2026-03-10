# SoundPulse ML Data Notes

This directory stores local dataset manifests, split exports, and metadata generated during curation.

Expected contents:

- curated manifest JSON files
- split summaries and leakage reports
- no raw audio committed to git

Conventions:

- `source_group_id` should identify clips that share source provenance so split leakage can be detected.
- `clip_id` must be unique across the full manifest.
- `class_slug` must match `packages/shared_contracts/v1/class_labels.json`.
