# YAMNet Kaggle Evaluation Report

## Run provenance

- Source notebook: `ml/notebooks/03_yamnet_kaggle_pipeline.ipynb`
- Artifact summary: `ml/export/artifacts/generated/yamnet_kaggle/artifact_summary.json`
- Prediction export: `ml/export/artifacts/generated/yamnet_kaggle/prototype_predictions.csv`
- Model artifact directory: `ml/export/artifacts/generated/yamnet_kaggle/`
- Evaluation status: derived from saved notebook outputs and exported predictions CSV

## Evaluation setup captured in notebook output

- Backbone: `google/yamnet/tensorFlow2/yamnet`
- Class order: `fire_alarm`, `clock_alarm`, `door_knock`, `doorbell`, `crying_baby`, `water_running`, `dog_barking`, `door_wood_creaking`, `speech`, `background`
- Embedding dimension: `1024`
- Test support: `528`
- Inference approach: frozen YAMNet embeddings with prototype classification from train-set class centroids

## Test classification report

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| fire_alarm | 0.49 | 0.30 | 0.37 | 64 |
| clock_alarm | 0.39 | 0.88 | 0.54 | 8 |
| door_knock | 0.25 | 0.25 | 0.25 | 64 |
| doorbell | 0.30 | 0.38 | 0.33 | 34 |
| crying_baby | 0.97 | 0.74 | 0.84 | 38 |
| water_running | 0.93 | 0.80 | 0.86 | 64 |
| dog_barking | 1.00 | 0.56 | 0.72 | 64 |
| door_wood_creaking | 0.37 | 0.47 | 0.41 | 64 |
| speech | 0.38 | 0.83 | 0.52 | 64 |
| background | 0.60 | 0.23 | 0.34 | 64 |

## Aggregate metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.51 |
| Macro precision | 0.57 |
| Macro recall | 0.54 |
| Macro F1 | 0.52 |
| Weighted precision | 0.58 |
| Weighted recall | 0.51 |
| Weighted F1 | 0.51 |

## Interpretation

This YAMNet prototype baseline is materially stronger than the recorded TinyCNN baseline on the current saved runs. It reaches usable recall on most target classes and performs especially well on `crying_baby`, `water_running`, `dog_barking`, and `speech`.

The weakest remaining areas in this run are `door_knock`, `doorbell`, `door_wood_creaking`, and `background`, which indicates threshold calibration and open-set handling are still needed before release decisions.