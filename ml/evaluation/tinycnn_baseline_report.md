# TinyCNN Baseline Evaluation Report

## Run provenance

- Source notebook: `ml/notebooks/02_tinycnn_mobilenetv4convsmall_training.ipynb`
- Export metadata: `ml/export/artifacts/generated/tinycnn_baseline/export_metadata.json`
- Model artifact directory: `ml/export/artifacts/generated/tinycnn_baseline/`
- Evaluation status: derived from saved notebook outputs

## Evaluation setup captured in notebook output

- Class order: `fire_alarm`, `clock_alarm`, `door_knock`, `doorbell`, `crying_baby`, `water_running`, `dog_barking`, `door_wood_creaking`, `speech`, `background`
- Sample rate: `16000 Hz`
- Embedding dimension: `128`
- Input image size: `128 x 128`
- Epochs trained in recorded run: `3`
- Test support: `752`

## Training snapshot

| Epoch | Train accuracy | Train loss | Validation accuracy | Validation loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.4029 | 1.9434 | 0.1750 | 2.2214 |
| 2 | 0.4777 | 1.6979 | 0.1732 | 2.2221 |
| 3 | 0.5143 | 1.5420 | 0.2142 | 2.2826 |

The saved run improved on training accuracy, but validation accuracy remained low and unstable, which is consistent with the weak test-set performance below.

## Test classification report

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| fire_alarm | 0.00 | 0.00 | 0.00 | 96 |
| clock_alarm | 0.00 | 0.00 | 0.00 | 8 |
| door_knock | 0.44 | 0.48 | 0.46 | 96 |
| doorbell | 0.00 | 0.00 | 0.00 | 34 |
| crying_baby | 0.00 | 0.00 | 0.00 | 38 |
| water_running | 0.32 | 0.60 | 0.42 | 96 |
| dog_barking | 0.17 | 0.77 | 0.27 | 96 |
| door_wood_creaking | 0.00 | 0.00 | 0.00 | 96 |
| speech | 0.25 | 0.05 | 0.09 | 96 |
| background | 0.00 | 0.00 | 0.00 | 96 |

## Aggregate metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.24 |
| Macro precision | 0.12 |
| Macro recall | 0.19 |
| Macro F1 | 0.12 |
| Weighted precision | 0.15 |
| Weighted recall | 0.24 |
| Weighted F1 | 0.16 |

## Interpretation

This TinyCNN baseline is not yet a competitive checkpoint. It carries most of its performance from `door_knock`, `water_running`, and `dog_barking`, while completely missing several operationally important classes in this run, including `fire_alarm`, `doorbell`, `crying_baby`, `door_wood_creaking`, and `background`.

The notebook also exported a prototype store and embedding TFLite artifact, so the model path is functional for downstream integration work even though the recorded evaluation quality is weak.