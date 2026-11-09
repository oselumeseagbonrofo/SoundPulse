# Same-Cap Backbone Comparison

## Protocol

Both backbones were re-evaluated with prototype classification on the same capped manifest protocol.

- Shared caps: train=128, validation=64, test=64
- Test support per model: 510
- Scoring method: nearest-prototype inference using L2-normalized train-split centroids

## Aggregate comparison

| Metric | TinyCNN | YAMNet | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.6706 | 0.7451 | +0.0745 |
| Macro precision | 0.7161 | 0.7411 | +0.0250 |
| Macro recall | 0.6979 | 0.7658 | +0.0679 |
| Macro F1 | 0.6636 | 0.7363 | +0.0727 |
| Weighted precision | 0.7552 | 0.7690 | +0.0138 |
| Weighted recall | 0.6706 | 0.7451 | +0.0745 |
| Weighted F1 | 0.6741 | 0.7450 | +0.0709 |

## Per-class recall

| Class | TinyCNN | YAMNet | Delta |
| --- | ---: | ---: | ---: |
| fire_alarm | 0.7091 | 0.7455 | +0.0364 |
| clock_alarm | 1.0000 | 1.0000 | +0.0000 |
| door_knock | 0.8281 | 0.8281 | +0.0000 |
| doorbell | 0.5588 | 0.6471 | +0.0883 |
| crying_baby | 0.8205 | 0.7949 | -0.0256 |
| water_running | 0.8750 | 0.8750 | +0.0000 |
| dog_barking | 0.4844 | 0.7500 | +0.2656 |
| door_wood_creaking | 0.7656 | 0.7188 | -0.0468 |
| speech | 0.4375 | 0.4844 | +0.0469 |
| background | 0.5000 | 0.8148 | +0.3148 |

## Conclusion

YAMNet remains the stronger backbone under the stricter same-cap prototype-evaluation protocol.

- TinyCNN predictions: ml\evaluation\tinycnn_same_cap_predictions.csv
- YAMNet predictions: ml\evaluation\yamnet_same_cap_predictions.csv
