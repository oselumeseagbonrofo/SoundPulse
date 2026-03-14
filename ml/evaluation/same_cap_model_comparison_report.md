# Same-Cap Backbone Comparison

## Protocol

Both backbones were re-evaluated with prototype classification on the same capped manifest protocol.

- Shared caps: train=128, validation=64, test=64
- Test support per model: 528
- Scoring method: nearest-prototype inference using train-split centroids

## Aggregate comparison

| Metric | TinyCNN | YAMNet | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.2936 | 0.5076 | +0.2140 |
| Macro precision | 0.2738 | 0.5673 | +0.2935 |
| Macro recall | 0.3200 | 0.5432 | +0.2232 |
| Macro F1 | 0.2618 | 0.5181 | +0.2563 |
| Weighted precision | 0.2930 | 0.5820 | +0.2890 |
| Weighted recall | 0.2936 | 0.5076 | +0.2140 |
| Weighted F1 | 0.2657 | 0.5108 | +0.2451 |

## Per-class recall

| Class | TinyCNN | YAMNet | Delta |
| --- | ---: | ---: | ---: |
| fire_alarm | 0.0312 | 0.2969 | +0.2657 |
| clock_alarm | 0.5000 | 0.8750 | +0.3750 |
| door_knock | 0.6562 | 0.2500 | -0.4062 |
| doorbell | 0.2941 | 0.3824 | +0.0883 |
| crying_baby | 0.5000 | 0.7368 | +0.2368 |
| water_running | 0.4375 | 0.7969 | +0.3594 |
| dog_barking | 0.4219 | 0.5625 | +0.1406 |
| door_wood_creaking | 0.1875 | 0.4688 | +0.2813 |
| speech | 0.0938 | 0.8281 | +0.7343 |
| background | 0.0781 | 0.2344 | +0.1563 |

## Conclusion

YAMNet remains the stronger backbone under the stricter same-cap prototype-evaluation protocol.

- TinyCNN predictions: ml\evaluation\tinycnn_same_cap_predictions.csv
- YAMNet predictions: ml\evaluation\yamnet_same_cap_predictions.csv
