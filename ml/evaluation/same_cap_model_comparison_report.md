# Same-Cap Backbone Comparison

## Protocol

All 5 backbones were evaluated with prototype classification on the same capped manifest protocol.

- Shared caps: train=128, validation=64, test=64
- Test support per model: 510
- Scoring method: nearest-prototype inference using L2-normalized train-split centroids

## Aggregate comparison

| Metric | TinyCNN | YAMNet | EfficientAT (mn10) | PANNs MobileNetV1 | CED-Tiny | Best Model |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Accuracy | 0.6706 | 0.7451 | 0.4549 | 0.7686 | **0.8157** | CED-Tiny |
| Macro precision | 0.7161 | 0.7411 | 0.6288 | 0.7760 | **0.8098** | CED-Tiny |
| Macro recall | 0.6979 | 0.7658 | 0.4959 | 0.7738 | **0.8331** | CED-Tiny |
| Macro F1 | 0.6636 | 0.7363 | 0.4898 | 0.7568 | **0.8107** | CED-Tiny |
| Weighted precision | 0.7552 | 0.7690 | 0.6250 | 0.8039 | **0.8337** | CED-Tiny |
| Weighted recall | 0.6706 | 0.7451 | 0.4549 | 0.7686 | **0.8157** | CED-Tiny |
| Weighted F1 | 0.6741 | 0.7450 | 0.4673 | 0.7733 | **0.8168** | CED-Tiny |

## Per-class recall

| Class | TinyCNN | YAMNet | EfficientAT (mn10) | PANNs MobileNetV1 | CED-Tiny | Best Model |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fire_alarm | 0.7091 | 0.7455 | 0.5455 | 0.7455 | **0.8000** | CED-Tiny |
| clock_alarm | **1.0000** | **1.0000** | **1.0000** | 0.8750 | **1.0000** | TinyCNN, YAMNet, EfficientAT (mn10), CED-Tiny |
| door_knock | 0.8281 | 0.8281 | 0.4375 | 0.8438 | **0.8750** | CED-Tiny |
| doorbell | 0.5588 | 0.6471 | 0.2353 | 0.5882 | **0.7647** | CED-Tiny |
| crying_baby | 0.8205 | 0.7949 | 0.3846 | 0.8462 | **0.8718** | CED-Tiny |
| water_running | 0.8750 | 0.8750 | 0.6250 | 0.8594 | **0.9062** | CED-Tiny |
| dog_barking | 0.4844 | **0.7500** | 0.1562 | 0.7188 | **0.7500** | YAMNet, CED-Tiny |
| door_wood_creaking | 0.7656 | 0.7188 | 0.4375 | **0.8438** | 0.7656 | PANNs MobileNetV1 |
| speech | 0.4375 | 0.4844 | 0.3594 | 0.5469 | **0.6719** | CED-Tiny |
| background | 0.5000 | 0.8148 | 0.7778 | 0.8704 | **0.9259** | CED-Tiny |

## Leaderboard by Macro F1

| Rank | Backbone | Embedding Dim | Macro F1 | Accuracy | Macro Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | CED-Tiny | 192 | 0.8107 | 0.8157 | 0.8331 |
| 2 | PANNs MobileNetV1 | 1024 | 0.7568 | 0.7686 | 0.7738 |
| 3 | YAMNet | 1024 | 0.7363 | 0.7451 | 0.7658 |
| 4 | TinyCNN | 1024 | 0.6636 | 0.6706 | 0.6979 |
| 5 | EfficientAT (mn10) | 960 | 0.4898 | 0.4549 | 0.4959 |

## Conclusion

**CED-Tiny** achieves the highest Macro F1 (`0.8107`) under the same-cap prototype-evaluation protocol.

- TinyCNN predictions: `ml\evaluation\tinycnn_same_cap_predictions.csv`
- YAMNet predictions: `ml\evaluation\yamnet_same_cap_predictions.csv`
- EfficientAT (mn10) predictions: `ml\evaluation\efficientat_mn10_same_cap_predictions.csv`
- PANNs MobileNetV1 predictions: `ml\evaluation\panns_mobilenetv1_same_cap_predictions.csv`
- CED-Tiny predictions: `ml\evaluation\ced_tiny_same_cap_predictions.csv`
