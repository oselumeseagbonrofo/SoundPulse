# YAMNet Threshold Calibration Report

## Protocol

Threshold calibration was rerun on the YAMNet prototype path using out-of-scope FSD50K clips rather than the earlier ESC-50-only proxy set.

- Known caps: train=128, validation=64, test=64
- Unknown calibration proxy: FSD50K dev out-of-scope, support=1024, categories=86
- Unknown holdout proxy: FSD50K eval out-of-scope, support=1024, categories=93
- Unknown cap policy: max_total=1024/1024, max_per_category=12
- Urgent-release metric classes: fire_alarm, clock_alarm, door_knock, doorbell

## Selected thresholds

- Distance threshold: 182.0886
- Unknown ratio threshold: 0.7160
- Loudness floor dBFS: -60.0

## Operating-point comparison on holdout

| Operating point | Accuracy | Macro F1 | Urgent recall | Unknown non-alert | Unknown recall | Unknown precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected_release_candidate | 0.4364 | 0.5003 | 0.4125 | 0.7061 | 0.7002 | 0.7853 |
| best_known_retention | 0.5636 | 0.5717 | 0.5497 | 0.2920 | 0.2861 | 0.8162 |
| best_unknown_rejection | 0.2390 | 0.2879 | 0.1096 | 0.8789 | 0.6621 | 0.7410 |

## Baseline vs selected on known test

| Metric | Baseline | Selected | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5921 | 0.4364 | -0.1557 |
| Macro F1 | 0.5748 | 0.5003 | -0.0745 |
| Macro recall | 0.5686 | 0.4142 | -0.1544 |
| Urgent recall | 0.6092 | 0.4125 | -0.1967 |
| Background non-alert rate | 0.0000 | 0.0000 | +0.0000 |

## Selected open-set holdout results

- Unknown precision: 0.7853
- Unknown recall: 0.7002
- Unknown non-alert rate: 0.7061
- Unknown false-alert rate: 0.2939

## Top sweep candidates on calibration

| Candidate ID | Distance | Ratio | Loudness | Balanced | Val macro F1 | Val urgent recall | Unknown non-alert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 841 | 182.0886 | 0.7160 | -60.0 | 0.5713 | 0.5281 | 0.3946 | 0.7080 |
| 842 | 182.0886 | 0.7160 | -58.3 | 0.5704 | 0.5281 | 0.3946 | 0.7080 |
| 843 | 182.0886 | 0.7160 | -55.0 | 0.5691 | 0.5257 | 0.3946 | 0.7129 |
| 721 | 124.7662 | 0.7160 | -60.0 | 0.5684 | 0.5241 | 0.3848 | 0.7139 |
| 722 | 124.7662 | 0.7160 | -58.3 | 0.5674 | 0.5241 | 0.3848 | 0.7139 |
| 723 | 124.7662 | 0.7160 | -55.0 | 0.5659 | 0.5217 | 0.3848 | 0.7178 |
| 844 | 182.0886 | 0.7160 | -52.0 | 0.5657 | 0.5245 | 0.3897 | 0.7139 |
| 601 | 87.7669 | 0.7160 | -60.0 | 0.5651 | 0.5132 | 0.3701 | 0.7334 |
| 845 | 182.0886 | 0.7160 | -50.0 | 0.5646 | 0.5245 | 0.3897 | 0.7188 |
| 602 | 87.7669 | 0.7160 | -58.3 | 0.5641 | 0.5132 | 0.3701 | 0.7334 |

## Holdout routing breakdown

- Selected known-test routed labels: {"clock_alarm": 7, "crying_baby": 22, "dog_barking": 34, "door_knock": 30, "door_wood_creaking": 12, "doorbell": 8, "fire_alarm": 28, "speech": 77, "unknown": 196, "water_running": 42}
- Selected unknown-holdout routed labels: {"background": 6, "crying_baby": 21, "door_knock": 65, "door_wood_creaking": 36, "doorbell": 29, "fire_alarm": 7, "speech": 126, "unknown": 717, "water_running": 17}

## Assessment

This remains an M7 calibration artifact rather than a ship-default threshold package. The difference now is that the proxy unknown pool is materially harder and more realistic, and the sweep export makes the tradeoff between known retention and unknown rejection explicit rather than hiding it behind one selected point.

Hard floors applied during selection: urgent recall ≥ 0.1579, macro F1 ≥ 0.25.

The selected operating point does not yet meet the PRD urgent-recall target of 0.90. Current urgent recall is 0.4125.

## Top holdout false alerts

- Female_singing -> speech: 10
- Bass_guitar -> speech: 7
- Cheering -> speech: 6
- Human_group_actions -> speech: 6
- Electric_guitar -> speech: 6
- Acoustic_guitar -> speech: 5
- Fill_(with_liquid) -> water_running: 5
- Crowd -> speech: 5
- Crash_cymbal -> door_knock: 5
- Cupboard_open_or_close -> door_knock: 5
- Gasp -> crying_baby: 5
- Frying_(food) -> water_running: 5
- Giggle -> speech: 5
- Human_voice -> crying_baby: 5
- Cough -> crying_baby: 4
