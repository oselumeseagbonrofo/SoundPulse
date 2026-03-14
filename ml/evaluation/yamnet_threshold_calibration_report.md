# YAMNet Threshold Calibration Report

## Protocol

Threshold calibration was rerun on the YAMNet prototype path using out-of-scope FSD50K clips rather than the earlier ESC-50-only proxy set.

- Known caps: train=128, validation=64, test=64
- Unknown calibration proxy: FSD50K dev out-of-scope, support=1024, categories=86
- Unknown holdout proxy: FSD50K eval out-of-scope, support=1024, categories=93
- Unknown cap policy: max_total=1024/1024, max_per_category=12
- Urgent-release metric classes: fire_alarm, clock_alarm, door_knock, doorbell

## Selected thresholds

- Distance threshold: 175.8489
- Unknown ratio threshold: 0.8858
- Loudness floor dBFS: -58.7

## Operating-point comparison on holdout

| Operating point | Accuracy | Macro F1 | Urgent recall | Unknown non-alert | Unknown recall | Unknown precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected_release_candidate | 0.4223 | 0.4928 | 0.3660 | 0.5342 | 0.4736 | 0.7566 |
| best_balanced_score | 0.3883 | 0.4728 | 0.3254 | 0.6553 | 0.6094 | 0.7491 |
| best_known_retention | 0.4962 | 0.5221 | 0.4164 | 0.2129 | 0.1035 | 0.7260 |
| best_unknown_rejection | 0.2121 | 0.2652 | 0.0460 | 0.9072 | 0.6846 | 0.7045 |

## Baseline vs selected on known test

| Metric | Baseline | Selected | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5076 | 0.4223 | -0.0853 |
| Macro F1 | 0.5181 | 0.4928 | -0.0253 |
| Macro recall | 0.5432 | 0.4690 | -0.0742 |
| Urgent recall | 0.4511 | 0.3660 | -0.0851 |
| Background non-alert rate | 0.2344 | 0.4375 | +0.2031 |

## Selected open-set holdout results

- Unknown precision: 0.7566
- Unknown recall: 0.4736
- Unknown non-alert rate: 0.5342
- Unknown false-alert rate: 0.4658

## Top sweep candidates on calibration

| Candidate ID | Distance | Ratio | Loudness | Balanced | Val macro F1 | Val urgent recall | Unknown non-alert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 732 | 119.4214 | 0.8425 | -58.7 | 0.4971 | 0.5046 | 0.2544 | 0.6602 |
| 731 | 119.4214 | 0.8425 | -60.0 | 0.4966 | 0.5028 | 0.2544 | 0.6582 |
| 733 | 119.4214 | 0.8425 | -55.0 | 0.4959 | 0.5042 | 0.2544 | 0.6631 |
| 722 | 119.4214 | 0.7839 | -58.7 | 0.4941 | 0.4748 | 0.1996 | 0.7471 |
| 742 | 119.4214 | 0.8500 | -58.7 | 0.4939 | 0.5032 | 0.2544 | 0.6504 |
| 721 | 119.4214 | 0.7839 | -60.0 | 0.4937 | 0.4728 | 0.1996 | 0.7461 |
| 741 | 119.4214 | 0.8500 | -60.0 | 0.4933 | 0.5013 | 0.2544 | 0.6484 |
| 723 | 119.4214 | 0.7839 | -55.0 | 0.4928 | 0.4744 | 0.1996 | 0.7500 |
| 743 | 119.4214 | 0.8500 | -55.0 | 0.4926 | 0.5028 | 0.2544 | 0.6533 |
| 734 | 119.4214 | 0.8425 | -52.0 | 0.4922 | 0.5016 | 0.2494 | 0.6650 |

## Holdout routing breakdown

- Selected known-test routed labels: {"background": 19, "clock_alarm": 14, "crying_baby": 28, "dog_barking": 36, "door_knock": 42, "door_wood_creaking": 33, "doorbell": 33, "fire_alarm": 16, "speech": 101, "unknown": 156, "water_running": 50}
- Selected unknown-holdout routed labels: {"background": 62, "crying_baby": 31, "door_knock": 69, "door_wood_creaking": 107, "doorbell": 71, "fire_alarm": 12, "speech": 157, "unknown": 485, "water_running": 30}

## Assessment

This remains an M7 calibration artifact rather than a ship-default threshold package. The difference now is that the proxy unknown pool is materially harder and more realistic, and the sweep export makes the tradeoff between known retention and unknown rejection explicit rather than hiding it behind one selected point.

Hard floors applied during selection: urgent recall ≥ 0.1265, macro F1 ≥ 0.25.

The selected operating point does not yet meet the PRD urgent-recall target of 0.90. Current urgent recall is 0.3660.

## Top holdout false alerts

- Female_singing -> speech: 12
- Coin_(dropping) -> door_wood_creaking: 9
- Cough -> crying_baby: 8
- Crowd -> speech: 8
- Cheering -> speech: 7
- Bass_guitar -> speech: 7
- Crushing -> door_wood_creaking: 7
- Fart -> door_wood_creaking: 7
- Human_group_actions -> speech: 7
- Engine_starting -> door_wood_creaking: 6
- Drawer_open_or_close -> door_wood_creaking: 6
- Car -> door_wood_creaking: 6
- Car_passing_by -> speech: 6
- Gasp -> crying_baby: 6
- Human_voice -> crying_baby: 6
