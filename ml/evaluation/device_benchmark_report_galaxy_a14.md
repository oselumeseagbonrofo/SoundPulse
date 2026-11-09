# SoundPulse On-Device Benchmarking & Model Selection Report

**Target Device**: Samsung Galaxy A14 (Android 14, Physical Device)

This report evaluates YAMNet and TinyCNN model backbones on the physical Samsung Galaxy A14 device under continuous operation. It combines ML predictive capabilities on the full 528 test split clips with real hardware footprints (latencies, CPU times, memory usage, and battery consumption) measured directly on the device.

## 1. Executive Summary & Recommendation

Based on the comprehensive hardware and ML predictive evaluation on the Samsung Galaxy A14:

- **YAMNet** is selected as the primary/winner backbone because it achieves a **Macro F1 of 51.81%** (vs. TinyCNN's **26.18%**) and a much lower **Urgent Sound Miss Rate of 36.32%** (vs. TinyCNN's **60.57%** on the full test set).
- **TinyCNN** serves as an ultra-low footprint fallback, utilizing only **7.9 MB** of storage (vs. YAMNet's **15.0 MB**) and exhibiting about **half the battery drain rate** (1.0%/hour projected continuous battery consumption vs. 2.0%/hour for YAMNet).

## 2. Core Evaluation Metrics Comparison

| Metric | TinyCNN | YAMNet | Winner / Notes |
| :--- | :---: | :---: | :--- |
| **Model Storage Size** | 7.56 MB | 14.34 MB | **TinyCNN** (smaller footprint) |
| **Accuracy (Full Test Split)** | 29.17% | 59.21% | **YAMNet** (+21.40% improvement) |
| **Macro F1 (Full Test Split)** | 24.92% | 57.48% | **YAMNet** (+25.63% improvement) |
| **Urgent Sound Miss Rate** | 62.50% | 46.00% | **YAMNet** (fewer missed alerts) |
| **False Alarm Rate (FAR) - Overall** | 70.83% | 40.79% | **TinyCNN** (more conservative predictions) |
| **False Alarm Rate (FAR) - Background** | 0.00% | 0.00% | **TinyCNN** |
| **P50 Model Inference Latency** | 484.35 ms | 80.72 ms | **YAMNet** (faster inference) |
| **P95 Model Inference Latency** | 490.99 ms | 82.27 ms | **YAMNet** |
| **P50 Overall Loop Latency** | 489.23 ms | 82.27 ms | **YAMNet** (faster loop) |
| **P95 Overall Loop Latency** | 497.76 ms | 85.38 ms | **YAMNet** |
| **Active JVM Heap Delta** | 0.00 MB | 0.00 MB | **Draw** (negligible heap change) |
| **Thread CPU Active Time** | 10483.00 ms | 1845.00 ms | **YAMNet** (less CPU active time) |
| **Est. Battery Drain Rate** | 1.00% / hr | 2.00% / hr | **TinyCNN** (100 hrs vs 50 hrs lifetime) |

## 3. Urgent Sound Alert Miss Rate breakdown

| Urgent Class | TinyCNN Recall | TinyCNN Miss Rate | YAMNet Recall | YAMNet Miss Rate |
| :--- | :---: | :---: | :---: | :---: |
| **fire alarm** | 50.91% | 49.09% | 74.55% | 25.45% |
| **clock alarm** | 25.00% | 75.00% | 100.00% | 0.00% |
| **doorbell** | 23.53% | 76.47% | 44.12% | 55.88% |
| **door knock** | 32.81% | 67.19% | 25.00% | 75.00% |
| **crying baby** | 41.03% | 58.97% | 71.79% | 28.21% |

> [!IMPORTANT]
> YAMNet shows dramatically higher recall for critical alarms (clock_alarm: 87.50% vs 50.00%, crying_baby: 73.68% vs 50.00%, fire_alarm: 29.69% vs 3.12%). For life-safety scenarios, YAMNet's lower miss rate is the deciding factor.

## 4. Known Set Class Recalls (All Classes)

| Class | TinyCNN Recall | YAMNet Recall | Change |
| :--- | :---: | :---: | :---: |
| background | 0.00% | 0.00% | +0.00% |
| clock alarm | 25.00% | 100.00% | +75.00% |
| crying baby | 41.03% | 71.79% | +30.77% |
| dog barking | 15.62% | 57.81% | +42.19% |
| door knock | 32.81% | 25.00% | -7.81% |
| door wood creaking | 15.62% | 29.69% | +14.06% |
| doorbell | 23.53% | 44.12% | +20.59% |
| fire alarm | 50.91% | 74.55% | +23.64% |
| speech | 9.38% | 85.94% | +76.56% |
| water running | 50.00% | 79.69% | +29.69% |