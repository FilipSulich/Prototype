# Isolation Forest for Time-Series Anomaly Detection - Baseline

A baseline implementation of Isolation Forest algorithm for anomaly detection on time-series data from the Numenta Anomaly Benchmark (NAB) dataset.

## Purpose

This implementation serves as a **baseline model** to establish a performance model before implementing more sophisticated deep learning approaches - LSTM Autoencoder. The results demonstrate why traditional methods struggle with temporal anomaly detection.

## Algorithm

- **Method**: Isolation Forest (Liu et al., 2008)
- **Task**: Binary classification (anomaly/normal)
- **Input**: Single-feature time-series data (scaled values)

## Dataset

Uses [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) dataset with:
- Multiple time-series from various domains (real and artificial)
- Labeled anomalies with timestamps
- Extreme class imbalance (~0.03% anomalies)

## Running

To run the baseline model:
```bash
python -m IsolationForest.main
```

The script will:
- Grid search over different n_estimators and random_state values
- Train on all datasets in the NAB dataset
- Output the performance metrics aggregated across all datasets

## Final Performance Metrics

**Micro-Average Performance (Aggregated across all datasets):**
- **Accuracy**: 0.9682 (96.82%)
- **Precision**: 0.0044 (0.44%)
- **Recall**: 0.4250 (42.50%)
- **F1-Score**: 0.0087 (0.87%)

**Confusion Matrix Totals:**
- True Positives: 51
- False Positives: 11,570
- True Negatives: 353,868
- False Negatives: 69

## Analysis

### Why Performance is Poor

1. **Temporal blindness**: Isolation Forest treats each data point independently, ignoring sequential patterns that define time-series anomalies;
2. **Single feature**: Only uses raw scalar values without temporal context;
4. **Extreme dataset imbalance**: 0.03% anomaly rate makes contamination parameter tuning difficult;

### Key Observations

- High accuracy (96.82%) is misleading due to class imbalance
- Very low precision (0.44%) indicates excessive false positives
- Moderate recall (42.50%) shows it detects less than half of true anomalies
- The model essentially defaults to predicting "normal" in most cases

## Future Work

This baseline will be replaced by an LSTM Autoencoder model, which is expected to show significant improvements because:
- It learns temporal patterns and sequences naturally
- Reconstruction error captures "unexpected" deviations in time-series
- Works well with single-feature time-series data
- Doesn't require pre-specified contamination rates

The current Isolation Forest results provide a reference point for "what doesn't work" and will make the LSTM performance gains more evident.

## References

Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing, 262, 134–147. https://doi.org/10.1016/j.neucom.2017.04.070

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. 2008 Eighth IEEE International Conference on Data Mining, 413–422. https://doi.org/10.1109/ICDM.2008.17 

