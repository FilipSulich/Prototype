# Data Setup Instructions

The NAB dataset is not included in this repository due to size constraints.

## Download NAB Dataset

1. Clone the NAB repository:
```bash
git clone https://github.com/numenta/NAB.git
```

2. Copy the data files:
```bash
cp -r NAB/data/* IsolationForest/data/
```

3. Copy the labels file:
```bash
cp NAB/labels/combined_labels.json IsolationForest/labels/
```

## Alternative: Direct Download

Download the NAB dataset directly from:
https://github.com/numenta/NAB

The dataset should be structured as:
```
IsolationForest/
├── data/
│   ├── artificialNoAnomaly/
│   ├── artificialWithAnomaly/
│   ├── realAWSCloudwatch/
│   ├── realAdExchange/
│   ├── realKnownCause/
│   ├── realTraffic/
│   └── realTweets/
└── labels/
    └── combined_labels.json
```

## Dataset Information

- **Source**: Numenta Anomaly Benchmark (NAB)
- **Size**: ~58 CSV files with time-series data
- **License**: AGPL 3.0
- **Citation**: Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing, 262, 134–147.
