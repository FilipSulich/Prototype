# Computer Vision and Anomaly Detection Prototype

This repository contains two machine learning models for different tasks: object pose matching using RGB images and time-series anomaly detection.

## Project Structure

### [SiameseCNN](./SiameseCNN)
Siamese Convolutional Neural Network for estimation of angular difference between objects on two images, trained on the T-LESS dataset.

**Key Features:**
- ResNet18-based architecture for RGB image pair matching
- Binary classification (match/no-match) + rotation angle magnitude prediction
- Interactive GUI for testing and visualization

**See the [SiameseCNN README](./SiameseCNN/README.md) for detailed documentation.**

### [IsolationForest](./IsolationForest)
Baseline anomaly detection implementation using Isolation Forest algorithm on time-series data from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)

**Key Features:**
- Baseline model for time-series anomaly detection
- Dynamic contamination parameter tuning

**See the [IsolationForest README](./IsolationForest/readme.md) for detailed documentation.**

## Quick Start

### SiameseCNN
```bash
# Train/Test the model
python -m SiameseCNN.src.model.main

# Launch GUI
streamlit run SiameseCNN/interface.py
```

### IsolationForest
```bash
# Run baseline model
python -m IsolationForest.main
```

## Dependencies

Each project has its own requirements. Install dependencies as needed:
```bash
pip install -r requirements.txt
```

## Development Notes

In some parts of the code, Claude Code was used to help with debugging, refactoring, and implementation.
