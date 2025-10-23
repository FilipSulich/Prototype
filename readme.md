# Siamese CNN for Object Matching - Prototype

A Siamese CNN prototype for matching objects across different viewpoints using the T-LESS dataset.

## Architecture

- **Backbone**: ResNet18 (pre-trained, frozen)
- **Task**: Binary classification (match/no-match) + rotation magnitude prediction
- **Input**: RGB image pairs with object bounding boxes

## Dataset

Uses T-LESS dataset with:
- `cam_R_m2c`: 3x3 rotation matrices (model-to-camera)
- `cam_t_m2c`: 3D translation vectors
- Currently extracts rotation magnitude (single scalar) from full rotation matrices

## Training
To train the model, go the the [`main.py`](src/model/main.py) class and make sure the `model.train()` line is uncommented. Next, run the following terminal command, which will begin the training process. If desired, training hyperparameters can be modified.
```bash
python -m src.model.main
```

## Graphical User Interface
To open a visual interface, run the following terminal command
```bash
streamlit run interface.py
```

Features:
- Class imbalance handling with pos_weight
- Data augmentation for training
- Early stopping with patience=7
- Learning rate scheduling

## Evaluation

Generates plots for:
- Training/validation loss
- Training/validation accuracy
- Angle MAE (Mean Absolute Error)
- ROC curve with AUC

## Limitations & Future Work

**Current limitation**: The model predicts only rotation magnitude (scalar), which loses directional information about 3D rotations. This makes it impossible to distinguish between different types of rotations (e.g., 45 degree roll vs 45 degree pitch).

**Chosen solution for next phases**: Implement a full 6DoF pose estimation model - DenseFusion, which predict complete pose with rotation and translation.