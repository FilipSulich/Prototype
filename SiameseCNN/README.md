# Siamese CNN for Object Matching - Prototype

A Siamese CNN prototype for matching objects across different viewpoints using the T-LESS dataset.

## Architecture

- **Backbone**: ResNet18 (pre-trained, frozen) (He et al., 2015)
- **Task**: Binary classification (match/no-match) + rotation magnitude prediction
- **Input**: RGB image pairs with object bounding boxes

## Dataset

Uses T-LESS dataset (Hodaň et al., 2017) with:
- `cam_R_m2c`: 3x3 rotation matrices (model-to-camera)
- `cam_t_m2c`: 3D translation vectors
- Currently extracts rotation magnitude (single scalar) from full rotation matrices

## Testing
To test the model, go the the [`main.py`](src/model/main.py) class and make sure the `model.evaluate()` line is uncommented. Next, run the following terminal command, which will begin the training process. 
```bash
python -m SiameseCNN.src.model.main
```
Features:
- Class imbalance handling with pos_weight - there was a big imbalance in the dataset regarding the positive and negative matches (less than 10 degree angular difference is a match)
- Data augmentation for training - images are augmented with noise to improve learning
- Early stopping  - if the model is not improving through 3 consecutive Epochs, the training stops
- Learning rate scheduling - if improvement is small, the learning rate decreases. 

## Graphical User Interface
To open a visual interface, run the following terminal command
```bash
streamlit run SiameseCNN/interface.py
```



## Evaluation

Generates plots for:
- Training/validation loss
- Training/validation accuracy
- Angle MAE (Mean Absolute Error)
- ROC curve with AUC

## Limitations & Future Work

**Current limitation**: The model predicts only rotation magnitude (scalar), which loses directional information about 3D rotations. This makes it impossible to distinguish between different types of rotations (e.g., 45 degree roll vs. 45 degree pitch).

**Chosen solution for next phases**: Implement a full 6DoF pose estimation model - DenseFusion (Wang et al., 2019), which predicts the complete pose of an object with rotation and translation.


## References
He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. arXiv. arXiv:1512.03385. https://doi.org/10.48550/arXiv.1512.03385 

Hodaň, T., Haluza, P., Obdržálek, Š., Matas, J., Lourakis, M., & Zabulis, X. (2017). T-LESS: An RGB-D dataset for 6D pose estimation of texture-less objects. 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), 880–888. IEEE. https://doi.org/10.1109/WACV.2017.103 

Wang, C., Xu, D., Zhu, Y., Martín-Martín, R., Lu, C., Fei-Fei, L., & Savarese, S. (2019). DenseFusion: 6D object pose estimation by iterative dense fusion. In Proceedings of the IEEE/CVF Conference on  Computer Vision and Pattern Recognition (CVPR). http://arxiv.org/abs/1901.04780 

In some parts of the code, Claude Code was used to help with debugging and refactoring.