# Data Setup Instructions

The T-LESS dataset is not included in this repository due to size constraints.

## Download T-LESS Dataset

1. Visit the official T-LESS dataset page:
   - Website: http://cmp.felk.cvut.cz/t-less/
   - GitHub: https://github.com/thodan/t-less_toolkit

2. Download the dataset components you need:
   - **Test images**: BOP19 test images
   - **Ground truth annotations**: `gt.yml` files with rotation matrices and translations
   - **Camera info**: `info.yml` files with camera parameters

3. Place the downloaded data in the appropriate structure:
```
SiameseCNN/
├── data/
│   └── test/
│       ├── 01/
│       │   ├── rgb/
│       │   │   ├── 0000.png
│       │   │   └── ...
│       │   ├── depth/
│       │   │   ├── 0000.png
│       │   │   └── ...
│       │   ├── gt.yml
│       │   └── info.yml
│       └── ...
└── json_data/
    ├── train_pairs.json
    ├── val_pairs.json
    └── test_pairs.json
```

## Generate JSON Pair Files

After downloading the data, generate the training pairs:

```bash
python -m SiameseCNN.src.data_creation.preprocess_images
```

This will create the required JSON files in `SiameseCNN/json_data/`.

## Dataset Information

- **Source**: T-LESS (Texture-Less Objects)
- **Paper**: Hodaň et al., 2017, "T-LESS: An RGB-D dataset for 6D pose estimation of texture-less objects"
- **Size**: 30 texture-less objects with RGB-D images
- **Contains**: RGB images, depth maps, rotation matrices, translation vectors
- **Use Case**: 6D pose estimation and object matching
