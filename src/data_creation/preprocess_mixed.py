import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import json
import random

def rotation_matrix_to_euler(R_flat):
    R_matrix = np.array(R_flat).reshape(3, 3)
    rotation = Rotation.from_matrix(R_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles

def calculate_angle_difference(R1_flat, R2_flat):
    R1 = np.array(R1_flat).reshape(3, 3)
    R2 = np.array(R2_flat).reshape(3, 3)
    R_relative = R2 @ R1.T
    rotation = Rotation.from_matrix(R_relative)
    angle = rotation.magnitude() * 180 / np.pi
    return angle

def load_gt_data(gt_path):
    with open(gt_path, 'r') as f:
        gt_data = yaml.safe_load(f)
    return gt_data

def generate_image_pairs(dataset_path, same_object_only=True):
    dataset_path = Path(dataset_path)
    pairs = []

    object_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.isdigit()])
    print(f"Processing {len(object_folders)} object folders from {dataset_path}")

    for obj_folder in object_folders:
        gt_path = obj_folder / 'gt.yml'
        rgb_dir = obj_folder / 'rgb'

        if not gt_path.exists():
            continue

        gt_data = load_gt_data(gt_path)

        image_objects = {}
        for img_id, objects in gt_data.items():
            if objects is None:
                continue
            for obj in objects:
                if img_id not in image_objects:
                    image_objects[img_id] = []
                image_objects[img_id].append(obj)

        image_ids = sorted(image_objects.keys())

        for i, ref_id in enumerate(image_ids):
            ref_objects = image_objects[ref_id]

            for ref_obj in ref_objects:
                ref_obj_id = ref_obj['obj_id']
                ref_rotation = ref_obj['cam_R_m2c']
                ref_bbox = ref_obj['obj_bb']

                for query_id in image_ids:
                    if query_id == ref_id:
                        continue

                    query_objects = image_objects[query_id]

                    for query_obj in query_objects:
                        query_obj_id = query_obj['obj_id']

                        if same_object_only and ref_obj_id != query_obj_id:
                            continue

                        query_rotation = query_obj['cam_R_m2c']
                        query_bbox = query_obj['obj_bb']

                        angle_diff = calculate_angle_difference(ref_rotation, query_rotation)
                        match_label = 1 if angle_diff <= 10.0 else 0

                        pair = {
                            'reference_image': str(rgb_dir / f'{ref_id:04d}.png'),
                            'reference_bbox': ref_bbox,
                            'query_image': str(rgb_dir / f'{query_id:04d}.png'),
                            'query_bbox': query_bbox,
                            'angle_difference': float(angle_diff),
                            'match_label': match_label,
                            'object_id': ref_obj_id
                        }

                        pairs.append(pair)

    return pairs

def balance_pairs(pairs, ratio=3):
    """Balance positive and negative pairs"""
    pos_pairs = [p for p in pairs if p['match_label'] == 1]
    neg_pairs = [p for p in pairs if p['match_label'] == 0]

    max_neg = min(len(neg_pairs), ratio * len(pos_pairs))
    neg_pairs = random.sample(neg_pairs, max_neg)

    balanced_pairs = pos_pairs + neg_pairs
    random.shuffle(balanced_pairs)

    print(f"  Positive pairs: {len(pos_pairs)}")
    print(f"  Negative pairs: {len(neg_pairs)}")
    print(f"  Total balanced: {len(balanced_pairs)}")

    return balanced_pairs

def split_by_images(all_pairs, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """Split into train/val/test ensuring no image overlap"""
    np.random.seed(random_seed)

    all_images = set()
    for pair in all_pairs:
        all_images.add(pair['reference_image'])
        all_images.add(pair['query_image'])

    all_images = list(all_images)
    print(f"\nTotal unique images: {len(all_images)}")

    np.random.shuffle(all_images)

    train_split = int(train_ratio * len(all_images))
    val_split = int((train_ratio + val_ratio) * len(all_images))

    train_images = set(all_images[:train_split])
    val_images = set(all_images[train_split:val_split])
    test_images = set(all_images[val_split:])

    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for pair in all_pairs:
        ref_img = pair['reference_image']
        query_img = pair['query_image']

        if ref_img in train_images and query_img in train_images:
            train_pairs.append(pair)
        elif ref_img in val_images and query_img in val_images:
            val_pairs.append(pair)
        elif ref_img in test_images and query_img in test_images:
            test_pairs.append(pair)

    print(f"\nAfter filtering:")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Discarded pairs: {len(all_pairs) - len(train_pairs) - len(val_pairs) - len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs


if __name__ == "__main__":
    cluttered_pairs = generate_image_pairs(
        dataset_path='data/train',
        same_object_only=True
    )

    clean_pairs = generate_image_pairs(
        dataset_path='data/test',
        same_object_only=True
    )
    all_pairs = cluttered_pairs + clean_pairs
    print(f"Total pairs before balancing: {len(all_pairs)}")
    print(f"  From cluttered: {len(cluttered_pairs)}")
    print(f"  From clean: {len(clean_pairs)}")

    balanced_pairs = balance_pairs(all_pairs, ratio=3)
    train_pairs, val_pairs, test_pairs = split_by_images(
        balanced_pairs,
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42
    )

    with open('json_data/train_pairs.json', 'w') as f:
        json.dump(train_pairs, f, indent=2)

    with open('json_data/val_pairs.json', 'w') as f:
        json.dump(val_pairs, f, indent=2)

    with open('json_data/test_pairs.json', 'w') as f:
        json.dump(test_pairs, f, indent=2)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
