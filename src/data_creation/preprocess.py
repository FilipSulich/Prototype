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

def generate_image_pairs(dataset_path, output_json, same_object_only=True):
    dataset_path = Path(dataset_path)
    pairs = []

    object_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.isdigit()])
    print(f"Processing {len(object_folders)} object folders")

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

                for query_id in image_ids[i + 1:]:
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

    pos_pairs = []
    neg_pairs = []
    for pair in pairs:
        if pair['match_label'] == 1:
            pos_pairs.append(pair)
        elif pair['match_label'] == 0:
            neg_pairs.append(pair)
        
    ratio = 3
    max_neg = min(len(neg_pairs), ratio * len(pos_pairs))
    neg_pairs = random.sample(neg_pairs, max_neg)

    balanced_pairs = pos_pairs + neg_pairs
    random.shuffle(balanced_pairs) # we need to filter the data to ensure a balanced ratio of positive and negative pairs (1:3)
    pairs = balanced_pairs

    with open(output_json, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"Generated {len(pairs)} pairs")
    print(f"Positive pairs (<=10°): {sum(1 for p in pairs if p['match_label'] == 1)}")
    print(f"Negative pairs (>10°): {sum(1 for p in pairs if p['match_label'] == 0)}")

    return pairs


def split_by_images(all_pairs, train_ratio=0.75, random_seed=42):
    np.random.seed(random_seed)
    
    all_images = set()
    for pair in all_pairs:
        all_images.add(pair['reference_image'])
        all_images.add(pair['query_image'])
    
    all_images = list(all_images)
    print(f"\nTotal unique images: {len(all_images)}")
    
    np.random.shuffle(all_images)
    split_idx = int(train_ratio * len(all_images))
    
    train_images = set(all_images[:split_idx])
    val_images = set(all_images[split_idx:])
    
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    
    overlap = train_images.intersection(val_images)
    print(f"Image overlap: {len(overlap)} (should be 0)")
    assert len(overlap) == 0, "ERROR: Images overlap between train and val!"
    
    train_pairs = []
    val_pairs = []
    
    for pair in all_pairs:
        ref_img = pair['reference_image']
        query_img = pair['query_image']
        
        if ref_img in train_images and query_img in train_images:
            train_pairs.append(pair)
        elif ref_img in val_images and query_img in val_images:
            val_pairs.append(pair)
    
    print(f"\nAfter filtering:")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Discarded pairs: {len(all_pairs) - len(train_pairs) - len(val_pairs)}")
    
    return train_pairs, val_pairs



all_train_pairs = generate_image_pairs(
    dataset_path='data/train',
    output_json='all_train_pairs.json',
    same_object_only=True
)

train_pairs, val_pairs = split_by_images(
    all_train_pairs, 
    train_ratio=0.75, 
    random_seed=42
)

with open('train_pairs.json', 'w') as f:
    json.dump(train_pairs, f, indent=2)

with open('val_pairs.json', 'w') as f:
    json.dump(val_pairs, f, indent=2)

test_pairs = generate_image_pairs(
    dataset_path='data/test',
    output_json='test_pairs.json',
    same_object_only=True
)