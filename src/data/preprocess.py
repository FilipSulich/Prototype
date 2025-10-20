import yaml
import numpy as np
from scipy.spatial.transform import Rotation
import os
from pathlib import Path
import json

def rotation_matrix_to_euler(R_flat):
    R_matrix = np.array(R_flat).reshape(3, 3)
    rotation = Rotation.from_matrix(R_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles


def calculate_angle_difference(R1_flat, R2_flat):
    R1 = np.array(R1_flat).reshape(3, 3)
    R2 = np.array(R2_flat).reshape(3, 3)

    # Relative rotation
    R_relative = R2 @ R1.T

    # Convert to angle-axis representation
    rotation = Rotation.from_matrix(R_relative)
    angle = rotation.magnitude() * 180 / np.pi  # Convert to degrees

    return angle


def load_gt_data(gt_path):
    with open(gt_path, 'r') as f:
        gt_data = yaml.safe_load(f)
    return gt_data


def generate_image_pairs(dataset_path, output_json, same_object_only=True):
    dataset_path = Path(dataset_path)
    pairs = []

    # Iterate through object folders (01, 02, ..., 13)
    object_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.isdigit()])
    print(len(object_folders))

    for obj_folder in object_folders[:1]:  # Only process first folder for testing
        obj_id = int(obj_folder.name)
        gt_path = obj_folder / 'gt.yml'
        rgb_dir = obj_folder / 'rgb'
        print(gt_path)

        if not gt_path.exists():
            print(f"Skipping {obj_folder.name}: gt.yml not found")
            continue

        gt_data = load_gt_data(gt_path)
        print(gt_data.keys())

        # Get all image IDs and their objects for this object folder
        image_objects = {}
        for img_id, objects in gt_data.items():
            if objects is None:
                continue
            for obj in objects:
                if img_id not in image_objects:
                    image_objects[img_id] = []
                image_objects[img_id].append(obj)

        # Generate pairs within this object folder
        image_ids = sorted(image_objects.keys())

        for i, ref_id in enumerate(image_ids):
            ref_objects = image_objects[ref_id]

            for ref_obj in ref_objects:
                ref_obj_id = ref_obj['obj_id']
                ref_rotation = ref_obj['cam_R_m2c']
                ref_bbox = ref_obj['obj_bb']

                # Create pairs with other images
                for query_id in image_ids[i + 1:]:
                    query_objects = image_objects[query_id]

                    for query_obj in query_objects:
                        query_obj_id = query_obj['obj_id']

                        # Only pair same objects if flag is set
                        if same_object_only and ref_obj_id != query_obj_id:
                            continue

                        query_rotation = query_obj['cam_R_m2c']
                        query_bbox = query_obj['obj_bb']

                        # Calculate angle difference
                        angle_diff = calculate_angle_difference(ref_rotation, query_rotation)

                        # Determine match label (within 5 degrees = match)
                        match_label = 1 if angle_diff <= 5.0 else 0

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

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"Generated {len(pairs)} pairs")
    print(f"Positive pairs (<=5°): {sum(1 for p in pairs if p['match_label'] == 1)}")
    print(f"Negative pairs (>5°): {sum(1 for p in pairs if p['match_label'] == 0)}")

    return pairs


if __name__ == '__main__':
    # Generate pairs from training data
    train_pairs = generate_image_pairs(
        dataset_path='data/train',
        output_json='train_pairs.json',
        same_object_only=True
    )

    # Generate pairs from test data
    test_pairs = generate_image_pairs(
        dataset_path='data/test',
        output_json='test_pairs.json',
        same_object_only=True
    )