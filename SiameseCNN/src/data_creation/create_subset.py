import json
import random

def create_subset(input_json, output_json, num_samples):
    """
    Create a subset of the dataset by randomly sampling a specified number of pairs.
    Args:
        input_json (str): Path to the input JSON file containing image pairs.
        output_json (str): Path to the output JSON file to save the subset.
        num_samples (int): Number of pairs to sample for the subset.
    """
    with open(input_json, 'r') as f:
        pairs = json.load(f)

    if len(pairs) > num_samples:
        subset = random.sample(pairs, num_samples)
    else:
        subset = pairs

    with open(output_json, 'w') as f:
        json.dump(subset, f, indent=2)

create_subset('SiameseCNN/json_data/train_pairs.json', 'SiameseCNN/json_data/train_pairs_small2.json', num_samples=10000)
create_subset('SiameseCNN/json_data/val_pairs.json', 'SiameseCNN/json_data/val_pairs_small2.json', num_samples=1000)
