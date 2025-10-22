import json
import random

def create_subset(input_json, output_json, num_samples):
    with open(input_json, 'r') as f:
        pairs = json.load(f)

    print(len(pairs))
    if len(pairs) > num_samples:
        subset = random.sample(pairs, num_samples)
    else:
        subset = pairs

    print(f"Subset size: {len(subset)} pairs")

    with open(output_json, 'w') as f:
        json.dump(subset, f, indent=2)

    print(f"Saved to {output_json}")

create_subset('train_pairs.json', 'train_pairs_small2.json', num_samples=100000)
create_subset('val_pairs.json', 'val_pairs_small2.json', num_samples=10000)
