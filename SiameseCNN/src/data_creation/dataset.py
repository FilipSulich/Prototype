import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json


class TLESSDataset(Dataset):
    def __init__(self, pairs_json, transform=None, crop_objects=True, augment=False):
        """
        TLESS Dataset for image pairs.
        Args:
            pairs_json (str): Path to the JSON file containing image pairs.
            transform (bool): Optional transform to be applied on a sample.
            crop_objects (bool): Whether to crop images to bounding boxes.
            augment (bool): Whether to apply data augmentation.
        """
        with open(pairs_json, 'r') as f:
            self.pairs = json.load(f)

        self.crop_objects = crop_objects
        self.transform = transform
        self.augment = augment

        if self.transform is None:
            if self.augment:
                # here we add some random augmentations for training - without it, the model was overfitting a lot
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                # test without augmentation
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        with Image.open(pair['reference_image']) as ref_img: # open reference image
            ref_img = ref_img.convert('RGB').copy() # convert to RGB
        with Image.open(pair['query_image']) as query_img: # open query image
            query_img = query_img.convert('RGB').copy() # convert to RGB

        if self.crop_objects:
            ref_bbox = pair['reference_bbox']
            query_bbox = pair['query_bbox']

            ref_img = ref_img.crop(( # crop to bounding box
                ref_bbox[0], # x
                ref_bbox[1], # y
                ref_bbox[0] + ref_bbox[2], # x + width
                ref_bbox[1] + ref_bbox[3] # y + height
            ))

            query_img = query_img.crop(( # crop to bounding box
                query_bbox[0], # x
                query_bbox[1], # y
                query_bbox[0] + query_bbox[2], # x + width
                query_bbox[1] + query_bbox[3] # y + height
            ))

        ref_img = self.transform(ref_img)
        query_img = self.transform(query_img)

        angle_diff = torch.tensor(pair['angle_difference'], dtype=torch.float32) 
        match_label = torch.tensor(pair['match_label'], dtype=torch.float32)

        return ref_img, query_img, match_label, angle_diff
    
    def get_pos_count(self): 
        """
        Get the number of positive pairs (match) in the dataset.
        """
        return sum(1 for pair in self.pairs if pair['match_label'] == 1) 
    
    def get_neg_count(self):
        """
        Get the number of negative pairs (no match) in the dataset.
        """
        return sum(1 for pair in self.pairs if pair['match_label'] == 0) 
