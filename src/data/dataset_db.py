import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import io
import base64
from prisma import Prisma
import asyncio


class TLESSDatasetDB(Dataset):
    def __init__(self, split='train', transform=None, crop_objects=True):
        self.split = split
        self.crop_objects = crop_objects
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        # Load all pairs and images into memory
        self.pairs = []
        self.images = {}
        self._load_data()

    def _load_data(self):
        """Load all data from database into memory"""
        async def fetch_data():
            prisma = Prisma()
            await prisma.connect()

            try:
                # Fetch all pairs for this split with their images
                pairs = await prisma.pair.find_many(
                    where={'split': self.split},
                    include={
                        'refImage': True,
                        'queryImage': True
                    }
                )

                for pair in pairs:
                    # Store images by ID to avoid duplicates
                    if pair.refImageId not in self.images:
                        self.images[pair.refImageId] = pair.refImage.imageData
                    if pair.queryImageId not in self.images:
                        self.images[pair.queryImageId] = pair.queryImage.imageData

                    # Store pair info
                    self.pairs.append({
                        'ref_image_id': pair.refImageId,
                        'ref_bbox': json.loads(pair.refBbox),
                        'query_image_id': pair.queryImageId,
                        'query_bbox': json.loads(pair.queryBbox),
                        'angle_difference': pair.angleDifference,
                        'match_label': pair.matchLabel
                    })

            finally:
                await prisma.disconnect()

        # Run async function
        asyncio.run(fetch_data())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load images from base64
        ref_img = Image.open(io.BytesIO(base64.b64decode(self.images[pair['ref_image_id']]))).convert('RGB')
        query_img = Image.open(io.BytesIO(base64.b64decode(self.images[pair['query_image_id']]))).convert('RGB')

        # Crop to object bounding box if enabled
        if self.crop_objects:
            ref_bbox = pair['ref_bbox']  # [x, y, width, height]
            query_bbox = pair['query_bbox']

            ref_img = ref_img.crop((
                ref_bbox[0],
                ref_bbox[1],
                ref_bbox[0] + ref_bbox[2],
                ref_bbox[1] + ref_bbox[3]
            ))

            query_img = query_img.crop((
                query_bbox[0],
                query_bbox[1],
                query_bbox[0] + query_bbox[2],
                query_bbox[1] + query_bbox[3]
            ))

        ref_img = self.transform(ref_img)
        query_img = self.transform(query_img)

        angle_diff = torch.tensor(pair['angle_difference'], dtype=torch.float32)
        match_label = torch.tensor(pair['match_label'], dtype=torch.float32)

        return ref_img, query_img, match_label, angle_diff
