
import asyncio
import json
from PIL import Image
import io
import base64
from pathlib import Path
from tqdm import tqdm
from prisma import Prisma


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with Image.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')


async def insert_image(prisma: Prisma, image_path: str, image_cache: dict):
    """Insert image into database and return its ID"""
    if image_path in image_cache:
        return image_cache[image_path]

    # Check if image already exists
    existing = await prisma.image.find_unique(where={'path': image_path})
    if existing:
        image_cache[image_path] = existing.id
        return existing.id

    # Insert new image
    image_base64 = image_to_base64(image_path)
    new_image = await prisma.image.create(
        data={
            'path': image_path,
            'imageData': image_base64
        }
    )
    image_cache[image_path] = new_image.id
    return new_image.id


async def populate_database(train_json='train_pairs.json', test_json='test_pairs.json'):
    """Populate database with training and test data"""
    prisma = Prisma()
    await prisma.connect()

    image_cache = {}

    try:
        # Process training pairs
        print("Loading training pairs...")
        with open(train_json, 'r') as f:
            train_pairs = json.load(f)

        print(f"Processing {len(train_pairs)} training pairs...")
        for pair in tqdm(train_pairs, desc="Training pairs"):
            ref_id = await insert_image(prisma, pair['reference_image'], image_cache)
            query_id = await insert_image(prisma, pair['query_image'], image_cache)

            await prisma.pair.create(
                data={
                    'refImageId': ref_id,
                    'refBbox': json.dumps(pair['reference_bbox']),
                    'queryImageId': query_id,
                    'queryBbox': json.dumps(pair['query_bbox']),
                    'angleDifference': pair['angle_difference'],
                    'matchLabel': pair['match_label'],
                    'objectId': pair['object_id'],
                    'split': 'train'
                }
            )

        # Process test pairs
        print("\nLoading test pairs...")
        with open(test_json, 'r') as f:
            test_pairs = json.load(f)

        print(f"Processing {len(test_pairs)} test pairs...")
        for pair in tqdm(test_pairs, desc="Test pairs"):
            ref_id = await insert_image(prisma, pair['reference_image'], image_cache)
            query_id = await insert_image(prisma, pair['query_image'], image_cache)

            await prisma.pair.create(
                data={
                    'refImageId': ref_id,
                    'refBbox': json.dumps(pair['reference_bbox']),
                    'queryImageId': query_id,
                    'queryBbox': json.dumps(pair['query_bbox']),
                    'angleDifference': pair['angle_difference'],
                    'matchLabel': pair['match_label'],
                    'objectId': pair['object_id'],
                    'split': 'test'
                }
            )

        # Print statistics
        num_images = await prisma.image.count()
        num_train = await prisma.pair.count(where={'split': 'train'})
        num_test = await prisma.pair.count(where={'split': 'test'})

        print(f"\nDatabase created successfully!")
        print(f"Total images: {num_images}")
        print(f"Training pairs: {num_train}")
        print(f"Test pairs: {num_test}")

        # Get database size
        db_path = Path('dataset.db')
        if db_path.exists():
            db_size = db_path.stat().st_size / (1024 * 1024)  # MB
            print(f"Database size: {db_size:.2f} MB")

    finally:
        await prisma.disconnect()


if __name__ == '__main__':
    asyncio.run(populate_database())