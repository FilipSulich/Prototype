"""
Upload images and pairs data to Neon database via Prisma.
This script:
1. Uploads all images from data/train and data/test
2. Uploads all pairs from train_pairs.json and test_pairs.json
"""
import asyncio
import json
import base64
from pathlib import Path
from PIL import Image as PILImage
from prisma import Prisma
from tqdm import tqdm


async def upload_images(db: Prisma):
    """Upload all images to the database."""
    print("📁 Uploading images to database...")

    data_dir = Path("data")
    image_cache = {}  

    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        scene_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for scene_dir in scene_dirs:
            scene_id = int(scene_dir.name)
            rgb_dir = scene_dir / "rgb"

            if not rgb_dir.exists():
                continue

            # Get all image files
            image_files = sorted(rgb_dir.glob("*.png"))

            print(f"  Uploading {len(image_files)} images from {split}/scene {scene_id:02d}...")

            for image_path in tqdm(image_files, desc=f"{split}/{scene_id:02d}"):
                frame_number = int(image_path.stem)

                # Create cache key
                cache_key = f"{scene_id}_{split}_{frame_number}"

                # Check if already uploaded
                existing = await db.image.find_first(
                    where={
                        "sceneId": scene_id,
                        "split": split,
                        "frameNumber": frame_number
                    }
                )

                if existing:
                    image_cache[cache_key] = existing.id
                    continue

                # Read image
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                # Get image metadata
                with PILImage.open(image_path) as img:
                    width, height = img.size
                    format_str = img.format.lower() if img.format else "png"

                # Encode as base64 for Prisma (decode to string)
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                # Upload to database
                image = await db.image.create(
                    data={
                        "sceneId": scene_id,
                        "split": split,
                        "frameNumber": frame_number,
                        "imageData": image_b64,
                        "width": width,
                        "height": height,
                        "format": format_str,
                        "fileSize": len(image_bytes)
                    }
                )

                image_cache[cache_key] = image.id

    print(f"✅ Uploaded {len(image_cache)} images to database")
    return image_cache


async def upload_pairs(db: Prisma, image_cache: dict):
    """Upload image pairs from JSON files."""
    print("\n🔗 Uploading image pairs to database...")

    for split in ["train", "test"]:
        pairs_file = Path(f"{split}_pairs.json")

        if not pairs_file.exists():
            print(f"⚠️  Warning: {pairs_file} does not exist, skipping...")
            continue

        print(f"  Loading {pairs_file}...")
        with open(pairs_file, "r") as f:
            pairs = json.load(f)

        print(f"  Uploading {len(pairs)} pairs from {split}_pairs.json...")

        batch_size = 100
        for i in tqdm(range(0, len(pairs), batch_size), desc=f"{split} pairs"):
            batch = pairs[i:i+batch_size]

            for pair in batch:
                # Parse image paths
                ref_path = Path(pair["reference_image"])
                query_path = Path(pair["query_image"])

                # Extract scene_id, split, frame_number from paths
                # Format: data/train/01/rgb/0000.png
                ref_parts = ref_path.parts
                ref_scene_id = int(ref_parts[2])
                ref_split = ref_parts[1]
                ref_frame = int(ref_path.stem)

                query_parts = query_path.parts
                query_scene_id = int(query_parts[2])
                query_split = query_parts[1]
                query_frame = int(query_path.stem)

                # Get image IDs from cache
                ref_cache_key = f"{ref_scene_id}_{ref_split}_{ref_frame}"
                query_cache_key = f"{query_scene_id}_{query_split}_{query_frame}"

                if ref_cache_key not in image_cache:
                    print(f"⚠️  Warning: Reference image not found in cache: {ref_path}")
                    continue

                if query_cache_key not in image_cache:
                    print(f"⚠️  Warning: Query image not found in cache: {query_path}")
                    continue

                ref_image_id = image_cache[ref_cache_key]
                query_image_id = image_cache[query_cache_key]

                # Parse bounding boxes
                ref_bbox = pair["reference_bbox"]
                query_bbox = pair["query_bbox"]

                # Create pair
                await db.imagepair.create(
                    data={
                        "refImageId": ref_image_id,
                        "refBboxX": ref_bbox[0],
                        "refBboxY": ref_bbox[1],
                        "refBboxWidth": ref_bbox[2],
                        "refBboxHeight": ref_bbox[3],
                        "queryImageId": query_image_id,
                        "queryBboxX": query_bbox[0],
                        "queryBboxY": query_bbox[1],
                        "queryBboxWidth": query_bbox[2],
                        "queryBboxHeight": query_bbox[3],
                        "angleDifference": pair["angle_difference"],
                        "matchLabel": pair["match_label"],
                        "objectId": pair["object_id"],
                        "split": split
                    }
                )

        print(f"✅ Uploaded {len(pairs)} pairs from {split}_pairs.json")


async def create_dataset_metadata(db: Prisma):
    """Create dataset metadata entry."""
    print("\n📊 Creating dataset metadata...")

    # Count images and pairs
    total_images = await db.image.count()
    total_pairs = await db.imagepair.count()
    train_pairs = await db.imagepair.count(where={"split": "train"})
    test_pairs = await db.imagepair.count(where={"split": "test"})

    # Create or update dataset entry
    dataset = await db.dataset.upsert(
        where={"name": "T-LESS"},
        data={
            "create": {
                "name": "T-LESS",
                "version": "1.0",
                "description": "T-LESS dataset for object pose estimation",
                "totalImages": total_images,
                "totalPairs": total_pairs,
                "trainPairs": train_pairs,
                "testPairs": test_pairs,
                "preprocessedAt": asyncio.get_event_loop().time()
            },
            "update": {
                "totalImages": total_images,
                "totalPairs": total_pairs,
                "trainPairs": train_pairs,
                "testPairs": test_pairs
            }
        }
    )

    print(f"✅ Dataset metadata:")
    print(f"   Total images: {total_images}")
    print(f"   Total pairs: {total_pairs}")
    print(f"   Train pairs: {train_pairs}")
    print(f"   Test pairs: {test_pairs}")


async def main():
    """Main upload function."""
    print("🚀 Starting database upload...\n")

    db = Prisma()
    await db.connect()

    try:
        # Step 1: Upload images
        image_cache = await upload_images(db)

        # Step 2: Upload pairs
        await upload_pairs(db, image_cache)

        # Step 3: Create dataset metadata
        await create_dataset_metadata(db)

        print("\n✅ Database upload complete!")

    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        raise
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())