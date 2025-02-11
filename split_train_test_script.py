import os
import shutil
import random

from tqdm import tqdm
dataest_path = "tobacco-dataset"
output_path = "tobacco-dataset-split"
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15
os.listdir(dataest_path)
# Create output directory
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(output_path, split), exist_ok=True)

# Split dataset
for category in os.listdir(dataest_path):
    category_path = os.path.join(dataest_path, category)
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)
    random.shuffle(images)

    total = len(images)
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    test_count = total - (train_count + valid_count)

    train_images = images[:train_count]
    valid_images = images[train_count:train_count + valid_count]
    test_images = images[train_count + valid_count:]

    for split, images in zip(["train", "valid", "test"], [train_images, valid_images, test_images]):
        split_dir = os.path.join(output_path, split, category)
        os.makedirs(split_dir, exist_ok=True)

        for img in tqdm(images, desc=f"Copying {split} {category} images"):
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(split_dir, img)
            shutil.copy(src_path, dst_path)
