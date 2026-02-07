#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

# Paths
train_imgs = Path('data/processed/yolo/train/images')
train_lbls = Path('data/processed/yolo/train/labels')
val_imgs = Path('data/processed/yolo/val/images')
val_lbls = Path('data/processed/yolo/val/labels')

# Create val directories
val_imgs.mkdir(parents=True, exist_ok=True)
val_lbls.mkdir(parents=True, exist_ok=True)

# Get all image files
all_imgs = sorted([f for f in train_imgs.glob('*') if f.is_file()])
print(f"Total images: {len(all_imgs)}")

# Shuffle and split (15% to validation)
random.seed(42)
random.shuffle(all_imgs)
split_idx = int(len(all_imgs) * 0.85)

val_imgs_list = all_imgs[split_idx:]
print(f"Moving {len(val_imgs_list)} images to validation")

# Move images and labels to validation
for img_path in val_imgs_list:
    # Move image
    shutil.move(str(img_path), str(val_imgs / img_path.name))
    
    # Move corresponding label
    lbl_name = img_path.stem + '.txt'
    lbl_path = train_lbls / lbl_name
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(val_lbls / lbl_name))

print(f"Done! Remaining training: {len(list(train_imgs.glob('*')))}, Validation: {len(list(val_imgs.glob('*')))}")
