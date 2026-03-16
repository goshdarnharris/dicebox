import os
import json
import numpy as np
import torch
import h5py
from PIL import Image
from torchvision.transforms import v2
from gen_finder_targets import generate_heatmap

# === Settings ===
image_dir = '../training_images'
annotations_file = os.path.join(image_dir, 'annotations.json')
output_file = 'augmented_training.h5'
copies_per_image = 10
downsample = 9  # must match train_die_finder.py

# Photometric augmentation only — no spatial transforms needed
# because the fully convolutional finder CNN is already spatially invariant.
augmentation = v2.Compose([
    v2.ColorJitter(brightness=0.25, contrast=0.25),
])

# === Load annotations ===
with open(annotations_file, "r") as f:
    all_annotations = json.load(f)

# === Generate ===
all_images = []
all_targets = []
for rel_path in sorted(all_annotations.keys()):
    print(f"Processing {rel_path}...")

    source_path = os.path.join(image_dir, rel_path)
    if not os.path.exists(source_path):
        print(f"  Warning: source image not found: {source_path}")
        continue

    # Extract die centers (exclude face=0 negatives)
    centers = [[x, y] for x, y, face in all_annotations[rel_path] if face > 0]

    # Generate heatmap target from annotations
    src = Image.open(source_path)
    w, h = src.size
    tgt = generate_heatmap(w, h, centers)
    tgt_arr = np.array(tgt, dtype=np.float32) / 255.0

    # Downsample source to match training input
    src_gray = src.convert("L")
    small_w, small_h = w // downsample, h // downsample
    src_small = src_gray.resize((small_w, small_h))
    src_arr = np.array(src_small, dtype=np.float32) / 255.0

    # Add original
    all_images.append(src_arr)
    all_targets.append(tgt_arr)

    # Generate augmented copies — build a batch as (N, 1, H, W) tensor
    src_tensor = torch.from_numpy(src_arr).unsqueeze(0)  # (1, H, W)
    batch = src_tensor.repeat(copies_per_image, 1, 1, 1)  # (N, 1, H, W)

    augmented = augmentation(batch)  # (N, 1, H, W)
    noise = torch.normal(0, 0.05, size=augmented.shape)
    augmented = torch.clamp(augmented + noise, 0, 1)

    augmented_np = augmented.numpy()
    for i in range(copies_per_image):
        all_images.append(augmented_np[i, 0, :, :])
        all_targets.append(tgt_arr)  # target is the same for all augmented versions

# === Save to HDF5 ===
images_arr = np.array(all_images, dtype=np.float32)
targets_arr = np.array(all_targets, dtype=np.float32)
with h5py.File(output_file, "w") as hf:
    hf.create_dataset("images", data=images_arr, compression="gzip")
    hf.create_dataset("targets", data=targets_arr, compression="gzip")

print(f"Generated {len(all_images)} image/target pairs -> {output_file}")
