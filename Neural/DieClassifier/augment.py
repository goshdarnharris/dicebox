import os
import json
import random
import numpy as np
import torch
import h5py
from PIL import Image
from torchvision import transforms

# === Settings ===
image_dir = '../training_images'
annotations_file = os.path.join(image_dir, 'annotations.json')
raw_dir = "raw_training"
output_file = "augmented_training.h5"
crop_size = 180
downsample = 9
input_size = crop_size // downsample
copies_per_image = 10

# === Augmentation ===
# Rotation is handled separately on a larger crop to avoid black corners.
# Only color augmentation is applied via transforms.
color_augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.15, contrast=0.25),
])
rotation_range = 90  # degrees
position_jitter = 18  # max pixel offset in each direction
# Outer crop must be large enough that rotating inner crop + position jitter never hits the edge.
# For 180x180 inner crop, diagonal is ~255px, plus 18px jitter margin on each side.
outer_crop_size = int(crop_size * 1.45) + position_jitter * 2


def add_gaussian_noise(tensor, sigma=0.025):
    """Add Gaussian noise to a tensor and clamp to [0, 1]."""
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)


def average_border_fill(image, box):
    """Crop with padding if box extends past image edges."""
    avg_color = tuple(np.array(image).reshape(-1, 3).mean(axis=0).astype(np.uint8))
    padded = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), avg_color)
    left = max(box[0], 0)
    upper = max(box[1], 0)
    right = min(box[2], image.width)
    lower = min(box[3], image.height)
    cropped = image.crop((left, upper, right, lower))
    paste_x = left - box[0]
    paste_y = upper - box[1]
    padded.paste(cropped, (paste_x, paste_y))
    return padded


# === Setup raw_training dir ===
os.makedirs(raw_dir, exist_ok=True)
for f in os.listdir(raw_dir):
    fp = os.path.join(raw_dir, f)
    if os.path.isfile(fp):
        os.remove(fp)

# === Load annotations ===
with open(annotations_file, "r") as f:
    all_annotations = json.load(f)

# === Generate ===
all_images = []
all_labels = []
all_sources = []  # (rel_path, x, y) for tracing back to annotations
half = crop_size // 2
for rel_path in sorted(all_annotations.keys()):
    print(f"Processing {rel_path}...")
    source_path = os.path.join(image_dir, rel_path)
    if not os.path.exists(source_path):
        print(f"  Warning: source image not found: {source_path}")
        continue

    src = Image.open(source_path).convert("RGB")
    safe_name = rel_path.replace("/", "_").replace("\\", "_")
    safe_base = os.path.splitext(safe_name)[0]

    # Collect all annotated die centers for this image
    die_entries = list(all_annotations[rel_path])
    centers = [(x, y) for x, y, face in die_entries]
    w, h = src.size

    # Generate random negative locations (class 0)
    negatives_per_image = 30
    min_dist = 1.2 * (crop_size // 4)
    for _ in range(negatives_per_image * 10):  # extra attempts to find valid spots
        if sum(1 for _, _, f in die_entries if f == 0) >= negatives_per_image:
            break
        rx = random.randint(half, w - half)
        ry = min(random.randint(half, h - half), random.randint(half, h - half))
        too_close = any(((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5 < min_dist for cx, cy in centers)
        if not too_close:
            die_entries.append([rx, ry, 0])
            centers.append((rx, ry))

    for x, y, face in die_entries:
        # Crop die from source image (standard size for raw_training)
        box = (x - half, y - half, x + half, y + half)
        crop = average_border_fill(src, box)

        # Save full-size crop to raw_training
        raw_name = f"{face}_{safe_base}_{x}_{y}.png"
        crop.save(os.path.join(raw_dir, raw_name))

        # Downsample to training input size, convert to grayscale
        crop_small = crop.convert("L").resize((input_size, input_size))
        arr = np.array(crop_small, dtype=np.float32) / 255.0

        # Add original (unaugmented)
        source = f"{rel_path}:{x},{y}"
        all_images.append(arr)
        all_labels.append(face)
        all_sources.append(source)

        # For augmented copies, crop a larger region so rotation uses real image data
        outer_half = outer_crop_size // 2
        outer_box = (x - outer_half, y - outer_half, x + outer_half, y + outer_half)
        outer_crop = average_border_fill(src, outer_box).convert("L")

        for i in range(copies_per_image):
            # Rotate the larger crop by a random angle
            angle = random.uniform(-rotation_range, rotation_range)
            rotated = outer_crop.rotate(angle, resample=Image.BILINEAR, expand=False)
            # Crop the inner region with random position offset
            center = outer_crop_size // 2
            dx = random.randint(-position_jitter, position_jitter)
            dy = random.randint(-position_jitter, position_jitter)
            inner_left = center - half + dx
            inner_top = center - half + dy
            inner = rotated.crop((inner_left, inner_top, inner_left + crop_size, inner_top + crop_size))
            # Apply color augmentation, downsample, add noise
            inner_small = inner.resize((input_size, input_size))
            augmented_pil = color_augmentation(inner_small)
            augmented_tensor = transforms.functional.to_tensor(augmented_pil)
            augmented_tensor = add_gaussian_noise(augmented_tensor, sigma=0.025)
            all_images.append(augmented_tensor.squeeze(0).numpy())
            all_labels.append(face)
            all_sources.append(source)

# === Save to HDF5 ===
images_arr = np.array(all_images, dtype=np.float32)
labels_arr = np.array(all_labels, dtype=np.int32)
sources_arr = np.array(all_sources, dtype=h5py.string_dtype())
with h5py.File(output_file, "w") as hf:
    hf.create_dataset("images", data=images_arr, compression="gzip")
    hf.create_dataset("labels", data=labels_arr)
    hf.create_dataset("sources", data=sources_arr)

print(f"Generated {len(all_images)} images -> {output_file}")
for c in range(7):
    print(f"  Class {c}: {np.sum(labels_arr == c)}")
