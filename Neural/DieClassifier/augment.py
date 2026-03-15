import os
import json
import random
import numpy as np
import tensorflow as tf
import h5py
from PIL import Image

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
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.3, fill_mode="nearest"),
    tf.keras.layers.RandomBrightness([-0.25, 0.25], value_range=[0.0, 1.0]),
    tf.keras.layers.RandomContrast(0.25),
])


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
    min_dist = crop_size // 4
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
        # Crop die from source image
        box = (x - half, y - half, x + half, y + half)
        crop = average_border_fill(src, box)

        # Save full-size crop to raw_training
        raw_name = f"{face}_{safe_base}_{x}_{y}.png"
        crop.save(os.path.join(raw_dir, raw_name))

        # Downsample to training input size, convert to grayscale
        crop_small = crop.convert("L").resize((input_size, input_size))
        arr = np.array(crop_small, dtype=np.float32) / 255.0

        # Add original
        all_images.append(arr)
        all_labels.append(face)

        # Generate augmented copies
        batch = np.stack([arr[:, :, np.newaxis]] * copies_per_image)
        augmented = augmentation(batch, training=True).numpy()
        noise = np.random.normal(0, 0.025, augmented.shape).astype(np.float32)
        augmented = np.clip(augmented + noise, 0, 1)
        for i in range(copies_per_image):
            all_images.append(augmented[i, :, :, 0])
            all_labels.append(face)

# === Save to HDF5 ===
images_arr = np.array(all_images, dtype=np.float32)
labels_arr = np.array(all_labels, dtype=np.int32)
with h5py.File(output_file, "w") as hf:
    hf.create_dataset("images", data=images_arr, compression="gzip")
    hf.create_dataset("labels", data=labels_arr)

print(f"Generated {len(all_images)} images -> {output_file}")
for c in range(7):
    print(f"  Class {c}: {np.sum(labels_arr == c)}")
