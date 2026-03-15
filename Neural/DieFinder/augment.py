import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from gen_finder_targets import generate_heatmap

# === Settings ===
image_dir = '../training_images'
annotations_file = os.path.join(image_dir, 'annotations.json')
output_images_dir = 'augmented_training/images'
output_targets_dir = 'augmented_training/targets'
copies_per_image = 10
downsample = 9  # must match train_die_finder.py

# Photometric augmentation only — no spatial transforms needed
# because the fully convolutional finder CNN is already spatially invariant.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness([-0.25, 0.25], value_range=[0.0, 1.0]),
    tf.keras.layers.RandomContrast(0.25),
])

# === Setup output dirs ===
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_targets_dir, exist_ok=True)
for d in [output_images_dir, output_targets_dir]:
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            os.remove(fp)

# === Load annotations ===
with open(annotations_file, "r") as f:
    all_annotations = json.load(f)

# === Generate ===
count = 0
for rel_path in sorted(all_annotations.keys()):
    print(f"Processing {rel_path}...")
    safe_name = rel_path.replace("/", "_").replace("\\", "_")
    safe_base = os.path.splitext(safe_name)[0]

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

    # Downsample source to match training input
    src_gray = src.convert("L")
    small_w, small_h = w // downsample, h // downsample
    src_small = src_gray.resize((small_w, small_h))
    src_arr = np.array(src_small, dtype=np.float32) / 255.0

    # Save original
    Image.fromarray((src_arr * 255).astype(np.uint8), mode="L").save(
        os.path.join(output_images_dir, f"{safe_base}_orig.png")
    )
    tgt.save(os.path.join(output_targets_dir, f"{safe_base}_orig.png"))
    count += 1

    # Generate augmented copies
    batch = np.stack([src_arr[:, :, np.newaxis]] * copies_per_image)
    augmented = augmentation(batch, training=True).numpy()
    noise = np.random.normal(0, 0.05, augmented.shape).astype(np.float32)
    augmented = np.clip(augmented + noise, 0, 1)

    for i in range(copies_per_image):
        aug_img = (augmented[i, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(aug_img, mode="L").save(
            os.path.join(output_images_dir, f"{safe_base}_aug{i:02d}.png")
        )
        tgt.save(os.path.join(output_targets_dir, f"{safe_base}_aug{i:02d}.png"))
        count += 1

print(f"Generated {count} image/target pairs in augmented_training/")
