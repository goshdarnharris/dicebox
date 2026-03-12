import os
import numpy as np
import tensorflow as tf
from PIL import Image

# === Settings ===
image_dir = '../../images/final_box_20_dice/'
targets_dir = 'targets'
output_images_dir = 'augmented_training/images'
output_targets_dir = 'augmented_training/targets'
copies_per_image = 20
downsample = 9  # must match train_die_finder.py

# Photometric augmentation only — no spatial transforms needed
# because the fully convolutional finder CNN is already spatially invariant.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness([-0.5, 0.5], value_range=[0.0, 1.0]),
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

# === Generate ===
count = 0
for fname in sorted(os.listdir(targets_dir)):
    if not fname.startswith("target_") or not fname.endswith(".png"):
        continue

    base = fname.replace("target_", "").replace(".png", "")
    source_path = os.path.join(image_dir, base + ".jpg")
    target_path = os.path.join(targets_dir, fname)

    if not os.path.exists(source_path):
        print(f"Warning: source image not found: {source_path}")
        continue

    # Load source as grayscale, downsample to match training input
    src = Image.open(source_path).convert("L")
    w, h = src.size
    small_w, small_h = w // downsample, h // downsample
    src_small = src.resize((small_w, small_h))
    src_arr = np.array(src_small, dtype=np.float32) / 255.0  # (h, w)

    # Load target heatmap (stays unchanged for all augmented_training copies)
    tgt = Image.open(target_path).convert("L")

    # Save original
    Image.fromarray((src_arr * 255).astype(np.uint8), mode="L").save(
        os.path.join(output_images_dir, f"{base}_orig.png")
    )
    tgt.save(os.path.join(output_targets_dir, f"{base}_orig.png"))
    count += 1

    # Generate augmented_training copies
    # augmentation expects (batch, h, w, channels)
    batch = np.stack([src_arr[:, :, np.newaxis]] * copies_per_image)
    augmented = augmentation(batch, training=True).numpy()
    noise = np.random.normal(0, 0.025, augmented.shape).astype(np.float32)
    augmented = np.clip(augmented + noise, 0, 1)

    for i in range(copies_per_image):
        aug_img = (augmented[i, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(aug_img, mode="L").save(
            os.path.join(output_images_dir, f"{base}_aug{i:02d}.png")
        )
        # Target is the same for all augmented_training versions of this image
        tgt.save(os.path.join(output_targets_dir, f"{base}_aug{i:02d}.png"))
        count += 1

print(f"Generated {count} image/target pairs in augmented_training/")
