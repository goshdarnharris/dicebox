import os
import numpy as np
import tensorflow as tf
from PIL import Image

# === Settings ===
captures_dir = "raw_training"
output_dir = "augmented_training"
crop_size = 180
downsample = 9
input_size = crop_size // downsample
copies_per_image = 20  # how many augmented_training variants to generate per original

# === Augmentation ===
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.3, fill_mode="nearest"),
    tf.keras.layers.RandomBrightness([-0.5,0.5], value_range=[0.0, 1.0]),
    tf.keras.layers.RandomContrast(0.25),
    #tf.keras.layers.RandomTranslation(0.1, 0.1),
    #tf.keras.layers.RandomZoom(0.1),
])

# === Generate ===
os.makedirs(output_dir, exist_ok=True)
# Clear previous
for f in os.listdir(output_dir):
    fp = os.path.join(output_dir, f)
    if os.path.isfile(fp):
        os.remove(fp)

count = 0
for fname in sorted(os.listdir(captures_dir)):
    if not fname.endswith(".png"):
        continue
    digit = fname.split("_")[0]
    if not digit.isdigit():
        continue
    label = int(digit)
    if label < 0 or label > 6:
        continue

    path = os.path.join(captures_dir, fname)
    img = tf.keras.utils.load_img(path, color_mode="grayscale", target_size=(input_size, input_size))
    arr = tf.keras.utils.img_to_array(img) / 255.0  # (input_size, input_size, 1)

    # Save the original (downsampled)
    base = os.path.splitext(fname)[0]
    orig_out = (arr[:, :, 0] * 255).astype(np.uint8)
    Image.fromarray(orig_out, mode="L").save(os.path.join(output_dir, f"{base}_orig.png"))
    count += 1

    # Generate augmented_training copies
    batch = np.stack([arr] * copies_per_image)  # (copies, h, w, 1)
    augmented = augmentation(batch, training=True).numpy()
    noise = np.random.normal(0, 0.025, augmented.shape).astype(np.float32)
    augmented = np.clip(augmented + noise, 0, 1)
    for i in range(copies_per_image):
        aug_img = (augmented[i, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(aug_img, mode="L").save(
            os.path.join(output_dir, f"{base}_aug{i}.png")
        )
        count += 1

print(f"Generated {count} images in {output_dir}/")
