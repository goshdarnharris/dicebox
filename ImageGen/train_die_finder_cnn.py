import os
import numpy as np
import tensorflow as tf
from PIL import Image

# === Settings ===
image_dir = '../images/final_box_20_dice/'
targets_dir = "finder_targets"
downsample = 9  # source images are downsampled by this factor before feeding to CNN
epochs = 2000
patience = 200
batch_size = 4  # small because each sample is a full image

# === Load Data ===
# Each training pair is: (downsampled source image, downsampled target heatmap)
# The CNN has 2 MaxPool(2) layers, so its output is 1/4 the input in each dimension.
# The target heatmap must be downsampled to match the CNN output size.

def load_dataset():
    inputs = []
    targets = []
    for fname in sorted(os.listdir(targets_dir)):
        if not fname.startswith("target_") or not fname.endswith(".png"):
            continue
        # Derive source image name: target_00.png -> 00.jpg
        base = fname.replace("target_", "").replace(".png", "")
        source_path = os.path.join(image_dir, base + ".jpg")
        target_path = os.path.join(targets_dir, fname)

        if not os.path.exists(source_path):
            print(f"Warning: source image not found: {source_path}")
            continue

        # Load and downsample source image
        src = Image.open(source_path).convert("L")
        w, h = src.size
        small_w, small_h = w // downsample, h // downsample
        src_small = src.resize((small_w, small_h))
        src_arr = np.array(src_small, dtype=np.float32) / 255.0

        # Load and downsample target heatmap to CNN output size (1/4 of input due to 2 MaxPools)
        tgt = Image.open(target_path).convert("L")
        out_w, out_h = small_w // 2, small_h // 2
        tgt_small = tgt.resize((out_w, out_h), Image.BILINEAR)
        tgt_arr = np.array(tgt_small, dtype=np.float32) / 255.0

        inputs.append(src_arr[:, :, np.newaxis])   # (h, w, 1)
        targets.append(tgt_arr[:, :, np.newaxis])  # (h/4, w/4, 1)

        print(f"  {base}: input {small_w}x{small_h} -> output {out_w}x{out_h}")

    return inputs, targets

print("Loading dataset...")
inputs, targets = load_dataset()
print(f"Loaded {len(inputs)} image pairs")

# Split into train/val
n = len(inputs)
n_val = max(1, n // 5)
n_train = n - n_val
X_train = np.array(inputs[:n_train])
Y_train = np.array(targets[:n_train])
X_val = np.array(inputs[n_train:])
Y_val = np.array(targets[n_train:])
print(f"Train: {n_train}, Val: {n_val}")

# === Model ===
# Fully convolutional — no Dense or Flatten layers.
# Input: downsampled grayscale image (None, None, 1)
# Output: heatmap at 1/4 resolution (None, None, 1) with sigmoid activation

def build_finder_model():
    inputs = tf.keras.Input(shape=(None, None, 1))

    x = tf.keras.layers.Conv2D(8, 15, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 7, padding="same", activation="relu")(x)
    #x = tf.keras.layers.MaxPooling2D(2)(x)
    #x = tf.keras.layers.SpatialDropout2D(0.1)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    #x = tf.keras.layers.SpatialDropout2D(0.1)(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

model = build_finder_model()
model.summary()

def weighted_bce(y_true, y_pred):
    """Binary crossentropy weighted by (target + bias) so die centers matter more."""
    bias = 0.05
    k = 10
    weight = y_true + bias
    weight *= k
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weight[..., 0])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=weighted_bce,
    metrics=["mae"],
)

# === Train ===
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.75, patience=30, min_lr=1e-7, verbose=1
    ),
]

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
)

# === Evaluate ===
val_loss, val_mae = model.evaluate(X_val, Y_val, verbose=0)
print(f"\nValidation loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

# === Save predicted heatmaps for inspection ===
wrongs_dir = "wrongs_finder"
os.makedirs(wrongs_dir, exist_ok=True)
for f in os.listdir(wrongs_dir):
    os.remove(os.path.join(wrongs_dir, f))

X_all = np.concatenate([X_train, X_val], axis=0)
Y_all = np.concatenate([Y_train, Y_val], axis=0)
for i in range(len(X_all)):
    pred = model.predict(X_all[i:i+1], verbose=0)[0, :, :, 0]
    actual = Y_all[i, :, :, 0]

    pred_img = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    actual_img = Image.fromarray((actual * 255).astype(np.uint8), mode="L")

    # Save side by side: actual | predicted
    label = "train" if i < len(X_train) else "val"
    combined = Image.new("L", (pred_img.width * 2, pred_img.height))
    combined.paste(actual_img, (0, 0))
    combined.paste(pred_img, (pred_img.width, 0))
    combined.save(os.path.join(wrongs_dir, f"{label}_{i}_actual_vs_pred.png"))

print(f"Saved {len(X_all)} actual vs predicted comparisons to {wrongs_dir}/")

# === Save ===
keras_path = "die_finder_cnn.keras"
model.save(keras_path)
print(f"\nSaved Keras model: {keras_path}")

# Export TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = "die_finder_cnn.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model: {tflite_path}")
