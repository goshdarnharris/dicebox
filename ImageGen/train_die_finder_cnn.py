import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Settings ===
captures_dir = "augmented"
input_size = 20
batch_size = 1024
epochs = 2000
patience = 100

# === Load Data ===
def load_dataset(captures_dir):
    images = []
    labels = []
    for fname in os.listdir(captures_dir):
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
        images.append(tf.keras.utils.img_to_array(img))
        # Binary: 0 = not a die, 1-6 = die center
        labels.append(0.0 if label == 0 else 1.0)
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.float32)

print("Loading dataset...")
X, y = load_dataset(captures_dir)
print(f"Loaded {len(X)} images")
print(f"  Not die: {int(np.sum(y == 0))}")
print(f"  Die center: {int(np.sum(y == 1))}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# === Class Weights ===
n_neg = np.sum(y_train == 0)
n_pos = np.sum(y_train == 1)
class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
print(f"Class weights: {class_weight}")

# === Model ===
# Fully convolutional — no Dense or Flatten layers.
# This allows the model to accept any input size at inference and produce
# a spatial heatmap of "die center probability" when GlobalAveragePooling is removed.

def build_finder_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Conv layers extract features at decreasing spatial resolution
    x = tf.keras.layers.Conv2D(16, 5, padding="same", activation="relu")(inputs)   # -> 20x20x8
    x = tf.keras.layers.MaxPooling2D(2)(x)                                         # -> 10x10x8
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)        # -> 10x10x16
    x = tf.keras.layers.MaxPooling2D(2)(x)                                         # -> 5x5x16
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)  # -> 5x5x32
    x = tf.keras.layers.SpatialDropout2D(0.1)(x)

    # 1x1 conv reduces to a single channel — each spatial position is a
    # "die center score". At training size (20x20) this is 5x5x1.
    x = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)      # -> 5x5x1

    # GlobalAveragePooling collapses spatial dims for training on per-image labels.
    # Remove this layer at inference to get the full spatial heatmap.
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)                           # -> 1

    return tf.keras.Model(inputs, outputs)

model = build_finder_model((input_size, input_size, 1))
model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# === Train ===
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.75, patience=10, min_lr=1e-7, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight,
    callbacks=callbacks,
)

# === Evaluate ===
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy: {val_acc:.3f}")

y_pred_raw = model.predict(X_val, verbose=0).flatten()
y_pred = (y_pred_raw > 0.5).astype(np.int32)
y_val_int = y_val.astype(np.int32)

print("\nClassification Report:")
print(classification_report(y_val_int, y_pred, target_names=["not die", "die center"]))
print("Confusion Matrix:")
print(confusion_matrix(y_val_int, y_pred))

# === Save Wrong Predictions ===
from PIL import Image
wrongs_dir = "wrongs_finder"
os.makedirs(wrongs_dir, exist_ok=True)
for f in os.listdir(wrongs_dir):
    os.remove(os.path.join(wrongs_dir, f))
wrong_count = 0
for i in range(len(y_val)):
    if y_pred[i] != y_val_int[i]:
        img_array = (X_val[i, :, :, 0] * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        fname = f"true{y_val_int[i]}_pred{y_pred[i]}_conf{y_pred_raw[i]:.2f}_{i}.png"
        img.save(os.path.join(wrongs_dir, fname))
        wrong_count += 1
print(f"\nSaved {wrong_count} wrong predictions to {wrongs_dir}/")

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
