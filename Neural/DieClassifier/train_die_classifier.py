import os
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# === Settings ===
dataset_file = "augmented_training.h5"
input_size = 20 # 180 image size reduced 9x by augmentation script
num_classes = 7
batch_size = 1024
epochs = 2000
patience = 100

# === Load Data ===
print("Loading dataset...")
with h5py.File(dataset_file, "r") as hf:
    X = hf["images"][:][..., np.newaxis]  # (N, 20, 20, 1)
    y = hf["labels"][:]
print(f"Loaded {len(X)} images")
for c in range(num_classes):
    print(f"  Class {c}: {np.sum(y == c)} images")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# === Class Weights ===
counts = np.bincount(y_train, minlength=num_classes)
total = len(y_train)
class_weight = {c: total / (num_classes * count) for c, count in enumerate(counts) if count > 0}
print(f"Class weights: {class_weight}")

# === Model ===
def build_conv_layers(x):
    # Input: 20x20x1 (grayscale). Pips are ~3px wide at this scale.

    # Conv2D: slide 8 different 5x5 filters across the image, each learning to detect
    # a different low-level pattern (edges, curves, dark spots). "same" padding keeps
    # the output the same spatial size. ReLU zeroes out negative values.
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    # 20x20x8

    # Second conv: 8 filters of 3x3, combining the 8 edge maps into mid-level
    # features (pip shapes, corners of dice).
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    # 10x10x16

    # Third conv: 16 filters of 3x3, combining mid-level features into high-level
    # patterns (pip arrangements, die structure).
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    # 5x5x32

    # Flatten: reshape
    # so they can be fed into a Dense (fully connected) layer.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    return x

inputs = tf.keras.Input(shape=(input_size, input_size, 1))
x = inputs  # augmentation is pre-applied by augment.py
x = build_conv_layers(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
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

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=[str(i) for i in range(num_classes)]))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# === Save Wrong Predictions ===
from PIL import Image
wrongs_dir = "wrong_results"
os.makedirs(wrongs_dir, exist_ok=True)
# Clear previous wrong_results
for f in os.listdir(wrongs_dir):
    os.remove(os.path.join(wrongs_dir, f))
wrong_count = 0
for i in range(len(y_val)):
    if y_pred[i] != y_val[i]:
        img_array = (X_val[i, :, :, 0] * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        fname = f"true{y_val[i]}_pred{y_pred[i]}_{i}.png"
        img.save(os.path.join(wrongs_dir, fname))
        wrong_count += 1
print(f"\nSaved {wrong_count} wrong predictions to {wrongs_dir}/")

# === Save ===
keras_path = "dice_cnn.keras"
model.save(keras_path)
print(f"\nSaved Keras model: {keras_path}")

# Export TFLite (int8 quantized)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_data():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1]]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
tflite_path = "dice_cnn.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model (int8): {tflite_path}")
