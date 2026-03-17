import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# === Settings ===
dataset_file = "augmented_training.h5"
input_size = 20  # 180 image size reduced 9x by augmentation script
num_classes = 7
batch_size = 1024*8
epochs = 10000
patience = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")
print(f"Using device: {device}")


# === Model ===
# Fully convolutional — no Flatten or Linear layers.
# On 20x20 input: produces 7x1x1 (same as before).
# On full downsampled image: produces a 7-channel heatmap in a single pass.
# Output stride = 9 (downsample) * 4 (two MaxPool2d) = 36 pixels in original image space.
DiceCNN = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),                          # -> 8x20x20
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # -> 16x10x10
    #nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.01),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.05), nn.MaxPool2d(2),  # -> 32x5x5
    nn.Conv2d(32, 64, 5), nn.ReLU(), nn.Dropout2d(0.05),               # -> 64x1x1 (replaces Flatten+Linear)
    nn.Conv2d(64, num_classes, 1),                                      # -> 7x1x1 (replaces final Linear)
)


# === Load Data ===
print("Loading dataset...")
with h5py.File(dataset_file, "r") as hf:
    X = hf["images"][:].astype(np.float32)  # (N, 20, 20)
    y = hf["labels"][:].astype(np.int64)
    sources = hf["sources"][:].astype(str)  # "rel_path:x,y"
print(f"Loaded {len(X)} images")
for c in range(num_classes):
    print(f"  Class {c}: {np.sum(y == c)} images")

X_train, X_val, y_train, y_val, src_train, src_val = train_test_split(
    X, y, sources, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# === Class Weights ===
counts = np.bincount(y_train, minlength=num_classes)
total = len(y_train)
class_weight = {c: total / (num_classes * count) for c, count in enumerate(counts) if count > 0}
print(f"Class weights: {class_weight}")
weight_tensor = torch.tensor(
    [class_weight.get(c, 1.0) for c in range(num_classes)], dtype=torch.float32
).to(device)

# === Prepare DataLoaders ===
# PyTorch conv2d expects (N, C, H, W), so add channel dim at axis=1
X_train_t = torch.from_numpy(X_train[:, np.newaxis, :, :])  # (N, 1, 20, 20)
y_train_t = torch.from_numpy(y_train)
X_val_t = torch.from_numpy(X_val[:, np.newaxis, :, :])
y_val_t = torch.from_numpy(y_val)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# === Build Model ===
model = DiceCNN.to(device)
pt_path = "dice_cnn.pt"
if os.path.exists(pt_path):
    model.load_state_dict(torch.load(pt_path, map_location=device))
    print(f"Resumed from {pt_path}")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=250, min_lr=1e-5, verbose=True
)

# === Train ===
best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0
interrupted = False

try:
    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(-1).squeeze(-1)  # (N, 7, 1, 1) -> (N, 7)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total += len(xb)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb).squeeze(-1).squeeze(-1)
                loss = criterion(out, yb)
                val_loss += loss.item() * len(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += len(xb)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:4d}/{epochs} - "
            f"loss: {train_loss:.4f} acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} - "
            f"lr: {current_lr:.2e}"
        )

        # --- EarlyStopping with restore_best_weights ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
except KeyboardInterrupt:
    print(f"\n\nInterrupted at epoch {epoch}. Saving best weights...")

# Restore best weights
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)
print(f"Best validation loss: {best_val_loss:.4f}")

# === Evaluate ===
model.eval()
all_preds = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = model(xb).squeeze(-1).squeeze(-1)
        all_preds.append(out.argmax(1).cpu().numpy())

y_pred = np.concatenate(all_preds)
val_acc = np.mean(y_pred == y_val)
print(f"\nValidation accuracy: {val_acc:.3f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=[str(i) for i in range(num_classes)]))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# === Save Wrong Predictions ===
wrongs_dir = "wrong_results"
os.makedirs(wrongs_dir, exist_ok=True)
# Clear previous wrong_results
for f in os.listdir(wrongs_dir):
    os.remove(os.path.join(wrongs_dir, f))
wrong_count = 0
for i in range(len(y_val)):
    if y_pred[i] != y_val[i]:
        img_array = (X_val[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        # Source format: "rel_path:x,y" — make filesystem safe
        safe_src = src_val[i].replace("/", "_").replace("\\", "_").replace(":", "_")
        fname = f"true{y_val[i]}_pred{y_pred[i]}_{safe_src}.png"
        img.save(os.path.join(wrongs_dir, fname))
        wrong_count += 1
print(f"\nSaved {wrong_count} wrong predictions to {wrongs_dir}/")

# === Save PyTorch Model ===
pt_path = "dice_cnn.pt"
torch.save(model.state_dict(), pt_path)
print(f"\nSaved PyTorch model: {pt_path}")

# === Export to ONNX ===
onnx_path = "dice_cnn.onnx"
model.eval()
model.to("cpu")
dummy_input = torch.randn(1, 1, input_size, input_size)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},
    opset_version=17,
)
print(f"Saved ONNX model: {onnx_path}")

# === TFLite Conversion ===
# To convert the ONNX model to int8 quantized TFLite, use onnx2tf:
#
#   pip install onnx2tf
#   onnx2tf -i dice_cnn.onnx -oiqt -qt per-tensor
#
# This will produce a saved_model directory with a TFLite file inside.
# For full int8 quantization with a representative dataset, you can also use:
#
#   import tensorflow as tf
#   converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
#   converter.optimizations = [tf.lite.Optimize.DEFAULT]
#   def representative_data():
#       for i in range(min(200, len(X_train))):
#           yield [X_train[i:i+1][np.newaxis]]  # (1, 1, 20, 20) float32
#   converter.representative_dataset = representative_data
#   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#   converter.inference_input_type = tf.uint8
#   converter.inference_output_type = tf.uint8
#   tflite_model = converter.convert()
#   with open("dice_cnn.tflite", "wb") as f:
#       f.write(tflite_model)
