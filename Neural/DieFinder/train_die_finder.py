import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image

# === Settings ===
dataset_file = "augmented_training.h5"
epochs = 2000
patience = 200
batch_size = 64
initial_lr = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# === Model ===
# Fully convolutional — no Dense or Flatten layers.
# Input: (N, 1, H, W) downsampled grayscale image
# Output: (N, 1, H/2, W/2) heatmap with sigmoid activation

class DieFinderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=15, padding=7)    # same padding
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, padding=3)    # same padding
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # same padding
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, padding=0)    # same padding

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x


# === Dataset ===

class DieFinderDataset(Dataset):
    def __init__(self, images, targets):
        # images: list of (H, W) float32, targets: list of (H/2, W/2) float32
        # Convert to torch tensors with channel dim: (1, H, W)
        self.images = [torch.from_numpy(img[np.newaxis, :, :]) for img in images]
        self.targets = [torch.from_numpy(tgt[np.newaxis, :, :]) for tgt in targets]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


# === Load Data ===
# Images are already downsampled by augment.py.
# Target heatmaps are downsampled to match CNN output size (1/2 of input due to 1 MaxPool).

print("Loading dataset...")
with h5py.File(dataset_file, "r") as hf:
    raw_images = hf["images"][:]
    raw_targets = hf["targets"][:]
print(f"Loaded {len(raw_images)} image/target pairs")

# Downsample targets to CNN output size (1/2 due to MaxPool)
inputs = []
targets = []
for i in range(len(raw_images)):
    src_arr = raw_images[i]
    h, w = src_arr.shape
    out_h, out_w = h // 2, w // 2

    tgt = Image.fromarray((raw_targets[i] * 255).astype(np.uint8), mode="L")
    tgt_small = tgt.resize((out_w, out_h), Image.BILINEAR)
    tgt_arr = np.array(tgt_small, dtype=np.float32) / 255.0

    inputs.append(src_arr)
    targets.append(tgt_arr)

print(f"  Input: {w}x{h} -> Output: {out_w}x{out_h}")

# Split into train/val
n = len(inputs)
n_val = max(1, n // 5)
n_train = n - n_val

train_dataset = DieFinderDataset(inputs[:n_train], targets[:n_train])
val_dataset = DieFinderDataset(inputs[n_train:], targets[n_train:])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Train: {n_train}, Val: {n_val}")


# === Loss ===

def weighted_bce(y_pred, y_true):
    """Binary crossentropy weighted by (target + bias) so die centers matter more."""
    bias = 0.05
    k = 10.0
    weight = (y_true + bias) * k
    bce = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    return (bce * weight).mean()


# === Training ===

model = DieFinderCNN().to(device)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {total_params} total params, {trainable_params} trainable")
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.75, patience=30, min_lr=1e-7, verbose=True
)

# Early stopping state
best_val_loss = float("inf")
best_epoch = 0
best_state = None

for epoch in range(1, epochs + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    train_batches = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = weighted_bce(pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += (pred - y_batch).abs().mean().item()
        train_batches += 1

    train_loss /= train_batches
    train_mae /= train_batches

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_batches = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = weighted_bce(pred, y_batch)

            val_loss += loss.item()
            val_mae += (pred - y_batch).abs().mean().item()
            val_batches += 1

    val_loss /= val_batches
    val_mae /= val_batches

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch}/{epochs} - "
        f"loss: {train_loss:.4f} - mae: {train_mae:.4f} - "
        f"val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f} - "
        f"lr: {current_lr:.2e}"
    )

    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    elif epoch - best_epoch >= patience:
        print(f"\nEarly stopping at epoch {epoch} (best was epoch {best_epoch})")
        break

# Restore best weights
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)
    print(f"Restored best weights from epoch {best_epoch}")

# === Evaluate ===
model.eval()
val_loss = 0.0
val_mae = 0.0
val_batches = 0
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        val_loss += weighted_bce(pred, y_batch).item()
        val_mae += (pred - y_batch).abs().mean().item()
        val_batches += 1

val_loss /= val_batches
val_mae /= val_batches
print(f"\nValidation loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

# === Save predicted heatmaps for inspection ===
wrongs_dir = "wrong_results"
os.makedirs(wrongs_dir, exist_ok=True)
for f in os.listdir(wrongs_dir):
    os.remove(os.path.join(wrongs_dir, f))

all_dataset = DieFinderDataset(inputs, targets)
all_loader = DataLoader(all_dataset, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(all_loader):
        x = x.to(device)
        pred = model(x).cpu().numpy()[0, 0]    # (H/2, W/2)
        actual = y.numpy()[0, 0]                # (H/2, W/2)

        pred_img = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
        actual_img = Image.fromarray((actual * 255).astype(np.uint8), mode="L")

        label = "train" if i < n_train else "val"
        combined = Image.new("L", (pred_img.width * 2, pred_img.height))
        combined.paste(actual_img, (0, 0))
        combined.paste(pred_img, (pred_img.width, 0))
        combined.save(os.path.join(wrongs_dir, f"{label}_{i}_actual_vs_pred.png"))

print(f"Saved {len(all_dataset)} actual vs predicted comparisons to {wrongs_dir}/")

# === Save PyTorch model ===
pt_path = "die_finder_cnn.pt"
torch.save(model.state_dict(), pt_path)
print(f"\nSaved PyTorch model: {pt_path}")

# === Export ONNX ===
model.eval()
model.to("cpu")
# Use actual input dimensions for the export (dynamic axes allow other sizes at runtime)
dummy_input = torch.randn(1, 1, h, w)
onnx_path = "die_finder_cnn.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=17,
)
print(f"Saved ONNX model: {onnx_path}")
