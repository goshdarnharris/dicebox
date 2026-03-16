import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max, corner_subpix
import os

# === Settings ===
_model_path = os.path.join(os.path.dirname(__file__), "DieClassifier", "dice_cnn.onnx")
_input_size = 20
_crop_size = 180
_stride = 18  # pixels between each classifier evaluation
_downsample = 9  # crop_size / input_size
_gaussian_sigma = 0.5
_min_distance = 54//_stride  # in heatmap space (~36px in original)
_threshold = 0.7  # minimum softmax confidence to count

# === Load model once on import ===
_session = ort.InferenceSession(_model_path)
_input_name = _session.get_inputs()[0].name


def _softmax(logits):
    """Apply softmax along last axis."""
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def analyze_throw(pil_image):
    """
    Analyze a full image of dice by sliding the classifier across the image.

    Args:
        pil_image: PIL Image (RGB) of the full throw

    Returns:
        List of (x, y, face_value, confidence, confidence) tuples.
        face_value is 1-6. Two confidence values for compatibility with ManualDiePicker.
    """
    gray = np.array(pil_image.convert("L"), dtype=np.float32) / 255.0
    img_h, img_w = gray.shape
    half = _crop_size // 2

    # Pad image so we can center crops on any pixel including edges
    padded = np.pad(gray, half, mode='edge')

    # Extract all patches at stride intervals across the full original image
    patches = []
    positions = []  # (x_center, y_center) in original image coords
    for y in range(0, img_h, _stride):
        for x in range(0, img_w, _stride):
            # In padded coords, the center is at (x + half, y + half)
            patch = padded[y:y + _crop_size, x:x + _crop_size]
            # Downsample using simple reshaping (crop_size/downsample = input_size)
            small = patch.reshape(_input_size, _downsample, _input_size, _downsample).mean(axis=(1, 3))
            patches.append(small)
            positions.append((x, y))

    if not patches:
        return []

    # Run classifier on all patches in one batch
    batch = np.array(patches, dtype=np.float32).reshape(-1, 1, _input_size, _input_size)
    logits = _session.run(None, {_input_name: batch})[0]  # (N, 7)
    probs = _softmax(logits)  # (N, 7)

    # Build 6-channel heatmap (one per face value 1-6)
    grid_h = len(range(0, img_h, _stride))
    grid_w = len(range(0, img_w, _stride))
    heatmaps = probs[:, 1:7].reshape(grid_h, grid_w, 6)  # ignore class 0

    # Find peaks in each channel
    results = []
    for face_idx in range(6):
        face_value = face_idx + 1
        hmap = heatmaps[:, :, face_idx]

        # Smooth to merge nearby responses
        hmap_smooth = gaussian_filter(hmap, sigma=_gaussian_sigma)

        coords = peak_local_max(
            hmap_smooth,
            min_distance=_min_distance,
            threshold_abs=_threshold,
        )

        if len(coords) == 0:
            continue

        # Sub-pixel refinement
        refined = corner_subpix(hmap_smooth, coords, window_size=5)

        for i, (row, col) in enumerate(coords):
            confidence = float(hmap_smooth[row, col])
            sub_row, sub_col = refined[i]
            if np.isnan(sub_row):
                sub_row, sub_col = float(row), float(col)
            # Convert grid coords back to original image coords
            orig_x = int(round(sub_col * _stride))
            orig_y = int(round(sub_row * _stride))
            results.append((orig_x, orig_y, face_value, confidence, confidence))

    # Remove duplicate detections where multiple face channels found the same location
    # Keep the one with highest confidence
    results.sort(key=lambda r: r[3], reverse=True)
    filtered = []
    for r in results:
        too_close = any(
            ((r[0] - f[0]) ** 2 + (r[1] - f[1]) ** 2) ** 0.5 < _crop_size // 2
            for f in filtered
        )
        if not too_close:
            filtered.append(r)

    return filtered


# === CLI ===
if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else ["C:\\Users\\james\\OneDrive\\Documents\\dicebox_git\\images\\final_box_20_dice\\04.jpg"]
    for path in paths:
      img = Image.open(path)

      print(f"Analyzing {path}...")
      results = analyze_throw(img)

      print(f"\nFound {len(results)} dice:")
      counts = [0] * 7
      for x, y, face, conf, _ in results:
          print(f"  ({x:4d}, {y:4d})  face={face}  confidence={conf:.3f}")
          counts[face] += 1

      print(f"\nSummary: {sum(counts)} dice")
      for i in range(1, 7):
          if counts[i] > 0:
              print(f"  {counts[i]}x {i}'s")
