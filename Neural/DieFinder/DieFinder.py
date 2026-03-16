import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage.feature import peak_local_max, corner_subpix
from scipy.ndimage import gaussian_filter
import os

# === Settings ===
_model_path = os.path.join(os.path.dirname(__file__), "die_finder_cnn.onnx")
_downsample = 9
_gaussian_sigma = 0.75  # sigma for smoothing the heatmap before peak detection
_min_distance = 2     # minimum pixels between peaks in downsampled space (~90px in original)
_threshold = 0.1       # minimum heatmap value to count as a detection

# === Load model once on import ===
_session = ort.InferenceSession(_model_path)
_input_name = _session.get_inputs()[0].name


def find_dice(pil_image, downsample=_downsample, gaussian_sigma=_gaussian_sigma,
              min_distance=_min_distance, threshold=_threshold):
    """
    Find dice centers in a PIL image.

    Args:
        pil_image: PIL Image (RGB or grayscale) at original resolution
        downsample: factor to shrink image before feeding to CNN
        gaussian_sigma: smoothing applied to heatmap before peak finding
        min_distance: minimum pixels between detected peaks (in downsampled space)
        threshold: minimum heatmap confidence to count as a detection

    Returns:
        List of (x, y, confidence) tuples in original image coordinates,
        sorted by confidence descending.
    """
    # Downsample and prepare input
    gray = pil_image.convert("L")
    orig_w, orig_h = gray.size
    small_w, small_h = orig_w // downsample, orig_h // downsample
    small = gray.resize((small_w, small_h))

    # ONNX model expects (N, C, H, W) float32
    arr = np.array(small, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 1, small_h, small_w)

    output = _session.run(None, {_input_name: arr})[0]
    heatmap = output[0, 0, :, :]  # (H/2, W/2)

    # Smooth with matched Gaussian filter to suppress noise
    heatmap_smooth = gaussian_filter(heatmap, sigma=gaussian_sigma)

    # Find local maxima
    coords = peak_local_max(
        heatmap_smooth,
        min_distance=min_distance,
        threshold_abs=threshold,
    )

    # Refine peaks to sub-pixel precision, then scale back to original image space
    refined = corner_subpix(heatmap_smooth, coords, window_size=5)
    results = []
    for i, (row, col) in enumerate(coords):
        confidence = float(heatmap_smooth[row, col])
        sub_row, sub_col = refined[i]
        # Fall back to integer coords if refinement failed (NaN)
        if np.isnan(sub_row):
            sub_row, sub_col = float(row), float(col)
        orig_x = int(round(sub_col * downsample * 2 + downsample))
        orig_y = int(round(sub_row * downsample * 2 + downsample))
        results.append((orig_x, orig_y, confidence))

    # Sort by confidence descending
    results.sort(key=lambda r: r[2], reverse=True)
    return results


# === CLI for quick testing ===
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python DieFinder.py <image_path>")
        sys.exit(1)

    img = Image.open(path)
    dice = find_dice(img)
    print(f"Found {len(dice)} dice:")
    for x, y, conf in dice:
        print(f"  ({x}, {y}) confidence={conf:.3f}")
