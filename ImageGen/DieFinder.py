import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import os

# === Settings ===
_model_path = os.path.join(os.path.dirname(__file__), "die_finder_cnn.keras")
_downsample = 9
_gaussian_sigma = 0.75  # sigma for smoothing the heatmap before peak detection
_min_distance = 3     # minimum pixels between peaks in downsampled space (~90px in original)
_threshold = 0.1       # minimum heatmap value to count as a detection

# === Load model once on import ===
_model = tf.keras.models.load_model(_model_path, compile=False)


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
    arr = np.array(small, dtype=np.float32) / 255.0
    arr = arr.reshape(1, small_h, small_w, 1)

    # Run CNN to get heatmap
    heatmap = _model.predict(arr, verbose=0)[0, :, :, 0]

    # Smooth with matched Gaussian filter to suppress noise
    heatmap_smooth = gaussian_filter(heatmap, sigma=gaussian_sigma)

    # Find local maxima
    coords = peak_local_max(
        heatmap_smooth,
        min_distance=min_distance,
        threshold_abs=threshold,
    )

    # Build results: scale coordinates back to original image space
    results = []
    for row, col in coords:
        confidence = float(heatmap_smooth[row, col])
        orig_x = int(col * downsample * 2 + downsample)
        orig_y = int(row * downsample * 2 + downsample)
        results.append((orig_x, orig_y, confidence))

    # Sort by confidence descending
    results.sort(key=lambda r: r[2], reverse=True)
    return results


# === CLI for quick testing ===
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "C:\\Users\\james\\OneDrive\\Documents\\dicebox_git\\images\\final_box_20_dice\\04.jpg"
    img = Image.open(path)
    dice = find_dice(img)
    print(f"Found {len(dice)} dice:")
    for x, y, conf in dice:
        print(f"  ({x}, {y}) confidence={conf:.3f}")
