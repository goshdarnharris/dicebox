import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.feature import peak_local_max, corner_subpix
from scipy.ndimage import gaussian_filter
import os

# === Settings ===
_model_path = os.path.join(os.path.dirname(__file__), "die_finder_cnn.tflite")
_downsample = 9
_gaussian_sigma = 0.75  # sigma for smoothing the heatmap before peak detection
_min_distance = 2     # minimum pixels between peaks in downsampled space (~90px in original)
_threshold = 0.1       # minimum heatmap value to count as a detection

# === Load model once on import ===
_interpreter = tf.lite.Interpreter(model_path=_model_path)
_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()


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

    # Resize interpreter for this input size (fully conv model accepts any size)
    input_shape = [1, small_h, small_w, 1]
    _interpreter.resize_tensor_input(_input_details[0]['index'], input_shape)
    _interpreter.allocate_tensors()

    input_dtype = _input_details[0]['dtype']
    if input_dtype == np.uint8:
        arr = np.array(small, dtype=np.uint8).reshape(input_shape)
    else:
        arr = (np.array(small, dtype=np.float32) / 255.0).reshape(input_shape)

    _interpreter.set_tensor(_input_details[0]['index'], arr)
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_details[0]['index'])[0, :, :, 0]

    # Dequantize if int8 output
    if output.dtype == np.uint8:
        out_detail = _output_details[0]
        scale, zero_point = out_detail['quantization']
        heatmap = (output.astype(np.float32) - zero_point) * scale
    else:
        heatmap = output

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

    path = sys.argv[1] if len(sys.argv) > 1 else "C:\\Users\\james\\OneDrive\\Documents\\dicebox_git\\images\\final_box_20_dice\\04.jpg"
    img = Image.open(path)
    dice = find_dice(img)
    print(f"Found {len(dice)} dice:")
    for x, y, conf in dice:
        print(f"  ({x}, {y}) confidence={conf:.3f}")
