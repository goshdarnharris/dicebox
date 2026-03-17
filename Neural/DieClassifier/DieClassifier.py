import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# === Settings ===
_model_path = os.path.join(os.path.dirname(__file__), "dice_cnn.onnx")
_input_size = 20

# === Load model once on import ===
_session = ort.InferenceSession(_model_path)
_input_name = _session.get_inputs()[0].name


def identify_die(pil_image, input_size=_input_size):
    """
    Classify a cropped die image.

    Args:
        pil_image: PIL Image of a single die crop (any size, will be resized)
        input_size: size to resize to before inference

    Returns:
        (face_value, confidence) where face_value is 0-6 (0 = not a die)
        and confidence is the softmax probability.
    """
    gray = pil_image.convert("L").resize((input_size, input_size))
    # ONNX model expects (N, C, H, W) float32
    arr = np.array(gray, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 1, input_size, input_size)
    output = _session.run(None, {_input_name: arr})[0]
    # Output is (1, 7, 1, 1) — squeeze to (7,)
    logits = output.squeeze()
    # Apply softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    face_value = int(np.argmax(probs))
    confidence = float(probs[face_value])
    return face_value, confidence, probs


def _softmax_axis0(logits):
    """Apply softmax along axis 0."""
    exp = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    return exp / exp.sum(axis=0, keepdims=True)


def classify_image(pil_image, downsample=9):
    """
    Run the classifier over an entire image using 16 shifted passes for stride-9 resolution.

    The model's native output stride is 36px (9x downsample * 4x from two MaxPools).
    To get 9px resolution, we run 16 passes (4x4 shifts of 1px in downsampled space)
    and interleave the results.

    Args:
        pil_image: PIL Image (RGB or grayscale) at original resolution
        downsample: factor to shrink image before feeding to CNN

    Returns:
        probs: (H, W, 7) numpy array of softmax probabilities at stride-9 resolution
        stride: effective pixel stride in original image coordinates (= downsample = 9)
    """
    gray = pil_image.convert("L")
    orig_w, orig_h = gray.size
    small_w, small_h = orig_w // downsample, orig_h // downsample

    small = gray.resize((small_w, small_h))
    arr = np.array(small, dtype=np.float32) / 255.0

    # Pad so the model can produce output covering the full image.
    # The model needs 20 input pixels to produce 1 output pixel.
    # With 2 MaxPool(2), each output pixel spans 4 input pixels.
    # To center the first output on the image edge, pad by half the receptive field.
    pad = _input_size // 2  # = 10
    arr = np.pad(arr, pad, mode='edge')

    pool_stride = 4  # 2^(num_maxpools)

    # The output grid covers all positions in the downsampled image at stride 1.
    # Each position (r, c) in the grid corresponds to a classifier centered at
    # downsampled pixel (r, c), i.e., original pixel (r * downsample, c * downsample).
    out_h = small_h
    out_w = small_w
    full_logits = np.full((7, out_h, out_w), -1e9, dtype=np.float32)  # fill with large negative (softmax → ~0)

    for dy in range(pool_stride):
        for dx in range(pool_stride):
            # Crop starting at offset (dy, dx) in the padded image.
            # Trim so dimensions are compatible with the model (need 20 + n*4 pixels).
            cropped = arr[dy:, dx:]
            crop_h = (cropped.shape[0] // pool_stride) * pool_stride
            crop_w = (cropped.shape[1] // pool_stride) * pool_stride
            cropped = cropped[:crop_h, :crop_w]

            if crop_h < _input_size or crop_w < _input_size:
                continue

            inp = cropped.reshape(1, 1, crop_h, crop_w).astype(np.float32)
            output = _session.run(None, {_input_name: inp})[0][0]  # (7, oh, ow)
            oh, ow = output.shape[1], output.shape[2]

            # The model's first output pixel corresponds to the receptive field
            # starting at the beginning of `cropped`. In the padded image, that's
            # position (dy, dx). The center of that receptive field in the padded
            # image is at (dy + pad, dx + pad) with pool stride 4.
            # But we padded by `pad`, so in the original downsampled image, the
            # center is at (dy + pad - pad, dx + pad - pad) = (dy, dx).
            # Wait -- the receptive field center for output pixel 0 is at
            # input pixel (pad-1) in the cropped array (center of the 20-pixel window).
            # In the padded array, that's (dy + pad - 1). Subtracting the pad we added,
            # in the original downsampled image that's (dy - 1).
            #
            # Actually let's think more carefully:
            # - The model sees a 20-pixel window and produces 1 output.
            # - With MaxPool stride 4, output[r] corresponds to input pixels [r*4, r*4+20).
            # - The center of that window is at input pixel r*4 + 10 - 1 = r*4 + 9.
            # - In the padded array, that's dy + r*4 + 9.
            # - In the original array (before padding), that's dy + r*4 + 9 - pad = dy + r*4 - 1.
            #
            # Hmm, off-by-one is tricky. Let's use: center = r*4 + 9 for a 20-wide window.
            # In original coords: orig_r = dy + r*4 + 9 - pad = dy + r*4 - 1

            for r in range(oh):
                for c in range(ow):
                    orig_r = dy + r * pool_stride
                    orig_c = dx + c * pool_stride
                    if 0 <= orig_r < out_h and 0 <= orig_c < out_w:
                        full_logits[:, orig_r, orig_c] = output[:, r, c]

    # Apply softmax across class dimension
    probs = _softmax_axis0(full_logits)  # (7, H, W)
    probs = probs.transpose(1, 2, 0)    # (H, W, 7)

    stride = downsample
    return probs, stride


# === CLI for quick testing ===
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python DieClassifier.py <image_path>")
        sys.exit(1)

    img = Image.open(path)
    face, conf = identify_die(img)
    if face == 0:
        print(f"Not a die (confidence={conf:.3f})")
    else:
        print(f"Die face: {face} (confidence={conf:.3f})")
