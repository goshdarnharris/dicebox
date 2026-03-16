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
    return face_value, confidence


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

    # Pad so the model can produce output for edge positions.
    # Pad by 10 (half of 20px receptive field) so output covers the full image.
    pad = 10
    small = gray.resize((small_w, small_h))
    arr = np.array(small, dtype=np.float32) / 255.0
    arr = np.pad(arr, pad, mode='edge')

    # Native model stride in downsampled space is 4 (two MaxPool2d(2)).
    # We want stride 1 in downsampled space (= 9px in original space).
    # Run 4x4=16 shifted passes and interleave results.
    pool_stride = 4  # 2^(num_maxpools)
    padded_h, padded_w = arr.shape

    # Calculate output size for the full-resolution heatmap
    out_h = (padded_h - 20) // 1 + 1  # every pixel position in downsampled space
    out_w = (padded_w - 20) // 1 + 1
    full_logits = np.zeros((7, out_h, out_w), dtype=np.float32)

    for dy in range(pool_stride):
        for dx in range(pool_stride):
            # Crop the padded image with this offset
            cropped = arr[dy:, dx:]
            # Trim to make dimensions compatible with the model (divisible by pool_stride)
            crop_h = (cropped.shape[0] // pool_stride) * pool_stride
            crop_w = (cropped.shape[1] // pool_stride) * pool_stride
            cropped = cropped[:crop_h, :crop_w]

            inp = cropped.reshape(1, 1, crop_h, crop_w).astype(np.float32)
            output = _session.run(None, {_input_name: inp})[0][0]  # (7, H', W')

            # Place results into the full-resolution grid
            # This output's positions in the full grid are at dy, dy+4, dy+8, ...
            oh, ow = output.shape[1], output.shape[2]
            for r in range(oh):
                for c in range(ow):
                    full_r = dy + r * pool_stride
                    full_c = dx + c * pool_stride
                    if full_r < out_h and full_c < out_w:
                        full_logits[:, full_r, full_c] = output[:, r, c]

    # Apply softmax across class dimension
    probs = _softmax_axis0(full_logits)  # (7, H, W)

    # Transpose to (H, W, 7)
    probs = probs.transpose(1, 2, 0)

    # Effective stride is now 1 in downsampled space = 9 in original space
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
