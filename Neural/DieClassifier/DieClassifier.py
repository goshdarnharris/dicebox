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
    output = _session.run(None, {_input_name: arr})[0][0]  # raw logits
    # Apply softmax
    exp = np.exp(output - np.max(output))
    probs = exp / exp.sum()
    face_value = int(np.argmax(probs))
    confidence = float(probs[face_value])
    return face_value, confidence


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
