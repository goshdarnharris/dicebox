import numpy as np
import tensorflow as tf
from PIL import Image
import os

# === Settings ===
_model_path = os.path.join(os.path.dirname(__file__), "dice_cnn.tflite")
_input_size = 20

# === Load model once on import ===
_interpreter = tf.lite.Interpreter(model_path=_model_path)
_interpreter.allocate_tensors()
_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()


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
    arr = np.array(gray, dtype=np.float32) / 255.0
    arr = arr.reshape(1, input_size, input_size, 1)
    _interpreter.set_tensor(_input_details[0]['index'], arr)
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_details[0]['index'])[0]
    face_value = int(np.argmax(output))
    confidence = float(output[face_value])
    return face_value, confidence


# === CLI for quick testing ===
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python DieIdentifier.py <image_path>")
        sys.exit(1)

    img = Image.open(path)
    face, conf = identify_die(img)
    if face == 0:
        print(f"Not a die (confidence={conf:.3f})")
    else:
        print(f"Die face: {face} (confidence={conf:.3f})")
