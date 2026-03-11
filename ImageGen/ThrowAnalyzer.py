import numpy as np
from PIL import Image
from DieFinder import find_dice
from DieIdentifier import identify_die

# === Settings ===
crop_size = 180


def average_border_fill(image, box):
    """Crop with padding if box extends past image edges."""
    avg_color = tuple(np.array(image).reshape(-1, 3).mean(axis=0).astype(np.uint8))
    padded = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), avg_color)
    left = max(box[0], 0)
    upper = max(box[1], 0)
    right = min(box[2], image.width)
    lower = min(box[3], image.height)
    cropped = image.crop((left, upper, right, lower))
    paste_x = left - box[0]
    paste_y = upper - box[1]
    padded.paste(cropped, (paste_x, paste_y))
    return padded


def analyze_throw(pil_image, crop_sz=crop_size):
    """
    Analyze a full image of dice.

    Args:
        pil_image: PIL Image (RGB) of the full throw
        crop_sz: size of the crop around each detected die center

    Returns:
        List of (x, y, face_value, die_confidence, id_confidence) tuples.
        face_value is 1-6 (results with face_value=0 are filtered out).
    """
    image = pil_image.convert("RGB")
    positions = find_dice(image)

    results = []
    half = crop_sz // 2
    for x, y, die_conf in positions:
        box = (x - half, y - half, x + half, y + half)
        crop = average_border_fill(image, box)
        face, id_conf = identify_die(crop)
        if face > 0:
            results.append((x, y, face, die_conf, id_conf))

    return results


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
      for x, y, face, die_conf, id_conf in results:
          print(f"  ({x:4d}, {y:4d})  face={face}  finder={die_conf:.3f}  identifier={id_conf:.3f}")
          counts[face] += 1

      print(f"\nSummary: {sum(counts)} dice")
      for i in range(1, 7):
          if counts[i] > 0:
              print(f"  {counts[i]}x {i}'s")
