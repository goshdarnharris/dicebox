import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, center_of_mass, sum as ndsum
from DieClassifier.DieClassifier import classify_image

# === Settings ===
_gaussian_sigma = 0.5
_activation_threshold = 0.5  # minimum softmax value to count as activated
_sum_threshold = 10.0         # minimum summed confidence to keep a detection
_dedup_distance = 90         # minimum distance between detections in original image pixels


def analyze_throw(pil_image):
    """
    Analyze a full image of dice using a single-pass fully convolutional classifier.

    Args:
        pil_image: PIL Image (RGB) of the full throw

    Returns:
        List of (x, y, face_value, confidence, confidence) tuples.
        face_value is 1-6. Two confidence values for compatibility with ManualDiePicker.
    """
    # Single forward pass produces a 7-channel heatmap
    probs, stride = classify_image(pil_image)  # (H', W', 7)

    # Find dice in each face channel (1-6, skip class 0)
    results = []
    for face_idx in range(6):
        face_value = face_idx + 1
        hmap = probs[:, :, face_value]

        # Smooth and threshold to find activated regions
        hmap_smooth = gaussian_filter(hmap, sigma=_gaussian_sigma)
        mask = hmap_smooth >= _activation_threshold

        # Label connected regions
        labeled, n_regions = label(mask)
        if n_regions == 0:
            continue

        # Get center of mass and summed confidence for each region
        region_indices = range(1, n_regions + 1)
        centers = center_of_mass(hmap_smooth, labeled, region_indices)
        sums = ndsum(hmap_smooth, labeled, region_indices)

        for region_i, (com_row, com_col) in enumerate(centers):
            confidence = float(sums[region_i])
            if confidence < _sum_threshold:
                continue
            orig_x = int(round(com_col * stride))
            orig_y = int(round(com_row * stride))
            results.append((orig_x, orig_y, face_value, confidence, confidence))

    # Remove duplicate detections where multiple face channels found the same location
    # Keep the one with highest confidence
    results.sort(key=lambda r: r[3], reverse=True)
    filtered = []
    for r in results:
        too_close = any(
            ((r[0] - f[0]) ** 2 + (r[1] - f[1]) ** 2) ** 0.5 < _dedup_distance
            for f in filtered
        )
        if not too_close:
            filtered.append(r)

    return filtered


# === CLI ===
if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python ThrowAnalyzer.py <image_path> [image_path ...]")
        sys.exit(1)

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
