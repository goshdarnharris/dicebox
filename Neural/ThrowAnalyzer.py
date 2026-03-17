import numpy as np
import time
from PIL import Image
from scipy.ndimage import gaussian_filter, label, center_of_mass, sum as ndsum
from DieClassifier.DieClassifier import classify_image

# === Settings ===
_gaussian_sigma = 0.5
_activation_threshold = 0.8  # minimum softmax value to count as activated
_sum_threshold = 20.0         # minimum summed confidence to keep a detection
_max_blob_area = 70          # max pixels in a blob (filters out large lens flare puddles)
_dedup_distance = 90         # minimum distance between detections in original image pixels
_ambiguity_dedup_distance = 60         # minimum distance between detections and ambiguity spots in original image pixels
_ambiguity_threshold = 0.7   # class-0 below this = "not confident it's empty"
_ambiguity_fill_radius = 2   # radius in heatmap pixels to fill around detected dice
_ambiguity_min_size = 5


def ambiguity_icon_size(area):
    """Return (radius, font_size) for an ambiguity marker given its blob area."""
    r = max(10, min(40, int(area ** 0.5 * 3)))
    f = max(12, min(36, int(area ** 0.5 * 2.5)))
    return r, f


def analyze_throw(pil_image):
    """
    Analyze a full image of dice using a single-pass fully convolutional classifier.

    Args:
        pil_image: PIL Image (RGB) of the full throw

    Returns:
        List of (x, y, face_value, confidence, confidence) tuples.
        face_value is 1-6. Two confidence values for compatibility with ManualDiePicker.
    """
    t0 = time.perf_counter()
    # Single forward pass produces a 7-channel heatmap
    probs, stride = classify_image(pil_image)  # (H', W', 7)
    t1 = time.perf_counter()

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

        # Get center of mass, summed confidence, and area for each region
        region_indices = range(1, n_regions + 1)
        centers = center_of_mass(hmap_smooth, labeled, region_indices)
        sums = ndsum(hmap_smooth, labeled, region_indices)
        areas = ndsum(np.ones_like(labeled), labeled, region_indices)

        for region_i, (com_row, com_col) in enumerate(centers):
            confidence = float(sums[region_i])
            area = int(areas[region_i])
            if confidence < _sum_threshold or area > _max_blob_area:
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

    # Find ambiguous regions: holes in class-0 that aren't explained by a detection
    class0 = probs[:, :, 0]
    class0_smooth = gaussian_filter(class0, sigma=_gaussian_sigma)

    # Start with "not confident it's empty" mask
    ambiguity_mask = class0_smooth < _ambiguity_threshold

    # Fill in holes where we found dice (these are explained, not ambiguous)
    h, w = ambiguity_mask.shape
    for x, y, *_ in filtered:
        # Convert back to heatmap coords
        hx = int(round(x / stride))
        hy = int(round(y / stride))
        r = _ambiguity_fill_radius
        y0, y1 = max(0, hy - r), min(h, hy + r + 1)
        x0, x1 = max(0, hx - r), min(w, hx + r + 1)
        ambiguity_mask[y0:y1, x0:x1] = False

    # Find remaining blobs — these are ambiguous regions
    ambiguities = []
    amb_labeled, amb_n = label(ambiguity_mask)
    if amb_n > 0:
        amb_indices = range(1, amb_n + 1)
        amb_centers = center_of_mass(ambiguity_mask.astype(float), amb_labeled, amb_indices)
        amb_areas = ndsum(np.ones_like(amb_labeled), amb_labeled, amb_indices)
        for i, (com_row, com_col) in enumerate(amb_centers):
            area = int(amb_areas[i])
            if area < _ambiguity_min_size:# or area > _max_blob_area:
                continue
            ax = int(round(com_col * stride))
            ay = int(round(com_row * stride))
            # Don't flag ambiguity too close to a detected die
            too_close = any(((ax - f[0])**2 + (ay - f[1])**2)**0.5 < _dedup_distance for f in filtered)
            if not too_close:
                ambiguities.append((ax, ay, area))

    t2 = time.perf_counter()
    print(f"ThrowAnalyzer: CNN {(t1-t0)*1000:.0f}ms, post-process {(t2-t1)*1000:.0f}ms, total {(t2-t0)*1000:.0f}ms, {len(ambiguities)} ambiguous")
    return filtered, ambiguities


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
      results, ambiguities = analyze_throw(img)

      print(f"\nFound {len(results)} dice, {len(ambiguities)} ambiguous:")
      counts = [0] * 7
      for x, y, face, conf, _ in results:
          print(f"  ({x:4d}, {y:4d})  face={face}  confidence={conf:.3f}")
          counts[face] += 1

      print(f"\nSummary: {sum(counts)} dice")
      for i in range(1, 7):
          if counts[i] > 0:
              print(f"  {counts[i]}x {i}'s")
