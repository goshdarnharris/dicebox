import os
import json
import numpy as np
from PIL import Image

# === Settings ===
image_dir = '../../images/final_box_20_dice/'
locations_file = "die_locations.json"
output_dir = "targets"
gaussian_sigma = 10  # (sigma * 2 = visiblish width)

# === Load die locations ===
with open(locations_file, "r") as f:
    all_locations = json.load(f)

os.makedirs(output_dir, exist_ok=True)

for image_name, centers in all_locations.items():
    # Get dimensions from original image
    img = Image.open(os.path.join(image_dir, image_name))
    w, h = img.size

    # Build heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in centers:
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        heatmap += np.exp(-dist_sq / (2 * gaussian_sigma ** 2))

    # Clip to [0, 1] in case overlapping Gaussians exceed 1
    heatmap = np.clip(heatmap, 0, 1)

    # Save as grayscale image
    out = (heatmap * 255).astype(np.uint8)
    out_name = f"target_{os.path.splitext(image_name)[0]}.png"
    Image.fromarray(out, mode="L").save(os.path.join(output_dir, out_name))
    print(f"{out_name}: {len(centers)} dice, {w}x{h}")

print(f"\nGenerated {len(all_locations)} target heatmaps in {output_dir}/")
