import os
import json
import numpy as np
from PIL import Image

# === Settings ===
output_dir = "targets"
gaussian_sigma = 10  # (sigma * 2 = visiblish width)


def generate_heatmap(w, h, centers):
    """Generate a Gaussian heatmap for the given die center positions."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if centers:
        yy, xx = np.mgrid[0:h, 0:w]
        for cx, cy in centers:
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            heatmap += np.exp(-dist_sq / (2 * gaussian_sigma ** 2))
        heatmap = np.clip(heatmap, 0, 1)
    return Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")


def save_heatmap(w, h, centers, safe_base, out_dir=output_dir):
    """Generate and save a heatmap target image."""
    os.makedirs(out_dir, exist_ok=True)
    img = generate_heatmap(w, h, centers)
    out_name = f"target_{safe_base}.png"
    img.save(os.path.join(out_dir, out_name))
    return out_name


# === CLI: regenerate all targets from die_locations.json ===
if __name__ == "__main__":
    image_dir = '../training_images'
    locations_file = "die_locations.json"

    with open(locations_file, "r") as f:
        all_locations = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for rel_path, centers in all_locations.items():
        img = Image.open(os.path.join(image_dir, rel_path))
        w, h = img.size
        safe_name = rel_path.replace("/", "_").replace("\\", "_")
        safe_base = os.path.splitext(safe_name)[0]
        out_name = save_heatmap(w, h, centers, safe_base)
        print(f"{out_name}: {len(centers)} dice, {w}x{h}")

    print(f"\nGenerated {len(all_locations)} target heatmaps in {output_dir}/")
