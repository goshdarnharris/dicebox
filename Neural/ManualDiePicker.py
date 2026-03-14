import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import os
import json
import numpy as np
from ThrowAnalyzer import analyze_throw

# Settings
image_dir = 'training_images/'
image_names = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
image_paths = [os.path.join(image_dir, name) for name in image_names]
crop_size = 180
crop_output_dir = "DieClassifier/raw_training"
locations_output = "DieFinder/die_locations.json"
os.makedirs(crop_output_dir, exist_ok=True)
processed_log = os.path.join(crop_output_dir, "processed.txt")
click_radius = crop_size // 4  # how close a click must be to an existing die to remove it

def load_processed():
    if not os.path.exists(processed_log):
        return set()
    with open(processed_log, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_processed(image_path):
    with open(processed_log, "a") as f:
        f.write(image_path + "\n")

# Load existing die locations
if os.path.exists(locations_output):
    with open(locations_output, "r") as f:
        all_locations = json.load(f)
else:
    all_locations = {}

processed = load_processed()
image_paths = [p for p in image_paths if p not in processed]
if not image_paths:
    print("All images have already been processed.")
    exit(0)

# === Globals ===
current_index = 0
original_image = None
tk_image = None
image_width = 0
image_height = 0

# Each annotation is (x, y, face_value, die_conf, id_conf)
# die_conf/id_conf are None for manually added dice
# face_value 0 = not a die (excluded from heatmap, included in classifier training)
annotations = []

# Pending click waiting for a keypress
pending_click = None  # (x, y) or None


# === Functions ===

def average_border_fill(image, box):
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

def find_nearest_annotation(x, y):
    """Find the index of the nearest annotation within click_radius, or None."""
    best_i = None
    best_dist = float('inf')
    for i, (ax, ay, *_) in enumerate(annotations):
        dist = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_i = i
    if best_dist <= click_radius:
        return best_i
    return None

def redraw_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    half = crop_size // 2
    for x, y, face, die_conf, id_conf in annotations:
        box = (x - half, y - half, x + half, y + half)
        color = "green" if face > 0 else "orange"
        canvas.create_rectangle(box[0], box[1], box[2], box[3], outline=color, width=2)
        # Center crosshair
        r = 6
        canvas.create_line(x - r, y, x + r, y, fill=color, width=2)
        canvas.create_line(x, y - r, x, y + r, fill=color, width=2)
        # Face value
        canvas.create_text(x, y - 20, text=str(face), fill=color, font=("Arial", 20, "bold"))
        # Confidences
        if die_conf is not None:
            canvas.create_text(x, y + 18, text=f"f:{die_conf:.2f} i:{id_conf:.2f}", fill=color, font=("Arial", 11))
    # Draw pending click if any
    if pending_click:
        px, py = pending_click
        box = (px - half, py - half, px + half, py + half)
        canvas.create_rectangle(box[0], box[1], box[2], box[3], outline="red", width=2)
    move_next_button()

def load_image(index):
    global original_image, tk_image, image_width, image_height, pending_click
    annotations.clear()
    pending_click = None

    original_image = Image.open(image_paths[index]).convert("RGB")
    image_width, image_height = original_image.size
    tk_image = ImageTk.PhotoImage(original_image)
    canvas.config(width=image_width, height=image_height)

    # Run ThrowAnalyzer to pre-populate annotations
    print("Running ThrowAnalyzer...")
    results = analyze_throw(original_image)
    for x, y, face, die_conf, id_conf in results:
        annotations.append((x, y, face, die_conf, id_conf))
    print(f"Auto-detected {len(results)} dice.")

    redraw_canvas()

def on_click(event):
    global pending_click
    x, y = event.x, event.y

    # Check if click is near an existing annotation
    nearest = find_nearest_annotation(x, y)
    if nearest is not None:
        removed = annotations.pop(nearest)
        print(f"Removed die at ({removed[0]}, {removed[1]}) face={removed[2]}")
        pending_click = None
        redraw_canvas()
        return

    # New die — show crop box and wait for keypress
    pending_click = (x, y)
    print(f"Click at ({x}, {y}). Awaiting number key 0-6...")
    redraw_canvas()

def on_keypress(event):
    global pending_click

    if pending_click is None or not event.char.isdigit():
        return
    digit = int(event.char)
    if digit < 0 or digit > 6:
        return

    x, y = pending_click
    annotations.append((x, y, digit, None, None))
    print(f"Added die at ({x}, {y}) face={digit}")
    pending_click = None
    redraw_canvas()

def on_undo(event):
    global pending_click
    if pending_click:
        pending_click = None
        print("Cancelled pending click.")
        redraw_canvas()
        return
    if not annotations:
        print("Nothing to undo.")
        return
    removed = annotations.pop()
    print(f"Undo: removed die at ({removed[0]}, {removed[1]}) face={removed[2]}")
    redraw_canvas()

def save_annotations():
    """Save cropped images for classifier and die locations for finder."""
    image_name = os.path.basename(image_paths[current_index])
    half = crop_size // 2

    # Save cropped images for classifier training
    for x, y, face, _, _ in annotations:
        box = (x - half, y - half, x + half, y + half)
        cropped = average_border_fill(original_image, box)
        filename = f"{face}_crop_{image_name}_{x}_{y}.png"
        cropped.save(os.path.join(crop_output_dir, filename))

    # Save die locations for finder training (exclude class-0)
    locations = [[x, y] for x, y, face, _, _ in annotations if face > 0]
    all_locations[image_name] = locations
    with open(locations_output, "w") as f:
        json.dump(all_locations, f, indent=2)

    print(f"Saved {len(annotations)} crops, {len(locations)} finder locations for {image_name}")

def next_image():
    global current_index, pending_click
    save_annotations()
    mark_processed(image_paths[current_index])
    current_index += 1
    if current_index >= len(image_paths):
        print("All images processed.")
        root.destroy()
        return
    pending_click = None
    canvas.delete("all")
    load_image(current_index)

# === GUI Setup ===
root = tk.Tk()
root.title("Dice Annotation Tool")

canvas = tk.Canvas(root)
canvas.pack()

canvas.bind("<Button-1>", on_click)
root.bind("<Key>", on_keypress)
root.bind("<BackSpace>", on_undo)

def move_next_button():
    global image_width, image_height
    next_btn = tk.Button(root, text="Next Image ▶", command=next_image)
    next_btn_window = canvas.create_window(0, 0, window=next_btn, anchor=tk.NW)
    canvas.update_idletasks()
    btn_width = next_btn.winfo_reqwidth()
    btn_height = next_btn.winfo_reqheight()
    canvas.coords(next_btn_window, image_width - btn_width - 5, image_height - btn_height - 5)

move_next_button()
load_image(current_index)
root.mainloop()
