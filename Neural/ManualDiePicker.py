import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import os
import json
import numpy as np
from ThrowAnalyzer import analyze_throw
from DieClassifier.DieClassifier import identify_die

# Settings
image_dir = 'training_images'
image_paths = sorted(
    os.path.join(root, f)
    for root, _, files in os.walk(image_dir)
    for f in files
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
)
crop_size = 180
annotations_file = os.path.join(image_dir, "annotations.json")
click_radius = crop_size // 4  # how close a click must be to an existing die to remove it

# Load existing annotations
if os.path.exists(annotations_file):
    with open(annotations_file, "r") as f:
        all_annotations = json.load(f)
else:
    all_annotations = {}

# Start on the first un-annotated image, but keep all images navigable
start_index = 0
for i, p in enumerate(image_paths):
    rel = os.path.relpath(p, image_dir).replace("\\", "/")
    if rel not in all_annotations:
        start_index = i
        break
else:
    # All annotated — start on the last one
    start_index = len(image_paths) - 1

# === Globals ===
current_index = start_index
original_image = None
tk_image = None
image_width = 0
image_height = 0

# Each annotation is (x, y, face_value, die_conf, id_conf)
# die_conf/id_conf are None for manually added dice
# face_value 0 = not a die (excluded from heatmap, included in classifier training)
annotations = []

# Annotations from the previous image, for copying to same-roll images
prev_annotations = []

# Pending click waiting for a keypress
pending_click = None  # (x, y) or None


# === Functions ===

def rel_path_for(index):
    return os.path.relpath(image_paths[index], image_dir).replace("\\", "/")

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
    move_buttons()
    dice_count = sum(1 for _, _, face, _, _ in annotations if face > 0)
    rel_path = rel_path_for(current_index)
    root.title(f"Dice Annotation — {rel_path} — {dice_count} dice — [{current_index+1}/{len(image_paths)}]")

def load_image(index):
    global original_image, tk_image, image_width, image_height, pending_click
    annotations.clear()
    pending_click = None

    original_image = Image.open(image_paths[index]).convert("RGB")
    image_width, image_height = original_image.size
    tk_image = ImageTk.PhotoImage(original_image)
    canvas.config(width=image_width, height=image_height)

    rel_path = rel_path_for(index)
    if rel_path in all_annotations:
        # Load existing annotations from JSON
        for x, y, face in all_annotations[rel_path]:
            annotations.append((x, y, face, None, None))
        print(f"Loaded {len(annotations)} saved annotations for {rel_path}")
    else:
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

    # New die — crop, run classifier, add with auto-guess
    half = crop_size // 2
    box = (x - half, y - half, x + half, y + half)
    crop = average_border_fill(original_image, box)
    face, conf = identify_die(crop)
    annotations.append((x, y, face, None, conf))
    pending_click = (x, y)
    print(f"Click at ({x}, {y}). Auto-ID: {face} ({conf:.2f}). Press 0-6 to override.")
    redraw_canvas()

def on_keypress(event):
    global pending_click

    if pending_click is None or not event.char.isdigit():
        return
    digit = int(event.char)
    if digit < 0 or digit > 6:
        return

    x, y = pending_click
    # Remove the auto-guess annotation and replace with manual override
    for i in range(len(annotations) - 1, -1, -1):
        ax, ay = annotations[i][0], annotations[i][1]
        if ax == x and ay == y:
            annotations.pop(i)
            break
    annotations.append((x, y, digit, None, None))
    print(f"Override: die at ({x}, {y}) set to face={digit}")
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
    """Save annotations to the central JSON file."""
    rel_path = rel_path_for(current_index)
    all_annotations[rel_path] = [[x, y, face] for x, y, face, _, _ in annotations]
    with open(annotations_file, "w") as f:
        json.dump(all_annotations, f, indent=2)
    print(f"Saved {len(annotations)} annotations for {rel_path}")

def clear_annotations():
    global pending_click
    annotations.clear()
    pending_click = None
    print("Cleared all annotations.")
    redraw_canvas()

def rerun_auto():
    global pending_click
    annotations.clear()
    pending_click = None
    print("Running ThrowAnalyzer...")
    results = analyze_throw(original_image)
    for x, y, face, die_conf, id_conf in results:
        annotations.append((x, y, face, die_conf, id_conf))
    print(f"Auto-detected {len(results)} dice.")
    redraw_canvas()

def copy_prev():
    global pending_click
    if not prev_annotations:
        print("No previous annotations to copy.")
        return
    annotations.clear()
    annotations.extend([(x, y, face, None, None) for x, y, face, _, _ in prev_annotations if face > 0])
    pending_click = None
    print(f"Copied {len(annotations)} annotations from previous image.")
    redraw_canvas()

def next_image():
    global current_index, pending_click, prev_annotations
    prev_annotations = list(annotations)
    current_index += 1
    if current_index >= len(image_paths):
        print("All images processed.")
        root.destroy()
        return
    pending_click = None
    canvas.delete("all")
    load_image(current_index)

def prev_image():
    global current_index, pending_click
    if current_index <= 0:
        print("Already at first image.")
        return
    current_index -= 1
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

def move_buttons():
    global image_width, image_height
    # Right side buttons
    next_btn = tk.Button(root, text="Next ▶", command=next_image)
    next_btn_window = canvas.create_window(0, 0, window=next_btn, anchor=tk.NW)
    copy_btn = tk.Button(root, text="Copy Prev", command=copy_prev)
    copy_btn_window = canvas.create_window(0, 0, window=copy_btn, anchor=tk.NW)
    save_btn = tk.Button(root, text="Save", command=save_annotations, bg="#4a4", fg="white")
    save_btn_window = canvas.create_window(0, 0, window=save_btn, anchor=tk.NW)
    # Left side buttons
    prev_btn = tk.Button(root, text="◀ Prev", command=prev_image)
    prev_btn_window = canvas.create_window(0, 0, window=prev_btn, anchor=tk.NW)
    clear_btn = tk.Button(root, text="Clear", command=clear_annotations)
    clear_btn_window = canvas.create_window(0, 0, window=clear_btn, anchor=tk.NW)
    auto_btn = tk.Button(root, text="Re-Auto", command=rerun_auto)
    auto_btn_window = canvas.create_window(0, 0, window=auto_btn, anchor=tk.NW)
    canvas.update_idletasks()
    btn_height = next_btn.winfo_reqheight()
    # Right side positioning
    next_w = next_btn.winfo_reqwidth()
    copy_w = copy_btn.winfo_reqwidth()
    save_w = save_btn.winfo_reqwidth()
    canvas.coords(next_btn_window, image_width - next_w - 5, image_height - btn_height - 5)
    canvas.coords(copy_btn_window, image_width - next_w - copy_w - 15, image_height - btn_height - 5)
    canvas.coords(save_btn_window, image_width - next_w - copy_w - save_w - 25, image_height - btn_height - 5)
    # Left side positioning
    prev_w = prev_btn.winfo_reqwidth()
    clear_w = clear_btn.winfo_reqwidth()
    auto_w = auto_btn.winfo_reqwidth()
    canvas.coords(prev_btn_window, 5, image_height - btn_height - 5)
    canvas.coords(clear_btn_window, prev_w + 15, image_height - btn_height - 5)
    canvas.coords(auto_btn_window, prev_w + clear_w + 25, image_height - btn_height - 5)

move_buttons()
load_image(current_index)
root.mainloop()
