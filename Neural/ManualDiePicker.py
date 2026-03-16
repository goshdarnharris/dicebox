import tkinter as tk
from PIL import Image, ImageTk
import os
import json
import numpy as np
from ThrowAnalyzer import analyze_throw
from DieClassifier.DieClassifier import identify_die

# === Settings ===
image_dir = 'training_images'
image_paths = sorted(
    os.path.join(root, f)
    for root, _, files in os.walk(image_dir)
    for f in files
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
)
crop_size = 180
annotations_file = os.path.join(image_dir, "annotations.json")
click_radius = crop_size // 4

# === Load existing annotations ===
if os.path.exists(annotations_file):
    with open(annotations_file, "r") as f:
        all_annotations = json.load(f)
else:
    all_annotations = {}

# === State ===
current_index = 0
original_image = None
tk_image = None
image_width = 0
image_height = 0
annotations = []       # list of (x, y, face, conf)
prev_annotations = []  # for "copy prev" feature
pending_click = None   # (x, y) or None

# Start on the first un-annotated image
for i, p in enumerate(image_paths):
    if os.path.relpath(p, image_dir).replace("\\", "/") not in all_annotations:
        current_index = i
        break
else:
    current_index = len(image_paths) - 1


# === Helpers ===

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
    padded.paste(cropped, (left - box[0], upper - box[1]))
    return padded


def find_nearest_annotation(x, y):
    best_i, best_dist = None, float('inf')
    for i, (ax, ay, *_) in enumerate(annotations):
        dist = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_i = dist, i
    return best_i if best_dist <= click_radius else None


def run_auto_detect():
    """Run ThrowAnalyzer and populate annotations."""
    print("Running ThrowAnalyzer...")
    results = analyze_throw(original_image)
    for x, y, face, conf, _ in results:
        annotations.append((x, y, face, conf))
    print(f"Auto-detected {len(results)} dice.")


def set_annotations(new_annotations):
    """Replace annotations and reset pending state."""
    global pending_click
    annotations.clear()
    annotations.extend(new_annotations)
    pending_click = None
    redraw_canvas()


def navigate(delta):
    """Move to an adjacent image."""
    global current_index, pending_click, prev_annotations
    new_index = current_index + delta
    if new_index < 0 or new_index >= len(image_paths):
        if new_index >= len(image_paths):
            print("All images processed.")
            root.destroy()
        else:
            print("Already at first image.")
        return
    prev_annotations = list(annotations)
    current_index = new_index
    pending_click = None
    canvas.delete("all")
    load_image(current_index)


# === Core functions ===

def redraw_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    half = crop_size // 2
    for x, y, face, conf in annotations:
        color = "green" if face > 0 else "orange"
        canvas.create_rectangle(x - half, y - half, x + half, y + half, outline=color, width=2)
        # Center crosshair
        r = 6
        canvas.create_line(x - r, y, x + r, y, fill=color, width=2)
        canvas.create_line(x, y - r, x, y + r, fill=color, width=2)
        # Label
        label = f"{face} ({conf:.2f})" if conf is not None else str(face)
        canvas.create_text(x, y - 20, text=label, fill=color, font=("Arial", 16, "bold"))
    if pending_click:
        px, py = pending_click
        canvas.create_rectangle(px - half, py - half, px + half, py + half, outline="red", width=2)
    create_buttons()
    dice_count = sum(1 for _, _, face, _ in annotations if face > 0)
    root.title(f"Dice Annotation — {rel_path_for(current_index)} — {dice_count} dice — [{current_index+1}/{len(image_paths)}]")


def load_image(index):
    global original_image, tk_image, image_width, image_height
    annotations.clear()

    original_image = Image.open(image_paths[index]).convert("RGB")
    image_width, image_height = original_image.size
    tk_image = ImageTk.PhotoImage(original_image)
    canvas.config(width=image_width, height=image_height)

    rel_path = rel_path_for(index)
    if rel_path in all_annotations:
        for x, y, face in all_annotations[rel_path]:
            annotations.append((x, y, face, None))
        print(f"Loaded {len(annotations)} saved annotations for {rel_path}")
    else:
        run_auto_detect()

    redraw_canvas()


# === Event handlers ===

def on_click(event):
    global pending_click
    x, y = event.x, event.y

    nearest = find_nearest_annotation(x, y)
    if nearest is not None:
        removed = annotations.pop(nearest)
        print(f"Removed die at ({removed[0]}, {removed[1]}) face={removed[2]}")
        pending_click = None
        redraw_canvas()
        return

    # Auto-classify and add
    half = crop_size // 2
    box = (x - half, y - half, x + half, y + half)
    crop = average_border_fill(original_image, box)
    face, conf = identify_die(crop)
    annotations.append((x, y, face, conf))
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
    # Replace the auto-guess with manual override
    for i in range(len(annotations) - 1, -1, -1):
        if annotations[i][0] == x and annotations[i][1] == y:
            annotations.pop(i)
            break
    annotations.append((x, y, digit, None))
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


# === Button actions ===

def save_annotations():
    rel_path = rel_path_for(current_index)
    all_annotations[rel_path] = [[x, y, face] for x, y, face, _ in annotations]
    with open(annotations_file, "w") as f:
        json.dump(all_annotations, f, indent=2)
    print(f"Saved {len(annotations)} annotations for {rel_path}")


def rerun_auto():
    set_annotations([])
    run_auto_detect()
    redraw_canvas()


def copy_prev():
    if not prev_annotations:
        print("No previous annotations to copy.")
        return
    set_annotations([(x, y, face, None) for x, y, face, _ in prev_annotations if face > 0])
    print(f"Copied {len(annotations)} annotations from previous image.")


# === GUI Setup ===

root = tk.Tk()
root.title("Dice Annotation Tool")
canvas = tk.Canvas(root)
canvas.pack()

canvas.bind("<Button-1>", on_click)
root.bind("<Key>", on_keypress)
root.bind("<BackSpace>", on_undo)


def create_buttons():
    """Create and position toolbar buttons."""
    buttons_left = [
        ("◀ Prev", lambda: navigate(-1)),
        ("Clear", lambda: set_annotations([])),
        ("Re-Auto", rerun_auto),
    ]
    buttons_right = [
        ("Save", save_annotations, {"bg": "#4a4", "fg": "white"}),
        ("Copy Prev", copy_prev),
        ("Next ▶", lambda: navigate(1)),
    ]

    btn_height = 0
    # Left side
    x_pos = 5
    for item in buttons_left:
        text, cmd = item[0], item[1]
        kwargs = item[2] if len(item) > 2 else {}
        btn = tk.Button(root, text=text, command=cmd, **kwargs)
        win = canvas.create_window(x_pos, 0, window=btn, anchor=tk.NW)
        canvas.update_idletasks()
        btn_height = btn.winfo_reqheight()
        canvas.coords(win, x_pos, image_height - btn_height - 5)
        x_pos += btn.winfo_reqwidth() + 10

    # Right side (placed right-to-left)
    x_pos = image_width - 5
    for item in reversed(buttons_right):
        text, cmd = item[0], item[1]
        kwargs = item[2] if len(item) > 2 else {}
        btn = tk.Button(root, text=text, command=cmd, **kwargs)
        canvas.update_idletasks()
        btn_w = btn.winfo_reqwidth()
        x_pos -= btn_w
        win = canvas.create_window(x_pos, image_height - btn_height - 5, window=btn, anchor=tk.NW)
        x_pos -= 10


create_buttons()
load_image(current_index)
root.mainloop()
