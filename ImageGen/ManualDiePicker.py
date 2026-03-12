import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import os
import numpy as np
from ThrowAnalyzer import analyze_throw

# Settings
image_dir = '../images/final_box_20_dice/'
image_names = ['%02d.jpg'%i for i in range(10)]
image_paths = [os.path.join(image_dir, name) for name in image_names]
crop_size = 180
output_dir = "DieClassifier/raw_training"
os.makedirs(output_dir, exist_ok=True)
processed_log = os.path.join(output_dir, "processed.txt")

def load_processed():
    if not os.path.exists(processed_log):
        return set()
    with open(processed_log, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_processed(image_path):
    with open(processed_log, "a") as f:
        f.write(image_path + "\n")

processed = load_processed()
image_paths = [p for p in image_paths if p not in processed]
if not image_paths:
    print("All images have already been processed.")
    exit(0)

# === Globals ===
current_index = 0
original_image = None
tk_image = None
last_crop_box = None
last_crop_coords = None
image_width = 0
image_height = 0
annotations = []  # list of (box, digit, save_path) for undo support


# === Functions ===

def load_image(index):
    global original_image, tk_image, image_width, image_height
    original_image = Image.open(image_paths[index]).convert("RGB")
    image_width, image_height = original_image.size

    tk_image = ImageTk.PhotoImage(original_image)
    canvas.config(width=image_width, height=image_height)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    move_next_button()

    # Run ThrowAnalyzer and draw results
    print("Running ThrowAnalyzer...")
    results = analyze_throw(original_image)
    half = crop_size // 2
    for x, y, face, die_conf, id_conf in results:
        box = (x - half, y - half, x + half, y + half)
        canvas.create_rectangle(box[0], box[1], box[2], box[3], outline="green", width=2)
        canvas.create_text(x, y, text=str(face), fill="green", font=("Arial", 20, "bold"))
    print(f"Found {len(results)} dice.")

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

def on_click(event):
    global last_crop_box, last_crop_coords
    x, y = event.x, event.y
    half = crop_size // 2
    box = (x - half, y - half, x + half, y + half)
    last_crop_box = box
    last_crop_coords = (x, y)
    canvas.create_rectangle(box[0], box[1], box[2], box[3], outline="red", width=2)
    print("Click registered. Awaiting number key 0–6...")

def on_keypress(event):
    global last_crop_box, last_crop_coords

    if not last_crop_box or not event.char.isdigit():
        return
    digit = int(event.char)
    if digit < 0 or digit > 6:
        return

    # Get and save the padded crop
    cropped = average_border_fill(original_image, last_crop_box)
    base = os.path.basename(image_paths[current_index])
    x, y = last_crop_coords
    filename = f"{digit}_crop_{base}_{x}_{y}.png"
    save_path = os.path.join(output_dir, filename)
    cropped.save(save_path)
    print(f"Saved: {save_path}")

    annotations.append((last_crop_box, digit, save_path))

    # Draw the digit label in the square
    label_x = (last_crop_box[0] + last_crop_box[2]) // 2
    label_y = (last_crop_box[1] + last_crop_box[3]) // 2
    canvas.create_text(label_x, label_y, text=str(digit), fill="white", font=("Arial", 20, "bold"))

    last_crop_box = None
    last_crop_coords = None

def redraw_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    for box, digit, _ in annotations:
        canvas.create_rectangle(box[0], box[1], box[2], box[3], outline="red", width=2)
        label_x = (box[0] + box[2]) // 2
        label_y = (box[1] + box[3]) // 2
        canvas.create_text(label_x, label_y, text=str(digit), fill="white", font=("Arial", 20, "bold"))
    move_next_button()

def on_undo(event):
    global last_crop_box, last_crop_coords
    if not annotations:
        print("Nothing to undo.")
        return
    box, digit, save_path = annotations.pop()
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Undo: removed {save_path}")
    last_crop_box = None
    last_crop_coords = None
    redraw_canvas()

def next_image():
    global current_index, last_crop_box, last_crop_coords
    mark_processed(image_paths[current_index])
    annotations.clear()
    current_index += 1
    if current_index >= len(image_paths):
        print("All images processed.")
        root.destroy()
        return
    last_crop_box = None
    last_crop_coords = None
    canvas.delete("all")
    load_image(current_index)

# === GUI Setup ===
root = tk.Tk()
root.title("Image Crop Tool")

canvas = tk.Canvas(root)
canvas.pack()
canvas_img = canvas.create_image(0, 0, anchor=tk.NW)

canvas.bind("<Button-1>", on_click)
root.bind("<Key>", on_keypress)
root.bind("<BackSpace>", on_undo)

def move_next_button():
    global original_image, tk_image, image_width, image_height
    next_btn = tk.Button(root, text="Next Image ▶", command=next_image)
    next_btn_window = canvas.create_window(0, 0, window=next_btn, anchor=tk.NW)
    canvas.update_idletasks()
    btn_width = next_btn.winfo_reqwidth()
    btn_height = next_btn.winfo_reqheight()
    canvas.coords(next_btn_window, image_width - btn_width - 5, image_height - btn_height - 5)

move_next_button()
#next_btn = tk.Button(root, text="Next Image ▶", command=next_image)
#next_btn_window = canvas.create_window(0, 0, window=next_btn, anchor=tk.NW)

load_image(current_index)
root.mainloop()
