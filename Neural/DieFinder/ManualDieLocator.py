import tkinter as tk
from PIL import Image, ImageTk
import os
import json

# Settings
image_dir = '../../images/final_box_20_dice/'
image_names = ['%02d.jpg' % i for i in range(10)]
image_paths = [os.path.join(image_dir, name) for name in image_names]
output_file = "die_locations.json"

# Load existing locations to skip already-processed images
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        all_locations = json.load(f)
else:
    all_locations = {}

image_paths = [p for p in image_paths if os.path.basename(p) not in all_locations]
if not image_paths:
    print("All images have already been processed.")
    exit(0)

# === Globals ===
current_index = 0
original_image = None
tk_image = None
image_width = 0
image_height = 0
die_centers = []  # list of (x, y) for current image

# === Functions ===

def load_image(index):
    global original_image, tk_image, image_width, image_height
    original_image = Image.open(image_paths[index]).convert("RGB")
    image_width, image_height = original_image.size
    tk_image = ImageTk.PhotoImage(original_image)
    canvas.config(width=image_width, height=image_height)
    canvas.itemconfig(canvas_img, image=tk_image)
    canvas.coords(canvas_img, 0, 0)
    root.title(f"Die Locator - {os.path.basename(image_paths[index])}")
    move_next_button()

def on_click(event):
    x, y = event.x, event.y
    die_centers.append((x, y))
    # Draw a crosshair at the click location
    r = 10
    canvas.create_line(x - r, y, x + r, y, fill="cyan", width=2)
    canvas.create_line(x, y - r, x, y + r, fill="cyan", width=2)
    canvas.create_oval(x - r, y - r, x + r, y + r, outline="cyan", width=2)
    print(f"Die center marked at ({x}, {y}) — total: {len(die_centers)}")

def redraw_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    for x, y in die_centers:
        r = 10
        canvas.create_line(x - r, y, x + r, y, fill="cyan", width=2)
        canvas.create_line(x, y - r, x, y + r, fill="cyan", width=2)
        canvas.create_oval(x - r, y - r, x + r, y + r, outline="cyan", width=2)
    move_next_button()

def on_undo(event):
    if not die_centers:
        print("Nothing to undo.")
        return
    x, y = die_centers.pop()
    print(f"Undo: removed ({x}, {y}) — remaining: {len(die_centers)}")
    redraw_canvas()

def save_locations():
    """Save current die centers to the JSON file."""
    image_name = os.path.basename(image_paths[current_index])
    all_locations[image_name] = die_centers.copy()
    with open(output_file, "w") as f:
        json.dump(all_locations, f, indent=2)
    print(f"Saved {len(die_centers)} die locations for {image_name}")

def next_image():
    global current_index
    save_locations()
    die_centers.clear()
    current_index += 1
    if current_index >= len(image_paths):
        print("All images processed.")
        root.destroy()
        return
    canvas.delete("all")
    load_image(current_index)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    move_next_button()

# === GUI Setup ===
root = tk.Tk()
root.title("Die Locator")

canvas = tk.Canvas(root)
canvas.pack()
canvas_img = canvas.create_image(0, 0, anchor=tk.NW)

canvas.bind("<Button-1>", on_click)
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
