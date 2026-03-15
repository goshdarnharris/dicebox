import tkinter as tk
from PIL import Image, ImageTk
import h5py
import sys

# === Settings ===
dataset_file = sys.argv[1] if len(sys.argv) > 1 else "augmented_training.h5"
scale = 2

hf = h5py.File(dataset_file, "r")
images = hf["images"]
targets = hf["targets"]
n = len(images)
current = [0]

root = tk.Tk()

def show(index):
    index = max(0, min(index, n - 1))
    current[0] = index

    img = Image.fromarray((images[index] * 255).astype("uint8"), mode="L")
    tgt = Image.fromarray((targets[index] * 255).astype("uint8"), mode="L")
    tgt_resized = tgt.resize(img.size, Image.BILINEAR)

    # Side by side: image | target
    combined = Image.new("L", (img.width * 2, img.height))
    combined.paste(img, (0, 0))
    combined.paste(tgt_resized, (img.width, 0))

    w, h = combined.size
    combined = combined.resize((w * scale, h * scale), Image.NEAREST)
    tk_img = ImageTk.PhotoImage(combined)
    canvas.tk_img = tk_img
    canvas.config(width=w * scale, height=h * scale)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    index_var.set(str(index))
    root.title(f"[{index}/{n}] image | target")

def on_key(event):
    if event.keysym in ("Right", "space"):
        show(current[0] + 1)
    elif event.keysym == "Left":
        show(current[0] - 1)
    elif event.keysym == "Home":
        show(0)
    elif event.keysym == "End":
        show(n - 1)
    elif event.keysym == "Escape":
        root.destroy()

def on_index_enter(event):
    try:
        show(int(index_var.get()))
    except ValueError:
        pass

canvas = tk.Canvas(root)
canvas.pack()

frame = tk.Frame(root)
frame.pack(fill="x")
tk.Label(frame, text="Index:").pack(side="left", padx=5)
index_var = tk.StringVar(value="0")
index_entry = tk.Entry(frame, textvariable=index_var, width=10)
index_entry.pack(side="left")
index_entry.bind("<Return>", on_index_enter)
tk.Label(frame, text=f"/ {n}").pack(side="left", padx=5)

root.bind("<Key>", on_key)
show(0)
root.mainloop()
hf.close()
