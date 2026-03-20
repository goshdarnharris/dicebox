import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk, PIL.ImageDraw, PIL.ImageFont

from dataclasses import dataclass
from typing import List

import ImageSource
import sys
sys.path.insert(0, 'Neural')
from ThrowAnalyzer import analyze_throw, ambiguity_icon_size

import threading
from probability_display import compute_probability_display

# Opposite face mapping: bottom 1 = top 6, bottom 2 = top 5, etc.
OPPOSITE_FACE = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}

display_w = 800
display_h = 480

@dataclass
class GUIState:
    bg_img:PIL.ImageTk.PhotoImage
    disp_result:List[int] # list of the counts of the 6 possible values [#1s #2s #3s #4s #5s #6s]
    pivot:int  # pivot position: 1-5. "pivot-" sums values <= pivot, "pivot+" sums values > pivot
    last_img: object      # last captured BGR image (for re-rendering on pivot change)
    last_results: list    # last detection results
    last_ambiguities: list  # last ambiguity locations

gui_state = GUIState(None, [], 3, None, [], [])

# Create main window
root = tk.Tk()
root.title("Dice test")
root.geometry(f"{display_w}x{display_h}+0+0")

def makeFullscreen(win):
    win.attributes("-fullscreen", True)
if ImageSource.onRaspi():
    root.after(500, lambda: makeFullscreen(root))

def on_close():
    print("Cleaning up before exit...")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# Create a canvas that fills the entire window
canvas = tk.Canvas(root)
canvas.pack(fill='both', expand=True)

def drawText(text, xcenter, ycenter, size:int = 14, anchor='center'):
    # Creates matching text objects, white on black to give a shadow effect.
    font = ("Helvetica", size, "bold")
    shadow_sep = 3
    text_id = canvas.create_text(xcenter + shadow_sep, ycenter + shadow_sep, text=text, fill="black", font=font, anchor=anchor)
    shadow_id = canvas.create_text(xcenter, ycenter, text=text, fill="white", font=font, anchor=anchor)
    return (text_id, shadow_id)

splash_img = PIL.ImageTk.PhotoImage(PIL.Image.open("splash.png").resize((display_w,display_h)))

gui_label_y = 25
gui_count_y = 70
gui_bg_img = canvas.create_image(0, 0, anchor="nw", image=splash_img)

# Layout: [pivot-] [stddev]  1 ▼ 2 ▼ 3 ▼ 4 ▼ 5 ▼ 6  [pivot+] [total]
# Pivot labels get extra space on the sides; face values are centered.
pivot_margin = 75   # center of pivot labels from edge
face_left = 150     # left edge of face value region
face_right = 650    # right edge of face value region
face_span = face_right - face_left

# Face value positions (6 values, evenly spaced in the center region)
face_positions = [int(face_left + face_span * (i + 0.5) / 6) for i in range(6)]
# Chevron positions (between face values)
chevron_positions = [(face_positions[i] + face_positions[i + 1]) // 2 for i in range(5)]

# Face value labels and counts
gui_face_labels = []
gui_face_counts = []
for i in range(6):
    gui_face_labels.append(drawText(str(i + 1), face_positions[i], gui_label_y, 24))
    gui_face_counts.append(drawText('—', face_positions[i], gui_count_y, 40))

# Chevron visual indicators
gui_chevrons = []
for i in range(5):
    gui_chevrons.append(drawText("▼", chevron_positions[i], gui_label_y, 20))

# Touch zones for chevrons — wide invisible rectangles across the top
touch_height = 95
def set_pivot(p):
    gui_state.pivot = p
    if gui_state.last_img is not None:
        display_pil = prepare_display(gui_state.last_img, gui_state.last_results, gui_state.last_ambiguities, p)
        gui_state.bg_img = PIL.ImageTk.PhotoImage(image=display_pil)
    refreshCanvas()

for i in range(5):
    left = face_positions[i]
    right = face_positions[i + 1]
    zone = canvas.create_rectangle(left, 0, right, touch_height, fill='', outline='')
    canvas.tag_bind(zone, "<Button-1>", lambda e, p=i+1: set_pivot(p))

# Pivot labels
pivot_left_label = drawText('3-', pivot_margin, gui_label_y, 28)
pivot_left_count = drawText('—', pivot_margin, gui_count_y, 44)
pivot_right_label = drawText('4+', display_w - pivot_margin, gui_label_y, 28)
pivot_right_count = drawText('—', display_w - pivot_margin, gui_count_y, 44)

# Stddev beneath pivot- on left, total beneath pivot+ on right (inline text)
stddev_text = drawText('', pivot_margin, gui_count_y + 50, 22)
total_text = drawText('', display_w - pivot_margin, gui_count_y + 50, 22)

# Exit button in bottom left
exit_button = tk.Button(root, text="X", command=root.quit, padx=2, pady=0, font=("Arial", 8))
exit_button.place(x=0, rely=1.0, anchor='sw')

def refreshCanvas():
    if threading.current_thread() is not threading.main_thread():
        root.after(0, refreshCanvas)
        return

    def writeTextConfig(item_id_tuple, text):
        for val in item_id_tuple:
            canvas.itemconfig(val, text=text)

    if gui_state.bg_img:
        canvas.itemconfig(gui_bg_img, image=gui_state.bg_img)

    # Update pivot labels and chevron highlight
    p = gui_state.pivot
    writeTextConfig(pivot_left_label, f"{p}-")
    writeTextConfig(pivot_right_label, f"{p+1}+")
    for i in range(5):
        color = "red" if i + 1 == p else "white"
        canvas.itemconfig(gui_chevrons[i][1], fill=color)

    if len(gui_state.disp_result) == 6:
        for i in range(6):
            count = gui_state.disp_result[i]
            writeTextConfig(gui_face_counts[i], str(count) if count > 0 else '—')
        # Pivot sums
        low_sum = sum(gui_state.disp_result[:p])
        high_sum = sum(gui_state.disp_result[p:])
        n_dice = sum(gui_state.disp_result)
        writeTextConfig(pivot_left_count, str(low_sum))
        writeTextConfig(pivot_right_count, str(high_sum))
        writeTextConfig(total_text, f"n={n_dice}")
        prob = compute_probability_display(gui_state.disp_result, p)
        if prob:
            text, font_size, color = prob
            writeTextConfig(stddev_text, text)
            canvas.itemconfig(stddev_text[1], fill=color)
            font = ("Helvetica", font_size, "bold")
            for item_id in stddev_text:
                canvas.itemconfig(item_id, font=font)
        else:
            writeTextConfig(stddev_text, '')
    else:
        for i in range(6):
            writeTextConfig(gui_face_counts[i], '—')
        writeTextConfig(pivot_left_count, '—')
        writeTextConfig(pivot_right_count, '—')
        writeTextConfig(total_text, '')
        writeTextConfig(stddev_text, '')

def prepare_display(img, results, ambiguities, pivot):
    """Prepare a display image: flip 180, draw die overlays with opposite face values."""
    orig_h, orig_w = img.shape[:2]
    scale_x = display_w / orig_w
    scale_y = display_h / orig_h

    # Resize, convert BGR->RGB, flip 180
    disp_img = cv2.resize(img, (display_w, display_h))
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(disp_img).rotate(180)

    draw = PIL.ImageDraw.Draw(pil_img)
    circle_r = 30  # 70px diameter / 2
    font_die = PIL.ImageFont.load_default(size=40)

    # Draw ambiguity markers first (behind die overlays)
    for ax, ay, area in ambiguities:
        dx = display_w - int(ax * scale_x)
        dy = display_h - int(ay * scale_y)
        ar, font_sz = ambiguity_icon_size(area)
        draw.ellipse((dx - ar, dy - ar, dx + ar, dy + ar),
                     fill=(80, 80, 80, 150), outline="yellow", width=2)
        draw.text((dx, dy), "?", fill="yellow", anchor="mm",
                  font=PIL.ImageFont.load_default(size=font_sz))

    # Draw die overlays
    for x, y, face, _, _ in results:
        dx = display_w - int(x * scale_x)
        dy = display_h - int(y * scale_y)
        top_face = OPPOSITE_FACE[face]
        color = (0, 180, 0, 200) if top_face > pivot else (200, 0, 0, 200)
        draw.ellipse((dx - circle_r, dy - circle_r, dx + circle_r, dy + circle_r),
                     fill=color, outline="white", width=2)
        draw.text((dx, dy), str(top_face), fill="white", anchor="mm", font=font_die)

    return pil_img

def process_and_display(img):
    """Run detection on img and schedule GUI update."""
    pil_full = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results, ambiguities = analyze_throw(pil_full)
    face_values = [face for _, _, face, _, _ in results]
    print(f"Detected: {face_values}, {len(ambiguities)} ambiguous")
    roll_counts = [face_values.count(val) for val in range(1, 7)]
    # REVERSE because we read bottom faces, count top faces
    roll_counts = roll_counts[::-1]

    display_pil = prepare_display(img, results, ambiguities, gui_state.pivot)
    new_bg = PIL.ImageTk.PhotoImage(image=display_pil)

    def update(bg=new_bg, counts=roll_counts, raw_img=img, raw_results=results, raw_amb=ambiguities):
        gui_state.bg_img = bg
        gui_state.disp_result = counts
        gui_state.last_ambiguities = raw_amb
        gui_state.last_img = raw_img
        gui_state.last_results = raw_results
        refreshCanvas()
    root.after(0, update)

def on_button_press(evt):
    gui_state.disp_result = []
    img = ImageSource.getImage()
    threading.Thread(target=process_and_display, daemon=True, args=(img,)).start()
    # Show raw image immediately (flipped)
    disp_img = cv2.resize(img, (display_w, display_h))
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(disp_img).rotate(180)
    gui_state.bg_img = PIL.ImageTk.PhotoImage(image=pil_img)
    refreshCanvas()

def workerThread():
    while True:
        img = ImageSource.getImage()
        process_and_display(img)

if ImageSource.onRaspi():
    # Continuous capture mode on raspi
    threading.Thread(target=workerThread, daemon=True).start()
else:
    # Click to capture on desktop
    canvas.bind("<Button-1>", on_button_press)

# TK event loop never returns.
root.mainloop()
