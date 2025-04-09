import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk  # To convert OpenCV images to a format Tkinter can display
import numpy as np

from dataclasses import dataclass
from typing import List

import ImageSource
# Our local vision library. Maybe change name?
import vision

import threading

display_w = 800
display_h = 480

@dataclass
class GUIState:
    bg_img:PIL.ImageTk.PhotoImage
    pip_img:PIL.ImageTk.PhotoImage
    disp_result:List[int] # list of the counts of the 6 possible values [#1s #2s #3s #4s #5s #6s]
    slider_val:int #1 = 1- or 2+, 2 = 2- or 3+, etc

gui_state = GUIState(None, None, [], 1)

def on_slider_move(evt):
    gui_state.slider_val = int(evt)
    refreshCanvas()

# Create main window
root = tk.Tk()
#root.attributes('-fullscreen', True)  # Make it fullscreen
root.title("Dice test")
root.geometry(str(display_w)+'x'+str(display_h)+"+0+0")

def makeFullscreen(win):
    win.attributes("-fullscreen", True)
if ImageSource.onRaspi():
    root.after(100, lambda:makeFullscreen(root))

def on_close():
    print("Cleaning up before exit...")
    root.destroy()  # Close the window

root.protocol("WM_DELETE_WINDOW", on_close)

# Create a canvas that fills the entire window
canvas = tk.Canvas(root)
canvas.pack(fill='both', expand=True)

def drawText(text, xcenter, ycenter, size:int = 14, anchor='center'):
    font = ("Helvetica", size, "bold")
    shadow_sep = 3
    #canvas.create_text(xcenter - shadow_sep, ycenter - shadow_sep, text=text, fill="black", font=font, anchor=anchor)
    canvas.create_text(xcenter + shadow_sep, ycenter + shadow_sep, text=text, fill="black", font=font, anchor=anchor)
    canvas.create_text(xcenter, ycenter, text=text, fill="white", font=font, anchor=anchor)

# Slider (Scale) – positioned proportionally
slider = tk.Scale(
    root,
    from_=1, to=5,
    orient='horizontal',
    command=on_slider_move,
    width=40,            # Thickness of the track in pixels
    #sliderlength=40,     # Size of the slider handle
    showvalue=False,
    font=('Arial', 16),  # Font for numeric labels
    #length=500           # Track length in pixels (horizontal dimension)
)
slider.place(relx=0.1, rely=0.9, relwidth=0.8, relheight=0.1)

def refreshCanvas():
    if threading.current_thread() is not threading.main_thread():
        # Only run from main thread
        root.after(0, refreshCanvas)
        return
    top_row_offset = 30
    bot_row_offset = 90
    if gui_state.bg_img:
        canvas.create_image(0, 0, image=gui_state.bg_img, anchor="nw")
    if gui_state.pip_img:
        canvas.create_image(0, 0, image=gui_state.pip_img, anchor="nw")
    # X- Label
    drawText(str(gui_state.slider_val  ) + '-', 50, top_row_offset, 30)
    # X+ Label
    drawText(str(gui_state.slider_val+1) + '+', display_w-50, top_row_offset, 30)
    # Count label
    drawText(str("COUNT"), display_w/2, top_row_offset, 30)
    def drawResultLabels(count, low, high):
        drawText(str(low), 60, bot_row_offset, 65)
        drawText(str(high), display_w - 60, bot_row_offset, 65)
        drawText(str(count), display_w / 2, bot_row_offset, 65)
    if len(gui_state.disp_result) == 6:
        low_count  = sum(gui_state.disp_result[:gui_state.slider_val])
        high_count = sum(gui_state.disp_result[gui_state.slider_val:])
        n_dice = sum(gui_state.disp_result)
        drawResultLabels(n_dice, low_count, high_count)
    else:
        drawResultLabels('...','...','...')

def detectionThread(img):
    results, overlay_img = vision.do_recognition(img, "livecam")
    print(results)
    roll_counts = [results.count(val) for val in range(1, 7)]
    overlay_img = cv2.resize(overlay_img, (display_w, display_h))
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGRA2RGBA)
    pil_img = PIL.Image.fromarray(overlay_img, "RGBA")
    gui_state.pip_img = PIL.ImageTk.PhotoImage(image=pil_img)
    gui_state.disp_result = roll_counts
    refreshCanvas()

def on_button_press(evt):
    # Clear the result
    gui_state.disp_result = []
    img = ImageSource.getImage()
    # Start doing detection in the background
    threading.Thread(target=detectionThread, daemon=True, args=(img,)).start()

    disp_img = cv2.resize(img, (display_w, display_h))
    # Convert from BGR (OpenCV) to RGB (Tkinter/PIL expects RGB)
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(disp_img)
    gui_state.bg_img = PIL.ImageTk.PhotoImage(image=pil_img)
    refreshCanvas()

def workerThread():
    while True:
        img = ImageSource.getImage()
        disp_img = cv2.resize(img, (display_w, display_h))
        # Convert from BGR (OpenCV) to RGB (Tkinter/PIL expects RGB)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(disp_img)
        gui_state.bg_img = PIL.ImageTk.PhotoImage(image=pil_img)
        refreshCanvas()
        # Start doing detection in the background
        detectionThread(img)

if ImageSource.onRaspi():
    threading.Thread(target=workerThread, daemon=True).start()
else:
    canvas.bind("<Button-1>", on_button_press)

root.mainloop()
