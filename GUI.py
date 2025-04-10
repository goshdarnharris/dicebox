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
    text_id = canvas.create_text(xcenter + shadow_sep, ycenter + shadow_sep, text=text, fill="black", font=font, anchor=anchor)
    shadow_id = canvas.create_text(xcenter, ycenter, text=text, fill="white", font=font, anchor=anchor)
    return (text_id, shadow_id)

splash_img = PIL.ImageTk.PhotoImage(PIL.Image.open("splash.png").resize((display_w,display_h)))

gui_top_row_offset = 30
gui_bot_row_offset = 90
gui_bg_img = canvas.create_image(0, 0, anchor="nw", image=splash_img)
gui_overlay_img = canvas.create_image(0, 0, anchor="nw")
# X- Label
gui_x_minus_label = drawText(str(gui_state.slider_val  ) + '-', 50, gui_top_row_offset, 30)
# X+ Label
gui_x_plus_label = drawText(str(gui_state.slider_val+1) + '+', display_w-50, gui_top_row_offset, 30)
# Count label
# Never changes so we don't keep reference
gui_count_label = drawText(str("COUNT"), display_w/2, gui_top_row_offset, 30)

gui_x_minus_val = drawText('...', 60, gui_bot_row_offset, 65)
gui_x_plus_val  = drawText('...', display_w - 60, gui_bot_row_offset, 65)
gui_count_val   = drawText('...', display_w / 2, gui_bot_row_offset, 65)

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

# Create the button
exit_button = tk.Button(root, text="X", command=root.quit, padx=10, pady=10)

# Position it in the bottom-left corner
exit_button.place(x=10, rely=1.0, anchor='sw')  #

def refreshCanvas():
    if threading.current_thread() is not threading.main_thread():
        # Only run from main thread
        root.after(0, refreshCanvas)
        return

    def writeTextConfig(item_id_tuple, text):
        for val in item_id_tuple:
            canvas.itemconfig(val, text=text)

    if gui_state.bg_img:
        canvas.itemconfig(gui_bg_img, image=gui_state.bg_img)
        #canvas.create_image(0, 0, image=gui_state.bg_img, anchor="nw")
    if gui_state.pip_img:
        canvas.itemconfig(gui_overlay_img, image=gui_state.pip_img)
        #canvas.create_image(0, 0, image=gui_state.pip_img, anchor="nw")
    # X- Label
    #drawText(str(gui_state.slider_val  ) + '-', 50, top_row_offset, 30)
    writeTextConfig(gui_x_minus_label, str(gui_state.slider_val  ) + '-')
    # X+ Label
    #drawText(str(gui_state.slider_val+1) + '+', display_w-50, top_row_offset, 30)
    writeTextConfig(gui_x_plus_label, text=str(gui_state.slider_val+1) + '+')
    # Count label
    #drawText(str("COUNT"), display_w/2, top_row_offset, 30)
    def drawResultLabels(count, low, high):
        writeTextConfig(gui_count_val, text=str(count))
        writeTextConfig(gui_x_minus_val, text=str(low))
        writeTextConfig(gui_x_plus_val, text=str(high))
        #drawText(str(low), 60, bot_row_offset, 65)
        #drawText(str(high), display_w - 60, bot_row_offset, 65)
        #drawText(str(count), display_w / 2, bot_row_offset, 65)
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
    # REVERSE ROLL COUNTS BECAUSE WE ARE READING THE BOTTOMS OF THE DICE, SO THE TOPS ARE OPPOSITE
    # 1->6, 2->5, etc.
    roll_counts = roll_counts[::-1]
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
        detectionThread(img)

if False and ImageSource.onRaspi():
    # This behavior is actually just too annoying, it flickers.
    threading.Thread(target=workerThread, daemon=True).start()
else:
    canvas.bind("<Button-1>", on_button_press)

root.mainloop()
