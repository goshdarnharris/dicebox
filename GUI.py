import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk  # To convert OpenCV images to a format Tkinter can display
import numpy as np

import ImageSource
# Our local vision library. Maybe change name?
import vision

import threading

display_w = 800
display_h = 480

def on_slider_move(evt):
    print("Slider moved!")
    print(evt)

# Create main window
root = tk.Tk()
#root.attributes('-fullscreen', True)  # Make it fullscreen
root.title("Dice test")
root.geometry(str(display_w)+'x'+str(display_h)+"+0+0")

def on_close():
    print("Cleaning up before exit...")
    root.destroy()  # Close the window

root.protocol("WM_DELETE_WINDOW", on_close)

# Create a canvas that fills the entire window
canvas = tk.Canvas(root)
canvas.pack(fill='both', expand=True)

# Label for the background image (fills entire window)
bg_label = tk.Label(root)
bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)

# Slider (Scale) – positioned proportionally
slider = tk.Scale(
    root,
    from_=2, to=5,
    orient='horizontal',
    command=on_slider_move,
    width=40,            # Thickness of the track in pixels
    #sliderlength=40,     # Size of the slider handle
    showvalue=False,
    font=('Arial', 16),  # Font for numeric labels
    #length=500           # Track length in pixels (horizontal dimension)
)
slider.place(relx=0.1, rely=0.9, relwidth=0.8, relheight=0.1)

def do_detection(img):
    results = vision.do_recognition(img, "livecam")
    print(results)
    roll_counts = [results.count(val) for val in range(1, 7)]
    # Fixme: hacky
    overlay_img = cv2.imread('livecam/overlay.png')
    # Trigger a GUI update from the main thread
    root.after(0, show_results, *(overlay_img, roll_counts))

# THIS REFERENCE MUST EXIST OR bg_img WILL BE GARBAGE COLLECTED AND THE IMAGE WILL NOT SHOW
bg_img = None
def show_results(overlay_img, roll_counts):
    # Runs on main thread
    global bg_img
    disp_img = cv2.resize(overlay_img, (display_w, display_h))
    # Convert from BGR (OpenCV) to RGB (Tkinter/PIL expects RGB)
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    # Wash out the image
    white = np.full(disp_img.shape, 255, dtype=np.uint8)
    disp_img = cv2.addWeighted(disp_img, 0.3, white, 0.7, 0)  # 0.7 original, 0.3 white
    # Convert the OpenCV image to a PIL Image, then to a PhotoImage for Tkinter
    pil_img = PIL.Image.fromarray(disp_img)
    bg_img = PIL.ImageTk.PhotoImage(image=pil_img)
    bg_label.config(image=bg_img)

def on_button_press(evt):
    print(evt)
    global bg_img
    img = ImageSource.getImage()
    # Start doing detection in the background
    threading.Thread(target=do_detection, daemon=True, args=(img,)).start()

    disp_img = cv2.resize(img, (display_w, display_h))
    # Convert from BGR (OpenCV) to RGB (Tkinter/PIL expects RGB)
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    # Wash out the image
    white = np.full(disp_img.shape, 255, dtype=np.uint8)
    disp_img = cv2.addWeighted(disp_img, 0.3, white, 0.7, 0)  # 0.7 original, 0.3 white
    # Convert the OpenCV image to a PIL Image, then to a PhotoImage for Tkinter
    pil_img = PIL.Image.fromarray(disp_img)
    bg_img = PIL.ImageTk.PhotoImage(image=pil_img)
    bg_label.config(image=bg_img)


bg_label.bind("<Button-1>", on_button_press)

root.mainloop()