import time
from picamera2 import Picamera2, Preview
from libcamera import controls
import cv2
import numpy as np
import code

# Our local vision library. Maybe change name?
import vision

picam = Picamera2()

#config = picam.create_preview_configuration()
#picam.configure(config)
#picam.start_preview(Preview.DRM)

config = picam.create_still_configuration({
    "size":(1536,864), 
    })

picam.configure(config)
picam.start()
picam.set_controls({
    "AfMode":controls.AfModeEnum.Manual, 
    "LensPosition":5.5,
    "ExposureTime":15000,
    })
time.sleep(0.5) #Wait for focus
#picam.capture_file("test-python.jpg")

# Keystone correction: These numbers hardcoded by looking at an output image for this mechanical setup.
# Will need future adjustment.
src_pts = np.float32([(48,45),(1450,60),(1350,780),(145,787)])
dst_pts = np.float32([(0,0),  (1536,0), (1536,864),(0,864)])

warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

import os

try:
    while True:
        # This will give us an RGBA image
        rgb_img = picam.capture_array()
        # Convert to cv2 compliant BGR
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        #cv2.imwrite("0.jpg", bgr_img)

        warped_img = cv2.warpPerspective(bgr_img, warp_matrix, (1536,864))
        #cv2.imwrite("1.jpg", warped_img)

        print("Starting recognition...")

        results = vision.do_recognition(warped_img, "livecam")

        print(results)
        roll_counts = [results.count(val) for val in range(1,7)]
        print(roll_counts)
        print("Counted %d dice"%len(results))
        os.system("eog livecam/overlay.png")
except KeyboardInterrupt:
    picam.close()
    print("Goodbye!")


