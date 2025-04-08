# Utility class to supply images to the rest of the system.
# Ths supplied images should already be keystone corrected
# Meant to abstract away the camera or cache so both look the same.

import time

import cv2
import numpy as np
import os
import random

try:
    from picamera2 import Picamera2, Preview
    from libcamera import controls

    picam = Picamera2()

    # config = picam.create_preview_configuration()
    # picam.configure(config)
    # picam.start_preview(Preview.DRM)

    config = picam.create_still_configuration({
        "size": (1536, 864),
    })

    picam.configure(config)
    picam.start()
    picam.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 5.5,
        "ExposureTime": 15000,
    })
    time.sleep(0.5)  # Wait for focus

    # Keystone correction: These numbers hardcoded by looking at an output image for this mechanical setup.
    # Will need future adjustment.
    src_pts = np.float32([(48, 45), (1450, 60), (1350, 780), (145, 787)])
    dst_pts = np.float32([(0, 0), (1536, 0), (1536, 864), (0, 864)])

    warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)



    def getImage():
        # This will give us an RGB image
        rgb_img = picam.capture_array()
        # Convert to cv2 compliant BGR
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        # Keystone correction
        corrected_img = cv2.warpPerspective(bgr_img, warp_matrix, (1536, 864))

        # img_name = "%s/%02d.jpg"%(dirname,i)
        # cv2.imwrite(img_name, corrected_img)
        # os.system("eog %s"%img_name)
        # i += 1
        return corrected_img
    def close():
        picam.close()
except Exception as e:
    print("Failed to open live camera feed. Using cached images.")
    print("Failure was: ", e)
    src_dir = 'images/small_dice_44_acrylic_rough_side_up'
    fnames = os.listdir(src_dir)

    def getImage():
        # Load an image at random from the source dir
        fpath = os.path.join(src_dir, random.choice(fnames))
        img = cv2.imread(fpath)
        return img
    def close():
        pass