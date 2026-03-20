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

    # 4608, 2592 is maximum resolution of rpicam3
    # But we're going to use lower res to reduce processing requirements
    max_res_divisor = 2

    config = picam.create_still_configuration({
        "size": (4608//max_res_divisor, 2592//max_res_divisor),
        #"size": (1536, 864),
    })

    picam.configure(config)
    picam.start()
    picam.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 4.5,
        "ExposureTime": 15000,
    })
    time.sleep(0.5)  # Wait for focus

    # Keystone correction: These numbers hardcoded by looking at an output image for this mechanical setup.
    # Will need future adjustment.
    src_pts_native = [(751, 621), (3796, 603), (3502, 2327), (1037, 2352)]
    src_pts_downscale = [(x/max_res_divisor, y/max_res_divisor) for (x,y) in src_pts_native]
    src_pts = np.float32(src_pts_downscale)
    dst_pts = np.float32([(0, 0), (1676, 0), (1676, 1196), (0, 1196)])

    warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    out_dir = 'training_imgs'
    img_i = 0

    def getImage():
        global img_i
        # This will give us an RGB image
        rgb_img = picam.capture_array()
        # Convert to cv2 compliant BGR
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        # Keystone correction
        corrected_img = cv2.warpPerspective(bgr_img, warp_matrix, (1676, 1196))

        img_name = "%s/%03d.jpg"%(out_dir,img_i)
        #cv2.imwrite(img_name, corrected_img)
        # os.system("eog %s"%img_name)
        img_i += 1
        return corrected_img
    def close():
        picam.close()
    def onRaspi()->bool: return True

except Exception as e:
    # If we end up here, we are not on the raspi so we will assume we're on a test rig.
    # We should load test images and supply them as if they came from the camera.
    print("Failed to open live camera feed. Using cached images.")
    print("Failure was: ", e)
    src_dir = 'Neural/training_images'
    image_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(src_dir)
        for f in files
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    def getImage():
        # Load an image at random from the source dir (including subdirs)
        fpath = random.choice(image_paths)
        print("Loading ", fpath)
        img = cv2.imread(fpath)
        return img
    def close():
        pass

    def onRaspi() -> bool: return False
