import cv2
import numpy as np
from pathlib import Path
import os
import sys

from datetime import datetime as dt 

import libpip
import solver

old_out = sys.stdout

class timestamped_output(object):
    """Stamped stdout."""
    
    nl = True
  
    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write('%s> %s' % (str(dt.now()), x))
            self.nl = False
        else:
            old_out.write(x)
    def flush(self):
        old_out.flush()
    
sys.stdout = timestamped_output()

def get_outlines(frame, pips):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png',gray)
    
    blur = cv2.medianBlur(gray, 3)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8))
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)
    grad = libpip.remove_from_image(pips, grad, fill=(0,0,0), factor=2.25)

    cv2.imwrite('grad.png',grad)

    grad_blur = cv2.medianBlur(grad, 3)

    cv2.imwrite('grad_blur.png',grad_blur)


    ret, grad_thresh = cv2.threshold(grad_blur, 15, 255, cv2.THRESH_BINARY)
    cv2.imwrite('grad_thresh.png',grad_thresh)

    return grad_thresh

def overlay_info(frame, dice, pips):
    # Overlay pips
    for pip in libpip.Pip.generator(pips):
        pos = pip.pos
        r = pip.radius

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 0, 0), 2)

    # Overlay dice number
    for pips,centroid_x,centroid_y in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            str(len(pips)), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(len(pips)),
                    (int(centroid_x - textsize[0] / 2),
                     int(centroid_y + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return frame

def main():
    os.makedirs('output', exist_ok=True)

    for path in os.listdir('images'):
        print(f'processing {path}')
        im = cv2.imread('images/' + path)

        

        if im is not None:
            image_name = Path(path).stem

            image_cropped = im[540:2350, 775:3760]
            image_downsampled = cv2.resize(image_cropped, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

            image_detect = image_downsampled

            cv2.imwrite(f'{image_name}.original.png', image_detect)

            pips = libpip.find_pips(image_detect)
            print("found pips")

            cv2.imwrite(f'{image_name}.pips.png', libpip.overlay_pips(pips, image_detect.copy()))
            print("getting outlines")
            outlines = get_outlines(image_detect,pips)
            print("extracting dice")
            # dice = solver.solve_image_clustering_outline_penalty(image_detect,outlines,pips)
            dice = solver.solve_graph_outlines(image_detect,outlines,pips)

            results = [len(ps[0]) for ps in dice]
            for roll in range(1,6):
                print(f'{roll}: {results.count(roll)}')

            out_pips = overlay_info(image_detect, dice, pips)
            cv2.imwrite(f'{image_name}.overlay.png',out_pips)

if __name__ == '__main__':
    main()
