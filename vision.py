import cv2
import numpy as np
from pathlib import Path
import os
import sys
import toml

from datetime import datetime as dt 
from contextlib import contextmanager

import libpip
import solver

old_out = sys.stdout

class timestamped_output(object):
    """Stamped stdout."""
    
    def __init__(self):
        self.nl = True
        self.indent_depth = 0
  
    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write(f'{dt.now()} > {"  "*self.indent_depth}{x}')
            self.nl = False
        else:
            old_out.write(x)
    @contextmanager
    def indent(self, depth = 1):
        self.indent_depth += depth
        try:
            yield
        finally:
            self.indent_depth -= depth

    def flush(self):
        old_out.flush()
    
output = timestamped_output()
sys.stdout = output

def get_outlines(frame, pips, prepend_imwrite = ""):
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv_no_pips = libpip.remove_from_image(pips, hsv, fill="auto", factor=1.25)
    # rgb_no_pips = cv2.cvtColor(hsv_no_pips, cv2.COLOR_HSV2BGR)
    # cv2.imwrite('outline.rgb.nopips.png',rgb_no_pips)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{prepend_imwrite}outline.gray.png',gray)

    blur = gray #cv2.medianBlur(gray, 3)
    cv2.imwrite(f'{prepend_imwrite}outline.blur.png',blur)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8))
    cv2.imwrite(f'{prepend_imwrite}outline.grad.png',grad)
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f'{prepend_imwrite}outline.grad.norm.png',grad)
    no_pips = libpip.remove_from_image(pips, grad, fill=(0,0,0), factor=1.3, expand = 5)

    cv2.imwrite(f'{prepend_imwrite}outline.nopips.png',grad)

    grad_blur = cv2.medianBlur(no_pips, 3)

    cv2.imwrite(f'{prepend_imwrite}outline.grad.blur.png',grad_blur)


    ret, grad_thresh = cv2.threshold(grad_blur, 15, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{prepend_imwrite}outline.grad.thresh.png',grad_thresh)
    grad_thresh = cv2.medianBlur(grad_thresh, 3)
    cv2.imwrite(f'{prepend_imwrite}outline.grad.thresh.blur.png',grad_thresh)

    # define the kernel 
    kernel = np.ones((3, 3), np.uint8) 
    
    # opening the image 
    outlines = cv2.morphologyEx(grad_thresh, cv2.MORPH_OPEN, 
                            kernel, iterations=1)
    

    cv2.imwrite(f'{prepend_imwrite}outline.png',outlines)
    return outlines

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

def test_inputs():
    for path in os.listdir('images'):
        if path not in ["truth.toml"] and not os.path.isdir(f"images/{path}"):
            yield path

def main():
    os.makedirs('output', exist_ok=True)

    with open("images/truth.toml", 'r') as f:
        truth = toml.load(f)

    test_results = {}
    for path in test_inputs():
        print(f'processing {path}')
        im = cv2.imread('images/' + path)       

        if im is not None:
            image_name = Path(path).stem
            os.makedirs(f"output/{image_name}", exist_ok=True)
            
            with output.indent(2):
                image_cropped = im[540:2350, 775:3760]
                image_downsampled = cv2.resize(image_cropped, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

                image_detect = image_downsampled

                cv2.imwrite(f'output/{image_name}/original.png', image_detect)

                image_detect = cv2.bilateralFilter(image_detect, 3, 50, 50)
                cv2.imwrite(f'output/{image_name}/bilateral.png', image_detect)

                pips = libpip.find_pips(image_detect)
                print("found pips")

                cv2.imwrite(f'output/{image_name}/pips.png', libpip.overlay_pips(pips, image_detect.copy()))
                print("getting outlines")
                outlines = get_outlines(image_detect, pips, prepend_imwrite = f"output/{image_name}/")
                print("extracting dice")
                # dice = solver.solve_image_clustering_outline_penalty(image_detect,outlines,pips)
                dice = solver.solve_graph_outlines(image_detect,outlines, pips, prepend_figs = f"output/{image_name}/")

                out_pips = overlay_info(image_detect, dice, pips)
                cv2.imwrite(f'output/{image_name}/overlay.png',out_pips)

                results = [len(ps[0]) for ps in dice]
                all_correct = True
                for roll in range(1,7):
                    if results.count(roll) == truth[path][str(roll)]:
                        print(f'{roll}: {results.count(roll)} (correct)')
                    else:
                        all_correct = False
                        print(f'{roll}: {results.count(roll)} (incorrect)')
            if all_correct:
                test_results[path] = 'PASS'
            else:
                test_results[path] = 'FAIL'
            print(test_results[path])

            
        else:
            print(f'could not read {path}')

    print("")
    print(f"RESULTS")
    with output.indent(2):
        for path in test_inputs():
            print(f'{path}: {test_results[path]}')


if __name__ == '__main__':
    main()
