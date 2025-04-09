import cv2
import numpy as np
from pathlib import Path
import os
import sys
import toml

from datetime import datetime as dt 
from contextlib import contextmanager
from functools import partial
from inspect import signature
from typing import List

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


def count_required_params(func):
    params = signature(func).parameters
    def test(param):
        a = param.default == param.empty
        b = param.kind == param.POSITIONAL_OR_KEYWORD
        c = param.kind == param.POSITIONAL_ONLY
        return a and (b or c)
    required = len([p for p in params.values() if test(p)])
    return required

class monad(object):
    def __init__(self, value):
        self.value = value
    def __rshift__(self, func):
        try:
            if count_required_params(func) > 1:
                def do(arg, *args, **kwargs):
                    return func(arg, self.value, *args, **kwargs)
                return do
            else:
                return monad(func(self.value))
        except ValueError:
            return monad(func(self.value))

def write_image(path):
    def write_image_func(im):
        print(path)
        cv2.imwrite(path, im)
        return im
    return write_image_func

def crop(x_slice, y_slice):
    def do_crop(im):
        return im[x_slice, y_slice]
    return do_crop

def get_outlines(image, pips, prepend_imwrite = ""):    
    open_kernel = np.ones((3, 3), np.uint8) 

    outlines = (monad(image) 
        >> partial(cv2.cvtColor, code = cv2.COLOR_BGR2GRAY)
            >> write_image(f'{prepend_imwrite}outline.gray.png')
        >> partial(cv2.morphologyEx, op = cv2.MORPH_GRADIENT, kernel = np.ones((6,6),np.uint8))
            >> write_image(f'{prepend_imwrite}outline.grad.png')
        >> partial(cv2.normalize, dst = None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
            >> write_image(f'{prepend_imwrite}outline.grad.norm.png')
        >> partial(libpip.remove_from_image, pips = pips, fill=(0,0,0), factor=1.6, expand = 5)
            >> write_image(f'{prepend_imwrite}outline.nopips.png')
        #>> partial(cv2.medianBlur, ksize = 3)
        #    >> write_image(f'{prepend_imwrite}outline.grad.blur.png')
        >> (lambda x: cv2.threshold(x, thresh = 10, maxval = 255, type = cv2.THRESH_BINARY)[1])
            >> write_image(f'{prepend_imwrite}outline.grad.thresh.png')
        #>> partial(cv2.medianBlur, ksize = 3)
        #    >> write_image(f'{prepend_imwrite}outline.grad.thresh.blur.png')
        >> partial(cv2.morphologyEx, op = cv2.MORPH_OPEN, kernel = open_kernel, iterations = 1)
            >> write_image(f'{prepend_imwrite}outline.png')
    )
    return outlines.value

def overlay_info(frame, pips, dice):
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

def do_recognition(im:cv2.Mat, debug_dir)->List[int]:
    image = monad(im)
    os.makedirs(debug_dir, exist_ok=True)
    
    with output.indent(2):
        image_detect = (image 
            >> write_image(f'{debug_dir}/original.png')
            >> partial(cv2.bilateralFilter, d = 30, sigmaColor = 25, sigmaSpace = 75)
                >> write_image(f'{debug_dir}/bilateral.png')
        )

        pips = image_detect >> libpip.find_pips
        if len(pips.value) == 0:
            return []

        print("found pips")
        (image_detect #write an image with pips
            >> np.copy 
            >> (pips >> libpip.overlay_pips) 
                >> write_image(f'{debug_dir}/pips.png')
        )

        outlines = image_detect >> (pips >> partial(get_outlines, prepend_imwrite = f"{debug_dir}/"))

        # dice = image_detect >> (outlines >> (pips >> partial(solver.solve_graph_outlines, prepend_figs = f"{debug_dir}/")))
        dice = solver.solve_graph_outlines(
            image_detect.value, 
            outlines.value, 
            pips.value, 
            prepend_figs = f"{debug_dir}/"
        )

        (image_detect 
            >> (pips >> partial(overlay_info, dice = dice)) 
                >> write_image(f'{debug_dir}/overlay.png')
        )

        results = [len(ps[0]) for ps in dice]
        return results

import code

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
            debug_dir = f"output/{image_name}"
            image_crop = (monad(im)
                >> crop(slice(540, 2350), slice(775, 3760)) 
                >> partial(cv2.resize, dsize = (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            )
            results = do_recognition(image_crop.value, debug_dir)
            with output.indent(2):
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
