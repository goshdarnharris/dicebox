import cv2
import numpy as np
import sklearn.cluster
import sklearn.metrics
import skimage
from pathlib import Path
import os
import sys
from dataclasses import dataclass
import networkx as nx
from networkx.algorithms import isomorphism

from datetime import datetime as dt 

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

def remove_pips(image, pips, fill=(0,0,0), factor = 1.0):
    for pip in pips.pip_iterator():
        pos = pip.pos
        r = pip.radius

        cv2.circle(image, (int(pos[0]), int(pos[1])),
                   int(r*factor), fill, -1)
    return image

def get_outlines(frame, blobs):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png',gray)
    
    blur = cv2.medianBlur(gray, 3)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8))
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)
    grad = remove_pips(grad, blobs, fill=(0,0,0), factor=2.25)

    cv2.imwrite('grad.png',grad)

    grad_blur = cv2.medianBlur(grad, 3)

    cv2.imwrite('grad_blur.png',grad_blur)


    ret, grad_thresh = cv2.threshold(grad_blur, 15, 255, cv2.THRESH_BINARY)
    cv2.imwrite('grad_thresh.png',grad_thresh)

    return grad_thresh

@dataclass(frozen=True)
class Pip(object):
    pos: tuple
    radius: float
    color: tuple
    die_color: tuple
    def to_tuple(self):
        return (*self.pos, self.radius, *self.color, *self.die_color)
    @staticmethod
    def from_tuple(t):
        return Pip(t[:2], t[2], t[3:6], t[6:])

@dataclass
class Pips(object):
    locations: np.array
    radii: np.array
    colors: np.array
    die_colors: np.array

    def __post_init__(self):
        assert len(self.locations) == len(self.radii) == len(self.colors) == len(self.die_colors)
        self.distances = np.linalg.norm(self.locations[:,None] - self.locations, axis=-1)
        self.size_error = np.abs(abs(self.radii[:,None] - self.radii)/self.radii)
        self.color_error = np.abs(self.colors[:,None].astype(np.float32) - self.colors.astype(np.float32))
        self.die_color_error = np.abs(self.die_colors[:,None].astype(np.float32) - self.die_colors.astype(np.float32))

        # print("LOCATIONS", self.locations)
        # print("RADII", self.radii)
        # print("COLORS", self.colors)
        # print("DIE COLORS", self.die_colors)

        # print("DISTANCES", self.distances)
        # print("DIFF SIZES", self.size_error)
        # print("DIFF COLORS", self.color_error)
        # print("DIFF DIE COLORS", self.die_color_error)
    def get_pip(self, idx):
        return Pip(tuple(self.locations[idx]), self.radii[idx], tuple(self.colors[idx]), tuple(self.die_colors[idx]))
    def pip_iterator(self):
        for i in range(len(self.locations)):
            yield self.get_pip(i)
    def overlay_pips(self, image):
        for pip in self.pip_iterator():
            cv2.circle(image, (int(pip.pos[0]), int(pip.pos[1])),
                    int(pip.radius), (255, 0, 0), 2)

            textsize = cv2.getTextSize(
                f"{pip.radius:0.2f}", cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

            cv2.putText(image, f"{pip.radius:0.2f}",
                        (int(pip.pos[0]),
                        int(pip.pos[1])),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return image
    def count(self):
        return len(self.locations)

def find_pips(frame):
    params = cv2.SimpleBlobDetector_Params()
    params.minInertiaRatio = 0.7
    detector = cv2.SimpleBlobDetector_create(params)

    frame_blurred = cv2.medianBlur(frame, 3)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('blurred.findpips.png',frame_blurred)

    blobs = detector.detect(frame_gray)


    pip_locations = np.array([np.array(b.pt) for b in blobs])
    pip_radii = np.array([b.size/2 for b in blobs])
    pip_indices = pip_locations.astype(np.int32)

    frame_blurred_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

    pip_colors = frame_blurred_hsv[pip_indices[:,1],pip_indices[:,0]]
    
    die_color_sample_pos_eps = np.ceil(3*pip_radii/2).astype(np.int32)
    die_color_samples = np.array([
        frame_blurred_hsv[pip_indices[:,1] + die_color_sample_pos_eps, pip_indices[:,0]],        
        frame_blurred_hsv[pip_indices[:,1], pip_indices[:,0] + die_color_sample_pos_eps],        
        frame_blurred_hsv[pip_indices[:,1] - die_color_sample_pos_eps, pip_indices[:,0]],        
        frame_blurred_hsv[pip_indices[:,1], pip_indices[:,0] - die_color_sample_pos_eps]
    ])

    die_colors = np.mean(die_color_samples, axis=0)

    return Pips(pip_locations, pip_radii, pip_colors, die_colors)

def get_dice_from_pips(image, outlines, pips):
    dbscan_eps = 50

    def die_metric(a, b):
        pip_a = Pip.from_tuple(a)
        pip_b = Pip.from_tuple(b)

        loss = sklearn.metrics.pairwise_distances([pip_a.pos],[pip_b.pos])
        if loss < dbscan_eps:
            #apply a penalty for difference in radius
            tolerance = 0.08
            hinge_factor = 2
            difference = np.abs(pip_a.radius - pip_b.radius)
            if difference > tolerance:
                loss = loss + hinge_factor*(difference - tolerance)
        if loss < dbscan_eps:
            #apply a penalty if the line between the two points crosses a dice outline
            profile = skimage.measure.profile_line(outlines, (pip_a.pos[1], pip_a.pos[0]), (pip_b.pos[1], pip_b.pos[0]))
            distance_penalty = np.sum(profile)
            return loss + distance_penalty
        return loss

    if pips.count() > 0:
        groups = []




        fit_array = np.array([p.to_tuple() for p in pips.pip_iterator()])
        clustering = sklearn.cluster.DBSCAN(
            eps=dbscan_eps, 
            min_samples=1,
            metric = die_metric
        ).fit(fit_array)

        # Find the largest label assigned + 1, that's the number of dice found
        num_dice = max(clustering.labels_) + 1

        dice = []
        # Calculate centroid of each dice, the average between all a dice's dots
        for i in range(num_dice):
            die_pips = np.array(list(pips.pip_iterator()))[clustering.labels_ == i]

            centroid_dice = np.mean([p.pos for p in die_pips], axis=0)

            dice.append([die_pips, *centroid_dice])

        return dice

    else:
        return []

def overlay_info(frame, dice, pips):
    # Overlay pips
    for pip in pips.pip_iterator():
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

            pips = find_pips(image_detect)
            print("found pips")
            cv2.imwrite(f'{image_name}.pips.png', pips.overlay_pips(image_detect.copy()))
            print("getting outlines")
            outlines = get_outlines(image_detect,pips)
            print("extracting dice")
            dice = get_dice_from_pips(image_detect,outlines,pips)

            results = [len(ps[0]) for ps in dice]
            for roll in range(1,6):
                print(f'{roll}: {results.count(roll)}')

            out_pips = overlay_info(image_detect, dice, pips)
            cv2.imwrite(f'{image_name}.overlay.png',out_pips)

if __name__ == '__main__':
    main()
