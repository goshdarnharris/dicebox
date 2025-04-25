import cv2
import numpy as np
from dataclasses import dataclass
import networkx as nx
import skimage

import libpysal.weights as lw

# FIXME: For some reason the first call to lw.Gabriel() is insanely slow (10s-ish)
# Front-load it so we don't have to wait during our first interaction
if 1:
    dummy = np.array( ((0,0), (1,1), (0,1)) )
    print("Starting dummy gabriel...")
    lw.Gabriel(dummy)
    print("Done.")

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
    @staticmethod
    def generator(pips : np.array):
        for tup in pips:
            yield Pip.from_tuple(tup)


@dataclass(frozen = True)
class Pips(object):
    lookup: np.array
    np_pips: np.array

def remove_from_image(image, pips, fill, factor = 1.0, expand = 0):
    for pip in Pip.generator(pips):
        pos = pip.pos
        r = pip.radius

        if len(fill) == 4 and fill == "auto":
            pfill = pip.die_color    
        else:
            pfill = fill        

        cv2.circle(image, (int(pos[0]), int(pos[1])),
                   int(r*factor + expand), pfill, -1)
    return image

def overlay_pips(image, pips):
    for pip in Pip.generator(pips):
        cv2.circle(image, (int(pip.pos[0]), int(pip.pos[1])),
                int(pip.radius), (255, 0, 0), 2)

        textsize = cv2.getTextSize(
            f"{pip.radius:0.2f}", cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        cv2.putText(image, f"{pip.radius:0.2f}",
                    (int(pip.pos[0]),
                    int(pip.pos[1])),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return image

def find_pips(frame):
    #frame_blurred = cv2.medianBlur(frame, 3)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    frame_gray = clahe.apply(frame_gray)
    #_, frame_gray = cv2.threshold(frame_gray, thresh=128, maxval=255, type=cv2.THRESH_BINARY)

    # Erode speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, kernel)

    # Gentle blur
    frame_gray = cv2.GaussianBlur(frame_gray, (5,5), 0)

    # Extend the edges because the blob detector struggles at the edges
    padding_px = 0
    padded_frame = cv2.copyMakeBorder(
        frame_gray,
        padding_px, padding_px, padding_px, padding_px,
        borderType=cv2.BORDER_REPLICATE
    )

    cv2.imwrite('livecam/blurred.findpips.png',padded_frame)

    if 0:
        # Original blob-based pip detection
        params = cv2.SimpleBlobDetector_Params()
        params.minInertiaRatio = 0.70
        params.minArea = 500.0
        params.maxArea = 1200.0
        params.minDistBetweenBlobs = 25.0
        detector = cv2.SimpleBlobDetector_create(params)
        blobs = detector.detect(padded_frame)
        #Relocate to original unpadded image
        for blob in blobs:
            blob.pt = (blob.pt[0] - padding_px, blob.pt[1] - padding_px)
        pip_locations = np.array([np.array(b.pt) for b in blobs])
        pip_radii = np.array([b.size/2 for b in blobs])
        pip_indices = pip_locations.astype(np.int32)
    else:
        # JHW: Hough circle-based pip detection was more reliable in my testing, especially near the edges of the image

        # Internally the circle detector works on Canny edges, so we will generate a test image just for our reference.
        # This image is for human diagnostic purposes only.
        canny_edges = cv2.Canny(padded_frame, 90, 45)
        cv2.imwrite('livecam/canny_edges.png', canny_edges)

        #Hough circle detection
        circles = cv2.HoughCircles(
            padded_frame,
            method=cv2.HOUGH_GRADIENT_ALT,
            dp=2.0,  # Inverse resolution (1 = same size as input)
            minDist=10,  # Min distance between circle centers
            param1=90,  # Canny high threshold (lower threshold = param1/2)
            #param2=25,  # Circle detection threshold (lower = more sensitive)
            param2=0.9,
            minRadius=9,  # Min expected radius
            maxRadius=16  # Max expected radius
        )
        # For some reason circles has an extra dimension
        circles = circles[0] if circles is not None else []
        for circle in circles:
            # each row is x, y, radius
            circle[0] -= padding_px
            circle[1] -= padding_px
        pip_locations = np.array([np.array(c[:2]) for c in circles])
        pip_radii = np.array([c[2] for c in circles])
        pip_indices = pip_locations.astype(np.int32)

    if len(pip_indices) == 0:
        # No pips detected
        return np.array([])

    frame_blurred_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    pip_colors = frame_blurred_hsv[pip_indices[:,1],pip_indices[:,0]]

    if 0:
        die_color_sample_pos_eps = np.ceil(1.1*pip_radii).astype(np.int32)
        die_color_sample_pos_eps_diag = np.ceil(np.sqrt(2)*die_color_sample_pos_eps).astype(np.int32)
        die_color_samples = [
            frame_blurred_hsv[pip_indices[:,1] + die_color_sample_pos_eps, pip_indices[:,0]],
            frame_blurred_hsv[pip_indices[:,1], pip_indices[:,0] + die_color_sample_pos_eps],
            frame_blurred_hsv[pip_indices[:,1] - die_color_sample_pos_eps, pip_indices[:,0]],
            frame_blurred_hsv[pip_indices[:,1], pip_indices[:,0] - die_color_sample_pos_eps],
            frame_blurred_hsv[pip_indices[:,1] + die_color_sample_pos_eps_diag, pip_indices[:,0] + die_color_sample_pos_eps_diag],
            frame_blurred_hsv[pip_indices[:,1] - die_color_sample_pos_eps_diag, pip_indices[:,0] - die_color_sample_pos_eps_diag],
            frame_blurred_hsv[pip_indices[:,1] + die_color_sample_pos_eps_diag, pip_indices[:,0] - die_color_sample_pos_eps_diag],
            frame_blurred_hsv[pip_indices[:,1] - die_color_sample_pos_eps_diag, pip_indices[:,0] + die_color_sample_pos_eps_diag]
        ]
        die_colors = np.median(die_color_samples, axis=0)
    else:
        #FIXME: The above is broken and I don't want to figure out what it's trying to do
        die_colors = pip_colors

    np_pips = np.concatenate((pip_locations, pip_radii[:,None], pip_colors, die_colors), axis=1)
    #code.interact(local=locals())
    return np_pips

def slice_by(pips, key):
    lookup = {'location': slice(0,2), 'radius': slice(2,3), 'color': slice(3,6), 'die_color': slice(6,9)}
    return pips[:,lookup[key]]

def filter_self_edges(indices):
    return indices[np.argwhere(indices[:,0] != indices[:,1])[:,0]]

def make_matching_edges(matching):
    matching_edges = np.argwhere(matching)
    matching_edges = filter_self_edges(matching_edges)
    return matching_edges

def make_size_graph(pips, error_threshold = 0.1):
    sizes = slice_by(pips, 'radius')
    size_error = np.abs(sizes[:,None,:] - sizes[None,:,:])

    graph = nx.Graph()
    graph.add_nodes_from(range(len(pips)))
    edges = make_matching_edges(np.all(size_error < error_threshold*sizes[:,None], axis=-1))
    # size_error = size_error[edges[:,0],edges[:,1]]
    # edge_labels = np.array([{'size_error': d} for d in size_error])[:,None]
    # labeled_edges = np.concatenate((edges, edge_labels), axis=-1)
    graph.add_edges_from(edges)
    return graph

def make_color_graph(pips, error_threshold_hsv = np.array([0.1, 0.1, 0.2])):
    color = slice_by(pips, 'color')
    color_error = np.abs(color[:,None,:] - color[None,:,:])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(pips)))
    edges = make_matching_edges(np.all((color_error < error_threshold_hsv*255), axis=-1))
    # color_error = color_error[edges[:,0],edges[:,1]]
    # edge_labels = np.array([{'color_error': d} for d in color_error])[:,None]
    # labeled_edges = np.concatenate((edges, edge_labels), axis=-1)
    graph.add_edges_from(edges)
    return graph

def make_die_color_graph(pips, error_threshold_hsv = np.array([0.05, 0.1, 0.2])):
    die_color = slice_by(pips, 'die_color')
    die_color_error = np.abs(die_color[:,None,:] - die_color[None,:,:])

    graph = nx.Graph()
    graph.add_nodes_from(range(len(pips)))
    edges = make_matching_edges(np.all(die_color_error < error_threshold_hsv*255, axis=-1))
    # die_color_error = die_color_error[edges[:,0],edges[:,1]]
    # edge_labels = np.array([{'die_color_error': d} for d in die_color_error])[:,None]
    # labeled_edges = np.concatenate((edges, edge_labels), axis=-1)
    graph.add_edges_from(edges)
    return graph

def make_distance_graph(pips, error_threshold = 1):
    positions = slice_by(pips, 'location')
    edges = set()
    if len(pips) > 3:
        # If we have enough pips, do a Gabriel triangulation. This is O(N logN) ish to calculate.
        # This is a triangulation where for every 3 fully connected nodes, you are guaranteed there are no
        # nodes in the circumscribed circle. Gives us a pretty cheap low density graph were nearest neighbors
        # are guaranteed to be connected.
        print("Start Gabriel...")
        gabriel = lw.Gabriel(positions)
        print("Done.")

        for i, neighbors in gabriel.neighbors.items():
            for j in neighbors:
                to_add = (i,j) if i<j else (j,i)
                edges.add(to_add)
    elif len(pips) > 1:
        # Not enough edges for a Gabriel graph, just fully connect the nodes
        for i in range(len(pips)-1):
            for j in range(i,len(pips),1):
                edges.add( (i,j) )
    # edges now contains the unique edges in the Gabriel graph
    # Gel the order
    edges = np.array(list(edges))

    if len(edges) == 0:
        #Return early
        graph = nx.Graph()
        graph.add_nodes_from(range(len(pips)))
        return graph

    # Construct an array of pairs of locations
    edge_locations = positions[edges]
    distances = np.linalg.norm(edge_locations[:,0] - edge_locations[:,1], axis=-1)

    radii = np.squeeze(slice_by(pips, 'radius'))

    edge_radii = radii[edges]
    avg_radii_by_edge = np.average(edge_radii,-1)

    normalized_dist = distances / avg_radii_by_edge
    labeled_edges = np.array([{'distance': d} for d in normalized_dist])[:,None]
    labeled_edges = np.concatenate((edges, labeled_edges), axis=-1)

    # Optimization: From observation, we know that we don't see edges within a die which are > 7ish pip-radii long
    # Exclude all edges with length > 10
    # TODO: Will need to revisit this if we use different dice
    short_edges = np.array([
        row for row in labeled_edges
        if row[2]['distance'] < 10.0
    ])

    graph = nx.Graph()
    graph.add_nodes_from(range(len(pips)))
    graph.add_edges_from(short_edges)
    return graph

def make_six_graph():
    six_graph = nx.Graph()
    edge_scales = [1,2,np.linalg.norm([1,2]),np.linalg.norm([2,2])]
    six_graph.add_edge(0, 3, distance = edge_scales[0])
    six_graph.add_edge(3, 6, distance = edge_scales[0])
    six_graph.add_edge(2, 5, distance = edge_scales[0])
    six_graph.add_edge(5, 8, distance = edge_scales[0])
    six_graph.add_edge(0, 2, distance = edge_scales[1])
    six_graph.add_edge(3, 5, distance = edge_scales[1])
    six_graph.add_edge(6, 8, distance = edge_scales[1])
    six_graph.add_edge(0, 6, distance = edge_scales[1])
    six_graph.add_edge(2, 8, distance = edge_scales[1])
    six_graph.add_edge(0, 5, distance = edge_scales[2])
    six_graph.add_edge(3, 2, distance = edge_scales[2])
    six_graph.add_edge(3, 8, distance = edge_scales[2])
    six_graph.add_edge(6, 5, distance = edge_scales[2])
    six_graph.add_edge(0, 8, distance = edge_scales[3])
    six_graph.add_edge(6, 2, distance = edge_scales[3])
    return six_graph

six_graph = make_six_graph()

def fast_profile_sum(image, p0, p1):
    # This is a rougher but much faster version of skimage.measure.profile_line. That function does interpolation
    # and antialiasing which make it slow. We don't care about that, we just want to sample every pixel the line crosses.
    rr, cc = skimage.draw.line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
    return np.float32(np.sum(image[rr, cc]))

def calculate_outline_loss(pips, edges, outlines, loss = 1):
    locations = slice_by(pips, 'location')
    losses = []
    for edge in edges:
        pip_a = locations[edge[0]]
        pip_b = locations[edge[1]]
        this_loss = fast_profile_sum(outlines, (pip_a[0], pip_a[1]), (pip_b[0], pip_b[1]))
        this_loss /= 255.0
        losses.append(this_loss)
    return losses
