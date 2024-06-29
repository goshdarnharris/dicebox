import cv2
import numpy as np
from dataclasses import dataclass
import networkx as nx
import skimage


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

def remove_from_image(pips, image, fill=(0,0,0), factor = 1.0):
    for pip in Pip.generator(pips):
        pos = pip.pos
        r = pip.radius

        cv2.circle(image, (int(pos[0]), int(pos[1])),
                   int(r*factor), fill, -1)
    return image

def overlay_pips(pips, image):
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

    np_pips = np.concatenate((pip_locations, pip_radii[:,None], pip_colors, die_colors), axis=1)
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
    edges = make_matching_edges(np.all(die_color_error < error_threshold_hsv*255, axis=-1))
    # die_color_error = die_color_error[edges[:,0],edges[:,1]]
    # edge_labels = np.array([{'die_color_error': d} for d in die_color_error])[:,None]
    # labeled_edges = np.concatenate((edges, edge_labels), axis=-1)
    graph.add_edges_from(edges)
    return graph

def make_distance_graph(pips, error_threshold = 1):
    positions = slice_by(pips, 'location')
    distances = np.linalg.norm(positions[:,None,:] - positions[None,:,:], axis=-1)
    graph = nx.Graph()
    edges = make_matching_edges(distances < error_threshold)
    # distances = pips.distances[edges[:,0],edges[:,1]]
    # edge_labels = np.array([{'distance': d} for d in distances])[:,None]
    # labeled_edges = np.concatenate((edges, edge_labels), axis=-1)
    graph.add_edges_from(edges)
    return graph

def calculate_outline_loss(pips, edges, outlines, loss = 1):
    locations = slice_by(pips, 'location')
    error = []
    for edge in edges:
        pip_a = locations[edge[0]]
        pip_b = locations[edge[1]]
        profile = skimage.measure.profile_line(outlines, (pip_a[1], pip_a[0]), (pip_b[1], pip_b[0]))
        error.append(np.sum(profile)*loss)
    return error
