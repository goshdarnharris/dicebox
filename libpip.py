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

import code

def find_pips(frame):
    params = cv2.SimpleBlobDetector_Params()
    params.minInertiaRatio = 0.75
    detector = cv2.SimpleBlobDetector_create(params)

    frame_blurred = cv2.medianBlur(frame, 3)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    cv2.imwrite('blurred.findpips.png',frame)

    blobs = detector.detect(frame_gray)


    pip_locations = np.array([np.array(b.pt) for b in blobs])
    pip_radii = np.array([b.size/2 for b in blobs])
    pip_indices = pip_locations.astype(np.int32)

    frame_blurred_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    pip_colors = frame_blurred_hsv[pip_indices[:,1],pip_indices[:,0]]
    
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
    radii = np.squeeze(slice_by(pips, 'radius'))
    # print("RADII",radii)
    zeros = np.zeros((len(pips), len(pips)))
    radii_square = radii[None,:] + zeros
    # print("RADII",radii_square)
    distances = np.linalg.norm(positions[:,None,:] - positions[None,:,:], axis=-1)
    # print("DISTANCES",distances)
    distances_pips = distances/radii_square
    # print("DISTANCES PIPS",distances_pips)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(pips)))
    edges = make_matching_edges(distances_pips < error_threshold)
    distances = distances[edges[:,0],edges[:,1]]
    edge_labels = np.array([{'distance': d} for d in distances])[:,None]
    labeled_edges = np.concatenate((edges, edge_labels), axis=-1)
    graph.add_edges_from(labeled_edges)
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

def calculate_outline_loss(pips, edges, outlines, loss = 1):
    locations = slice_by(pips, 'location')
    losses = []
    for edge in edges:
        pip_a = locations[edge[0]]
        pip_b = locations[edge[1]]
        profile = skimage.measure.profile_line(outlines, (pip_a[1], pip_a[0]), (pip_b[1], pip_b[0]))
        this_loss = np.sum(profile.astype(np.float32)/255)*loss
        losses.append(this_loss)
    return losses
