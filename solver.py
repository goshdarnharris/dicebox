import libpip
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import cv2
import sklearn.cluster
import sklearn.metrics
import skimage
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class PipGraphs(object):
    pips: libpip.Pips
    size_tolerance: float
    color_tolerance: np.array
    die_color_tolerance: np.array
    distance_limit: float

    def __post_init__(self):
        pip_graph = nx.Graph()
        for idx in range(len(self.pips.locations)):
            pip_graph.add_node(idx)

        def filter_self_edges(indices):
            return indices[np.argwhere(indices[:,0] != indices[:,1])[:,0]]

        self.size_match = pip_graph.copy()
        match_indices = np.argwhere(self.pips.size_error < self.size_tolerance*self.pips.radii[:,None])
        self.size_match.add_edges_from(filter_self_edges(match_indices))

        self.color_match = pip_graph.copy()
        match_indices = np.argwhere(np.all((self.pips.color_error < self.color_tolerance*255), axis=-1))
        self.color_match.add_edges_from(filter_self_edges(match_indices))

        self.die_color_match = pip_graph.copy()
        match_indices = np.argwhere(np.all(self.pips.die_color_error < self.die_color_tolerance*255, axis=-1))
        self.die_color_match.add_edges_from(filter_self_edges(match_indices))

        self.distance_match = pip_graph.copy()
        match_indices = np.argwhere(self.pips.distances < self.distance_limit)
        match_distances = self.pips.distances[match_indices[:,0],match_indices[:,1]]
        match_weights = np.array([{'weight': d} for d in match_distances])[:,None]
        match_indices = np.concatenate((match_indices, match_weights), axis=-1)
        self.distance_match.add_edges_from(filter_self_edges(match_indices))

        self.all_match = nx.intersection_all([self.size_match, self.color_match, self.die_color_match, self.distance_match])
        # self.all_match = nx.intersection(self.size_match, self.color_match)
        self.distance_edges_shortest_first = sorted(self.distance_match.edges(data='weight'), key = lambda x: x[2])

        nx.set_edge_attributes(self.all_match, self.distance_match.edges)
            
    def shortest_first_iterator(self):
        return distance_edges_shortest_first
    def consume_pip(self, pip):
        self.size_match.remove_node(pip)
        self.color_match.remove_node(pip)
        self.die_color_match.remove_node(pip)
        self.distance_match.remove_node(pip)
        self.all_match.remove_node(pip)

class SixMatcher(isomorphism.GraphMatcher):
    def __init__(self, pip_graph):
        six_graph = nx.Graph()
        self.edge_scales = [1,2,np.linalg.norm([1,2]),np.linalg.norm([2,2])]

        six_graph.add_edge(0, 3, distance = self.edge_scales[0])
        six_graph.add_edge(3, 6, distance = self.edge_scales[0])
        six_graph.add_edge(2, 5, distance = self.edge_scales[0])
        six_graph.add_edge(5, 8, distance = self.edge_scales[0])
        six_graph.add_edge(0, 2, distance = self.edge_scales[1])
        six_graph.add_edge(3, 5, distance = self.edge_scales[1])
        six_graph.add_edge(6, 8, distance = self.edge_scales[1])
        six_graph.add_edge(0, 6, distance = self.edge_scales[1])
        six_graph.add_edge(2, 8, distance = self.edge_scales[1])
        six_graph.add_edge(0, 5, distance = self.edge_scales[2])
        six_graph.add_edge(3, 2, distance = self.edge_scales[2])
        six_graph.add_edge(3, 8, distance = self.edge_scales[2])
        six_graph.add_edge(6, 5, distance = self.edge_scales[2])
        six_graph.add_edge(0, 8, distance = self.edge_scales[3])
        six_graph.add_edge(6, 2, distance = self.edge_scales[3])

        super().__init__(pip_graph, six_graph)
        self.pip_graph = pip_graph
        self.die_graph = six_graph
    # def semantic_feasibility(self, pip_node, die_node):
    #     #find the shortest edge from the pipe node
    #     # print(self.pip_graph[pip_node])
    #     edge_lengths_pip = [e['weight'] for e in self.pip_graph[pip_node].values()]
    #     shortest_distance_pip = min(edge_lengths_pip)

    #     n_edges_at_scale = []
    #     for scale in self.edge_scales:
    #         matches = np.where(np.isclose(edge_lengths_pip, scale*shortest_distance_pip, rtol=0.075))[0]
    #         n_edges_at_scale.append(len(matches))
    #     n_edges_at_scale = np.array(n_edges_at_scale)
    #     # print("PIP", pip_node, "DIE", die_node)
    #     # print("SHORTEST", shortest_distance_pip)
    #     # print("EDGES", edge_lengths_pip)
    #     # for idx,edges in enumerate(n_edges_at_scale):
    #     #     print(f"\tSCALE {idx}", edges)

    #     maybe_corner = False
    #     maybe_edge = False
    #     if np.any(n_edges_at_scale >= np.array([1,2,1,1])): # corner pip
    #         maybe_corner = True
    #     if np.any(n_edges_at_scale >= np.array([2,1,2,0])):
    #         maybe_edge = True
    #     # corner pips
    #     if die_node in [0,2,6,8] and maybe_corner: return True
    #     # edge pips
    #     elif die_node in [3,5] and maybe_edge: return True
    #     return False    

def solve_sixes(image, graphs):
    #For the 6, the die is the subgraph where:
    #  - there are 6 pips
    #  - for the smallest distance edge weight in the graph,
    #     - there are 4 edges with that weight
    #     - there are 2 nodes with exactly 2 edges at that weight
    #     - there are 4 nodes with exactly 1 edge at that weight
    #  - for the next largest edge weight in the graph,
    #     - there are 3 edges with that weight
    #     - the edges form 3 disjoint subgraphs (each node is only reachable from one other node by edges with that weight)
    #  - for the next largest edge weight in the graph,
    #     - there are 4 edges with that weight
    #     - the edges form 2 disjoint subgraphs
    #  - for the next largest edge weight in the graph,
    #     - there are 2 edges with that weight
    #     - the edges form 2 disjoint subgraphs
    

    matcher = SixMatcher(graphs.all_match)
    
    subgraphs = {}
    for idx,subgraph_monomorphism in enumerate(matcher.subgraph_monomorphisms_iter()):
        subgraph = graphs.all_match.subgraph(subgraph_monomorphism.keys())
        print(idx, subgraph.nodes, subgraph_monomorphism)
        for pip_idx in subgraph:
            pip = graphs.pips.get_pip(pip_idx)
            subgraphs.get(pip,[]).append(idx)
            cv2.circle(image, (int(pip.pos[0]), int(pip.pos[1])),
                   int(pip.radius), (255, 0, 0), 2)
    for pip_idx,subgraph_indices in subgraphs.items():
        pip = graphs.pips.get_pip(pip_idx)
        subgraphs_str = ','.join([str(i) for i in subgraph_indices])
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            subgraphs_str, cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(image, subgraphs_str,
                    (int(pip.pos[0] - textsize[0] / 2),
                    int(pip.pos[1] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imwrite('sixes.png',image)


def solve_graph_monomorphism(image, pips, size_tolerance = 0.1, color_tolerance = np.array([0.1, 0.1, 0.2]), die_color_tolerance = np.array([0.05, 0.1, 0.2]), distance_limit = 75):
    #Create graphs of size matches, pip/color matches, and distance between pips
    # pip_graph = nx.Graph()
    # for pip in libpip.Pip.generator(pips):
    #     pip_graph.add_node(pip)
    # pip_size_match_graph = pip_graph.copy()
    # match_indices = np.argwhere(pips.size_error < size_tolerance*pips.radii[:,None])
    # pip_size_match_graph.add_edges_from(match_indices)

    # print("constructed pip size graph")
    # pip_color_match_graph = pip_graph.copy()
    # print("constructed pip color graph")    
    # pip_die_color_match_graph = pip_graph.copy()
    # print("constructed pip die color graph")
    # pip_distance_graph = pip_graph.copy()
    # print("constructed pip distance graph")

    
    # for a in libpip.Pip.generator(pips):
    #     for b in libpip.Pip.generator(pips):
    #         if a != b:
    #             size_diff, size_match = a.matches_size(b)
    #             color_diff, color_match = a.matches_color(b)
    #             die_color_diff, die_color_match = a.matches_die_color(b)
    #             distance, distance_match = a.matches_distance(b)
    #             if size_match:
    #                 pip_size_match_graph.add_edge(a, b, weight = size_diff)
    #             if color_match:
    #                 pip_color_match_graph.add_edge(a, b, weight = color_diff)
    #             if die_color_match:
    #                 pip_die_color_match_graph.add_edge(a, b, weight = die_color_diff)
    #             if distance_match:
    #                 pip_distance_graph.add_edge(a, b, weight = distance)

    # pip_graphs = PipGraphs(pip_size_match_graph, pip_color_match_graph, pip_die_color_match_graph, pip_distance_graph)
    pip_graphs = PipGraphs(pips, size_tolerance, color_tolerance, die_color_tolerance, distance_limit)
    print("constructed pip graphs")
    #Now what?
    # for all dice, we are looking for groupings of pips that all have the same distance to their closest pip
    
    # for die >= 3,
    #   a die candidate is a subgraph where 
    solve_sixes(image, pip_graphs)

def solve_image_clustering_outline_penalty(image, outlines, pips):
    dbscan_eps = 50

    def die_metric(a, b):
        pip_a = libpip.Pip.from_tuple(a)
        pip_b = libpip.Pip.from_tuple(b)

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

    if len(pips) > 0:
        groups = []
        fit_array = np.array([p.to_tuple() for p in libpip.Pip.generator(pips)])
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
            die_pips = np.array(list(libpip.Pip.generator(pips)))[clustering.labels_ == i]

            centroid_dice = np.mean([p.pos for p in die_pips], axis=0)

            dice.append([die_pips, *centroid_dice])

        return dice

    else:
        return []

def graph_write(graph, path):
    nx.draw(graph, with_labels = True)
    plt.savefig(path)
    plt.close()

def solve_graph_outlines(image, outlines, pips, prepend_figs = ""):
    #size_graph = libpip.make_size_graph(pips, error_threshold = 0.2)
    #print("SIZE:", len(size_graph.edges))
    #color_graph = libpip.make_color_graph(pips, error_threshold_hsv = np.array([0.2, 0.2, 0.2]))
    #print("COLOR:", len(color_graph.edges))
    #die_color_graph = libpip.make_die_color_graph(pips, error_threshold_hsv = np.array([0.2, 0.2, 0.2]))
    #print("DIE COLOR:", len(die_color_graph.edges))
    distance_graph = libpip.make_distance_graph(pips, error_threshold = 12)
    print("DISTANCE:", len(distance_graph.edges))

    #dice_graph = nx.intersection_all([size_graph, color_graph, die_color_graph, distance_graph])
    dice_graph = nx.intersection_all([distance_graph])
    nx.set_edge_attributes(dice_graph, distance_graph.edges)

    edges = dice_graph.edges

    print("Start outline loss...")
    edge_outline_losses = libpip.calculate_outline_loss(pips, edges, outlines, loss = 0.35)
    print("Done.")

    for edge, loss in zip(edges, edge_outline_losses):
        if loss > 2.0: # Empirical
            dice_graph.remove_edge(*edge)

    dice = []
    for die_candidate in nx.connected_components(dice_graph):
        candidate_pips = pips[list(die_candidate),:]
        pip_locations = libpip.slice_by(candidate_pips, 'location')
        centroid = np.mean(pip_locations, axis=0)
        dice.append([candidate_pips, *centroid])
    print("solved dice")

    # six_graph = libpip.six_graph
    # graph_write(six_graph, f'{prepend_figs}six_graph.png')
    # annotated_image = image.copy()
    # for idx, die_candidate in enumerate(nx.connected_components(dice_graph)):
    #     if len(die_candidate) == 6:
    #         candidate_graph = dice_graph.subgraph(die_candidate)
    #         graph_write(candidate_graph, f'{prepend_figs}candidate_graph.{idx}.png')
    #         #get shortest edge in the die
    #         shortest_edge = min(candidate_graph.edges(data=True), key = lambda x: x[2]['distance'])[2]['distance']

    #         print("SIX:", np.sort(np.array([e[2]['distance'] for e in six_graph.edges(data=True)])))
    #         print("CAN:", np.sort(np.array([e[2]['distance'] for e in candidate_graph.edges(data=True)])/shortest_edge))

    #         def edge_match(dict_die, dict_candidate):
    #             distance_candidate = dict_candidate['distance']
    #             distance_die = dict_die['distance']
    #             result = np.isclose(distance_die, distance_candidate/shortest_edge, rtol=0.15)
    #             # if not result:
    #             #     print("EDGE MISMATCH", distance_candidate/shortest_edge, distance_die)
    #             return result

    #         gm = isomorphism.GraphMatcher(
    #             six_graph, 
    #             candidate_graph, 
    #             edge_match = edge_match                
    #         )
    #         print(f"{idx} ISO SIX:", gm.subgraph_is_monomorphic(), 
    #             f"{len(candidate_graph.nodes)}:{len(candidate_graph.edges)}", 
    #             f"{len(six_graph.nodes)}:{len(six_graph.edges)}"
    #         )
            
    #         candidate_pips = pips[list(die_candidate),:]
    #         pip_locations = libpip.slice_by(candidate_pips, 'location')
    #         centroid_x, centroid_y = np.mean(pip_locations, axis=0)

    #         textsize = cv2.getTextSize(
    #             str(idx), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

    #         cv2.putText(annotated_image, str(idx),
    #             (int(centroid_x - textsize[0] / 2),
    #             int(centroid_y + textsize[1] / 2)),
    #             cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # cv2.imwrite(f'{prepend_figs}six.annotated.png',annotated_image)

    return dice