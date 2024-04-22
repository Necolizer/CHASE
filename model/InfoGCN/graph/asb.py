import sys
import numpy as np

sys.path.extend(['../'])
from . import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
inward = [
    (5, 17), (17, 18), (18, 19), (19, 4),  # pinky
    (5, 14), (14, 15), (15, 16), (16, 3),  # ring
    (5, 11), (11, 12), (12, 13), (13, 2),  # middle
    (5, 8), (8, 9), (9, 10), (10, 1),  # index
    (5, 6), (6, 7), (7, 0),  # thumb
    (6, 8), (8, 11), (11, 14), (14, 17),  # palm
                    ]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

num_node_1 = 10
indices_1 = [1, 2, 3, 4, 5, 7, 9, 12, 15, 18]
self_link_1 = [(i, i) for i in range(num_node_1)]
# inward_ori_index_1 = [(5, 7), (5, 9), (5, 12), (5, 15), (5, 18), (9, 1), (12, 2), (15, 3), (18, 4)]
inward_ori_index_1 = [(4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (6, 0), (7, 1), (8, 2), (9, 3)]
inward_1 = [(i, j) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 6
indices_2 = [4, 0, 1, 2, 3, 5]
self_link_2 = [(i ,i) for i in range(num_node_2)]
# inward_ori_index_2 = [(5, 0), (5, 10), (5, 13), (5, 16), (5, 19)]
inward_ori_index_2 = [(1, 0), (1, 2), (1, 3), (1, 4), (1, 5)]
inward_2 = [(i, j) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
