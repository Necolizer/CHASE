import sys
import numpy as np

sys.path.extend(['../'])
from . import tools

num_node = 15
# num_node = 6
self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
# inward_ori_index = [(6, 5), (5, 4), (4, 3), (3, 2), (2, 1)]
inward_ori_index = [(1, 2), (2, 13), (3, 13), (4, 3), 
                    (5, 13), (6, 5), (7, 6), (8, 7), 
                    (9, 1), (10, 9), (11, 10), (12, 11),
                    (14, 15), (15, 8)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
