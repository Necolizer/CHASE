import sys
import numpy as np
sys.path.extend(['../'])
from . import tools

num_node = 23
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
        (0, 1),  # hip -> ab
        (1, 2),  # ab -> chest
        (2, 3),  # chest -> neck
        (3, 4),  # neck -> head
        (3, 5),  # neck, L shoulder
        (5, 6),  # L shoulder, L U arm
        (6, 7),  # L u arm, L l f arm
        (7, 8),  # L f arm, L hand
        (3, 9),  # neck, R shoulder
        (9, 10),  # R shoulder, R U arm
        (10, 11),  # R u arm, R l f arm
        (11, 12),  # R f arm, R hand
        (0, 13),  # hip, LShin
        (13, 14),  # LShin, LTigh
        (14, 15),  # LTigh, LFoot
        (15, 16),  # LFoot, Ltoe
        (0, 17),  # hip, RShin
        (17, 18),  # RShin, RTigh
        (18, 19),  # RTigh, RFoot
        (19, 20),  # RFoot,RLtoe
        (6, 13),  # LShin to L u arm  (hip to shoulder)
        (10, 17),  # RShin to R u arm  (hip to shoulder)
    ]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
