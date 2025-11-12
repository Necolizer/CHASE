import sys

sys.path.extend(['../'])
from . import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),(6,8),(7,9),(8,10),(9,11),(2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, hyper_joints=0, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.hyper_joints = hyper_joints
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_ensemble':
            A = tools.get_spatial_graph_ensemble(num_node, self_link, inward, outward, 8)
        elif labeling_mode == 'virtual_ensemble':
            A = tools.get_virtual_graph_ensemble(num_node, self_link, inward, outward, self.hyper_joints, 8)
        else:
            raise ValueError()
        return A
