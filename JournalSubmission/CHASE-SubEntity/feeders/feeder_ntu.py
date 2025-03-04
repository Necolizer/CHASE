import numpy as np
from torch.utils.data import Dataset
from feeders import tools
import torch
import math

head_torso = [0, 0, 1, 20, 2, 3]
left_arm = [4, 5, 6, 7, 22, 21]
right_arm = [8, 9, 10, 11, 24, 23]
left_leg = [1, 0, 12, 13, 14, 15]
right_leg = [1, 0, 16, 17, 18, 19]
bodypart_5_list = [head_torso, left_arm, right_arm, left_leg, right_leg]

left_half =  [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22]
right_half = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 20, 23, 24]
bodypart_2_list = [left_half, right_half]

whole = [i for i in range(25)]
bodypart_1_list = [whole]

PartMap = {
    'wholebody': bodypart_1_list,
    '2parts': bodypart_2_list,
    '5parts': bodypart_5_list,
}

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False, random_spatial_shift=False, entity_rearrangement=False, action_type='mutual',
                 data_type='wholebody'):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        entity_rearrangement: If true, use entity rearrangement (interactive actions)
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.random_spatial_shift = random_spatial_shift
        self.entity_rearrangement = entity_rearrangement
        self.action_type = action_type
        self.data_type = data_type
        self.part_list = PartMap[data_type.lower()]
        self.load_data()

    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            
            if self.action_type.lower() == 'mutual':
                # Mutual Actions
                index_needed = np.where(((self.label > 48) & (self.label < 60)) | (self.label > 104))
                self.label = self.label[index_needed]
                self.data = self.data[index_needed]
                self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)
                self.label = np.where((self.label > 104), self.label-94, self.label)
            elif self.action_type.lower() == 'individual':
                # Individual Actions
                index_needed = np.where(((self.label <= 48)) | ((self.label >= 60) & (self.label <= 104)))
                self.label = self.label[index_needed]
                self.data = self.data[index_needed]
                self.label = np.where(((self.label >= 60) & (self.label <= 104)), self.label-11, self.label)
            else:
                pass

            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]

            if self.action_type.lower() == 'mutual':
                # Mutual Actions
                index_needed = np.where(((self.label > 48) & (self.label < 60)) | (self.label > 104))
                self.label = self.label[index_needed]
                self.data = self.data[index_needed]
                self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)
                self.label = np.where((self.label > 104), self.label-94, self.label)
            elif self.action_type.lower() == 'individual':
                # Individual Actions
                index_needed = np.where(((self.label <= 48)) | ((self.label >= 60) & (self.label <= 104)))
                self.label = self.label[index_needed]
                self.data = self.data[index_needed]
                self.label = np.where(((self.label >= 60) & (self.label <= 104)), self.label-11, self.label)
            else:
                pass

            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

        if self.action_type.lower() == 'individual':
            self.data = self.data[:,:,:,:,0:1]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        C, T, V, M = data_numpy.shape
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        data_numpy = torch.from_numpy(data_numpy)
        if self.random_spatial_shift:
            data_numpy = tools.random_spatial_shift(data_numpy, norm=0.01)
        if self.random_rot:
            data_numpy = tools.random_rot_enhanced(data_numpy, thetas=[0.3, math.pi/4, 0.3])
        if self.entity_rearrangement:
            data_numpy = data_numpy[:,:,:,torch.randperm(data_numpy.size(3))]
        if self.bone:
            ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))
            bone_data_numpy = torch.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        data_numpy = torch.cat([
            data_numpy[:, :, index_list, :] for index_list in self.part_list
        ], dim=-1)

        return data_numpy, label, index