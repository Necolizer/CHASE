import os
import pickle
import numpy as np
import random
import copy
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from dotmap import DotMap


COCO_KEYPOINT_INDEXES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

COCO_KEYPOINT_HORIZONTAL_FLIPPED = {
    0: 0,
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: 6,
    6: 5,
    7: 8,
    8: 7,
    9: 10,
    10: 9,
    11: 12,
    12: 11,
    13: 14,
    14: 13,
    15: 16,
    16: 15
}

 
KEYPOINT_PURTURB_RANGE = 1.0


class Volleyball(Dataset):
    def __init__(self, **args):
        self.args = DotMap(args)
        self.split = self.args.split
        
        
        if self.args.olympic_split:
            self.dataset_splits = {
                'train': [1, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                          41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                'test': [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                         19, 20, 21, 22, 23, 24, 25, 26, 27]
            }
        else:
            self.dataset_splits = {
                'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39,
                          40, 41, 42, 48, 50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26,
                          27, 28, 30, 33, 46, 49, 51],
                'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
            }
        
        
        self.idx2class = {
            0: {'r_set', 'r-set'},
            1: {'l_set', 'l-set'},
            2: {'r_spike', 'r-spike'},
            3: {'l_spike', 'l-spike'},
            4: {'r_pass', 'r-pass'},
            5: {'l_pass', 'l-pass'},
            6: {'r_winpoint', 'r-winpoint'},
            7: {'l_winpoint', 'l-winpoint'}
        }
        self.class2idx = dict()
        if self.args.print_cls_idx:
            print('class index:') 
        for k in self.idx2class:
            for v in self.idx2class[k]:
                self.class2idx[v] = k
                if self.args.print_cls_idx:
                    print('{}: {}'.format(v, k))
        self.group_activities_weights = torch.FloatTensor([1., 1., 1., 1., 1., 1., 1., 1.]).cuda()
        
                    
        self.person_actions_all = pickle.load(
                open(os.path.join(self.args.dataset_dir, self.args.person_action_label_file_name), "rb"))
        self.person_actions_weights = torch.FloatTensor([0.2, 1., 1., 2., 3., 1., 4., 4., 0.2, 1.]).cuda()
        # ACTIONS = ['NA', 'blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']
        # { 'NA': 0,
        # 'blocking': 1, 
        # 'digging': 2, 
        #  'falling': 3, 
        #  'jumping': 4,
        #  'moving':5 , 
        #  'setting': 6, 
        #  'spiking': 7, 
        #  'standing': 8,
        #  'waiting': 9}
        
        
        
        self.annotations = []
        self.annotations_each_person = []
        self.clip_joints_paths = []
        self.clips = []
        if self.args.ball_trajectory_use:
            self.clip_ball_paths = []
        self.prepare(self.args.dataset_dir)
            
        if self.args.horizontal_flip_augment and self.split == 'train':
            self.classidx_horizontal_flip_augment = {
                0: 1,
                1: 0,
                2: 3,
                3: 2,
                4: 5,
                5: 4,
                6: 7,
                7: 6
            }
            if self.args.horizontal_flip_augment_purturb:
                self.horizontal_flip_augment_joint_randomness = dict()
                
        if self.args.horizontal_move_augment and self.split == 'train':
            self.horizontal_move_augment_joint_randomness = dict()
                
        if self.args.vertical_move_augment and self.split == 'train':
            self.vertical_move_augment_joint_randomness = dict()
            
        if self.args.agent_dropout_augment:
            self.agent_dropout_augment_randomness = dict()
            
        self.collect_standardization_stats()
        

        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.clip_joints_paths))]
        elif self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.clip_joints_paths))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        
    def prepare(self, dataset_dir):
        """
        Prepare the following lists based on the dataset_dir, self.split
            - self.annotations 
            - self.annotations_each_person 
            - self.clip_joints_paths
            - self.clips
            (the following if needed)
            - self.clip_ball_paths
            - self.horizontal_flip_mask
            - self.horizontal_mask
            - self.vertical_mask
            - self.agent_dropout_mask
        """  
        annotations_thisdatasetdir = defaultdict()
        clip_joints_paths = []

        for annot_file in glob.glob(os.path.join(dataset_dir, 'videos/*/annotations.txt')):
            video = annot_file.split('/')[-2]
            with open(annot_file, 'r') as f:
                lines = f.readlines()
            for l in lines:
                clip, label = l.split()[0].split('.jpg')[0], l.split()[1]
                annotations_thisdatasetdir[(video, clip)] = self.class2idx[label]  

        for video in self.dataset_splits[self.split]:
            clip_joints_paths.extend(glob.glob(os.path.join(dataset_dir, self.args.joints_folder_name, str(video), '*.pickle')))
            
        count = 0
        for path in clip_joints_paths:
            video, clip = path.split('/')[-2], path.split('/')[-1].split('.pickle')[0]
            self.clips.append((video, clip))
            self.annotations.append(annotations_thisdatasetdir[(video, clip)])
            self.annotations_each_person.append(self.person_actions_all[(int(video), int(clip))])
            if self.args.ball_trajectory_use:
                self.clip_ball_paths.append(os.path.join(dataset_dir, self.args.ball_trajectory_folder_name, video, clip + '.txt'))
            count += 1
        # print('total number of clips is {}'.format(count))

        self.clip_joints_paths += clip_joints_paths
      
        assert len(self.annotations) == len(self.clip_joints_paths)
        assert len(self.annotations) == len(self.annotations_each_person)
        assert len(self.clip_joints_paths) == len(self.clips)
        if self.args.ball_trajectory_use:
            assert len(self.clip_joints_paths) == len(self.clip_ball_paths)
        
        true_data_size = len(self.annotations)
        true_annotations = copy.deepcopy(self.annotations)
        true_annotations_each_person = copy.deepcopy(self.annotations_each_person)
        true_clip_joints_paths = copy.deepcopy(self.clip_joints_paths)
        true_clips = copy.deepcopy(self.clips)
        if self.args.ball_trajectory_use:
            true_clip_ball_paths = copy.deepcopy(self.clip_ball_paths)
        
        # if horizontal flip augmentation and is training
        if self.args.horizontal_flip_augment and self.split == 'train':
            self.horizontal_flip_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
                
        # if horizontal move augmentation and is training
        if self.args.horizontal_move_augment and self.split == 'train':
            self.horizontal_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
      
        # if vertical move augmentation and is training
        if self.args.vertical_move_augment and self.split == 'train':
            self.vertical_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
                
        # if random agent dropout augmentation and is training
        if self.args.agent_dropout_augment and self.split == 'train':
            self.agent_dropout_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
            
    
    def __len__(self):
        return len(self.clip_joints_paths)
    
         
    def collect_standardization_stats(self):
        # get joint x/y mean, std over train set
        if self.split == 'train':
            if self.args.recollect_stats_train or (
                not os.path.exists(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'))):
                joint_xcoords = []
                joint_ycoords = []
                joint_dxcoords = []
                joint_dycoords = [] 
                if self.args.ball_trajectory_use:
                        ball_xcoords = []
                        ball_ycoords = []
                        ball_dxcoords = []
                        ball_dycoords = [] 

                for index in range(self.__len__()):   # including augmented data!
                    with open(self.clip_joints_paths[index], 'rb') as f:
                        joint_raw = pickle.load(f)
                        
                    frames = sorted(joint_raw.keys())[self.args.frame_start_idx:self.args.frame_end_idx+1:self.args.frame_sampling]

                    if self.args.ball_trajectory_use:
                        ball_trajectory_data = self.read_ball_trajectory(self.clip_ball_paths[index])
                        ball_trajectory_data = ball_trajectory_data[self.args.frame_start_idx+10:self.args.frame_end_idx+1+10:self.args.frame_sampling]
                        # ball trajectory annotation has 41 frames annotated but joint/track file only has 20 frames.
                        assert len(ball_trajectory_data) == len(frames)
                        # (T, 2)
                       
                    # if horizontal flip augmentation and is training
                    if self.args.horizontal_flip_augment:
                        if index < len(self.horizontal_flip_mask):
                            if self.horizontal_flip_mask[index]:
                                if self.args.horizontal_flip_augment_purturb:
                                    self.horizontal_flip_augment_joint_randomness[index] = defaultdict()
                                    joint_raw = self.horizontal_flip_augment_joint(
                                        joint_raw, frames, 
                                        add_purturbation=True, randomness_set=False, index=index)
                                else:
                                    joint_raw = self.horizontal_flip_augment_joint(joint_raw, frames)
                                    
                                if self.args.ball_trajectory_use:
                                    ball_trajectory_data = self.horizontal_flip_ball_trajectory(ball_trajectory_data)
                                    
                    
                    # if horizontal move augmentation and is training
                    if self.args.horizontal_move_augment:
                        if index < len(self.horizontal_mask):
                            if self.horizontal_mask[index]:
                                if self.args.ball_trajectory_use:
                                    if self.args.horizontal_move_augment_purturb:
                                        self.horizontal_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw, ball_trajectory_data = self.horizontal_move_augment_joint(
                                            joint_raw, frames,  
                                            add_purturbation=True, randomness_set=False, index=index, ball_trajectory=ball_trajectory_data)
                                    else:
                                        joint_raw, ball_trajectory_data = self.horizontal_move_augment_joint(joint_raw, frames, ball_trajectory=ball_trajectory_data)
                                else:
                                    if self.args.horizontal_move_augment_purturb:
                                        self.horizontal_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw = self.horizontal_move_augment_joint(
                                            joint_raw, frames,  
                                            add_purturbation=True, randomness_set=False, index=index)
                                    else:
                                        joint_raw = self.horizontal_move_augment_joint(joint_raw, frames)
                            
                    # if vertical move augmentation and is training
                    if self.args.vertical_move_augment:
                        if index < len(self.vertical_mask):
                            if self.vertical_mask[index]:
                                if self.args.ball_trajectory_use:
                                    if self.args.vertical_move_augment_purturb:
                                        self.vertical_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw, ball_trajectory_data = self.vertical_move_augment_joint(
                                            joint_raw, frames,  
                                            add_purturbation=True, randomness_set=False, index=index, ball_trajectory=ball_trajectory_data)
                                    else:
                                        joint_raw, ball_trajectory_data = self.vertical_move_augment_joint(joint_raw, frames, ball_trajectory=ball_trajectory_data)
                                else:
                                    if self.args.vertical_move_augment_purturb:
                                        self.vertical_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw = self.vertical_move_augment_joint(
                                            joint_raw, frames,  
                                            add_purturbation=True, randomness_set=False, index=index)
                                    else:
                                        joint_raw = self.vertical_move_augment_joint(joint_raw, frames)
                                    
                    # To compute statistics, no need to consider the random agent dropout augmentation,
                    # but we can set the randomness here.
                    # if random agent dropout augmentation and is training
                    if self.args.agent_dropout_augment:
                        if index < len(self.agent_dropout_mask):
                            if self.agent_dropout_mask[index]:
                                chosen_frame = random.choice(frames)
                                chosen_person = random.choice(range(self.args.N))
                                self.agent_dropout_augment_randomness[index] = (chosen_frame, chosen_person)
            
                    
                    joint_raw = self.joints_sanity_fix(joint_raw, frames)
                    if self.args.ball_trajectory_use:
                        ball_trajectory_data = self.ball_trajectory_sanity_fix(ball_trajectory_data)
                    

                    for tidx, frame in enumerate(frames):
                        joint_xcoords.extend(joint_raw[frame][:,:,0].flatten().tolist())
                        joint_ycoords.extend(joint_raw[frame][:,:,1].flatten().tolist())

                        if tidx != 0:
                            pre_frame = frames[tidx-1]
                            joint_dxcoords.extend((joint_raw[frame][:,:,0]-joint_raw[pre_frame][:,:,0]).flatten().tolist())
                            joint_dycoords.extend((joint_raw[frame][:,:,1]-joint_raw[pre_frame][:,:,1]).flatten().tolist())
                        else:
                            joint_dxcoords.extend((np.zeros((self.args.N, self.args.J))).flatten().tolist())
                            joint_dycoords.extend((np.zeros((self.args.N, self.args.J))).flatten().tolist())
                            
                    if self.args.ball_trajectory_use:
                        ball_xcoords.extend(list(ball_trajectory_data[:, 0]))
                        ball_ycoords.extend(list(ball_trajectory_data[:, 1]))
                        
                        for t in range(len(ball_trajectory_data)):
                            if t == 0:
                                ball_dxcoords.append(0)
                                ball_dycoords.append(0)
                            else:
                                ball_dxcoords.append(ball_trajectory_data[t, 0] - ball_trajectory_data[t-1, 0])
                                ball_dycoords.append(ball_trajectory_data[t, 1] - ball_trajectory_data[t-1, 1])
                             

                # -- collect mean std
                if self.args.ball_trajectory_use:
                    joint_xcoords_mean, joint_xcoords_std = np.mean(joint_xcoords), np.std(joint_xcoords)
                    joint_ycoords_mean, joint_ycoords_std = np.mean(joint_ycoords), np.std(joint_ycoords)
                    joint_dxcoords_mean, joint_dxcoords_std = np.mean(joint_dxcoords), np.std(joint_dxcoords)
                    joint_dycoords_mean, joint_dycoords_std = np.mean(joint_dycoords), np.std(joint_dycoords)
                    
                    ball_xcoords_mean, ball_xcoords_std = np.mean(ball_xcoords), np.std(ball_xcoords)
                    ball_ycoords_mean, ball_ycoords_std = np.mean(ball_ycoords), np.std(ball_ycoords)
                    ball_dxcoords_mean, ball_dxcoords_std = np.mean(ball_dxcoords), np.std(ball_dxcoords)
                    ball_dycoords_mean, ball_dycoords_std = np.mean(ball_dycoords), np.std(ball_dycoords) 


                    self.stats = {
                        'joint_xcoords_mean': joint_xcoords_mean, 'joint_xcoords_std': joint_xcoords_std,
                        'joint_ycoords_mean': joint_ycoords_mean, 'joint_ycoords_std': joint_ycoords_std,
                        'joint_dxcoords_mean': joint_dxcoords_mean, 'joint_dxcoords_std': joint_dxcoords_std,
                        'joint_dycoords_mean': joint_dycoords_mean, 'joint_dycoords_std': joint_dycoords_std,
                        'ball_xcoords_mean': ball_xcoords_mean, 'ball_xcoords_std': ball_xcoords_std,
                        'ball_ycoords_mean': ball_ycoords_mean, 'ball_ycoords_std': ball_ycoords_std,
                        'ball_dxcoords_mean': ball_dxcoords_mean, 'ball_dxcoords_std': ball_dxcoords_std,
                        'ball_dycoords_mean': ball_dycoords_mean, 'ball_dycoords_std': ball_dycoords_std
                    }

                else:
                    joint_xcoords_mean, joint_xcoords_std = np.mean(joint_xcoords), np.std(joint_xcoords)
                    joint_ycoords_mean, joint_ycoords_std = np.mean(joint_ycoords), np.std(joint_ycoords)
                    joint_dxcoords_mean, joint_dxcoords_std = np.mean(joint_dxcoords), np.std(joint_dxcoords)
                    joint_dycoords_mean, joint_dycoords_std = np.mean(joint_dycoords), np.std(joint_dycoords) 

                    self.stats = {
                        'joint_xcoords_mean': joint_xcoords_mean, 'joint_xcoords_std': joint_xcoords_std,
                        'joint_ycoords_mean': joint_ycoords_mean, 'joint_ycoords_std': joint_ycoords_std,
                        'joint_dxcoords_mean': joint_dxcoords_mean, 'joint_dxcoords_std': joint_dxcoords_std,
                        'joint_dycoords_mean': joint_dycoords_mean, 'joint_dycoords_std': joint_dycoords_std
                    }
                    
                    
                os.makedirs(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name), exist_ok=True)
                with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'), 'wb') as f:
                    pickle.dump(self.stats, f)
                    
                if self.args.horizontal_flip_augment and self.args.horizontal_flip_augment_purturb:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_flip_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.horizontal_flip_augment_joint_randomness, f)
                        
                if self.args.horizontal_move_augment and self.args.horizontal_move_augment_purturb:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_move_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.horizontal_move_augment_joint_randomness, f)
                        
                if self.args.vertical_move_augment and self.args.vertical_move_augment_purturb:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'vertical_move_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.vertical_move_augment_joint_randomness, f)
                        
                if self.args.agent_dropout_augment:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'agent_dropout_augment_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.agent_dropout_augment_randomness, f)
                    
            else:
                try:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'), 'rb') as f:
                        self.stats = pickle.load(f)
                except FileNotFoundError:
                    print('Dataset statistics (e.g., mean, std) are missing! The dataset statistics pickle file should be generated during training.')
                    os._exit(0)
                    
                if self.args.horizontal_flip_augment and self.args.horizontal_flip_augment_purturb:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_flip_augment_joint_randomness.pickle'), 'rb') as f:
                        self.horizontal_flip_augment_joint_randomness = pickle.load(f)
                        
                if self.args.horizontal_move_augment and self.args.horizontal_move_augment_purturb:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_move_augment_joint_randomness.pickle'), 'rb') as f:
                        self.horizontal_move_augment_joint_randomness = pickle.load(f)
                        
                if self.args.vertical_move_augment and self.args.vertical_move_augment_purturb:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'vertical_move_augment_joint_randomness.pickle'), 'rb') as f:
                        self.vertical_move_augment_joint_randomness = pickle.load(f)
                
                if self.args.agent_dropout_augment:
                    with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 
                                           'agent_dropout_augment_randomness.pickle'), 'rb') as f:
                        self.agent_dropout_augment_randomness = pickle.load(f)
        else:
            try:
                with open(os.path.join(self.args.dataset_dir, 'generated_dataset_stat', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'), 'rb') as f:
                    self.stats = pickle.load(f)
            except FileNotFoundError:
                print('Dataset statistics (e.g., mean, std) are missing! The dataset statistics pickle file should be generated during training.')
                os._exit(0)
                
                
    def read_ball_trajectory(self, filepath):
        with open(filepath , 'r') as f:
            ball_trajectory_lines = f.readlines()
        ball_trajectory = []
        for line in ball_trajectory_lines:
            x, y = line.rstrip().split()
            ball_trajectory.append([int(x), int(y)])
        return np.array(ball_trajectory)
            
    
    def joints_sanity_fix(self, joint_raw, frames):
        # note that it is possible the width_coords>1280 and height_coords>720 due to imperfect pose esitimation
        # here we fix these cases
        
        for t in joint_raw:
            for n in range(len(joint_raw[t])):
                for j in range(len(joint_raw[t][n])):
                    # joint_raw[t][n, j, 0] = int(joint_raw[t][n, j, 0])
                    # joint_raw[t][n, j, 1] = int(joint_raw[t][n, j, 1])
                    
                    if joint_raw[t][n, j, 0] >= self.args.image_w:
                        joint_raw[t][n, j, 0] = self.args.image_w - 1
                        
                    if joint_raw[t][n, j, 1] >= self.args.image_h:
                        joint_raw[t][n, j, 1] = self.args.image_h - 1
                    
                    if joint_raw[t][n, j, 0] < 0:
                        joint_raw[t][n, j, 0] = 0
                        
                    if joint_raw[t][n, j, 1] < 0:
                        joint_raw[t][n, j, 1] = 0 
                        
        # modify joint_raw - loop over each frame and pad the person dim because it can have less than N persons
        for f in joint_raw:
            n_persons = joint_raw[f].shape[0]
            if n_persons < self.args.N:  # padding in case some clips has less than N persons 
                joint_raw[f] = np.concatenate((
                    joint_raw[f], 
                    np.zeros((self.args.N-n_persons, self.args.J, joint_raw[f].shape[2]))), 
                    axis=0)
        return joint_raw
    
    
    def ball_trajectory_sanity_fix(self, ball_trajectory):
        # ball_trajectory: (T, 2)
        for t in range(len(ball_trajectory)):
            if ball_trajectory[t, 0] >= self.args.image_w:
                ball_trajectory[t, 0] = self.args.image_w - 1
                
            if ball_trajectory[t, 1] >= self.args.image_h:
                ball_trajectory[t, 1] = self.args.image_h - 1

            if ball_trajectory[t, 0] < 0:
                ball_trajectory[t, 0] = 0

            if ball_trajectory[t, 1] < 0:
                ball_trajectory[t, 1] = 0 
        return ball_trajectory
            
    
    
    def horizontal_flip_augment_joint(self, joint_raw, frames, add_purturbation=False, randomness_set=False, index=0):
        for t in frames:
            for n in range(len(joint_raw[t])):
                if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                    continue
                for j in range(len(joint_raw[t][n])):
                    joint_raw[t][n, j, 0] = self.args.image_w - joint_raw[t][n, j, 0]  # flip joint coordinates
                    if add_purturbation:
                        if not randomness_set:
                            self.horizontal_flip_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                                -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                        joint_raw[t][n, j, 0] += self.horizontal_flip_augment_joint_randomness[index][(t, n, j)]
                    joint_raw[t][n, j, 2] = COCO_KEYPOINT_HORIZONTAL_FLIPPED[joint_raw[t][n, j, 2]]  # joint class type has to be flipped
                joint_raw[t][n] = joint_raw[t][n][joint_raw[t][n][:, 2].argsort()]  # sort by joint type class id
        return joint_raw
    
    
    def horizontal_move_augment_joint(self, joint_raw, frames, add_purturbation=False, randomness_set=True, index=0, max_horizontal_diff=10.0, ball_trajectory=None):
        horizontal_change = np.random.uniform(low=-max_horizontal_diff, high=max_horizontal_diff)
        for t in frames:
            for n in range(len(joint_raw[t])):
                if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                    continue
                for j in range(len(joint_raw[t][n])):
                    joint_raw[t][n, j, 0] += horizontal_change  # horizontally move joint 
                    if add_purturbation:
                        if not randomness_set:
                            self.horizontal_move_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                                -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                        joint_raw[t][n, j, 0] += self.horizontal_move_augment_joint_randomness[index][(t, n, j)]
        if ball_trajectory is not None:
            for t in range(len(ball_trajectory)):
                 ball_trajectory[t, 0] += horizontal_change
            return joint_raw, ball_trajectory
        else:
            return joint_raw
        
    
    def vertical_move_augment_joint(self, joint_raw, frames, add_purturbation=False, randomness_set=True, index=0, max_vertical_diff=10.0, ball_trajectory=None):
        vertical_change = np.random.uniform(low=-max_vertical_diff, high=max_vertical_diff)
        for t in frames:
            for n in range(len(joint_raw[t])):
                if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                    continue
                for j in range(len(joint_raw[t][n])):
                    joint_raw[t][n, j, 1] += vertical_change  # vertically move joint 
                    if add_purturbation:
                        if not randomness_set:
                            self.vertical_move_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                                -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                        joint_raw[t][n, j, 1] += self.vertical_move_augment_joint_randomness[index][(t, n, j)]
        if ball_trajectory is not None:
            for t in range(len(ball_trajectory)):
                 ball_trajectory[t, 1] += vertical_change
            return joint_raw, ball_trajectory
        else:
            return joint_raw
    
    
    def agent_dropout_augment_joint(self, joint_feats, frames, index=0, J=17):
        # joint_feats: (N, J, T, d)
        chosen_frame = self.agent_dropout_augment_randomness[index][0] 
        chosen_person = self.agent_dropout_augment_randomness[index][1] 
        feature_dim = joint_feats.shape[3]

        joint_feats[chosen_person, :, frames.index(chosen_frame), :] = torch.zeros(J, feature_dim)
        return joint_feats
    
    
    def horizontal_flip_ball_trajectory(self, ball_trajectory):
        # ball_trajectory: (T, 2)
        for t in range(len(ball_trajectory)):
             ball_trajectory[t, 0] = self.args.image_w - ball_trajectory[t, 0]
        return ball_trajectory
            

    def __getitem__(self, index):
        current_joint_feats_path = self.clip_joints_paths[index] 
        (video, clip) = self.clips[index]
        label = self.annotations[index]
        person_labels = self.annotations_each_person[index]
        
        joint_raw = pickle.load(open(current_joint_feats_path, "rb"))
        # joint_raw: T: (N, J, 3)
        # 3: [joint_x, joint_y, joint_type]
        
        frames = sorted(joint_raw.keys())[self.args.frame_start_idx:self.args.frame_end_idx+1:self.args.frame_sampling]
        
        if self.args.ball_trajectory_use:
            ball_trajectory_data = self.read_ball_trajectory(self.clip_ball_paths[index])
            ball_trajectory_data = ball_trajectory_data[self.args.frame_start_idx+10:self.args.frame_end_idx+1+10:self.args.frame_sampling]
            # ball trajectory annotation has 41 frames annotated but joint/track file only has 20 frames.
            assert len(ball_trajectory_data) == len(frames)
            # (T, 2)
                        
        person_labels = torch.LongTensor(person_labels[frames[0]].squeeze())[:self.args.N]  # person action remains to be the same across all frames 
        # person_labels: (N, )
        
        # if horizontal flip augmentation and is training
        if self.args.horizontal_flip_augment and self.split == 'train':
            if index < len(self.horizontal_flip_mask):
                if self.horizontal_flip_mask[index]:
                    if self.args.horizontal_flip_augment_purturb:
                        joint_raw = self.horizontal_flip_augment_joint(
                            joint_raw, frames, add_purturbation=True, randomness_set=True, index=index)
                    else:
                        joint_raw = self.horizontal_flip_augment_joint(joint_raw, frames)
                    label = self.classidx_horizontal_flip_augment[label]  # label has to be flipped!
                    
                    if self.args.ball_trajectory_use:
                        ball_trajectory_data = self.horizontal_flip_ball_trajectory(ball_trajectory_data)
                        
        # if horizontal move augmentation and is training
        if self.args.horizontal_move_augment and self.split == 'train':
            if index < len(self.horizontal_mask):
                if self.horizontal_mask[index]:
                    if self.args.ball_trajectory_use:
                        if self.args.horizontal_move_augment_purturb:
                            joint_raw, ball_trajectory_data = self.horizontal_move_augment_joint(
                                joint_raw, frames, add_purturbation=True, randomness_set=True, 
                                index=index, ball_trajectory=ball_trajectory_data)
                        else:
                            joint_raw, ball_trajectory_data = self.horizontal_move_augment_joint(
                                joint_raw, frames, ball_trajectory=ball_trajectory_data) 
                    else:
                        if self.args.horizontal_move_augment_purturb:
                            joint_raw = self.horizontal_move_augment_joint(
                                joint_raw, frames, add_purturbation=True, randomness_set=True, index=index)
                        else:
                            joint_raw = self.horizontal_move_augment_joint(joint_raw, frames)  
                        
        # if vertical move augmentation and is training
        if self.args.vertical_move_augment and self.split == 'train':
            if index < len(self.vertical_mask):
                if self.vertical_mask[index]:
                    if self.args.ball_trajectory_use:
                        if self.args.vertical_move_augment_purturb:
                            joint_raw, ball_trajectory_data = self.vertical_move_augment_joint(
                                joint_raw, frames, add_purturbation=True, 
                                randomness_set=True, index=index, 
                                ball_trajectory=ball_trajectory_data)
                        else:
                            joint_raw, ball_trajectory_data = self.vertical_move_augment_joint(
                                joint_raw, frames, ball_trajectory=ball_trajectory_data) 
                    else:
                        if self.args.vertical_move_augment_purturb:
                            joint_raw = self.vertical_move_augment_joint(
                                joint_raw, frames, add_purturbation=True, 
                                randomness_set=True, index=index)
                        else:
                            joint_raw = self.vertical_move_augment_joint(joint_raw, frames)                  
                    
        joint_raw = self.joints_sanity_fix(joint_raw, frames)
        if self.args.ball_trajectory_use:
            ball_trajectory_data = self.ball_trajectory_sanity_fix(ball_trajectory_data)
        
        
        # get joint_coords_all for image coordinates embdding
        if self.args.image_position_embedding_type != 'None':
            joint_coords_all = []
            for n in range(self.args.N):
                joint_coords_n = []

                for j in range(self.args.J):
                    joint_coords_j = []

                    for tidx, frame in enumerate(frames):
                        joint_x, joint_y, joint_type = joint_raw[frame][n,j,:]
                        
                        joint_x = min(joint_x, self.args.image_w-1)
                        joint_y = min(joint_y, self.args.image_h-1)
                        joint_x = max(0, joint_x)
                        joint_y = max(0, joint_y)

                        joint_coords = []
                        joint_coords.append(joint_x)  # width axis 
                        joint_coords.append(joint_y)  # height axis
                            
                        joint_coords_j.append(joint_coords)
                    joint_coords_n.append(joint_coords_j)   
                joint_coords_all.append(joint_coords_n)
        
        joint_feats = torch.Tensor(np.array(joint_coords_all))
        
        # if random agent dropout augmentation and is training                
        if self.args.agent_dropout_augment and self.split == 'train':
            if index < len(self.agent_dropout_mask):
                if self.agent_dropout_mask[index]:
                    joint_feats = self.agent_dropout_augment_joint(
                            joint_feats, frames, index=index, J=self.args.J)


        joint_feats = joint_feats.permute(3, 2, 1, 0).contiguous()
        # joint_feats ([C, T, V, M])

        people_idx_sort = torch.argsort(joint_feats[0, :, :, :].view(-1, self.args.N).mean(0, keepdim=False), dim=-1)
        left_group_people_idx = people_idx_sort[:int(self.args.N//2)]  # (N/2)
        right_group_people_idx = people_idx_sort[int(self.args.N//2):]  # (N/2)

        left_group_people = joint_feats[:, :, :, left_group_people_idx]
        right_group_people = joint_feats[:, :, :, right_group_people_idx]

        left_group_people_labels = person_labels[left_group_people_idx]
        right_group_people_labels = person_labels[right_group_people_idx]

        if self.args.ball_trajectory_use:
            # (T, 2)
            ball_feats = torch.Tensor(ball_trajectory_data)
            ball_feats = ball_feats.unsqueeze(-1).expand(-1, -1, self.args.J).unsqueeze(-1)
            ball_feats = ball_feats.permute(1, 0, 2, 3).contiguous() # C, T, V, M
            left_group_people = torch.cat((left_group_people, ball_feats), dim=-1)
            right_group_people = torch.cat((right_group_people, ball_feats), dim=-1)
            left_group_people_labels = torch.cat((left_group_people_labels, torch.tensor([10], dtype=left_group_people_labels.dtype)), dim=-1) # person_label==10 means volleyball
            right_group_people_labels = torch.cat((right_group_people_labels, torch.tensor([10], dtype=right_group_people_labels.dtype)), dim=-1) # person_label==10 means volleyball
        
        joint_feats = torch.cat((left_group_people, right_group_people), dim=-1)
        person_labels = torch.cat((left_group_people_labels, right_group_people_labels), dim=-1)
        
        assert not torch.isnan(joint_feats).any() 

        # joint_feats ([C, T, V, M]) = (2, 10, 17, M) # if ball_trajectory_use == True then M=M+2
        # person_labels ([M, 1])
        # label (int)

        return joint_feats, label, person_labels, index
