import os
import pickle
from glob import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

from feeders import tools

def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

class Feeder(Dataset):
    """
    Data loader for the Harper (3D) dataset.
    This data loader is designed to provide data for forecasting, but can easily adapted as per your needs.
    """

    def __init__(self, 
                data_path: str, 
                split: str,
                p_interval=1,
                window_size=-1,
                random_rot=False, 
                entity_rearrangement=False,
                debug=False,
            ) -> None:
        # Sanity checks
        assert os.path.exists(data_path), f"Path {data_path} does not exist. Please download the dataset first"
        assert split in ["train", "test"], f"Split {split} not recognized. Use either 'train' or 'test'"
        data_folder = os.path.join(data_path, split)
        assert os.path.exists(data_folder), f"Path {data_folder} does not exist. It is in the correct format? Refer to the README"

        self.data_path = data_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_rot = random_rot
        self.entity_rearrangement = entity_rearrangement
        self.debug = debug

        # Load data
        pkls_files: list[str] = glob(os.path.join(data_folder, "*.pkl"))
        self.all_sequences: list[dict[int, dict]] = [load_pkl(f) for f in pkls_files]
        self.sample_name: list[str] = [os.path.basename(name).replace(".pkl", "") for name in pkls_files]

        self.action2label = {
            "act1": 0, "act2": 1, "act3": 2, "act4": 3, "act5": 4,
            "act6": 5, "act7": 6, "act8": 7, "act9": 8, "act10": 9,
            "act11": 10, "act12": 11, "act13": 12, "act14": 13, "act15": 14,
        }

    def __len__(self):
        return len(self.all_sequences)

    def __pad_tensor(self, person_kpts, robot_kpts):
        # pad the person keypoints with 0
        person_kpts = F.pad(person_kpts, (0,0,0,robot_kpts.size(1)-person_kpts.size(1)), "constant", 0)

        poses = torch.stack([person_kpts, robot_kpts], dim=-1)

        return poses.permute(2,0,1,3) # T,V,C,M -> C,T,V,M

    def __getitem__(self, idx):
        curr_data = self.all_sequences[idx]
        info = self.sample_name[idx].split('_')
        subject = info[0]
        action = info[1]
        freq = info[2]
        
        human = [curr_data[i]["human_joints_3d"] for i in range(len(curr_data))]
        human = torch.tensor(human, dtype=torch.float32)
        spot = [curr_data[i]["spot_joints_3d"] for i in range(len(curr_data))]
        spot = torch.tensor(spot, dtype=torch.float32)

        poses = self.__pad_tensor(human, spot)

        poses = np.array(poses)
        valid_frame_num = np.sum(poses.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        poses = tools.valid_crop_resize(poses, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            poses = tools.random_rot(poses)
        else:
            poses = torch.from_numpy(poses)
        if self.entity_rearrangement:
            poses = poses[:,:,:,torch.randperm(poses.size(3))]

        return poses, self.action2label[action], idx