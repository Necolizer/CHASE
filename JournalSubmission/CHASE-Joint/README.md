# CHASE-Joint

Under review. Please stay tuned for updates.

## Prerequisites
To clone the `main` branch only (for code) and exclude the `gh-pages` branch (for project page), use the following `git` command:
```shell
git clone -b main https://github.com/Necolizer/CHASE.git
pip install -r requirements.txt
cd ./JournalSubmission/CHASE-Joint
```

## Datasets

### NTU Mutual 11 & 26, H2O, and Assembly101
Follow the instructions in section [Prepare the Datasets](https://github.com/Necolizer/ISTA-Net?tab=readme-ov-file#3-prepare-the-datasets) to prepare NTU Mutual 11 & 26, H2O, and Assembly101.

### Collective Activity and Volleyball
See section [Dataset Preparation](https://github.com/hongluzhou/composer?tab=readme-ov-file#dataset-preparation) to get Collective Activity and Volleyball. You could directly download the data using their provided google drive links.

### HARPER
Download the 30Hz 3D skeleton data and the external RGB videos from [this link](https://univr-my.sharepoint.com/:f:/g/personal/federico_cunico_univr_it/Esk9qR4fKyFBg05UdXK0YSYBY8JvLHpY2Bis2xyX1pcVWg). You could download the 3D skeleton data using the script provided in [HARPER](https://github.com/intelligolabs/HARPER):
```python
PYTHONPATH=. python download/harper_only_3d_downloader.py --dst_folder ./data
```

This will generate the following tree structure:
```
data
├── harper_3d_120
│   ├── test
│   │   ├── subj_act_120hz.pkl
│   │   ├── ...
│   │   └── subj_act_120hz.pkl
│   └── train
│       ├── subj_act_120hz.pkl
│       ├── ...
│       └── subj_act_120hz.pkl
└── harper_3d_30
    ├── test
    │   ├── subj_act_30hz.pkl
    │   ├── ...
    │   └── subj_act_30hz.pkl
    └── train
        ├── subj_act_30hz.pkl
        ├── ...
        └── subj_act_30hz.pkl
```


## Run the Code
To run the code:
```shell
python main.py --config config/[yourBackboneName]/[dataset]/[yourSetting]_chase.yaml
python main_group.py --config config/[yourBackboneName]/[cadORvol]/[yourSetting]_chase.yaml
```

For DeGCN backbone, run the following command instead:
```shell
python main_degcn.py --config config/[yourBackboneName]/[dataset]/[yourSetting]_chase.yaml
python main_group_degcn.py --config config/[yourBackboneName]/[cadORvol]/[yourSetting]_chase.yaml
```

## Citation
If you find this work or code helpful in your research, please consider citing:
```
Please stay tuned for updates.
```

## Acknowledgement
This project is built on top of the follows, please consider citing them if you find them useful: [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), [InfoGCN](https://github.com/stnoah1/infogcn), [STTFormer](https://github.com/heleiqiu/STTFormer), [HD-GCN](https://github.com/Jho-Yonsei/HD-GCN), [DeGCN](https://github.com/WoominM/DeGCN_pytorch), and [Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN).