# CHASE: Learning Convex Hull Adaptive Shift for Skeleton-based Multi-Entity Action Recognition
<a href='https://arxiv.org/abs/2410.07153'>
  <img src='https://img.shields.io/badge/Paper-arXiv-green?style=flat&logo=arxiv' alt='arXiv PDF'>
</a>
<a href='https://necolizer.github.io/CHASE/'>
  <img src='https://img.shields.io/badge/Project-Page-orange?style=flat&logo=googlechrome&logoColor=orange' alt='project page'>
</a>
<a href='https://github.com/Necolizer/CHASE/blob/main/LICENSE'>
  <img src='https://img.shields.io/badge/License-MIT-yellow?style=flat' alt='license'>
</a>

Here's the official implementation of *CHASE: Learning Convex Hull Adaptive Shift for Skeleton-based Multi-Entity Action Recognition* accepted in [NeurIPS 2024](https://nips.cc/Conferences/2024).

## 0. Table of Contents

* [1. Change Log](#1-change-log)
* [2. Prerequisites](#2-prerequisites)
* [3. Datasets](#3-datasets)
* [4. Run the Code](#4-run-the-code)
* [5. Checkpoints](#5-checkpoints)
* [6. Acknowledgement](#6-acknowledgement)
* [7. Citation](#7-citation)

## 1. Change Log
- [2024/10/11] Our paper is available in [arXiv](https://arxiv.org/abs/2410.07153). Visit our [project website](https://necolizer.github.io/CHASE/)!
- [2024/09/27] This work is accepted by [NeurIPS 2024](https://nips.cc/Conferences/2024). We make our scripts and checkpoints public.

## 2. Prerequisites
To clone the `main` branch only (for code) and exclude the `gh-pages` branch (for project website), use the following `git` command:
```shell
git clone -b main https://github.com/Necolizer/CHASE.git
```

```shell
pip install -r requirements.txt 
```

## 3. Datasets
### 3.1 NTU Mutual 11 & 26, H2O, Assembly101
Please refer to [ISTA-Net](https://github.com/Necolizer/ISTA-Net) and follow the instructions in section [Prepare the Datasets](https://github.com/Necolizer/ISTA-Net?tab=readme-ov-file#3-prepare-the-datasets) to prepare these datasets.

### 3.2 Collective Activity, Volleyball
Please refer to [COMPOSER](https://github.com/hongluzhou/composer) repo's section [Dataset Preparation](https://github.com/hongluzhou/composer?tab=readme-ov-file#dataset-preparation). You could directly download the data using their provided google drive links.

## 4. Run the Code
### 4.1 NTU Mutual 11, NTU Mutual 26
```shell
python main.py --config config/[yourBackboneName]/[ntu11ORntu26]/[yourSetting]_chase.yaml
```

### 4.2 H2O
**Train & Validate**
```shell
python main.py --config config/[yourBackboneName]/h2o/h2o_chase.yaml
```

**Generate JSON File for Test Result Submission**
```shell
python main.py --config config/[yourBackboneName]/h2o/h2o_get_test_results_chase.yaml --weights path/to/your/checkpoint
```

Submit zipped json file `action_labels.json` in CodaLab Challenge [H2O - Action](https://codalab.lisn.upsaclay.fr/competitions/4820) to get the test accuracy scores.

### 4.3 Assembly101
**Train & Validate**
```shell
# Action (mandatory): 1380 classes
python main.py --config config/[yourBackboneName]/asb/asb_action_chase.yaml
```

**Generate JSON File for Test Result Submission**
```shell
# Action (mandatory): 1380 classes
python main.py --config config/[yourBackboneName]/asb/asb_action_get_test_results_chase.yaml --weights path/to/your/action/checkpoint
```

Submit zipped json file `preds.json` in CodaLab Challenge [Assembly101 3D Action Recognition](https://codalab.lisn.upsaclay.fr/competitions/5256) to get the test accuracy scores.

> ATTENTION: `preds.json` for 'Action' is about 673Mb before compression.

### 4.4 Collective Activity, Volleyball
```shell
python main_group.py --config config/[yourBackboneName]/[cadORvol]/[yourSetting]_chase.yaml
```

## 5. Checkpoints
Checkpoints of the best backbone for each benchmark are provided in this [Hugging Face repo](https://huggingface.co/Necolizer/CHASE).

## 6. Acknowledgement
Grateful to the authors of [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), [InfoGCN](https://github.com/stnoah1/infogcn), [STTFormer](https://github.com/heleiqiu/STTFormer), [HD-GCN](https://github.com/Jho-Yonsei/HD-GCN), [ISTA-Net](https://github.com/Necolizer/ISTA-Net), [COMPOSER](https://github.com/hongluzhou/composer) repository. Thanks to the authors for their great work.


## 7. Citation

If you find this work or code helpful in your research, please consider citing:
```
@inproceedings{wen2024chase,
    title={CHASE: Learning Convex Hull Adaptive Shift for Skeleton-based Multi-Entity Action Recognition},
    author={Yuhang Wen and Mengyuan Liu and Songtao Wu and Beichen Ding},
    booktitle={Thirty-eighth Conference on Neural Information Processing Systems (NeurIPS)},
    year={2024},
}

@INPROCEEDINGS{wen2023interactive,
    author={Wen, Yuhang and Tang, Zixuan and Pang, Yunsheng and Ding, Beichen and Liu, Mengyuan},
    booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
    title={Interactive Spatiotemporal Token Attention Network for Skeleton-Based General Interactive Action Recognition}, 
    year={2023},
    pages={7886-7892},
    doi={10.1109/IROS55552.2023.10342472}
}
```