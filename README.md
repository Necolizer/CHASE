# [IJCV'26] De-biasing Skeleton-based Action Recognition with Convex Hull Adaptive Shift 

Here's the official implementation of 
1. [De-biasing Skeleton-based Action Recognition with Convex Hull Adaptive Shift](https://link.springer.com/article/10.1007/s11263-026-03037-1) accepted by International Journal of Computer Vision (IJCV).
2. [CHASE: Learning Convex Hull Adaptive Shift for Skeleton-based Multi-Entity Action Recognition](https://arxiv.org/abs/2410.07153) accepted by [NeurIPS 2024](https://nips.cc/virtual/2024/poster/94816).

![](https://github.com/Necolizer/CHASE/blob/gh-pages/static/images/EntityBias.svg)

![](https://github.com/Necolizer/CHASE/blob/gh-pages/static/images/Viz.svg)


## 1. [IJCV'26] De-biasing Skeleton-based Action Recognition with Convex Hull Adaptive Shift

Detailed Implementation: See folder `JournalSubmission` [README 1](./JournalSubmission/CHASE-Joint/README.md) & [README 2](./JournalSubmission/CHASE-SubEntity/README.md)


## 2. [NeurIPS'24] CHASE: Learning Convex Hull Adaptive Shift for Skeleton-based Multi-Entity Action Recognition

Detailed Implementation: See folder `NeurIPS24` [README](./NeurIPS24/README.md)

To clone the `main` branch only (for code) and exclude the `gh-pages` branch (for project page), use the following `git` command:
```shell
git clone -b main https://github.com/Necolizer/CHASE.git
pip install -r requirements.txt
cd ./NeurIPS24
```

For datasets:
- Please refer to [ISTA-Net](https://github.com/Necolizer/ISTA-Net) and follow the instructions in section [Prepare the Datasets](https://github.com/Necolizer/ISTA-Net?tab=readme-ov-file#3-prepare-the-datasets) to prepare NTU Mutual 11 & 26, H2O, and Assembly101.
- Please refer to [COMPOSER](https://github.com/hongluzhou/composer) repo's section [Dataset Preparation](https://github.com/hongluzhou/composer?tab=readme-ov-file#dataset-preparation) to get Collective Activity and Volleyball. You could directly download the data using their provided google drive links.

To run the code:
```shell
python main.py --config config/[yourBackboneName]/[dataset]/[yourSetting]_chase.yaml
python main_group.py --config config/[yourBackboneName]/[cadORvol]/[yourSetting]_chase.yaml
```

Checkpoints of the best backbone for each benchmark are provided in this [Hugging Face repo](https://huggingface.co/Necolizer/CHASE).

## 3. Citation

If you find this work or code helpful in your research, please consider citing:
```
@article{liu2026debiasing,
  title = {De-biasing Skeleton-Based Action Recognition with Convex Hull Adaptive Shift},
  author = {Liu, Mengyuan and Wen, Yuhang and Zhang, Yi and Wu, Songtao and Liu, Hong and Yuan, Junsong and Ding, Beichen},
  journal = {International Journal of Computer Vision},
  volume = {134},
  number = {10},
  pages = {443},
  year = {2026},
  doi = {10.1007/s11263-026-03037-1},
}

@inproceedings{NEURIPS2024_wen2024chase,
    author = {Wen, Yuhang and Liu, Mengyuan and Wu, Songtao and Ding, Beichen},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
    pages = {9388--9420},
    publisher = {Curran Associates, Inc.},
    title = {CHASE: Learning Convex Hull Adaptive Shift for Skeleton-based Multi-Entity Action Recognition},
    url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/11f5520daf9132775e8604e89f53925a-Paper-Conference.pdf},
    volume = {37},
    year = {2024}
}
```
