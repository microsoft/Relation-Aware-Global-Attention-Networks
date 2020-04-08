## Description
This repository holds the codes and methods for the following CVPR-2020 paper:
- [Relation-aware Global Attention for Person Re-identification](https://arxiv.org/pdf/1904.02998.pdf)

We hope it will inspire more excellent works and encourage you to adopt RGA into other tasks. If you find our paper and repository useful, please consider citing our paper:

```
@article{zhang2020relation,
  title={Relation-Aware Global Attention for Person Re-identification},
  author={Zhang, Zhizheng and Lan, Cuiling and Zeng, Wenjun and Jin, Xin and Chen, Zhibo},
  journal={CVPR},
  year={2020}
}
```

## Introduction
Person re-identification (re-id) aims to match a specific person across different times, places, or cameras, which has drawn a surge of interests from both industry and academia. The challenge lies in how to extract discriminative features (for identifying the same person and distinguishing different persons) from person images where there are background clutter, diversity of poses, occlusion, etc. For this task, attention align well with its target and has been demonstrated effective. However, previous attention works do not fully exploit the correlations from a global view and ignore the global scope structural information.

Inspired by the obeservation for human attention: making a global scope comparison to determine the importance. We propose to globally learn the attention for each feature node by taking a global view of the relations among the features. With the global scope relations having valuable structural (clustering-like) information, we propose to mine semantics from relations for deriving attention through a learned function. Specifically, for a feature node, we build a compact representation by stacking its pairwise relations with respect to all feature nodes as a vector and mine patterns from it for attention learning.

![image](https://github.com/microsoft/Relation-Aware-Global-Attention-Networks/blob/master/diagrams/spatial_channel_RGA.png)

Specifically, we design a relation-aware global attention (RGA) module which compactly represents the global scope relations and derives the attention based on them via two convolutional layers for spatial and channel dimensions, respectively.

## Installation

1. Git clone this repo.
2. Intall dependencies by `pip install -r requirements.txt` (if you hope to use the same environment configuaration as we used for the results in our paper.)
Sepcifically, we train the all models to get the reported results in our paper on a single NVIDIA Tesla P40 card (with 24GB GPU memory).

## ReID Dataset Preparation
Image-reid datasets (here we use CUHK03 dataset as an example for description):
- [CUHK03](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf) (`cuhk03`)
The keys to use these datasets are enclosed in the parentheses. See [torchreid/datasets/\_\_init__.py](torchreid/datasets/__init__.py) for details. The data managers of image reid are implemented in [torchreid/data_manager.py](torchreid/data_manager.py).
1. Create a folder named `cuhk03/` under `/YOUR_DATASET_PATH/`.
2. Download dataset to `data/cuhk03/` from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and extract `cuhk03_release.zip`, so you will have `data/cuhk03/cuhk03_release`.
3. Download new split from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03). What you need are `cuhk03_new_protocol_config_detected.mat` and `cuhk03_new_protocol_config_labeled.mat`. Put these two mat files under `data/cuhk03`. Finally, the data structure would look like
```
cuhk03/
    cuhk03_release/
    cuhk03_new_protocol_config_detected.mat
    cuhk03_new_protocol_config_labeled.mat
    ...
```
4. In default mode, we use new split protocol (767/700).
5. *Please remember to modify the variable `DATD_DIR` in our provided bash script for specifying the path of your dataset accordingly.)

## Pretrain Model Preparation

1. Create a folder named `weights/pretrain` under the root path of this repo.
2. Download the pre-trained ResNet-50 model to `weights/pretrain` from https://download.pytorch.org/models/resnet50-19c8e357.pth.

## Training and Evaluation

For your convenience, we provide the bash script with our recommended hyper-parameters settings. Please `cd` to the root path of this repo and run:

`bash ./scripts/run_rgasc_cuhk03.sh`

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
