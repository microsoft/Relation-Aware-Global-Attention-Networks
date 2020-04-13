# Relation-Aware Global Attention (RGA)

## Introduction

Attention mechanisms are widely used to learn discriminative features by strengthening important features and suppressing irrelevant ones, which have been demonstrated to be useful in many vision tasks. However, many previous works locally learn the attention using convolutions with small receptive fields, ignoring the mining of knowledge from global structure patterns. Intuitively, to accurately determine the level of importance of one feature node, it is better to know the information of all the feature nodes (for comparison). Motivated by this, we propose an effective Relation-Aware Global Attention (RGA) module which captures the global structural information for better attention learning. Specifically, for each feature position, in order to compactly grasp the structural information of global scope and the local appearance information, we stack the relations, i.e., its pairwise correlations/affinities with all the feature positions (e.g., in raster scan order), and the feature itself together, to learn the attention with a shallow convolutional model.  

We validate the effectiveness of RGA modules by applying them to the person re-identification (re-id) task. Note that our implementation on person re-id is targeted for the applications of finding lost child, and the customer density analysis in retail stores. 

![image](https://github.com/microsoft/Relation-Aware-Global-Attention-Networks/blob/master/diagrams/spatial_channel_RGA.png)
Figure 1: Diagram of our proposed Spatial Relation-aware Global Attention (RGA-S) and Channel Relation-aware Global Attention (RGA-C). When computing the attention at a feature position, in order to grasp information of global scope, we stack the pairwise relation items, i.e., its correlations/affinities with all the feature positions, and the unary item, i.e., the feature of this position, for learning the attention with convolutional operations. For each feature node, such compact global relation representation contains both the global scope affinities and the location information and is helpful for learning semantics and inferring attention.


## Installation

1. Git clone this repo.
2. Install dependencies by `pip install -r requirements.txt` to have the same environment configuration as the one we used. Note that we trained all models on a single NVIDIA Tesla P40 card.

## Re-ID Dataset Preparation
Here we use the [CUHK03](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf) dataset as an example for description.

1. Create a folder named `cuhk03/` under `/YOUR_DATASET_PATH/`. Download dataset to `/YOUR_DATASET_PATH/cuhk03/` from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and extract `cuhk03_release.zip`. Then you will have `/YOUR_DATASET_PATH/cuhk03/cuhk03_release`.
2. Download the train/test split protocol from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03). Put the two mat files `cuhk03_new_protocol_config_detected.mat` and `cuhk03_new_protocol_config_labeled.mat` under `data/cuhk03`. In the default mode, we use this new split protocol (767/700). Finally, the data structure will look like
```
cuhk03/
    cuhk03_release/
    cuhk03_new_protocol_config_detected.mat
    cuhk03_new_protocol_config_labeled.mat
    ...
```

## Pre-trained Model Preparation

1. Create a folder named `weights/pre_train` under the root path of this repo.
2. Download the pre-trained ResNet-50 model from https://download.pytorch.org/models/resnet50-19c8e357.pth and place it under `weights/pre_train`.

## Train

For your convenience, we provide the bash script with our recommended settings of hyper-parameters. Please `cd` to the root path of this repo and run:

`bash ./scripts/run_rgasc_cuhk03.sh`


## Reference

The work with this technique applied to the person re-identification task has been accepted by CVPR'20. 

- [Relation-aware Global Attention for Person Re-identification](https://arxiv.org/pdf/1904.02998.pdf)

We hope that this technique will benefit more computer vision related applications and inspire more works.
If you find this technique and repository useful, please cite the paper. Thanks!

```
@article{zhang2020relation,
  title={Relation-Aware Global Attention for Person Re-identification},
  author={Zhang, Zhizheng and Lan, Cuiling and Zeng, Wenjun and Jin, Xin and Chen, Zhibo},
  journal={CVPR},
  year={2020}
}
```


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
