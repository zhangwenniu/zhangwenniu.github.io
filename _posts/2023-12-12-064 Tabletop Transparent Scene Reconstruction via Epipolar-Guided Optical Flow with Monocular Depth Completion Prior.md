---
layout: mypost
title: 064 Tabletop Transparent Scene Reconstruction via Epipolar-Guided Optical Flow with Monocular Depth Completion Prior
categories: [论文阅读, 表面重建, 读完论文]
---


# 文章信息

## 标题

Dense Reconstruction of Transparent Objects by Altering Incident Light Paths Through Refraction

## 作者

Xiaotong Chen1, Zheming Zhou2, Zhuo Deng2, Omid Ghasemalizadeh2, Min Sun2, Cheng-Hao Kuo2, Arnie Sen2

1 X. Chen is with the Department of Robotics, University of Michigan, Ann Arbor, MI, USA. cxt@umich.edu 2 Z. Zhou, Z. Deng, O. Ghasemalizadeh, M. Sun, and C.H. Kuo are with Amazon Lab126, Sunnyvale, CA, USA. {zhemiz, zhuod, ghasemal, minnsun, chkuo, senarnie}@amazon.com

## 发表信息



## 引用信息

```
@INPROCEEDINGS{10233838,
  author={Wang, Ziyu and Yang, Wei and Cao, Junming and Hu, Qiang and Xu, Lan and Yu, Junqing and Yu, Jingyi},
  booktitle={2023 IEEE International Conference on Computational Photography (ICCP)}, 
  title={NeReF: Neural Refractive Field for Fluid Surface Reconstruction and Rendering}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/ICCP56744.2023.10233838}
}
```

## 论文链接

[ieee link](https://ieeexplore.ieee.org/abstract/document/10233838)

[iccp link](https://ieeexplore.ieee.org/xpl/conhome/10233258/proceeding)

[iccp 2023 link](https://www.computer.org/csdl/proceedings-article/iccp/2023/10233838/1Qao8fuqNJC)

[arxiv link](https://arxiv.org/abs/2203.04130)

[semantic scholar link](https://www.semanticscholar.org/paper/NeReF%3A-Neural-Refractive-Field-for-Fluid-Surface-Wang-Yang/fd2c0e45a95933cacc2a69c79ec1f74553d0cb39)

## 后人对此文章的评价


# 文章内容

## 摘要

> 

## 介绍

21-231209. Tabletop Transparent Scene Reconstruction via Epipolar-Guided Optical Flow with Monocular Depth Completion Prior. 本文于2023年10月15号挂在Arxiv上的，据arxiv上的备注说，本文是投IEEE-RAS Humanoids 2023 paper的。文章的方法缩写为D-EOF，缩写来自Monocular Depth Prior-based Epipolar-Guided Optical Flow，基于ClearPose多透明物体数据集的透明物体点云重建任务，作者将工作分为两阶段。第一阶段使用单视角的深度补全网络与透明物体的分割网络，预测单一视角下的透明物深度。第二阶段利用对极线约束相邻视角下透明物体的边界位置一致性，相邻视角的位置变化是通过光流法确定的。文章汇报了在ClearPose的透明物体数据集下，对透明物体点云的重建效果。由于没有合适的比较方法，文章对比单个透明物体的重建方法Through Looking Glass、通用场景下的TSDF表面重建方法。文章汇报了训练分割网络RCNN的轮数是5轮，三维的轮廓标定点在捆集调整训练30轮，文中并没有说明具体时间，但是考虑到整个数据集较大，训练一轮的时间可能较长。

## 本文的组织结构


- Abstract
- I. Introduction
- II. Related Work
  - A. Transparent Surface Reconstruction
  - B. Transparent Scene Depth Estimation
- III. Transparent Scene Reconstruction Pipeline
  - A. Single-view Depth Completion and Segmentation
  - B. Multi-view Boundary-Inspired and Epipolar-Guided Optical Flow
    - 1) Boundary-Inspired 2D Landmarks Generation
    - 2) Epipolar-Guided Optical Flow Correspondence
    - 3) Bundle Adjustment Formulation
- IV. Experiments
  - A. Dataset and Network Training
  - B. Evaluation Metrics of 3D Reconstruction
  - C. Implementation Details and Baselines
  - D. Results and Disscussions
    - 1) 2D Correspondece Estimation Evaluation
    - 2) EOF Evaluation
    - 3) End-to-end Evaluation
- V. Conclusion


# Key Points

# Abstract 

