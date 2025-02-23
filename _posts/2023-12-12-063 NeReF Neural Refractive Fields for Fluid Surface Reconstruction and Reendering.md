---
layout: mypost
title: 063 NeReF Neural Refractive Field for Fluid Surface Reconstruction and Rendering
categories: [透明, 表面重建]
---


# 文章信息

## 标题

Dense Reconstruction of Transparent Objects by Altering Incident Light Paths Through Refraction

## 作者

Kai Han1 · Kwan-Yee K. Wong1 · Miaomiao Liu2

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

​20-231208. NeReF: Neural Refractive Field for Fluid Surface Reconstruction and Implicit Representation. 本文发表于2023年的ICCP（International Conference on Computational Photography）。文章利用NeRF的体积表达，使用MLP估计空间中的体密度和法向量，重建液体表面。虽然是NeRF的表达，但并没有采用体积颜色渲染图像，也没有使用渲染图像与真实图像的颜色一致性表达信息。文章利用体渲染的公式得到相机视角下的加权深度与加权的法向量，因而能够利用菲涅尔定理计算像素对应射线在液体表面的折射方向，最终与提供纹理信息的底部参考平面相交，该相交的位置与纹理信息底部的真实位置信息之间的位置差距作为约束函数，与深度的梯度平滑项约束共同作为文章的损失函数。成像的像素在参考平面上的真实位置通过光流法确定。在数据集方面，作者在水缸上面放置10个相机，同时拍摄底部参考平面以及加入液体后的水面图像，其中9个相机的图像用于做训练，1个相机的图像用于做测试。本文没有合适的对比方法，分别与2011年的Dynamic fluid surface acquisition using a camera array方法、NeRF的做了比较，新视角下的成像质量较高。在实验效果方面，作者在基于重建的物体表面情况下，更改底部的纹理图像，利用折射光路的追踪与底部图像交点的对应关系下，预测新背景的图形，本质上没有使用体渲染，而是利用液体表面的几何信息与折射定律做的背景板上的光纤扭曲成像。作者在arxiv上放的早期版本有一些笔误，可直接看作者在iccp上发表的论文。

## 本文的组织结构


- Abstract
- 1 Introduction
- 2 Related Work
- 3 Shape Recovery of Transparent Objects
  - 3.1 Notations and Problem Formulation
  - 3.2 Setup and Assumption
  - 3.3 Dense Refraction Correspondeces
  - 3.4 Light Path Triangulation
  - 3.5 Surface Normal Reconstruction
- 4 Recovery of Thin Transparent Objects
  - 4.1 Setup and Assumptions
  - 4.2 Surface Reconstruction
- 5 Discussion
  - 5.1 Total Internal Reflection
  - 5.2 Object Analysis
  - 5.3 Single Refraction Approximation
- 6 Experimental Evaluation
  - 6.1 Synthetic Data
    - First method on a convex object
    - First method on a concave object
    - Second method on a thin convex cone
    - Second method on a spherical shell
  - 6.2 Real Data
- 7 Conslusions


# Key Points

# Abstract 

