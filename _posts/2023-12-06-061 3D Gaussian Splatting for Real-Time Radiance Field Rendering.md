---
layout: mypost
title: 060 NeUDF Learning Neural Unsigned Distance Fields with Volume Rendering
categories: [论文阅读, 表面重建, 读完论文]
---


# 文章信息

## 标题

3D Gaussian Splatting for Real-Time Radiance Field Rendering

用于实时辐射场渲染的三维高斯投影方法

## 作者



## 发表信息

文章发表于2023年的SIGGRAPH，并被评为2023年度的Best Paper.


## 引用信息


## 论文链接

[cvpr 2023 link](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_NeUDF_Leaning_Neural_Unsigned_Distance_Fields_With_Volume_Rendering_CVPR_2023_paper.html)

[Home Page](http://geometrylearning.com/neudf/)

[Github Link](https://github.com/IGLICT/NeUDF)

## 后人对此文章的评价


# 文章内容

## 摘要

> 

## 介绍

​18-231206. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. 论文发表于2023年的SIGGRAPH上，并被评为2023年SIGGRAPH的best paper. 文章将无界全范围场景下的新视角合成的速度和质量提上了一个很高的台阶，在训练三万轮（大约半个小时的训练时间）的时候，成像质量堪比或优于MipNeRF360；在训练7千轮（大约6-8分钟）的时候，成像质量也能够优于Instant-NGP。由于采用光栅化高斯球形投影方法，渲染速度能够达到一秒渲染93-135帧，此前最快的Instant-NGP的渲染速度是1秒渲染9帧，Plenoxels每秒8帧，MipNeRF大约十秒渲染一帧，实时渲染的性能相当可观。文章的方法主要由三点构成，第一使用各向异性的三维高斯作为场景的表示方法，在三维空间中朝向不同的高斯球，按照颜色、透明度，向成像平面加权投影，得到成像结果；第二是尺度适应性裁剪，高斯球的大小由协方差矩阵中的方差决定，方差越大表示高斯球的尺寸越大，可以用于表征空间中的大面积同质区域，文章适当地将大尺寸的高斯球分解为小尺寸的高斯球，逐步增强细节表征能力，最终的训练模型能达到场景中300-500k个小高斯球；第三是光栅化渲染算法，光栅化混合算法相比于NeRF的查询MLP更节省计算时间，最大程度提高渲染速度。

## 本文的组织结构


- 1 Introduction
- 2 Related Work
  - 2.1 Traditional Scene Reconstruction and Rendering
  - 2.2 Neural Rendering and Radiance Fields
  - 2.3 Point-Based Rendering and Radiance Fields
- 3 Overview
- 4 Differentiable 3D Gaussian Splatting
- 5 Optimization with adaptive density control of 3d Gaussians
  - 5.1 Optimization
  - 5.2 Adaptive Control of Gaussians
- 6 Fast Diifferentiable Rasterizer for Gaussians
- 7 Implemenation, Results and Evaluation
  - 7.1 Implementaion
  - 7.2 Results and Evalution
  - 7.3 Ablations
    - Initialization from SfM. 
    - Densification.
    - Unlimited depth complexity of splats with gradients. 
    - Anisotropic Covariance. 
    - Spherical Harmonics.
  - 7.4 Limitations
- 8 Discussion and Conclusions
- Acknowledgements
- References

- A Details of gradient computation
- B Optimization and Densification algorithm
- C Details of the Rasterizer
- D Per-Scene Error Metrics

文章的正文部分包括了11页，相比于CVPR论文普遍的8页长度，内容要更丰富一些。文章的编排过程中，给相关工作的篇幅比较大，看得出来版面比较富裕，工作也比较充实。


# Key Points

# Abstract 

