---
layout: mypost
title: 056 Extracting Triangular 3D Models, Materials, and Lighting From Images
categories: [论文阅读, 重建, 读完论文]
---


# 文章信息

## 标题

Extracting Triangular 3D Models, Materials, and Lighting From Images

从图像中提取三维三角面片模型，材质，亮度

## 作者

Douglas E. Zongker 1 
Dawn M. Werner 1 
Brian Curless1 David H. Salesin 1,2

1 University of Washington
2 Microsoft Research



## 发表信息

Jacob Munkberg 1 Jon Hasselgren 1 Tianchang Shen 1,2,3 Jun Gao 1,2,3 Wenzheng Chen 1,2,3 Alex Evans 1 Thomas M ̈ uller 1 Sanja Fidler 1,2,3

1 NVIDIA 2 University of Toronto 3 Vector Institute

基本上是英伟达的全员参与。Vector Institute是多伦多大学创建的向量学院。

本文发表收录于2022年的CVPR。


## 引用信息

截至2023年12月2日，被引用次数为154次。

```
@InProceedings{Munkberg_2022_CVPR,
    author    = {Munkberg, Jacob and Hasselgren, Jon and Shen, Tianchang and Gao, Jun and Chen, Wenzheng and Evans, Alex and M\"uller, Thomas and Fidler, Sanja},
    title     = {Extracting Triangular 3D Models, Materials, and Lighting From Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8280-8290}
}
```

## 论文链接

[cvpr link](https://openaccess.thecvf.com/content/CVPR2022/html/Munkberg_Extracting_Triangular_3D_Models_Materials_and_Lighting_From_Images_CVPR_2022_paper.html)

[pdf download link](https://openaccess.thecvf.com/content/CVPR2022/papers/Munkberg_Extracting_Triangular_3D_Models_Materials_and_Lighting_From_Images_CVPR_2022_paper.pdf)

[Github Link](https://github.com/NVlabs/nvdiffrec)

[Project Homepage](https://nvlabs.github.io/nvdiffrec/)


## 后人对此文章的评价


# 文章内容

## 摘要

> We present an efficient method for joint optimization of topology, materials and lighting from multi-view image observations. Unlike recent multi-view reconstruction approaches, which typically produce entangled 3D representations encoded in neural networks, we output triangle meshes with spatially-varying materials and environment lighting that can be deployed in any traditional graphics engine unmodified. We leverage recent work in differentiable rendering, coordinate-based networks to compactly represent volumetric texturing, alongside differentiable marching tetrahedrons to enable gradient-based optimization directly on the surface mesh. Finally, we introduce a differentiable formulation of the split sum approximation of environment lighting to efficiently recover all-frequency lighting. Experiments show our extracted models used in advanced scene editing, material decomposition, and high quality view interpolation, all running at interactive rates in triangle-based renderers (rasterizers and path tracers).

> 我们提出了一个有效从多视角图像中联合优化拓扑、材质以及光照的方法。不溶于最近的多视角重建方法，这些方法经常生成耦合起来的三维表征，表征通常编码于神经网络中。我们的输出是三角面片以及其随着空间变化的材质、环境光，可以直接用于传统的计算机图形学引擎中，不用做修改调整。我们采用最近可微分渲染的工作，基于坐标的网络以精简地表达体积纹理，沿着可微分的四面体移动方法，保证基于梯度的优化，该优化直接作用于物体表面的三角面片。最后，我们引入了一个可微分的公式，使用分总的方法近似环境光，以高效地恢复所有频率的光照。实验表明我们提取到的模型可以用于先进的场景编辑，材质解耦合，以及高质量的视角插值。所有的训练都在可以交互的速率上，使用基于三角的渲染器（光栅器和光路追踪器）。

## 介绍



## 本文的组织结构

- Abstract
- 1. Introduction
- 2. Related Work
  - 2.1. Multi-view 3D Reconstruction
  - 2.2. BRDF and Lighting Estimation
- 3. Our Approach
  - 3.1. Learning Topology
  - 3.2. Shading Model
  - 3.3. Image Based Lighting
- 4. Experiments
  - 4.1. Scene Editing and Simulation
  - 4.2. View Interpolation
    - Synthetic datasets
    - Real-world datasets
  - 4.3. Comparing Spherical Gaussians and Split Sum
- 5. Limitations and Conclusions
- References
- 6. Supplemental Materials
- 7. Novel applications
  - 7.1. Level-of-detail From Images
  - 7.2. Appearance-Aware NeRF 3D Model Extractor
  - 7.3. 3D Model Extraction with Known Lighting
- 8. Results
  - 8.1. Scene Editing and Simulation
  - 8.2. View interpolation
  - 8.3. Geometry
  - 8.4. Quality of Segmentation Masks
  - 8.5. Multi-View Stereo Datasets
- 9. Implementation
  - 9.1. Optimization
  - 9.2. Losses and Regularizers
    - Image Loss
    - Light Regularizer
    - Material Regularizer
    - Laplacian Regularizer
    - SDF Regularizer
  - 9.3. Split Sum Implementation Details
- 10. Scene Credits
- 


# Key Points

# Abstract 

