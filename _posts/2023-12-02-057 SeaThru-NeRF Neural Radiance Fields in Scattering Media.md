---
layout: mypost
title: 057 SeaThru-NeRF：Neural Radiance Fields in Scattering Media
categories: [论文阅读, NeRF, 读完论文]
---


# 文章信息

## 标题

SeaThru-NeRF: Neural Radiance Fields in Scattering Media

SeaThru-NeRF: 在散射介质中的神经辐射场

## 作者



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

> Research on neural radiance fields (NeRFs) for novel view generation is exploding with new models and extensions. However, a question that remains unanswered is what happens in underwater or foggy scenes where the medium strongly influences the appearance of objects. Thus far, NeRF and its variants have ignored these cases. However, since the NeRF framework is based on volumetric rendering, it has inherent capability to account for the medium’s effects, once modeled appropriately. We develop a new rendering model for NeRFs in scattering media, which is based on the SeaThru image formation model, and suggest a suitable architecture for learning both scene information and medium parameters. We demonstrate the strength of our method using simulated and real-world scenes, correctly rendering novel photorealistic views underwater. Even more excitingly, we can render clear views of these scenes, removing the medium between the camera and the scene and reconstructing the appearance and depth of far objects, which are severely occluded by the medium. Our code and unique datasets are available on the project’s website.
>
> 神经辐射场对新视角生成的研究已经有许多新的模型和扩展方法。在水下场景或者是雾状场景中，介质对物体的外观有较强的影响，在这种情况下的新视角合成仍然是一个问题。目前为止，NeRF及其变种方法都忽略了这个问题。然而，由于NeRF的框架基于体渲染方法，它本身就有能力考虑介质的效果，如果建模得当的话。我们采用一个新的渲染模型，将NeRF扩展到散射介质中，这基于SeaThru的成像模型，带来了一个合适的架构，用于学习场景信息和介质的参数。我们证明本方法的优势，使用模拟和实际拍摄的场景中，正确的渲染水下的新视角图像。更加激动人心的是，我们可以渲染这些场景下的清晰视角，可以移除相机与场景之间的介质，能够重建远处物体的外观以及深度，远处的场景会被介质严重的遮挡。我们的代码和数据集在项目的主页上可以看到。

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

