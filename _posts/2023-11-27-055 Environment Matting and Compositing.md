---
layout: mypost
title: 055 Environment Matting and Compositing
categories: [论文阅读, 透明重建, 读完论文]
---


# 文章信息

## 标题

Environment Matting and Compositing

环境抠图与组合

## 作者

Douglas E. Zongker 1 
Dawn M. Werner 1 
Brian Curless1 David H. Salesin 1,2

1 University of Washington
2 Microsoft Research



## 发表信息

本文被2023年评选的SIGGRAPH计算机图形学领域内的重要文献第二卷所收录。

Home - Collections - ACM Seminal Works - Seminal Graphics Papers: Pushing the Boundaries, Volume 2 - Environment Matting and Compositing

Seminal Graphics Papers: Pushing the Boundaries, Volume 2August 2023Article No.: 56 Pages 537–546 https://doi.org/10.1145/3596711.3596768
Published:02 August 2023

SIGGRAPH领域内的重要文献：[https://dl.acm.org/doi/book/10.1145/3596711](https://dl.acm.org/doi/book/10.1145/3596711)


该文章的原始发表信息为：

[https://dl.acm.org/doi/10.1145/311535.311558](https://dl.acm.org/doi/10.1145/311535.311558), 发表于1999年的SIGGRAPH上.


## 引用信息

被引用次数306次。

```
@inproceedings{10.1145/311535.311558,
author = {Zongker, Douglas E. and Werner, Dawn M. and Curless, Brian and Salesin, David H.},
title = {Environment Matting and Compositing},
year = {1999},
isbn = {0201485605},
publisher = {ACM Press/Addison-Wesley Publishing Co.},
address = {USA},
url = {https://doi.org/10.1145/311535.311558},
doi = {10.1145/311535.311558},
abstract = {This paper introduces a new process, environment matting, which captures not just a foreground object and its traditional opacity matte from a real-world scene, but also a description of how that object refracts and reflects light, which we call an environment matte. The foreground object can then be placed in a new environment, using environment compositing, where it will refract and reflect light from that scene. Objects captured in this way exhibit not only specular but glossy and translucent effects, as well as selective attenuation and scattering of light according to wavelength. Moreover, the environment compositing process, which can be performed largely with texture mapping operations, is fast enough to run at interactive speeds on a desktop PC. We compare our results to photos of the same objects in real scenes. Applications of this work include the relighting of objects for virtual and augmented reality, more realistic 3D clip art, and interactive lighting design.},
booktitle = {Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques},
pages = {205–214},
numpages = {10},
keywords = {blue-screen matting, colored transparency, augmented reality, environment matte, image-based rendering, reflection, refraction, clip art, alpha channel, interactive lighting design, environment map, blue spill},
series = {SIGGRAPH '99}
}

@inbook{10.1145/3596711.3596768,
author = {Zongker, Douglas E. and Werner, Dawn M. and Curless, Brian and Salesin, David H.},
title = {Environment Matting and Compositing},
year = {2023},
isbn = {9798400708978},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
edition = {1},
url = {https://doi.org/10.1145/3596711.3596768},
abstract = {This paper introduces a new process, environment matting, which captures not just a foreground object and its traditional opacity matte from a real-world scene, but also a description of how that object refracts and reflects light, which we call an environment matte. The foreground object can then be placed in a new environment, using environment compositing, where it will refract and reflect light from that scene. Objects captured in this way exhibit not only specular but glossy and translucent effects, as well as selective attenuation and scattering of light according to wavelength. Moreover, the environment compositing process, which can be performed largely with texture mapping operations, is fast enough to run at interactive speeds on a desktop PC. We compare our results to photos of the same objects in real scenes. Applications of this work include the relighting of objects for virtual and augmented reality, more realistic 3D clip art, and interactive lighting design.},
booktitle = {Seminal Graphics Papers: Pushing the Boundaries, Volume 2},
articleno = {56},
numpages = {10}
}

```


## 论文链接




## 后人对此文章的评价


# 文章内容

## 摘要

> This paper introduces a new process, environment matting,which captures not just a foreground object and its traditional opacity matte from a real-world scene, but also a description of how that object refracts and reflects light, which we call an environment matte. The foreground object can then be placed in a new environment, using environment compositing, where it will refract and reflect light from that scene. Objects captured in this way exhibit not only specular but glossy and translucent effects, as well as selective attenuation and scattering of light according to wavelength. Moreover, the environment compositing process, which can be performed largely with texture mapping operations, is fast enough to run at interactive speeds on a desktop PC. We compare our results to photos of the same objects in real scenes. Applications of this work include the relighting of objects for virtual and augmented reality, more realistic 3D clip art, and interactive lighting design.

> 

## 介绍

9-231127. Environment Matting and Compositing. 文章发表于1999年的SIGGRAPH上，后被2023年评选为近50年间有影响力文章，收录于Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 文章提出环境抠图技术，提取透明物体的前景之外，同时了解透明物体是如何对光线作用的，并用于该透明物体在新环境下的视角合成。

​10-231128. Environment Matting and Compositing. 本文提出透明前景物体的环境抠图方法，提供出透明前景物体如何与环境光作用，并用于将透明物体放置在新环境下的图像合成。图像拍摄由背景及左右两侧共三面的绿色、洋红色条纹格雷码作为环境光拍摄得到，该色彩选定原因是在RGB空间的正交性及拥有相近的亮度。文章假设成像的像素由前景颜色、按比例透光后的背景颜色、从周围环境中反射或折射所得到的颜色，这三部分加和组成。文中主要解决第三部分，即从周围环境中反射或折射得到的颜色的近似方式，该数值由反射比率函数以及环境纹理图的映射求加权乘积和得到。文章同时展示该方法在带颜色香槟、反光平面、透明棱台、不同深度下的放大镜新视角合成的效果。

## 本文的组织结构

- Abstractt
- 1 Introduction
  - 1.1 Related work
  - 1.2 Overview
- 2 The environment matte
- 3 Environment matting
  - 3.1 A coarse estimate of coverage
  - 3.2 The foreground color and reflectance coefficients
  - 3.3 The area extents and a refined estimate of coverage
  - 3.4 Sidedrops
- 4 Environment compositing
- 5 Results
- 6 Depth correction
- 7 Conclusion
- 8 Acknowledgements
- 9 References

# Key Points

# Abstract 

# 1 Introduction

## 1.1 Related work

- Dorsey et al. [10] render a scene under various lighting conditions and then synthesize a new image by taking a linear combination of the ren-derings. In effect, they store at each pixel the contributions from a set of light sources.
- Miller and Mondesir [18] ray trace individual objects and store a ray tree at each pixel for fast compositing into environments consisting of the front and back faces of an environment mapped cube. 做射线追踪的时候，为每个像素都存储一个追踪树，用于快速组合环境映射立方体的前表面和后表面信息。
- However, the scenes are synthetic, and the general effects of glossy and translucent surfaces must be modeled using methods such as distributed ray tracing [7], requiring multiple ray trees per pixel. 一个像素可能需要多个射线追踪树。
- Blue screen matting, pioneered by Vlahos [25], relies on a single-color background sufficiently different from foreground objects. 蓝幕抠图，需要一个与前景显著不同的颜色。
- Reflection from the background onto the foreground, called blue spill,however, remains troublesome even with two backdrops and results in incorrect object transparency after matte extraction. 从背景反射到前景的问题，称作蓝色溢出，但是，即使采用两个背景，仍然会导致透明物体的抠图提取错误。
- To acquire our environment matte, we could illuminate one point of light at a time and sweep over the environment around the object. 为了获得环境抠图，我们可以在一个时间照明一个点光源，沿着物体周围划过一圈。
- By projecting a hierarchy of progressively finer stripe patterns, the required number of images can be reduced to O(log n) [21]. 使用结构化的渐进带状条纹，需要的图片数目可以缩减。
- Our environment matting approach is based on the hierarchical stripe methods. 本文的环境抠图方法，基于结构带方法。

## 1.2 Overview

# 2 The environment matte

- We begin with the traditional compositing equation and then augment it with a new structure, the environment matte, which captures how light in the environment is refracted and reflected by a foreground element. 从一个从传统的组合方程开始，增强该方程以一个新的结构，环境抠图，这获取环境中的光线是如何被前景元素所折射和反射的。
- To start, we will assume that the only light reaching the foreground object is light coming from distant parts of the scene. This is essentially the “environment mapping assumption.”  我们假设只有来自远处的光线会到达前景物体。这是基本上的“环境映射假设”。
- 


# 3 Environment matting

## 3.1 A coarse estimate of coverage

## 3.2 The foreground color and reflectance coefficients

## 3.3 The area extents and a refined estimate of coverage

- This optimization problem still has four degrees of freedom for the unknown rectangle A1: left, right, top, and bottom (l, r, t, b). We can reduce the dimensionality further by using horizontal and vertical stripes for backgrounds. For horizontally striped backgrounds, the area determination is independent of l and r; similarly, for vertically striped backgrounds, the area determination is independent of t and b. 

## 3.4 Sidedrops

# 4 Environment compositing

# 5 Results

# 6 Depth correction

# 7 Conclusion

# 8 Acknowledgements

# 9 References
