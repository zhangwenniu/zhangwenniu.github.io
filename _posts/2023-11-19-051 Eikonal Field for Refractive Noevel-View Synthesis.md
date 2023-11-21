---
layout: mypost
title: 051 Eikonal Fields for Refractive Novel-View Synthesis
categories: [论文阅读, NeRF, 读完论文]
---


# 文章信息

## 标题

Eikonal Fields for Refractive Novel-View Synthesis

用于折射新视角合成的Eikonal场

## 作者

Mojtaba Bemana MPI Informatik Germany mbemana@mpi-inf.mpg.de

Karol Myszkowski MPI Informatik Germany karol@mpi-inf.mpg.de

Jeppe Revall Frisvad Technical University of Denmark Denmark jerf@dtu.dk

Hans-Peter Seidel MPI Informatik Germany hpseidel@mpi-inf.mpg.de

Tobias Ritschel University College London UK t.ritschel@ucl.ac.uk

作者主要来自于德国的MPI Informatik(马克思-普朗克-信息研究所)。

> Informatics: Accelerated progress in computing through new algorithms
> Algorithms and their applications are and have always been the main focus of the Institute. They are the core of what makes computer systems useful and productive. They influence every aspect of our daily lives and are the basis for industrial change. Throughout the last decade, major parts of our research effort have focused on multimodal computing. The grand challenge is to understand, search, and organize large, distributed, noisy, incomplete, and diverse information in a robust, efficient, and intelligent manner. Our research ranges from foundations (algorithms and complexity, automation of logic) to a variety of multimodal domains (computer graphics and vision, geometric computation, intelligent information systems, adaptive networks). In recent years, research on foundations of machine learning, as well as the investigation of machine learning and artificial intelligence methods at the intersection to the aforementioned research domains, has become an important part of the research of our institute. The overarching mission of the Institute is to be one of the world’s top players and strategic trend-setters on these topics. Most of the major advances in computer science have come through the combination of new theoretical insights and application-oriented experimental validation, all driven by outstanding researchers. Our goal is, thus, to have impact through publications, software, services, and data resources enabled by our research, and people alike.



## 发表信息

在Arxiv上面放了三版的链接，分别是：

- 2022年2月2日的v1, [v1](https://arxiv.org/abs/2202.00948v1) Wed, 2 Feb 2022 10:49:08 UTC (9,342 KB)
- 2022年2月11日的v2, [v2](https://arxiv.org/abs/2202.00948v2) Fri, 11 Feb 2022 15:04:17 UTC (9,342 KB)
- 2022年6月15日的v3. [v3](https://arxiv.org/abs/2202.00948) Wed, 15 Jun 2022 13:25:40 UTC (6,598 KB)

最终发表于2022年的SIGGRAPH，Conference Proceedings.  [Home Page](https://eikonalfield.mpi-inf.mpg.de/)

论文下载链接[Download Link from Arxiv](https://arxiv.org/pdf/2202.00948.pdf)

被dl.acm.org收录的网址[Link of dl.acm.org](https://dl.acm.org/doi/abs/10.1145/3528233.3530706)

Github的链接：[Github Link](https://github.com/m-bemana/eikonalfield)




## 引用信息

来自Arxiv的主页[https://arxiv.org/abs/2202.00948](https://arxiv.org/abs/2202.00948)

```
@misc{bemana2022eikonal,
      title={Eikonal Fields for Refractive Novel-View Synthesis}, 
      author={Mojtaba Bemana and Karol Myszkowski and Jeppe Revall Frisvad and Hans-Peter Seidel and Tobias Ritschel},
      year={2022},
      eprint={2202.00948},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
```

## 论文链接

[https://arxiv.org/abs/2202.00948](https://arxiv.org/abs/2202.00948)


## 后人对此文章的评价

我印象中NeTO的评价：用于新视角合成，但并不适用于透明物体的表面重建工作。在高折射率、射线偏折较为明显的情况下工作。

> NeTO的评价：
> More recently, Bemana et al. [2] leverage NeRF for novel view synthesis of transparent objects and show good performance to render novel views. However, since it targets novel view synthesis rather than reconstruction, it’s difficult to extract reliable geometry from the method. Different from the above methods, we leverage volume rendering to simulate the refractiontracing path for geometry optimization.

# 文章内容

## 摘要

> ABSTRACT
> We tackle the problem of generating novel-view images from collections of 2D images showing refractive and reflective objects. Current solutions assume opaque or transparent light transport along straight paths following the emission-absorption model. Instead, we optimize for a field of 3D-varying index of refraction (IoR) and trace light through it that bends toward the spatial gradients of said IoR according to the laws of eikonal light transport. 
> CCS CONCEPTS • Computing methodologies → Image-based rendering. 
> KEYWORDS refraction; deep learning; eikonal rendering

本文处理从一系列含有折射和反射物体的图像中，生成新视角图像。现有的方法都假设挡光物体或者沿着直线传播，遵循发射-吸收的模型。相比之下，我们优化一个随着空间变化的3D折射率的场，并沿着折射率所弯折的空间进行光路追踪，沿着的是IOR的空间梯度方向，该方法基于Eikonal的光传输理论。

最后一句可以翻译为：相反，我们优化一个三维空间变化的折射率场（IOR），并通过它最总光线，这写光线朝向该IOR的空间梯度弯曲，符合光传输的Eikonal定律。

CCS 概念
计算机方法-> 基于图像的渲染。

关键词：折射；深度学习；Eikonal渲染。

## 介绍

1-231119。Eikonal Fields for Refractive Novel-View Synthesis，发表于2022 SIGGRAPH。解决强折射物体的新视角合成问题，基于MLP估计的折射率计算法向量，弯折空间光线并积分渲染。方法具体而言，第一阶段优化挡光物体假设情况，第二阶段存在透明物体折射假设的情况。实验效果方面，能够体现出水对物体的弯折效果，但清晰度不够。存在的问题方面，一是时间较慢、多阶段，二是估计的IOR与真实情况不符，解耦合不准确。

1-231119. Eikonal Fields for Refractive Novel-View Synthesis. This paper is published on 2022 SIGGRAPH(Special Interest Groups for Computer GRAPHICS). The work aims to solve novel view synthesis(nvs) problem for objects with highly refractive properties. As for method, they estimate normal vector by computing the gradients of index of refraction(IOR) field, which is learned by a MLP. They bend the ray along the refracted direction, then use volume rendering to render the picture. Specifically, they adopt a two stage method. In the first stage, they optimize nerf fields in an opaque assumption. In the second stage, they optimize a IOR field to hold transparent objects. Key properties are stored in grids vertices. For experiments performance, they can render a picture with objects bent behind water or glasses performance caused by refraction. The limitations are that the render time is long and they adopt two-stage render procedure, as well as maintain multiple MLP to learn different fields. On the other hand, the estimated IOR is not the same with real-scene objects, the decomposition is not accurate enough. 



