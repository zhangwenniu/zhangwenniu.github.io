---
layout: mypost
title: 054 Differentiable refraction-tracing for mesh reconstruction of transparent objects
categories: [透明, 折射, 表面重建]
---


# 文章信息

## 标题

Differentiable refraction-tracing for mesh reconstruction of transparent objects

用于透明物体网格重建的可以微分的折射追踪算法

## 作者

JIAHUI LYU∗, Shenzhen University，深圳大学。

BOJIAN WU∗, Alibaba Group，吴博剑，在2020年论文发表的时候，已经作为阿里巴巴的团队参加论文的编写了。吴博剑的主页是：[https://bojianwu.github.io/](https://bojianwu.github.io/)

DANI LISCHINSKI, The Hebrew University of Jerusalem

学校全称：耶路撒冷希伯来大学；国家：以色列。希伯来大学（The Hebrew University of Jerusalem），简称希大，是犹太民族的第一所大学，同时也是犹太民族在其祖先发源地获得文化复兴的象征，全球大学高研院联盟成员。希大始创于1918年，落成于1925年。其校园除斯科普司山校区外，还有吉瓦特拉姆校区、雷霍伏特校区和英科雷姆校区。希大现已发展成为一所充满活力、集教学和研究于一体的世界一流大学。

作者Dani Lischinski的个人主页是[https://www.cs.huji.ac.il/~danix/](https://www.cs.huji.ac.il/~danix/), 他的研究方向是图形学、图像和视频处理、计算机视觉，涉及面比较广。

DANIEL COHEN-OR, Shenzhen University，深圳大学。该作者的主页是[https://danielcohenor.com/](https://danielcohenor.com/)，主要研究方向是计算机图形学、视觉计算、几何建模。近期，主要在做生成模型方向的工作。

HUI HUANG†, Shenzhen University，深圳大学，通讯作者。
†Corresponding author: Hui Huang (hhzhiyan@gmail.com)

黄惠，加拿大不列颠哥伦比亚大学（UBC）数学博士，深圳大学首批腾讯冠名特聘教授，教育部长江学者特聘教授，计算机科学与技术一级博士点学科带头人，国家科技创新领军，科技部中青年创新领军，国自然优秀青年基金获得者，英国皇家牛顿高级学者，广东省杰出人才，南粤巾帼十杰，广东省三八红旗手标兵，广东省自然科学基金研究团队、广东省高等学校创新团队、深圳市高层次人才孔雀团队带头人。现任深圳大学计算机与软件学院院长，可视计算研究中心主任，广东省3D内容制作工程技术中心主任。研究领域为计算机图形学，在三维获取与点云表示、几何建模与场景理解等前沿方向保持国际领先水平。主持（完成）国家自然科学基金8项（重点类 5 项）和科技部 973 前期研究专项（结题优秀）。现为国际顶级期刊ACM TOG和IEEE TVCG编委，SIGGRAPH技术论文咨询委员会和EUROGRAPHICS执行委员会唯一华人代表。2020年获中国计算机学会自然科学一等奖，2022年获中国图象图形学学会高等教育成果二等奖，2022、2023连续入选AMiner计算机图形学领域“全球最具影响力学者提名”和“全球顶尖女性学者”。主页[https://vcc.tech/~huihuang](https://vcc.tech/~huihuang)




## 发表信息

论文发表时间：2020年，被SIGGRAPH Asia录用。文章的缩写为DRT。该文章后续被TOG录用（ACM Transactions on Graphics）, TOG主页[https://dl.acm.org/journal/tog](https://dl.acm.org/journal/tog).


## 引用信息

```
@article{DRT,
title = {Differentiable Refraction-Tracing for Mesh Reconstruction of Transparent Objects},
author = {Jiahui Lyu and Bojian Wu and Dani Lischinski and Daniel Cohen-Or and Hui Huang},
journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH ASIA 2020)},
volume = {39},
number = {6},
pages = {195:1--195:13},
year = {2020},
}
```


## 论文链接

该文章展示在深圳大学可视计算研究中心（Visual Computing Research Center, VCC）的主页上，HomePage为[https://vcc.tech/research/2020/DRT](https://vcc.tech/research/2020/DRT).

论文下载的链接是: [https://csse.szu.edu.cn/attachment/cglr/1646560808_DRT.pdf](https://csse.szu.edu.cn/attachment/cglr/1646560808_DRT.pdf)

论文的在dl.acm.org上的链接是: [https://dl.acm.org/doi/abs/10.1145/3414685.3417815](https://dl.acm.org/doi/abs/10.1145/3414685.3417815).

GitHub上的主页链接是: [https://github.com/lvjiahui/DRT](https://github.com/lvjiahui/DRT) .

论文在arxiv上的链接是: [https://arxiv.org/abs/2009.09144](https://arxiv.org/abs/2009.09144)

文章在深圳大学计算机与软件学院上的主页是: [https://csse.szu.edu.cn/pages/research/details?id=118](https://csse.szu.edu.cn/pages/research/details?id=118)


## 后人对此文章的评价


# 文章内容

## 摘要

> Capturing the 3D geometry of transparent objects is a challenging task, ill-suited for general-purpose scanning and reconstruction techniques, since these cannot handle specular light transport phenomena. Existing state-ofthe-art methods, designed specifically for this task, either involve a complex setup to reconstruct complete refractive ray paths, or leverage a data-driven approach based on synthetic training data. In either case, the reconstructed 3D models suffer from over-smoothing and loss of fine detail. This paper introduces a novel, high precision, 3D acquisition and reconstruction method for solid transparent objects. Using a static background with a coded pattern, we establish a mapping between the camera view rays and locations on the background. Differentiable tracing of refractive ray paths is then used to directly optimize a 3D mesh approximation of the object, while simultaneously ensuring silhouette consistency and smoothness. Extensive experiments and comparisons demonstrate the superior accuracy of our method.


> 获取透明物体的三维几何信息是一个挑战性的任务，不适合使用通用任务的扫描与重建方法，因为这些通用方法无法处理镜面光传播效应。已有的为透明物体重建的技术，要么包含复杂的设定用于重建完整的折射光路，要么基于数据驱动的方法在合成数据集上重建。在这两种情况下，重建的三维模型都存在过渡平滑或者缺失细节的情况。本文提出了一个新的、高精度的三维获取和重建方法，用于对实心透明物体重建。通过使用一个模式编码的背景，我们构建一个在背景板上的位置与相机视角的映射。接下来采用可微分的折射光路追踪，优化物体的三维近似网格，同时保证剪影的一致性和平滑性。大量的实验和比较都证明了我们方法的高准确性。


CCS Concepts: • Computing methodologies → Computer graphics; Shape modeling; Mesh geometry models.

Computing Classification System Concepts: 计算方法 -> 计算机图形学；形状建模；网格几何模型。

Additional Key Words and Phrases: 3D reconstruction, transparent objects, differentiable rendering

其他的关键词及短语：三维重建，透明物体，可微分渲染

## 介绍

6-231124. Differentiable Refraction-Tracing for Mesh Reconstruction of Transparent Objects.  发表于2020年的SIGGRAPH，后续被ACM TOG录用。

文章解决实心透明物体的重建问题。实心透明物体放置在旋转台上，使用预知模式的黑白条带屏幕作为背景，用固定位置的相机拍摄图片。

通过环境抠图（Environment matting）方法获得掩膜以及透明物体对背景的畸变效果，利用视舱（Visual Hull）方法得到初始化的网格（mesh）。文章通过渐进式的方法，逐步增加网格的密度。

损失函数分为三部分，基于透明物体对光线折射所计算的射线位置与背景板位置的一致性约束，基于预测三维网格在对应视角上投影的剪影与真实剪影的一致性约束，对物体表面相邻顶点间法向量的一致性约束。全部过程是可微的，仅在假设的两次折射光线上对网格顶点优化，假设的玻璃折射率为1.5。

​8-231126. Differentiable Refraction-Tracing for Mesh Reconstruction of Transparent Objects. 文章使用视舱方法从拍摄图像中提取初始三角网格模型。

消融实验表明，前景剪影的约束对最终收敛效果影响最大：面片的投影边界与真实边界做比较，计算梯度时，使边界的负梯度方向指向该边所在中点的物体内外部的相反方向，即内部点梯度指向外部，外部点梯度指向内部，构造时中点梯度的方向指向边界外部，使用内外部符号函数控制反传梯度的方向。

在获取数据方面，实心透明物体的真实值是通过喷漆及扫描相机得到的。文章的四个问题，第一是拍摄时候相机拍摄方向水平，导致透明物体水平方向折痕缺少重建视角；二是假设两次折射，限制空腔或中空透明物的重建；三是物体的均匀性假设，要求预知折射率；四是拍摄环境在暗室内，不考虑反光对物体的影响。

# Key Points

- Our progressive mesh reconstruction. 文章采用的是逐步渐进式的网格精细化重建方法。
- Fig. 1. Reconstructing a transparent Hand object. The five images, from left to right, show a sequence of ray-traced models, progressively optimized by our method. The ground-truth geometry, obtained by painting and scanning the object and a real photograph of the original object are shown on the right. 物体的真实几何信息，由在物体表面喷漆并扫描得到。


# 1 Introduction

- Thus, such approaches are not applicable to objects made from a transparent, refractive material, due to the complex manner in which they interact with light. 因此，这些方法并不适用于透明的、折射材质，因为这些物体与光线的交互方式比较复杂。
- capturing correspondences between camera rays and the rays incident on the object from behind. 获取相机光线与物体后面光线入射之间的对应关系。
- yields a point cloud from which the final model is consolidated. Wu的方法最后输出是一组点云。
- Li et al. [2020] employ a data-driven approach, which leverages a large number of synthetic images as its training set, and requires capturing the environment map. 采用了大量的合成数据集，同时需要拍摄环境图。
- In contrast to Wu et al. [2018], our approach is based on optimizing correspondences between camera rays and locations on a static background monitor, thereby cutting the acquisition time by half, and avoiding additional cumulative errors. 不同于Wu等人2018年的方法，本文的工作基于优化相机的射线以及静态背景板上的位置信息的差值，因此将获取几何信息的时间减半了，避免了额外的累积误差。
- More importantly, the proposed method optimizes the reconstructed mesh directly, and is able to capture the fine geometric details of the object’s surface. 本文提出的方法直接优化重建出来的mesh网格，能够获取物体表面精细的几何信息。
- Furthermore, our approach leverages automatic differentiation, which can be better integrated with popular deep learning frameworks and benefit from GPU-accelerated optimization. 使用自动微分渲染，可以融合进深度学习的框架中，受到GPU加速的优化框架的好处。
- Starting from a rough initial mesh, obtained from the visual hull of the object, our method progressively refines the mesh in a coarse-to-fine fashion. 粗糙的初始化网格是从物体的视舱轮廓中获取到的，我们的方法采用由粗到细的方式逐步微调物体的几何结构。
- (1) Refraction loss, which minimizes the distance between the observed background refractions and the simulated ones; (2) Silhouette loss, which ensures that the boundary of the optimized mesh matches the captured silhouettes; (3) Smoothness loss, ensuring smoothness of the optimized mesh. 三组损失函数：折射损失，用于优化射线与背景板的一致性信息；剪影损失，保证网格的边界与剪影一致；平滑性损失，保证优化网格的平滑性。
- 前几天看了几篇老的文章。新文章的特点是，比较注重深度学习的架构，在损失函数上面找突破、求亮点，而非在物理信息上加入先验。
- Our approach only makes use of refractive ray paths through the object that feature exactly two refractions, once upon entering, and once upon exiting the transparent object 采用两次折射的确定性方式。
- Thus, our optimization ignores some of the additional ray paths that may be observed during acquisition, i.e., those involving more than two intersections with the object’s surface and/or total internal reflection. 如果光线传播过程中，遇到超过两次与物体的相交次数，或者发生了完全内反射，忽略这些射线，不对这些射线优化。


# 2 Related Work

## 2.1 Environment Matting

- Matting is a process concerned with extracting from an image a scalar foreground opacity map, commonly referred to as the alpha channel [Levin et al. 2008; Porter and Duff 1984]. 抠图是从一个图像中提取标量前景不透明度图的过程，经常称为alpha通道。
- Environment matting is an extension of alpha matting that also captures how a transparent foreground object distorts its background, so it may be composited over a new one. 环境抠图是alpha抠图的扩展，同时还要求知道透明的前景物体是如何对背景扭曲的，因此它能够被组合到新的图像中。
- The pioneering work of Zongker et al. [1999] extracts the environment matte from a series of projected horizontal and vertical stripe patterns, with the assumption that each pixel is only related to a rectangular background region. 环境抠图技术的先驱工作，假定每个像素都与背景的一个方形区域有关。
- To improve environment matting accuracy and to better approximate real-world scenarios, Chuang et al. [2000] propose to locate multiple contributing sources from surrounding environments.提出从周围的环境中定位多个贡献因素。具体是什么样的贡献源，我还没有看论文，并不知道是什么。
- Other works present solutions in domains other than the image, such as the wavelet domain [Peers and Dutré 2003], or the frequency domain [Qian et al. 2015]. 有一些工作致力于从小波领域或者频域分析。
- Our approach could be viewed as an extension of environment matting to the task of transparent object reconstruction, in the sense that it progressively optimizes the reconstructed shape of the object so as to better match a collection of environment mattes captured from multiple views. 本文的方法可以被视作环境抠图的扩展，用于透明物体重建任务。本文的方法渐进式的优化物体重建出的形状，用于匹配一系列从多个视角中获取到的环境抠图。

## 2.2 Transparent surface reconstruction

- Non-intrusive methods use the refractive properties of transparent objects to recover their shape by analyzing the distortions of reference background images [Ben-Ezra and Nayar 2003; Tanaka et al. 2016; Wetzstein et al. 2011]. 透明薄物体的表面重建工作，对背景的弯折程度有限。
- Recovering the object shape from the optical distortion it induces is typically applicable to a single refractive surface or a parametric model, since light transport resulting from multiple reflections and refractions is much more difficult to analyze. 怎么会有单次表面折射呢？
- it is also possible to capture the reflective components of light transport, and estimate the shape geometry by observing exterior specular highlights [Morris and Kutulakos 2007; Yeung et al. 2011]. 获取带有折射物体的反光分量，并通过外部的高光获取物体的几何形状。
- Since reflections occur on the outermost surface, it is possible to reconstruct objects with complex geometries and inhomogeneous internal materials. 这句话分析到我心坎上了，高光是存在于物体的外表面上的，有可能对复杂几何以及内部非均匀材料的几何形状。
- However, the acquisition process is quite involved and considerable manual effort is needed to control the lighting conditions precisely enough to obtain reasonable results. 利用透光物体表面的反光性质重建透明物体的表面信息，这本身需要精细的设计，才有可能获取物体的合理几何形状。
- Stets et al. [2019] and Sajjan et al. [2020] propose to use encoderdecoder architectures for estimating the segmentation mask, depth map and surface normals from a single input image of a transparent object. 使用编码器-解码器的架构，用于估计单张图像中透明物体的分割掩膜、深度图、表面法向量。
- Due to the difficulty of obtaining a sufficient amount of real training data, these data-driven methods rely on synthetic training images. 真实的训练数据比较难获取，所以采用合成图像加以训练。
- In contrast, we use a controlled acquisition setup to capture refractive light paths and use direct per-object shape optimization, which does not require a training set consisting of similar shapes. 逐个场景训练微调，在受控条件下获取折射的光路。


## 2.3 Light path triangulation

- Light path triangulation is an extension of classical stereo triangulation, which uses the relationship between direction of refraction and the surface normal to infer geometry from light transport. 光路三角化是传统的立体视觉三角化的扩展，使用折射方向和表面法向量的关系，从光线传播中推测几何形状。这是个新的领域，之前没见过。
- Kutulakos and Steger [2008] provide a theoretical analysis of the reconstruction feasibility based on the number of specular reflections and refractions along the ray paths. 看起来像是分析有多少次折射的时候，才可能重建物体的表面形状。
- Next, Tsai et al. [2015] reveal depth-normal ambiguity while assuming that the light rays refract twice. 假定光线有两次折射的时候，存在深度和法向量的歧义性。
- To eliminate the ambiguity, Qian et al. [2017] propose a position-normal consistency based optimization framework to recover front and back surface depth maps. 优化前表面和后表面的深度图，用于消除位置和法向量之间的歧义性。


- Differently from these methods, we do not employ an irradiance-based loss function that measures the discrepancy between the pixel values of the rendered image and the ground truth.
- Rather, our refraction loss is based directly on ray-pixel correspondences, which reflect the geometry of the underlying light transport.
- The geometry of light paths is directly determined by the shape geometry, which is what we seek to recover, compared to the final RGB pixel colors, which are influenced by additional factors, such as the BRDF.
- there may exist multiple refractive surfaces that might satisfy the observed refraction of the background pattern. 可能存在多组符合折射背景模式的物体表面的组合。这跟之前组会上讨论的是一致的。


## 5.1 Acquisition

- The intrinsic and extrinsic parameters of the camera, and the relative positions of monitor and turntable with respect to the camera are calibrated [Zhang 2000], before the acquisition commences. 相机的内参、外参，监视器的位置、相对相机位置的旋转都预先标定好的。
- To capture an object, the turntable is rotated to a set of 72 evenly sampled viewing angles. 最初使用72个均匀放置的视角拍摄。后面说的NeTO，采用了32张视角作为训练。
- a Gray-coded background pattern is displayed on the monitor for simultaneously extracting silhouette and estimating ray-pixel correspondences using environment matting. 背景板放在后面，用于同步的提取剪影，并估计射线与像素的对应关系，使用环境抠图的方法。
- The Gray-coded background is produced by displaying a sequence of 11 images with vertical stripes and 11 images with horizontal stripes (see Fig. 11). Note that, in order to avoid the influence of ambient light, the entire acquisition process is conducted in a dark room, and the background monitor is used as the only light source. 射线编码的背景，使用11张图像（竖条纹）和11张图像（横条纹）。那这样的话，还是72张图像，不就变成72*11=792张图像了吗？
- Note that, in order to avoid the influence of ambient light, the entire acquisition process is conducted in a dark room, and the background monitor is used as the only light source. 为了避免环境光的影响，整体的拍摄过程都在一个比较暗的房间中进行，背景的监视器用作唯一的光源。
