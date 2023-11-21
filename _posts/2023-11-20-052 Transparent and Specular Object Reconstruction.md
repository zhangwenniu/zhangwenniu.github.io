---
layout: mypost
title: 052 Transparent and Specular Object Reconstruction
categories: [论文阅读, 透明重建, 读完论文]
---


# 文章信息

## 标题

Transparent and Specular Object Reconstruction

综述-透明物体和镜面物体的重建

## 作者

Ivo Ihrke 1,2,
1 University of British Columbia, Canada, ihrke@mmci.uni-saarland.de
2 Saarland University, MPI Informatik, Germany
不列颠哥伦比亚大学，位于加拿大。一个集各种地名于一身的大学。

Kiriakos N. Kutulakos 3,
3 University of Toronto, Canada, kyros@cs.toronto.edu

Hendrik P. A. Lensch 4,
4 Ulm University, Germany, hendrik.lensch@uni-ulm.de

Marcus Magnor 5
5 TU Braunschweig, Germany, magnor@cg.tu-bs.de

Wolfgang Heidrich 1
1 University of British Columbia, Canada, heidrich@cs.ubc.ca

主要作者和最后的通讯作者都来自于加拿大的不列颠哥伦比亚大学。

## 发表信息

文章发表于 Computer Graphics forum.

> Edited By: Helwig Hauser and Pierre Alliez
> Impact factor (2022):2.5
> Journal Citation Reports (Clarivate, 2023): 52/108 (Computer Science, Software Engineering (Science))
> Online ISSN:1467-8659
> Print ISSN:0167-7055
> © The Eurographics Association and John Wiley & Sons Ltd.

该期刊2022年的影响因子IF是2.5，来源于Wiley Online Library. [https://onlinelibrary.wiley.com/journal/14678659](https://onlinelibrary.wiley.com/journal/14678659). 

2023年最新的期刊文章参见：[https://onlinelibrary.wiley.com/toc/14678659/2023/42/6](https://onlinelibrary.wiley.com/toc/14678659/2023/42/6).

本文发表时间是2010年的11月10日，DOI号为10.1111/j.1467-8659.2010.01753.x，该文章所在的网址为[https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01753.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01753.x).


## 引用信息

来自Wiley Online Library的bib citation [https://onlinelibrary.wiley.com/action/showCitFormats?doi=10.1111%2Fj.1467-8659.2010.01753.x](https://onlinelibrary.wiley.com/action/showCitFormats?doi=10.1111%2Fj.1467-8659.2010.01753.x)

```
@article{https://doi.org/10.1111/j.1467-8659.2010.01753.x,
author = {Ihrke, Ivo and Kutulakos, Kiriakos N. and Lensch, Hendrik P. A. and Magnor, Marcus and Heidrich, Wolfgang},
title = {Transparent and Specular Object Reconstruction},
journal = {Computer Graphics Forum},
volume = {29},
number = {8},
pages = {2400-2426},
keywords = {range scanning, transparent, specular, and volumetric objects, I.4.8 Scene Analysis, Range Data, Shape I.2.10 Vision and Scene Understanding, 3D Scene Analysis},
doi = {https://doi.org/10.1111/j.1467-8659.2010.01753.x},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2010.01753.x},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-8659.2010.01753.x},
abstract = {Abstract This state of the art report covers reconstruction methods for transparent and specular objects or phenomena. While the 3D acquisition of opaque surfaces with Lambertian reflectance is a well-studied problem, transparent, refractive, specular and potentially dynamic scenes pose challenging problems for acquisition systems. This report reviews and categorizes the literature in this field. Despite tremendous interest in object digitization, the acquisition of digital models of transparent or specular objects is far from being a solved problem. On the other hand, real-world data is in high demand for applications such as object modelling, preservation of historic artefacts and as input to data-driven modelling techniques. With this report we aim at providing a reference for and an introduction to the field of transparent and specular object reconstruction. We describe acquisition approaches for different classes of objects. Transparent objects/phenomena that do not change the straight ray geometry can be found foremost in natural phenomena. Refraction effects are usually small and can be considered negligible for these objects. Phenomena as diverse as fire, smoke, and interstellar nebulae can be modelled using a straight ray model of image formation. Refractive and specular surfaces on the other hand change the straight rays into usually piecewise linear ray paths, adding additional complexity to the reconstruction problem. Translucent objects exhibit significant sub-surface scattering effects rendering traditional acquisition approaches unstable. Different classes of techniques have been developed to deal with these problems and good reconstruction results can be achieved with current state-of-the-art techniques. However, the approaches are still specialized and targeted at very specific object classes. We classify the existing literature and hope to provide an entry point to this exiting field.},
year = {2010}
}
```

## 论文链接

[https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01753.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01753.x).

## 后人对此文章的评价



# 文章内容

## 摘要


> This state of the art report covers reconstruction methods for transparent and specular objects or phenomena. While the 3D acquisition of opaque surfaces with Lambertian reflectance is a well-studied problem, transparent, refractive, specular and potentially dynamic scenes pose challenging problems for acquisition systems. This report reviews and categorizes the literature in this field. 
> 
> Despite tremendous interest in object digitization, the acquisition of digital models of transparent or specular objects is far from being a solved problem. On the other hand, real-world data is in high demand for applications such as object modelling, preservation of historic artefacts and as input to data-driven modelling techniques. With this report we aim at providing a reference for and an introduction to the field of transparent and specular object reconstruction. 
> 
> We describe acquisition approaches for different classes of objects. Transparent objects/phenomena that do not change the straight ray geometry can be found foremost in natural phenomena. Refraction effects are usually small and can be considered negligible for these objects. Phenomena as diverse as fire, smoke, and interstellar nebulae can be modelled using a straight ray model of image formation. Refractive and specular surfaces on the other hand change the straight rays into usually piecewise linear ray paths, adding additional complexity to the reconstruction problem. Translucent objects exhibit significant sub-surface scattering effects rendering traditional acquisition approaches unstable. Different classes of techniques have been developed to deal with these problems and good reconstruction results can be achieved with current state-of-the-art techniques. However, the approaches are still specialized and targeted at very specific object classes. We classify the existing literature and hope to provide an entry point to this exiting field. 
> 
> Keywords: range scanning, transparent, specular, and volumetric objects 
> 
> ACM CCS: I.4.8 Scene Analysis, Range Data, Shape I.2.10 Vision and Scene Understanding, 3D Scene Analysis.

本篇最新的报告覆盖了对透明及反光物体和现象的重建方法。尽管获取具有朗博反射特性的不透光物体表面的三维获取已经是一个研究的比较好的问题，但是透明的、折射的、镜面的和可能是动态的场景仍然难以获取。本文回顾和分类了该领域中的文献。

尽管在物体的数字化方面有大量的研究热潮，但是透明或者反光物体的数字模型获取仍然远不是一个已经解决的问题。另一方面，物体建模、历史文物保护、作为数据驱动的建模技术等应用需要真实世界的数据。在这篇报告中，我们主要提供一个参考，并介绍透明及镜面物体的重建。

我们描述不同类物体的获取方法。在自然现象中，透明物体或者透明现象并不改变笔直射线传播的几何结构，这是最重要的自然现象。折射效应在自然现象中通常比较小，对于这些物体而言一般可以忽略。这些现象包括火焰、烟雾、星际星云等，可以通过直线方式做成像的建模。在另一方面，折射表面和镜面将直线传播的射线改编为分段的线性射线光路，这为重建问题带来了额外的复杂性。半透明物体显示出显著的次表面散射效应，这使得传统的重建获取方法变得不稳定。不同种类的技术已经可以用于处理这些问题，使用当前（2010年时候）的最先进方法，可以实现好的重建效果。然而，这些方法是特定用于某些特定的物体类别。我们对现有的文献分类，希望能够提供这个激动人心领域的进入点。

关键词：范围扫描，透明，镜面，体积物体

ACM CCS(Association for Computing Machinery, Computing Classification System， 计算机协会计算机分类学): I.4.8 场景分析，距离数据，形状；I.2.10 视觉和场景理解，3D场景分析。

## 介绍

## 关键信息

- Stich et al. [STM06] propose to detect discontinuities in epipolar plane images using a constant baseline multi-view stereo setup
- They show that for surface materials exhibiting a ‘diffuse+specular’ reflectance the radiance tensor is of rank two.
- This way, the reflectance does not change even in the presence of glossy materials and surface highlights become features that can be used for reconstruction purposes. 
- The reconstruction of surface geometry for specular objects is complicated by the fact that light is reflected off the surface. Therefore, there are no surface features that can be observed directly. When changing the view point, features appear to move on the surface and the law of reflection has to be taken into account.
- The patterns are generated using a computer monitor. Since the monitor is placed in close proximity of the object the inherent depth-normal ambiguity has to be considered. 考虑深度和法向量的一致性问题。
- The authors resolve it using an iterative approach. An initial guess for the depth value is propagated and corresponding normals are computed. The normal field is then integrated to obtain an updated depth estimate from which updated normals are computed. The process is iterated until the surface shape converges. 使用迭代的方法解决问题。先有初始的数值，接下来不断优化。如果有两个场，优化其中一个，固定之后优化另一个。反复迭代。
- Oren and Nayar [ON96] consider specular surface reconstruction in a structure-from-motion [HZ00] setting. The apparent motion of features in the image plane of a moving camera is analysed. The authors develop a classification between ‘real’ features, that is, world points not reflected by a specular object and ‘virtual’ features, that is, features influenced by specular reflection.
- A similar analysis has been performed by the same authors to explain the movement of highlights on specular surfaces [SKS∗02], see Section 2.2. 高光问题。
- Instead of relying on distant, calibrated patterns, lately researchers have investigated the dense tracking of specularly moving features reflected from a distant, unknown environment map. 考虑高光的移动。
- Roth and Black [RB06] introduced the notion of specular flow, similar to optical flow [HS81, LK81] for image movement due to diffuse surfaces. 镜面光的光流。
- The material distribution is modelled in a probabilistic way and an expectation–maximization algorithm is employed to infer a segmentation between regions moving due to diffuse optical flow and regions with apparent movement due to specular reflection. 用概率的方法建模，使用期望最大化的算法计算。
- It is shown that the incorporation of specular information yields a notably better reconstruction than in the case of only using the diffuse model. 使用镜面信息会显著提高重建结果，相比于只使用漫反射信息的情况。
- three-dimensional surface shape obtained by integrating the normal field. 对法向量场积分能够得到表面的形状吗？
- If standard stereo techniques are applied to such features, the depth estimate results in a point in front of the surface for concave surfaces and in its back when the surface shape is convex [Bla85, BB88a] since specular highlights do not remain stationary on a surface when the viewpoint is changed. 对于凹面的物体，使用标准的立体视觉技术，估计的深度点在表面前；对于凸的物体，估计的深度点在表面之后。当视点发生变化的时候，镜面高光并不保持静态。
- The depth-normal ambiguity can be avoided if the illumination is distant with respect to the object size, for example [Ike81, SWN88, RB06, AVBSZ07] or if polarization measurements are being used [SSIK99]. 如果光照在相对于物体的尺寸比较远的距离，或者使用极性度量，可以避免深度和法向量的不一致性。
- Using the physical Torrance–Sparrow BRDF model [TS67], the radiance fall-off in extended specular highlights is analysed and it is shown that second-order surface information, that is, the directions and magnitudes of principal curvature can be extracted from a single highlight. 好像听说过Torrance-Sparrow BRDF模型。对法向量再进行求导，二阶表面信息是什么呢？
- Zisserman et al. [ZGB89] study the movement of specular highlights due to known movement of the imaging sensor. 如果已知相机的位姿，对预估高光的出现位置是有帮助的。
- Sanderson et al. [SWN88] propose an approach to scan specular surfaces termed structured highlight scanning. 结构化的高光扫描，用于扫描镜面表面。
- By sequentially activating the light sources and observing the corresponding highlights, a normal field of the surface can be reconstructed. 通过顺序的激活点光源，观测对应的高光，表面的法向量场是可以重建出来的。
- They find that the accuracy of this approach diminishes with increasing surface curvature. 当曲率提高的时候，一些方法的精度会下降。
- Additionally, specular highlights are tracked through the image sequence and a variational framework for shape recovery from these sparse features is developed. 在图像序列中，对高光位置进行跟踪。
- The variational problem is solved using a level-set formulation [Set99, OF03] with diffuse features as boundary conditions. The application is a generalization of structure-from-motion approaches, where specular surfaces like windows and metallic surfaces are permitted to be present in the scene. 有这样一种扩展的运动恢复结构的方法，该方法中允许存在镜面的物体表面，比如说窗户和金属的表面。
- A completely different approach to exploit highlight information from surface highlights is presented by Saito et al. [SSIK99]. Their technique is based on partial polarization of light due to reflection off non-metallic surfaces. Examples for such surfaces include asphalt, snow, water or glass. If light is polarized by reflection, the polarization is minimal in the plane of reflection, that is, in the plane containing the incident light ray, the surface normal, and the viewing ray. 有这样一种方法来探索表面的高光信息。他们的技术基于由非金属表面的反射引起的偏振光，比如说沥青、雪、水、玻璃的表面。如果光线通过反射得到偏振，那么偏振在反射的平面上是最小的，也就是包括入射光方向、法向量方向、视角方向的平面。
- Again, the normal field is integrated to obtain the final surface shape. 又提到积分法向量场以得到表面形状。怎么积分的呢？
- Francken et al. [FHCB08] propose a similar setup as previously discussed. However, instead of generating the distant gradient illumination by employing spherically distributed LED’s like Ma et al. [MHP∗07], they use an LCD screen to display the pattern. The LCD screen simultaneously serves as a polarized light source, enabling the separation of diffuse and specular reflection components as in [MHP∗07]. 早些时候有将光照放在物体上方，呈半球状。在此时，已经有人采用显示器作为偏振光的光源了。DRT等工作可能沿袭此类思路。
- Translucent objects (Figure 1, class 4) are difficult to acquire for traditional range scanning techniques due to the non-locality of light transport introduced by multiple scattering just beneath the object surface. 使用传统距离扫描的技术很难获得半透明的物体，因为表面下方的多次散射会带来光传播的非局部性。
- Active light techniques often observe blurred impulse responses and the position of the highest intensity measurement might not coincide with the actual surface position that was illuminated [CLFS07]. 主动光技术经常观测到模糊脉冲响应，最大的强度并不总是出现在照明的表面位置。
- Since specular reflection is not influenced by sub-surface light transport, specular highlights appear in the same positions as they would for a surface not exhibiting global light transport within the object. 镜面高光反射并没有收到次表面光线传播的影响，镜面的高光出现在与不显示物体全局光照特性的物体相同的位置上。
- This property has been used by Chen et al. [CGS06] to acquire the mesostructure of sub-surface scattering objects and by Ma et al. [MHP∗07] to obtain detailed surface normals for translucent materials like human skin. 半透明的材质是包括人的皮肤的。
- The phase-shifting technique [CLFS07] discussed previously relies on polarization-based removal of surface highlights. 基于偏振移除表面的高光。
- In [CSL08], Chen et al. improve their separation strat-egy by modulating the low-frequency pattern used for phase shifting with a 2D high frequency pattern that allows for the separation of local direct illumination effects and light contributions due to global light transport [NKGR06]. 注意使用高频分量和低频分量，在设计球谐函数的时候，是可以将这些因素考虑进去的。

