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


## 介绍

​3-231121. Transparent and Specular Object Reconstruction. 全文20页，涉及到153篇参考文献，主要囊括2010年以前对反光、折射、雾状物体的重建方法。对折射物体而言，主要方法是设计结构化的激光扫描，或者将透明物体置于相同折射率的荧光溶液中，使用断层扫描的方式重建。折射物体一般需要预设折射率信息，做材质的均匀性假设，非均匀介质的重建更加复杂。文章信息量大，在雾状物体重建部分，除了重建烟雾或者火焰之外，甚至涉及到如何重建作为雾状结构的星云。截止2010年，尚没有通用的解决反光物体、折射透明物体的重建方法。


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

### 4. Refractive Surface Acquisition

- The problem of acquiring complete surface descriptions of refractive objects with possibly inhomogeneous material properties is very complex. In its most general form inclusions like air bubbles, cracks or even opaque or specular materials would have to be considered. The image formation for such objects is non-trivial and to date no reconstruction approaches exist for the general problem. Researchers have so far restricted themselves to sub-problems like single surface reconstruction where a well defined surface represents the transition from one medium to the other. Often the refractive index of the object needs to be known. Almost all methods assume that the refractive material is homogeneous. 获取非均匀材质的折射物体的完整表面是非常复杂的。如果考虑最宽泛的物体的话，包括气泡、裂缝、甚至是不透明的或者镜面材质，这些都要考量进去。这些物体的成像是不容易的，现有最新的重建方法并不能够解决这类通用问题。研究者目前都致力于解决一些受限条件下的子问题，比如说重建不同介质之间的交界单表面。一般需要预知折射率的数值。几乎所有的方法都假定折射的材质是均匀齐次的。
- The acquisition of refractive surfaces is more complex than the corresponding specular surface case because the ray path depends on the refractive index in addition to the dependence on the surface normal. 从畸变中恢复物体形状的方法，对透明物体更加复杂，相比于反光物体而言的话。因为需要考虑到折射率、表面的法向量。
- In computer vision, the problem of refractive surface reconstruction was introduced by Murase [Mur90, Mur92]. 竟然还有两个能够点的上名字的领域最早期的工作者。
- A sequence of distorted images due to water movement is recorded by the camera and analysed using optical flow [HS81, LK81]. 使用光流法计算一系列拍摄的图像，图像拍摄水面及水面下方放置的模板图像，分析畸变。
- undistorted in the sense that refraction is taking place at a planar interface only. 有些时候会考虑折射只发生在介质的交接表面处，而不考虑在非均匀介质中的折射情况。
- The gradient vectors are then integrated to obtain the final surface up to scale. 梯度向量是如何积分的？
- The authors lift several restrictions of Murase’s work by employing a stereo setup and using a known background pattern. With this extended setup it is shown that an unknown refractive index can be recovered in conjunction with accurate per-pixel depth and normal estimates. 通过增加已知的背景板信息，能够恢复出折射率和准确的逐像素深度和法向量的估值。
- The tracking of refracted scene features might be complicated by the fact that refraction results in sometimes severe magnification or minification of the background pattern. Additionally, if the object is not completely transparent, absorption might change the intensity of the observed features, complicating feature tracking. A solution to this problem, an extension to standard optical flow formulations, has been presented by Agarwal et al. [AMKB04]. 折射场景特征会让问题更加复杂，原因是，折射会导致在一些背景模式的扩大或衰减。另外，如果物体并不是完全透明的话，会光线的吸收可能会让观察到的强度衰减。
- Kutulakos and Steger [KS05, KS07] investigate several applications of direct ray measurements. The authors provide a thorough theoretical analysis of reconstruction possibilities based on pixel-independent ray measurements. They categorize reconstruction problems involving refractive and specular surfaces as pairs <N,M,K>,whereN is the number of view-points that are necessary for reconstruction, M is the number of specular or refractive surface points on a piecewise linear light path and K is the number of calibrated reference points on a ray exitant from the object. Kutulakos and Steger [KS05, KS07] 研究了许多射线测量的应用。他们对重建的可能性进行分析，有基于与像素无关的射线度量。奖重建问题分类，包括折射和镜面的承兑问题。N是视点的数目，M是静光路上的折射、反射的分段数量，K是射线上离开物体的标定的参考点。
- [MK05], Section 4.1.1, have already been discussed. The authors investigate the tractability of general <N,M,K>-reconstruction algorithms and show that a pixelwise independent reconstruction is not possible for more than two specular or refractive surface intersections, regardless of the number of input views and the number of reference points on each exitant ray. 这里有一项研究表明，如果有超过两次反射或者折射，无论视角有多少，参考点有多少，都无法重建。
- It is also shown that more than two known points on an exitant ray do not contribute information to the reconstruction problem. 这里所说的exitant ray上的点是什么，也尚不清楚。
- An example of the refractive index distribution above a gas burner is shown in Figure 10. 还有做气体折射率分布的。
- separate the direct reflection component from the indirect lighting effects by exploiting the physical properties of light transport, that is, light travels linearly before hitting the object and there is a radial fall-off of the incident irradiance. 能够通过分析，将直接光和间接光分离开。
- Based on these constraints it is possible to reconstruct very detailed depth and normal maps of refractive objects with complex, inhomogeneous interior, see Figure 11. 三维重建，重建出深度和法向量信息，其实就是早期重建任务的重点了。
- The surface is initialized with the visual hull. 经常见到这种几何初始化的方法。
- The measurement process consists of acquiring four differently polarized images by rotating the linear polarizer in front of the camera. 旋转相机前面的偏振子。
- Under certain circumstances light is not refracted by refractive objects. This is the case if the wavelength of the illumination is sufficiently high, that is, in the case of X-ray illumination, and when the refractive index of the medium surrounding the refractive object is the same as the object’s refractive index. X-ray scanning of refractive objects is straight forward [KTM∗02]. 在一些特定情况下，光并不会被折射物体所折射。当光的波长足够高的时候，也就是说在X射线的照射情况下，当介质和物体的折射率是相同的时候，光不会被折射。折射物体的X射线扫描是直线传播的。
- Refractive index matching is achieved by mixing water with, usually toxic, chemicals. 如果将玻璃放置在相同折射率的环境中，这一般是放置在有毒的化学液体中。
- In [TBH06] potassium thiocyanate is used, solutions of which in water can achieve refractive indices of n ≈ 1.55. 通常使用钾硫氰酸盐，它的水溶液能够让折射率达到1.55。
- If the refractive object is completely transparent, it ideally disappears in a refractive index matched immersing medium. Therefore, it is necessary to dye the surrounding medium in this case. However, if the refractive object is itself absorptive dyeing the surrounding medium can be omitted. The authors acquire 360 images spaced evenly around the object and solve a standard tomographic reconstruction problem. 如果折射的物体是完全透明的，他会消失在折射率匹配的介质中。因此，需要将周围的介质染色。但是，如果折射的物体本身是吸收光的，那么就可以避免将周围的介质染色。
- A related technique that also tries to avoid the refractive properties of transparent objects is investigated by Eren et al. [EAM∗09]: the infrared spectrum is not refracted by glass objects, therefore using an IR laser instead of visible light allows for laser range scanning of glass objects. The method is termed ‘scanning from heating’ because the glass is heated up by the incident IR radiation which is then recorded using an IR-sensitive camera. 另一种试图规避折射率的方法，是采用红外光扫描。
- The resolution of this technique is, however, restricted since the wavelength of the incident illumination is much larger than for visible light and thus cannot be focussed as well. 使用红外光扫描的方法精度有限，因为红外光的波长比较长，导致无法准确的聚焦在某个地方。
- In addition, glass dissipates the heat quite well and blurry impulse responses result. 使用红外党的方法，玻璃会让热量消散，会让激光脉冲的结果模糊。

### 5. Volumetric Phenomena

- Unlike in space carving approaches [KS00] the scene is assumed to be either completely or partially transparent. 体现象中，假定场景是完全透明或者部分透明。
- Furthermore, all methods presented in this section assume that light rays pass straight through the scene and that refractive effects can be neglected. 体现象中，假设光线沿直线传播，不发生折射。


#### 5.1. Tomographic approaches

- 1The light reaching a sensor element is usually a combination of emitted light, and light that is scattered into the direction of the observer. On its way through the volume it is generally subject to attenuation due to out-scatter and extinction. 传播到传感器原件的光，通常是出射光和散射光的组合。通过体的光线通常都遵守衰减率，因为会向外散射或者消失。
- Integral measurements are usually called projections and the task of recovering an n-dimensional function from its (n –1)dimensional projections is known as tomography. 讲积分测量的任务，以及层析成像的技术。

##### 5.1.1. Fire and smoke

- A collection of Gaussian blobs with varying standard deviation is used as a reconstruction basis for the tomographic problem. The blobs are initially evenly distributed. Their positions and standard deviations are then optimized in an iterative manner. 采用高斯球滴的方法，使用变化的标准差作为重建的基准。使用迭代的方法优化。现在其实很少见到说迭代了，很多都是在说训练。
- Reconstruction resolution is 1283 (left panel) and an octree-representation with effective resolution of 2563 (right panel). 八叉树的表示方式很早就有人在用了。


##### 5.1.3. Biological specimen

#### 5.2. Transparency in multi-view stero

- In this subsection we discuss volumetric reconstruction approaches that are deviating from a classical tomographic reconstruction formulation. Most algorithms can be considered as specialized tomographic approaches though. The distinction is thus not strict. Future research could establish links between the methods presented here and classical tomography approaches, potentially leading to more efficient or more accurate reconstruction algorithms in both domains. 当时主流的技术应该还是层析扫描技术。
- The authors formulate the image formation in such scenes by using a formulation similar to environment matting approaches [ZWCS99]. Environment matting技术看起来是无法绕过去的一项传统技术。
- However, smoke as a participating medium exhibits strong global illumination effects such as single and multiple scattering. 烟雾展现出比较强的光照效应，比如一次或者多次散射。

##### 5.3.1 Laser sheet scanning

- In addition to directly measuring volumetric smoke density distributions. 有一种思路，是直接度量烟雾密度分布的。
- Similar to Hawkins et al. [HED05], the pixel values are interpreted as a measure of smoke density. 烟雾的像素值可以解释为烟雾的密度。
- a radial basis function (RBF) style interpolation is found to yield adequate results. 使用辐射基函数作为表示的方法。
- However, because the line of sight stays constant, and the authors assume the smoke to move sufficiently slow for a constant smoke volume assumption, the volume densities are expanded in the dual of the projected basis. 假设烟雾的移动是足够缓慢的，可以假定体雾为常量。

### 6. Conclusions

- Currently there exist approaches that can deal relatively well with different subclasses of objects. However, the algorithms are still very specific and not generally applicable. Furthermore, many techniques require considerable acquisition effort and careful calibration. 处理困难任务的时候，需要使用特殊设计的算法，以及标定方法。
- Except for X-ray tomography, there are as of yet no commercially available scanners for refractive, specular, subsurface scattering or volumetric objects. 除了X射线层析扫描之外，还没有可以处理反光、透明物体的商用技术。
- Glass objects, for example, are seldom solid, at least for objects that are interesting from a rendering perspective. 玻璃物体很少是实心的。
- Especially in the case of refractive objects, it is usually not sufficient to recover the first surface of an object. 重建透明物体的外表面仍然比较困难。
- While it would be desirable from a computer graphics point of view to acquire such objects, at the moment it is not clear how to achieve this goal. 在当时，仍然很难找到方法来解决透明物体的重建工作。
- Rendering techniques can handle much more complex objects than can currently be acquired. Apart from the problem of geometry acquisition, surface reflectance properties have to be estimated simultaneously to achieve high quality renderings. 渲染工作比获取物体能做的事情更多。除了获取物体的几何信息，也需要获取物体的反射特性，以得到更好的渲染效果。
- There exist approaches to capture the surface properties of objects once the geometry is known [LKG∗03, GLL∗04], but the simultaneous acquisition is still challenging even for objects that do not give rise to major global illumination effects [TAL∗07]. 有工作可以在已知几何信息的情况下重建物体的反射信息，但是无法同时重建反射信息和几何信息。
- Even though they arrive at a theoretical result that shows that for light paths intersecting more then two surfaces of the reflective/refractive type a reconstruction of the surface shape is impossible in principle [KS07], this result only holds for per-pixel independent reconstructions. 即时有一些分析认为，重建多次折射的物体是不可能的，但是这些分析是建立在配个像素独立重建的假设上进行的。
- Light field cameras, for example, allow for the capture of a high angular resolution while simultaneously spacing view points very closely. This qualitatively new amount of data would be suitable to analyse difficult objects exhibiting global light transport effects as discussed in this report. 亮度场相机，允许获取光脚分辨率，同时保证空间视点比较近。
- Another promising example, multispectral imaging, could potentially aid in detecting surfaces that are difficult to acquire otherwise, as demonstrated by the infrared range scanning technique of Eren et al. [EAM∗09]. 使用多光谱图像可能有助于获取物体的几何性质。