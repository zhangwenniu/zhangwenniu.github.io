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

# 文章内容

## 摘要

> This paper introduces a new process, environment matting,which captures not just a foreground object and its traditional opacity matte from a real-world scene, but also a description of how that object refracts and reflects light, which we call an environment matte. The foreground object can then be placed in a new environment, using environment compositing, where it will refract and reflect light from that scene. Objects captured in this way exhibit not only specular but glossy and translucent effects, as well as selective attenuation and scattering of light according to wavelength. Moreover, the environment compositing process, which can be performed largely with texture mapping operations, is fast enough to run at interactive speeds on a desktop PC. We compare our results to photos of the same objects in real scenes. Applications of this work include the relighting of objects for virtual and augmented reality, more realistic 3D clip art, and interactive lighting design.

## 介绍

9-231127. Environment Matting and Compositing. 文章发表于1999年的SIGGRAPH上，后被2023年评选为近50年间有影响力文章，收录于Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 文章提出环境抠图技术，提取透明物体的前景之外，同时了解透明物体是如何对光线作用的，并用于该透明物体在新环境下的视角合成。

​10-231128. Environment Matting and Compositing. 本文提出透明前景物体的环境抠图方法，提供出透明前景物体如何与环境光作用，并用于将透明物体放置在新环境下的图像合成。图像拍摄由背景及左右两侧共三面的绿色、洋红色条纹格雷码作为环境光拍摄得到，该色彩选定原因是在RGB空间的正交性及拥有相近的亮度。文章假设成像的像素由前景颜色、按比例透光后的背景颜色、从周围环境中反射或折射所得到的颜色，这三部分加和组成。文中主要解决第三部分，即从周围环境中反射或折射得到的颜色的近似方式，该数值由反射比率函数以及环境纹理图的映射求加权乘积和得到。文章同时展示该方法在带颜色香槟、反光平面、透明棱台、不同深度下的放大镜新视角合成的效果。

# 文章的关键内容

## 1.1 Related work

- Dorsey et al. [10] render a scene under various lighting conditions and then synthesize a new image by taking a linear combination of the ren-derings. In effect, they store at each pixel the contributions from a set of light sources.
- Miller and Mondesir [18] ray trace individual objects and store a ray tree at each pixel for fast compositing into environments consisting of the front and back faces of an environment mapped cube. 做射线追踪的时候，为每个像素都存储一个追踪树，用于快速组合环境映射立方体的前表面和后表面信息。
- However, the scenes are synthetic, and the general effects of glossy and translucent surfaces must be modeled using methods such as distributed ray tracing [7], requiring multiple ray trees per pixel. 一个像素可能需要多个射线追踪树。
- Blue screen matting, pioneered by Vlahos [25], relies on a single-color background sufficiently different from foreground objects. 蓝幕抠图，需要一个与前景显著不同的颜色。
- Reflection from the background onto the foreground, called blue spill,however, remains troublesome even with two backdrops and results in incorrect object transparency after matte extraction. 从背景反射到前景的问题，称作蓝色溢出，但是，即使采用两个背景，仍然会导致透明物体的抠图提取错误。
- To acquire our environment matte, we could illuminate one point of light at a time and sweep over the environment around the object. 为了获得环境抠图，我们可以在一个时间照明一个点光源，沿着物体周围划过一圈。
- By projecting a hierarchy of progressively finer stripe patterns, the required number of images can be reduced to O(log n) [21]. 使用结构化的渐进带状条纹，需要的图片数目可以缩减。
- Our environment matting approach is based on the hierarchical stripe methods. 本文的环境抠图方法，基于结构带方法。

# 2 The environment matte

- We begin with the traditional compositing equation and then augment it with a new structure, the environment matte, which captures how light in the environment is refracted and reflected by a foreground element. 从一个从传统的组合方程开始，增强该方程以一个新的结构，环境抠图，这获取环境中的光线是如何被前景元素所折射和反射的。
- To start, we will assume that the only light reaching the foreground object is light coming from distant parts of the scene. This is essentially the “environment mapping assumption.”  我们假设只有来自远处的光线会到达前景物体。这是基本上的“环境映射假设”。

## 3.3 The area extents and a refined estimate of coverage

- This optimization problem still has four degrees of freedom for the unknown rectangle A1: left, right, top, and bottom (l, r, t, b). We can reduce the dimensionality further by using horizontal and vertical stripes for backgrounds. For horizontally striped backgrounds, the area determination is independent of l and r; similarly, for vertically striped backgrounds, the area determination is independent of t and b. 

# 论文难点分析【kimi】

Kimi: 这篇论文《Environment Matting and Compositing》由Douglas E. Zongker等人撰写，主要介绍了一种新的图像处理技术，称为环境遮罩（environment matting），它能够捕捉前景物体及其对光线的折射和反射特性，并将这些特性用于新的环境合成（environment compositing）中，以实现更真实的视觉效果。下面我将为你详细解读这篇论文的核心内容：

### 背景知识
- **Matting和Compositing**：在图形学中，matting过程是从背景图像中提取任意形状的前景元素，并描述其在每个点的不透明度。compositing过程则是将前景元素放置在新的背景图像上，使用遮罩来遮挡前景元素遮挡的新背景部分。传统的matting和compositing技术在电影、视频和计算机图形制作中被广泛应用，但它们未能模拟真实世界中的两个关键效果：**折射**和**反射**。

### 研究方法
- **Environment Matte**：作者提出了一种新的结构，称为环境遮罩，它不仅捕捉前景物体的颜色和不透明度，还描述了物体如何折射和反射环境中的光线。环境遮罩允许前景物体在新环境中产生折射和反射效果，包括光泽和半透明效果，以及根据波长选择性衰减和散射光线。
- **Environment Compositing**：通过环境遮罩，可以将前景物体合成到新环境中，并且能够快速运行在桌面PC上，实现交互式速度。

### 实验和结果
- **实验设置**：作者构建了一个实验系统，使用数字相机和多个显示器来获取环境遮罩。他们将三个显示器放置在与立方体环境的三个面一致的位置，并使用20英寸的Trinitron显示器，调整显示以使洋红色和绿色图像尽可能一致。使用Kodak DCS 520数字相机拍摄物体，并循环显示每个显示器上的图像。
- **结果**：通过环境遮罩技术，作者成功地将玻璃烛台、玻璃书挡等物体合成到新的背景中，展示了复杂的折射和反射效果。例如，玻璃烛台的两个凸起产生了复杂的折射图案，环境遮罩能够准确地重现这些效果。此外，环境遮罩还能够处理彩色透明度，更准确地再现背景颜色通过有色物体过滤后的效果。

### 关键结论
- **环境遮罩的有效性**：环境遮罩能够捕捉并重现物体的折射和反射特性，使得合成的图像更加真实。通过与传统alpha通道合成的对比，环境遮罩在处理透明和反光物体时表现出明显的优势。
- **交互式速度**：环境合成过程主要使用纹理映射操作，可以在桌面PC上以交互式速度运行，这对于虚拟和增强现实应用尤为重要。
- **应用前景**：这项技术可以应用于虚拟和增强现实中的对象重照明、更真实的3D剪贴画以及交互式照明设计。

### 未来工作
- **深度校正**：作者提出了一种方法，可以捕捉不同深度的环境遮罩，从而在合成时考虑物体与背景之间的相对深度。这需要在不同深度的背景下拍摄物体，并通过线性插值来构造任意深度的矩形区域。
- **改进方法**：作者计划进一步研究如何减少捕获环境遮罩所需的图像数量，以及如何更准确地处理光泽和半透明表面的反射和折射。

### 总结
这篇论文的核心贡献在于提出了一种新的环境遮罩技术，能够捕捉并重现物体的复杂光学特性，使得在新环境中的合成效果更加真实。通过实验验证，作者展示了该技术在处理透明和反光物体时的优越性能，并探讨了其在虚拟和增强现实中的应用潜力。未来的工作将集中在提高捕获过程的效率和准确性，以及扩展技术的应用范围。


# 方法部分详解【kimi】

#### 1. 环境遮罩（Environment Matting）

环境遮罩的过程旨在从真实场景中提取前景物体及其对光线的折射和反射特性。具体步骤如下：

1. **使用结构化光系统**：作者使用了结构化光系统来捕捉物体的折射和反射特性。物体被放置在多个显示器前，这些显示器显示不同的图案（称为背景幕和侧幕）。每个图案都被拍摄两次：一次是没有物体的参考图像，一次是有物体的对象图像。

2. **提取参数**：通过比较参考图像和对象图像，提取每个像素的前景颜色 $ F $、像素覆盖值 $ \alpha $ 和环境遮罩参数 $ \{R_i, A_i\} $。这些参数描述了物体如何折射和反射环境中的光线。

3. **分阶段处理**：为了简化问题，作者将提取过程分为四个阶段：
   - **粗略估计覆盖度**：首先将每个像素分类为覆盖或未覆盖。使用形态学操作（如开运算和闭运算）来清理 alpha 通道，去除孤立的覆盖或未覆盖像素。
   - **确定前景颜色和反射系数**：对于覆盖的像素，使用两个不同颜色的背景幕图像来求解前景颜色 $ F $ 和反射系数 $ R_1 $。
   - **确定区域范围和精细估计覆盖度**：通过最小化目标函数来确定背景区域的范围 $ A_1 $，并精细估计边界像素的 $ \alpha $ 值。
   - **处理侧幕**：通过类似的方法处理侧幕，以捕捉来自不同方向的光线。

#### 2. 环境合成（Environment Compositing）

环境合成的过程将提取的环境遮罩应用于新的背景图像，以实现真实的折射和反射效果。具体步骤如下：

1. **实现环境合成方程**：使用提取的环境遮罩参数，通过环境合成方程 $ C = F + (1 - \alpha) B + \sum_{i=1}^m R_i M(T_i, A_i) $ 来合成新的图像。其中，$ C $ 是合成后的颜色，$ F $ 是前景颜色，$ B $ 是背景颜色，$ R_i $ 是反射系数，$ M(T_i, A_i) $ 是纹理映射操作。

2. **纹理映射**：为了提高合成速度，作者使用了快速的纹理映射技术，如求和面积表（summed area tables），以避免 aliasing 问题。对于具有放大效果的物体，使用更高分辨率的纹理映射来存储背景图像。

3. **深度校正**：为了处理不同深度的背景，作者提出了一种方法，通过在不同深度的背景下拍摄物体，提取多个环境遮罩，并使用线性插值来构造任意深度的背景区域。

### 关键数值结果和方法细节

- **提取过程的时间**：提取过程大约需要 10 到 20 分钟每张环境图，运行在 Intel Pentium II 400MHz 的计算机上。
- **合成速度**：合成过程非常快，可以达到每秒 4 到 40 帧的速度。
- **纹理映射的分辨率**：为了处理 lenticular 物体，使用了比显示分辨率更高的纹理映射分辨率。

### 总结

这篇论文的方法部分详细介绍了如何通过结构化光系统和数学优化方法来提取环境遮罩，并通过纹理映射技术实现高效的环境合成。这种方法能够捕捉物体的复杂光学特性，并在新的环境中实现真实的视觉效果。


# 算法详解【kimi】

1. **环境遮罩方程**：
   环境遮罩方程是文章的核心内容，它将传统的图像合成方程扩展到能够处理折射和反射效果。传统的数字合成方程为：
   $$
   C = F + (1 - \alpha)B
   $$
   其中，$ C $ 是最终合成的颜色，$ F $ 是前景元素的颜色，$ B $ 是背景的颜色，$ \alpha $ 是前景元素的不透明度。在环境遮罩中，此方程被扩展为：
   $$
   C = F + (1 - \alpha)B + \sum_{i=1}^m R_i \cdot M(T_i, A_i)
   $$
   其中，$ R_i $ 是反射和折射系数，$ M(T_i, A_i) $ 是对纹理映射 $ T_i $ 在区域 $ A_i $ 上的平均值。这个方程表明，最终的合成颜色不仅取决于前景和背景颜色，还取决于从环境中反射和折射的光线。

2. **反射和折射系数的计算**：
   为了计算反射和折射系数 $ R_i $，文章提出了一种优化方法。通过将物体放置在带有结构化图案的背景幕和侧幕前，并拍摄多张图像，可以提取出 $ R_i $ 的值。例如，对于背景幕 $ T_1 $，通过比较物体在不同背景颜色下的图像，可以得到：
   $$
   R_1(\alpha) = \frac{C - C'}{B - B'} - (1 - \alpha)
   $$
   其中，$ C $ 和 $ C' $ 是物体在不同背景颜色 $ B $ 和 $ B' $ 下的像素颜色。

3. **纹理映射和区域估计**：
   文章使用纹理映射和区域估计来捕捉物体对环境光线的反射和折射。通过将环境光线分解为多个纹理映射 $ T_i $，可以分别计算每个纹理映射在物体上的贡献。区域 $ A_i $ 表示纹理映射 $ T_i $ 在物体上的有效区域。为了估计区域 $ A_i $，文章使用了结构化光技术，并通过优化目标函数来确定最佳的区域范围。

4. **高斯加权纹理映射**：
   为了处理具有光泽表面的物体，文章提出了一种高斯加权的纹理映射方法。与传统的箱式滤波器不同，这种方法使用高斯权重来计算纹理映射的平均值。这可以更好地近似现实世界中物体对光线的反射和折射。

### 算法公式及含义

1. **环境遮罩方程**：
   $$
   C = F + (1 - \alpha)B + \sum_{i=1}^m R_i \cdot M(T_i, A_i)
   $$
   这是核心方程，用于合成新环境中的物体图像。各项的含义如下：
   - $ C $：最终合成的像素颜色。
   - $ F $：前景物体的像素颜色。
   - $ \alpha $：前景物体的不透明度（覆盖度）。
   - $ B $：背景图像的像素颜色。
   - $ R_i $：表示物体反射和折射光线的系数。
   - $ M(T_i, A_i) $：对纹理映射 $ T_i $ 在区域 $ A_i $ 上的平均值。

2. **反射和折射系数的计算**：
   $$
   R_1(\alpha) = \frac{C - C'}{B - B'} - (1 - \alpha)
   $$
   该公式用于计算背景幕 $ T_1 $ 的反射和折射系数。其中，$ C $ 和 $ C' $ 是物体在不同背景颜色 $ B $ 和 $ B' $ 下的像素颜色。通过比较这两个图像，可以分离出物体的反射和折射特性。

3. **目标函数的优化**：
   用于确定区域 $ A_i $ 和反射系数 $ R_i $ 的目标函数如下：
   $$
   E = \sum_{j=1}^n \left\| C_j - \left( F(\alpha) + (1 - \alpha) B_j + R_1(\alpha) \cdot M(T_j, A_1) \right) \right\|^2
   $$
   其中，$ C_j $ 和 $ B_j $ 分别是物体图像和参考图像的像素颜色。目标函数通过最小化多个图像的误差来优化参数 $ \alpha $、$ R_i $ 和区域 $ A_i $。

### 算法的改进和扩展

1. **高斯加权纹理映射**：
   为了处理具有光泽表面的物体，文章提出了一种高斯加权的纹理映射方法。这可以更好地近似现实世界中物体对光线的反射和折射，从而提高合成图像的质量。

2. **深度校正**：
   文章提出了一种深度校正的方法，用于处理不同深度的背景。通过捕捉不同深度的环境遮罩，并使用线性插值来构造任意深度的背景区域，可以合成出更逼真的图像。

3. **多视图环境遮罩**：
   文章还探讨了捕捉多个视图下的环境遮罩的方法。通过采集不同视图下的图像，并使用视图插值技术，可以生成具有逼真反射和折射效果的动画。
