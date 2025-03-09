---
layout: mypost
title: 055, G044 Environment Matting and Compositing
categories: [透明, 折射]
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

```bibtex
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




# 核心内容讲解

这篇论文《Environment Matting and Compositing》由Douglas E. Zongker等人撰写，主要介绍了一种新的图像处理技术，称为环境遮罩（environment matting），它能够捕捉前景物体及其对光线的折射和反射特性，并将这些特性用于新的环境合成（environment compositing）中，以实现更真实的视觉效果。下面我将为你详细解读这篇论文的核心内容：

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


# 方法部分详解

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

# 算法详解

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


# 原文翻译

## 环境遮罩和合成
Douglas E. Zongker, Dawn M. Werner, Brian Curless, David H. Salesin
1华盛顿大学 2微软研究

### 摘要
本文介绍了一种新的过程，称为环境遮罩（environment matting），它不仅从真实场景中捕捉前景物体及其传统的不透明遮罩，还捕捉了该物体如何折射和反射光线的描述，我们称之为环境遮罩。然后可以使用环境合成（environment compositing）将前景物体放置在新环境中，在该环境中它将折射和反射来自该场景的光线。以这种方式捕捉的物体不仅表现出镜面反射，还表现出光泽和半透明效果，以及根据波长选择性地衰减和散射光线。此外，环境合成过程主要通过纹理映射操作进行，足够快，可以在桌面PC上以交互速度运行。我们将结果与真实场景中相同物体的照片进行比较。该工作的应用包括虚拟和增强现实中的物体重照明、更真实的3D剪贴画和交互式照明设计。

### 1 引言
遮罩和合成是图形学中的基本操作。在遮罩过程中，从背景图像中提取任意形状的前景元素。通过此过程提取的遮罩描述了前景元素在每个点的不透明度。在合成过程中，前景元素被放置在新的背景图像上，使用遮罩来遮挡前景元素遮挡的新背景部分。遮罩和合成最初是为电影和视频制作开发的[11]，例如，经常用于将演员的图像（在工作室中拍摄，背景受控）放置到另一个环境中。1984年，Porter和Duff[20]引入了遮罩的数字模拟——alpha通道，并展示了带有alpha的合成图像在创建复杂数字图像中的用途。1996年，Smith和Blinn[25]描述了从真实场景中提取alpha通道的数学框架，给定一对背景图像或前景元素颜色的某些假设。

尽管遮罩和合成在电影、视频和计算机图形制作中证明了其巨大的用途，但它们仍然未能模拟两个对现实感至关重要的关键效果：
- 透明物体表现出的折射；
- 在掠角处看到的光亮物体和物体表现出的反射。

此外，对于半透明或光泽材料，折射和反射进一步与各种光散射效应耦合。对于有色材料，折射和反射还可能表现出根据波长选择性衰减和散射光线的现象。

在本文中，我们将传统的遮罩和合成过程推广到包括所有这些效果。我们称之为环境遮罩和合成的结果过程，不仅能够捕捉前景物体及其传统遮罩，还能描述该物体如何折射和反射光线，我们称之为环境遮罩。然后可以将前景物体放置在新环境中，在该环境中它将折射和反射来自该场景的光线。虽然我们捕捉的环境遮罩仅提供了物体如何真正折射和反射光线的近似，但我们发现它们在真实场景中产生了令人信服的结果。此外，我们的环境遮罩表示使得环境合成主要可以通过简单的纹理映射进行。因此，对于合理大小的环境，这个过程可以在廉价的PC上以交互速度进行。图1展示了以这种方式捕捉的水杯并合成到两个新背景中。

### 1.1 相关工作
本文描述的工作结合了计算机图形学中许多不同领域的研究。

研究人员通过多种方式增强像素，以实现更灵活的图像合成。Porter和Duff的每个像素的alpha允许图像以层的形式获取或渲染，然后组合[20]。层可以独立修改，然后简单地重新组合。如上所述，他们的合成方法不考虑反射、折射或有色透明度。

Gershbein[12]通过深度、表面法线和着色参数增强像素，然后为可以以交互速度移动的光源执行每像素着色计算。Dorsey等人[10]在各种照明条件下渲染场景，然后通过对渲染结果进行线性组合来合成新图像。类似地，Nimeroff等人[19]计算在日光条件下预渲染的图像的线性组合。通过适当地近似日光的角度分布，他们构建了可操控的基函数，以模拟任意太阳位置的日光。该方法已在具有漫反射表面和光滑变化的照明场景中得到验证。

几位研究人员探索了在缓存结果的同时进行光线追踪的方法，以实现高效的重新渲染。S´equin和Smyrl[23]光线追踪场景并在每个像素存储着色表达式。在修改场景中物体的一些简单着色参数后，重新评估着色表达式以生成图像，而无需投射任何新光线。Bri´ere和Poulin[3]通过在光线树中存储光线路径的几何信息扩展了这一思想。重新着色可以像以前一样进行，但通过高效更新光线树可以快速处理场景几何的变化。Miller和Mondesir[18]光线追踪单个物体，并在每个像素存储光线树，以便快速合成到由环境映射立方体的前后表面组成的环境中。这些方法受益于光线追踪的通用性，并模拟了有色反射和折射等现象。然而，这些场景是合成的，光泽和半透明表面的整体效果必须使用分布式光线追踪[7]等方法建模，需要每个像素多个光线树。

与增强像素颜色的合成场景相比，从真实图像中获取信息的任务更加复杂。蓝屏遮罩，由Vlahos[25]开创，依赖于与前景物体充分不同的单色背景。如上所述，Smith和Blinn[25]使用两个背景来解除前景物体颜色的限制。然而，即使使用两个背景，背景对前景的反射（称为蓝色溢出）仍然是个麻烦，并导致遮罩提取后物体透明度不正确。我们的方法试图将这种反射正确建模为从背景重新定向的光线。

为了获取我们的环境遮罩，我们可以一次点亮一个光点并扫描物体周围的环境。覆盖这个区域需要$O(n^2)$张图像，其中$n^2$与物体周围环境的面积成正比。相反，我们从结构光范围扫描文献中获得灵感。使用扫描光平面，$O(n)$张图像可以通过光学三角测量获取形状[1]。通过投影逐渐细化的条纹图案，所需图像数量可以减少到$O(\log n)$[21]。使用强度梯度，理论上可以将图像数量减少到两张[5]；然而，这种方法对噪声高度敏感。提出了混合方法来管理图像数量和噪声敏感性之间的权衡[6, 16]。我们的环境遮罩方法基于分层条纹方法。然而，结构光测距仪假设物体是漫反射的，并试图将传感器的视线与投影光源的视线相交。在我们的情况下，表面通常假设相对光亮，甚至是折射的，传感器的视线可以任意重新映射到漫反射结构光图案上。

最后，已经开发了许多方法来将反射和折射物体渲染到环境中。Blinn和Newell[2]引入了环境（即反射）映射，以模拟几何物体对存储为纹理的环境的反射。Greene[14]改进了这一思想，以模拟一些有限的着色效果。Voorhies和Foran[26]后来展示了使用纹理映射硬件的交互式环境映射。此外，Kay和Greenberg[17]开发了一种快速渲染折射物体的近似方法。这些方法基于物体几何，并且据我们所知，没有一种方法能够准确地表示折射或在交互速度下模拟空间变化的半透明和光泽。


Miller和Mondesir的超精灵渲染[18]与我们的渲染方法最接近；然而，它使用超采样进行抗锯齿。通过使用求和面积表[8]，我们可以实现纹理抗锯齿，以及在交互速度下实现显著的光泽或半透明效果。

### 1.2 概述
接下来的三个部分依次讨论：我们的新表示方法，用于捕捉前景元素的折射和反射属性的环境遮罩（第2节）；从真实场景中提取环境遮罩的环境遮罩过程（第3节）；以及将前景元素放置到新环境中的环境合成过程（第4节）。对主要结果的讨论（第5节）之后，概述了一些初步结果，这些结果扩展了我们的方法，允许在合成时设置物体和背景的相对深度（第6节）。我们以一些未来研究的想法结束（第7节）。


### 2 环境遮罩（The Environment Matte）

我们从传统的合成方程开始，然后引入一个新的结构——环境遮罩，用于捕捉环境中的光线如何被前景元素折射和反射。

在传统的数字合成中，将前景元素放置在背景上的颜色$C$由“遮罩方程”给出[^25^]，在每个像素处计算如下：

$$
C = F + (1 - \alpha) B
$$

在传统的遮罩方程中，元素的遮罩（或alpha）一直扮演着双重角色：它同时用于表示前景元素覆盖像素的程度，以及该元素的不透明度。在我们的方法中，我们仅用alpha表示覆盖情况。因此，没有前景元素的像素的alpha值为0，而完全被覆盖的像素的alpha值为1——即使前景元素覆盖该像素是半透明或不透明的。同样，alpha的分数值用于表示前景元素的部分覆盖——再次强调，无论该元素的透明度如何。

类似地，在传统合成中，元素的颜色可以被视为几种不同成分的组合。这些包括：

1. 前景物体可能具有的任何发射成分；
2. 来自场景中光源的任何反射；
3. 以及前景物体被拍摄时所在环境中其余部分的任何额外反射或光线透射。

在我们的方法中，元素的前景颜色仅用于表征上述前两个成分。

```markdown
另一种方法是仅以前景色的形式表示发射组件，并将任何光源视为环境的一部分。然而，由于光源通常比其周围环境亮几个数量级，我们发现将其贡献分离出来更为实际——这与着色器通常独立考虑光源的贡献与其他环境因素的方式大致相同。（一个有趣的选择是使用Debevec和Malik[9]引入的“高动态范围图像”）。
```

此外，我们引入了一个新的结构——环境遮罩，用于捕捉环境中的光线如何被前景元素折射和反射。因此，环境遮罩将捕捉前景元素的任何透射效应——以一种与其覆盖情况或alpha无关的方式。得到的“环境遮罩方程”将具有以下形式：

$$
C = F + (1 - \alpha) B + \Phi
$$

其中，$\Phi$表示任何来自环境的光线反射或折射通过前景元素的贡献。

我们对环境遮罩的表示方法有以下两个要求。首先，该表示方法应支持快速合成算法。因此，我们选择将环境表示为一组纹理映射（类似于“环境映射”[^2^]）。因此，我们的环境遮罩包含指向这些纹理映射的索引，以及一组附加参数。其次，我们希望该表示方法相对简洁——即，每个环境映射仅需要少量、固定的数据。

因此，我们从一般公式开始，描述环境中光线通过前景元素在每个像素处的传输，然后推导出满足我们要求的近似方法。这个推导将使我们能够明确地指出每个误差来源，并在描述近似方法时对其进行特征化。

首先，我们假设到达前景物体的唯一光线来自场景的远处部分。这本质上是“环境映射假设”。我们使用这个假设来创建以下简化的光传输模型。

与Blinn和Newell的原始公式[^2^]类似，我们可以将环境描述为来自所有方向$\omega$的光线$E(\omega)$。那么，从前景元素的可见部分$f$通过给定像素到达点$p$的总光线量$\Phi$可以描述为对环境贡献到像素中点$p$的光线的积分，该积分被某个反射函数$R(\omega \rightarrow p)$衰减：

$$
\Phi = \int \int R(\omega \rightarrow p) E(\omega) \, d\omega \, dp
$$

注意，反射函数$R(\omega \rightarrow p)$包括通过前景元素看到的所有吸收和散射的总体效应。此外，$\Phi$、$R$和$E$都隐含地依赖于波长。

我们的第一个近似假设是反射函数$R(\omega \rightarrow p)$实际上在给定像素的覆盖区域内是常数，这使我们能够用一个与像素内位置无关的新函数$R(\omega)$来表示这个公式：

$$
\Phi = \int R(\omega) E(\omega) \, d\omega
$$

接下来，我们将环境的积分分解为对一组$m$个纹理映射$T_i(x)$的求和，每个纹理映射代表来自环境不同部分的光线（例如，来自立方体的不同侧面）：

$$
\Phi = \sum_{i=1}^{m} \int R_i(x) T_i(x) \, dx
$$

这里，积分是在每个纹理映射的整个面积上进行的，$R_i(x)$是一个新的反射函数，描述了来自纹理映射$T_i$上点$x$的光线对像素$p$的贡献。

最后，我们再做一个简化假设：来自纹理映射$T_i$的贡献可以用某个常数$K_i$乘以纹理映射中某个轴对齐矩形区域$A_i$内的总光量来近似。对于许多镜面表面，这种近似在实践中是合理的。我们在第5节中讨论了它的局限性。

大多数标准的纹理映射方法实际上计算轴对齐区域的纹理的平均值，因此我们设$R_i = K_i A_i$。设$M(T_i, A_i)$是一个纹理映射算子，它返回纹理$T_i$中轴对齐区域$A_i$的平均值，我们有：

$$
\Phi = \sum_{i=1}^{m} K_i \int_{A_i} T_i(x) \, dx \\
= \sum_{i=1}^{m} K_i A_i M(T_i, A_i) \\
= \sum_{i=1}^{m} R_i M(T_i, A_i)
$$

我们使用上述近似来表示$\Phi$。因此，我们的总体“环境遮罩方程”变为：

$$
C = F + (1 - \alpha) B + \sum_{i=1}^{m} R_i M(T_i, A_i)
$$

（注意，反射系数$R_i$本质上是“预乘”的”，因为它们不需要乘以元素像素的覆盖度$\alpha$。这个结果来自于我们对$\alpha$的早期定义，我们只对像素的覆盖部分进行积分。）

如上所述，每个波长的光最好独立处理。然而，我们使用标准计算机图形学近似，将光视为只有三个颜色分量：红色、绿色和蓝色。因此，在我们的实现中，量$\Phi$、$R_i$和$T_i$被视为3分量向量，而$T_i$是一个二维数组，类型相同。环境遮罩方程中的加法和减法被视为逐分量的加法和减法。

图1 一个水杯，以数字方式合成到背景图像上，保留了折射效果。

图2 环境遮罩处理使用结构化纹理来捕捉光线如何从背景（右侧轴）以及不同侧面（左侧轴）反射和折射。该过程还捕捉穿过像素未覆盖部分看到的来自背景的光（中间轴）。

图3 捕捉环境遮罩所使用的实验设置的照片。前景中的相机拍摄被结构化光图案包围的物体，这些光图案显示在计算机监视器上。图像经过处理以仅提取被背景图案覆盖的区域，结果如插图所示。

图4 从左至右：一个alpha遮罩合成，一个环境遮罩合成，以及一个物体前的背景图像照片。顶行展示了一个有肋纹的玻璃烛台；底行展示了一个粗糙表面的玻璃书立。

图5 从左至右：一个alpha遮罩合成，一个环境遮罩合成，以及一个物体前的背景图像照片。顶行展示了三杯分别染成红色、绿色和蓝色的水；底行展示了一个倾斜以反射来自背景光的馅饼盘。

图6 使用环境遮罩捕获的具有光泽反射的物体。从左至右：当采集过程中使用箱式滤波近似时复合图像中产生的伪影，使用高斯近似时的改进，以及实际照片。

图7 使用多侧面捕获的环境遮罩。左上图显示了前景颜色和未覆盖像素项的效果。接下来的三个图像显示了通过背面、左侧和右侧纹理映射反射来的光。最终的合成是这四幅图像的总和，如左下角所示。这与右下角物体周围环绕这些图像的照片进行了比较。

图8 使用光源的照片作为侧边来互动地重新照明物体。左和右侧面（如插图所示）是通过在纹理映射内移动光源图片生成的。

图9 失败案例。

图10 在新颖深度处渲染的环境遮罩。(a)-(d)部分是在四个不同深度处将放大镜与背景合成的图像。所有这四个深度都不同于原始物体图像中的背景深度。为了比较，(e)和(f)分别是在对应于(a)和(d)的深度处拍摄的。当背景向后移动时，请注意放大幅度如何增加直到图像水平和垂直翻转。反转发生时的深度对应于焦点的存在，超过这一点，放大倍数减小。


### 3 环境遮罩（Environment Matting）

在这里，我们考虑从真实世界物体的照片中提取前景颜色$F$、像素覆盖值$\alpha$和环境遮罩$\Phi = \{(R_1, A_1), \dots, (R_m, A_m)\}$的问题。

我们的方法受到结构光系统用于获取深度的启发。然而，与这些先前的系统不同，这些系统在单个点上采样物体，我们希望捕捉所有通过像素的光线是如何被散射到环境中的。（实际上，我们对相反的情况更感兴趣——即环境中所有光线是如何被散射到单个像素朝向眼睛的——但以“反向光线追踪”的方式思考更容易[^13^]。）

为此，我们在前景物体的后面和侧面放置一些带有图案的纹理，我们称之为背景幕和侧幕（见图2）。每个图案化的背景幕都被拍摄两次：一次是没有前景物体的背景幕图像，一次是前景物体在背景幕前面的图像。在下文中，我们将没有物体的背景幕图像称为参考图像，将前景物体在背景幕前面的图像称为物体图像。然后，我们求解一个非线性优化问题，以确定与所有图像数据最一致的一组参数。我们使用只在一个维度上变化的图案，从而将寻找矩形的问题分解为寻找两个一维区间的任务。理论上，任何线性独立的水平和垂直条纹排列都应该适用于我们的图案序列。然而，在实践中，我们觉得使用连贯的图案有助于减少轻微错位导致错误的可能性。因此，我们选择了对应于一维格雷码的条纹图案。我们选择了品红色和绿色的条纹，因为这两种颜色在RGB空间中是正交的，并且具有相似的亮度。我们的技术还需要拍摄物体在两个纯色背景幕前的视图。

环境遮罩问题的维度不幸地相当大：前景颜色$F$和每个反射系数$R_i$（每个颜色分量一个）有三个自由度，每个区域范围$A_i$有四个自由度，$\alpha$还有一个自由度。为了使这个提取问题更易于处理，我们将解决方案分为四个阶段。我们首先只考虑背景幕——直接位于物体后面的环境面，按照惯例，我们将它编号为环境纹理列表中的第一个。在第一阶段，我们使用不同的背景幕来计算$\alpha$的粗略估计。然后，我们确定任何被前景元素覆盖的像素的$F$和$R_1$。接下来，我们求解$A_1$，以及沿着元素轮廓的$\alpha$的更精细估计。最后
，一旦我们找到了$F$、$\alpha$、$R_1$和$A_1$，我们就可以确定其他环境面的$R$和$A_i$值。

#### 3.1 粗略估计覆盖度（A coarse estimate of coverage）

我们首先计算每个像素的覆盖度的粗略估计。我们首先将环境遮罩中的每个像素划分为两类：覆盖的和未覆盖的。如果在任意一张背景图像中，参考图像和对应的物体图像的颜色差异超过一个小值$\epsilon$，则认为该像素是覆盖的。接下来，我们使用形态学操作[^24^]（先进行开运算，再进行闭运算，均使用5×5的方块作为结构元素）来清理得到的alpha通道，移除任何孤立的覆盖或未覆盖像素。

未覆盖的（背景）像素将被赋予0的alpha值。覆盖的像素需要进一步划分为前景像素（其alpha值为1）和位于物体轮廓边界上的像素（我们需要为这些像素确定一个分数alpha值）。我们通过腐蚀（erode）第一步得到的二值图像，并从未经腐蚀的图像中减去腐蚀后的图像来完成第二次划分。边界像素的分数alpha值的计算将在下面第3.3节中讨论的矩形估计过程中同时进行。

我们也可以选择另一种方法，即独立地在每个像素处确定alpha值，从条纹背景幕中计算——实际上，将每个像素都视为边界像素，并计算该像素的最佳alpha值以及它的矩形。这正是我们最初采用的方法，但在处理真实照片数据时，总会不可避免地出现一些单像素误差。这些误差在静态合成图像中通常不明显，但当合成物体相对于背景移动时，这些误差会非常突出。它们会表现为与物体同步移动的“灰尘颗粒”，或者背景透过来的微小“孔洞”。我们的形态学方法虽然牺牲了一些数学上的优雅性，但能够提供更干净的结果。此外，由于它能够以较低的成本确定图像中大多数像素的alpha值，因此使用形态学方法大大加快了采集过程。

#### 3.2 前景颜色和反射系数（The foreground color and reflectance coefficients）

对于覆盖的像素（$\alpha > 0$的像素），我们需要确定前景颜色$F$以及每个环境面的反射系数$R_i$。为此，我们首先拍摄物体在两个不同纯色背景幕前的照片，并求解$R_1$和$F$。

对于给定的像素，设$B$是第一个背景幕从相机看到的颜色，$B'$是第二个背景幕的颜色。设$C$和$C'$分别是物体在这两个背景幕前时，相机看到的前景物体的颜色。这四个颜色通过环境遮罩方程（1）关联起来：

$$
C = F + (1 - \alpha)B + R_1 B
$$

$$
C' = F + (1 - \alpha)B' + R_1 B'
$$

现在我们有两个方程和两个未知数，可以很容易地求解$R_1$和$F$，它们可以表示为$\alpha$的函数：

$$
R_1(\alpha) = \frac{C - C'}{B - B'} - (1 - \alpha) \quad (2)
$$

$$
F(\alpha) = C - (1 - \alpha + R_1)B \quad (3)
$$

#### 3.3 区域范围和覆盖度的精细估计（The area extents and a refined estimate of coverage）

为了对边界像素的alpha值进行精细化，并确定最能近似场景中反射和折射的背景区域$A_1$，我们对每个覆盖像素的拍摄图像序列最小化以下目标函数：

$$
E_1 = \sum_{j=1}^{n} \left\| C^j - \left[ F(\alpha) + (1 - \alpha)B^j + R_1(\alpha) M(T_1^j, A_1) \right] \right\|^2
$$

这里，$B^j$和$C^j$分别是参考图像和物体图像中该像素的颜色，当第$j$个图案作为背景幕显示时。类似地，纹理映射$T_1^j$是通过将第$j$个图案作为背景幕拍摄得到的参考照片获得的。函数$F(\alpha)$和$R_1(\alpha)$是根据上面的公式（2）和（3）计算的。最后，平方的大小是通过计算RGB空间中颜色之间的平方差之和得到的。我们的目标是找到最小化目标函数$E_1$的矩形区域$A_1$[^2^]。【请注意，如果使用轴对齐矩形作为光传输行为的精确模型，那么最小化这个目标函数将对应于为像素值赋予高斯误差模型，并以“最大似然”的方式求解最优参数。实际上，轴对齐矩形只是一个近似模型，因此我们实际上是在寻找模型参数对数据的最小二乘最佳拟合。】

这个优化问题仍然有四个自由度，用于未知矩形$A_1$：左、右、顶、底（$l, r, t, b$）。我们可以通过使用水平和垂直条纹背景幕来进一步降低维度。对于水平条纹背景幕，区域的确定与$l$和$r$无关；类似地，对于垂直条纹背景幕，区域的确定与$t$和$b$无关。因此，原本五维的最小化问题$E_1$（在$\alpha, l, r, t, b$上）可以有效地分解为两个三维问题：在一组图案上对（$\alpha, l, r$）进行最小化，在另一组图案上对（$\alpha, t, b$）进行最小化。

我们首先假设一个$\alpha$的值：对于前景像素，$\alpha = 1$；而对于边界像素，我们将尝试多个值。然后，我们通过测试大量候选区间，寻找在垂直条纹图案上最小化$E_1$的区间$[l, r]$。为了加快这一搜索过程，我们采用多分辨率技术，在某个粗略尺度上找到最佳区间，然后反复细分，寻找更窄区间中的更好近似值。我们使用相同的技术处理水平条纹图案，以找到垂直范围$[t, b]$，从而确定矩形。对于边界像素，我们对多个$\alpha$值重复这一搜索，并输出使目标函数最小化的$\alpha$和区间$A_1 = (l, r, t, b)$。当然，这种更大范围的搜索会显著减慢速度，但它仅在物体轮廓处需要进行。此外，对$\alpha$的粗略搜索就足够了——我们发现尝试九个值，$\alpha \in \{0, \frac{1}{8}, \frac{2}{8}, \dots, 1\}$，就能得到很好的结果。

#### 3.4 侧幕（Sidedrops）

我们描述的技术允许我们捕捉来自相机正后方背景幕的光线如何被物体折射和反射。为了捕捉来自环境其他部分的环境遮罩，我们在拍摄物体时，用相同的结构化图案照亮侧幕，而不是背景幕。

提取过程几乎相同。我们不再需要计算$\alpha$或$F$；这些值与之前相同。$R$的计算也稍微简单一些。当两个不同的纯色$S$和$S'$显示在侧幕上时，对应的环境遮罩方程变为：

$$
C = F + (1 - \alpha)B_1 + R_i S
$$

$$
C' = F + (1 - \alpha)B_1 + R_i S'
$$

这里，$B_1$是背景幕的颜色（通常接近黑色）。将这两个方程相减，可以得到$R_i$的解：

$$
R_i = \frac{C - C'}{S - S'}
$$

然后，对于每个额外的侧幕$i$，需要最小化的目标函数$E$为：

$$
E = \sum_{j=1}^{n} \left\| C^j - \left[ F(\alpha) + (1 - \alpha)B_1 + R_i(\alpha) M(T_i^j, A_i) \right] \right\|^2
$$

由于我们无法为每个结构化图案拍摄侧幕的参考照片（侧幕对相机不可见），因此我们使用背景幕的对应图案照片来获得$B$和$B'$，以及纹理映射$T_j$。

通过在物体周围放置侧幕，并一次照亮一个侧幕，理论上我们可以获得所有击中相机的光线的描述。到目前为止，我们只对真实物体捕捉了背景幕和最多两个侧幕的环境遮罩。

### 4 环境合成（Environment Compositing）

一旦我们获得了前景物体的环境遮罩，我们就可以将其合成到新的环境中，以保留反射和折射效果。

这个过程称为环境合成（environment compositing），它只是方程（1）的实现。它涉及到前景和背景颜色的加和，以及每个描述环境的纹理映射的加权贡献。

如同任何纹理映射过程，我们需要进行某种过滤以避免混叠（aliasing）。我们使用求和面积表（summed area tables）[8]，它允许我们快速计算轴对齐矩形区域的平均值。为了处理放大效果的物体，我们通常使用比显示它们作为背景幕时更高的分辨率来存储背景图像的纹理映射。

### 5 结果（Results）

我们组装了一个成像系统，使用数字相机和多个显示器来获取一些物体的环境遮罩，以展示我们方法的能力。图3展示了成像系统的照片。我们将三个显示器放置在与立方体环境的三个面一致的位置。我们使用相同的20英寸Trinitron显示器，并调整显示，使得洋红色和绿色图像在所有显示器上尽可能一致。我们使用Kodak DCS 520数字相机拍摄每个物体，并依次循环显示每个显示器上的图像。显示器之间的相互反射并不重要。使用的条纹图案足够多，以使最小条纹的宽度对应于最终遮罩的一个像素。因此，为了提取512×512的遮罩，我们将使用18个条纹图像，9个水平和9个垂直。显示器还显示了配准标记，使得可以从每张照片中提取包含结构化图案的部分。这个提取过程包括一个简单的变形步骤，试图至少部分校正显示器的曲率和镜头的畸变。

提取过程每个环境面的时间大约为10到20分钟，运行在Intel Pentium II 400MHz的计算机上。合成速度要快得多，可以以每秒4到40帧的速度进行。

为了在我们的并排比较中获得一致的颜色，我们拍摄了显示器上的各种背景，并使用这些数字照片作为合成中的背景。在图4和图5中，得到的合成图像与相同背景下实际物体的照片进行了比较。在这些图的每一行中，左边的图片显示了使用传统alpha遮罩（通过Smith-Blinn三角剖分方法获得）得到的合成图像。中间的图片显示了使用环境遮罩方法得到的合成结果，右边的图片是物体在显示背景图像的显示器前的实际照片（在获取环境遮罩和三角剖分数据时拍摄）。

第一个物体是一个玻璃烛台。两个凸起的肋条产生了非常复杂的折射图案，环境遮罩能够准确地再现这些效果。

第二个物体是一个玻璃书挡，其表面粗糙，散射通过它的光线，产生了半透明效果。注意，这个物体在传统alpha遮罩下完全消失，因为每个像素都被判断为透明或几乎透明。

第三组图像（图5的上排）展示了将反射/折射系数$R_1$表示为RGB三元组的优势。物体是装有红色、绿色和蓝色水的香槟杯。环境遮罩能够更准确地再现背景颜色通过有色水过滤后的效果。注意，环境遮罩在捕捉杯底的反射方面也比普通alpha图像更成功。

第四组图像（图5的下排）提供了一个光泽反射的例子。物体是一个金属馅饼盘，倾斜到接近掠角的位置。对于这个物体，我们使用了高斯加权纹理提取算子，如下一个例子中详细讨论的那样。

图6展示了当反射足够漫射时，我们假设物体反射的光线可以用纹理区域的常数加权平均来描述的假设开始失效。最左边的图像显示了由于这种近似的不准确性而产生的明显条纹。为了改善这种效果，我们尝试将每个像素的反射建模为纹理上的椭圆高斯加权平均，类似于Ward的方法[27]。为了提取遮罩，我们首先在采集过程中修改纹理算子$M$，使其返回纹理区域的高斯加权平均值，而不是箱式滤波平均值。高斯选择使得矩形的每一边对应于中心的3σ差异。使用求和面积表进行快速渲染需要箱式滤波，因此在渲染时，我们将每个获取的矩形缩小到对应于3/2σ的宽度，使用箱式滤波作为原始高斯的粗略近似。（一个潜在的更好替代方法是使用椭圆加权平均（EWA）滤波器[15]，表示为使用高斯滤波器构建的图像金字塔[4]进行渲染。）在采集过程中使用高斯滤波近似，然后在渲染过程中使用箱式滤波近似，在某种程度上改善了结果。然而，找到一种足够通用的方法来处理漫射和镜面表面，同时提供高效渲染，仍然是未来研究的课题。

图7展示了使用背景幕和两个侧幕捕捉的物体。物体是一个闪亮的金属烛台和一个侧放的金属花瓶。这些物体被选择和定位，使得尽可能多的表面反射来自我们实验设置中有限的侧幕区域。图中展示了前景颜色和未遮挡背景的效果，接着是每个纹理映射（后、左、右）的单独贡献。总合成图像（左下）是这四个图像的总和，并与实际物体周围这些图像的照片（右下）进行比较。注意，即使是复杂的反射也被捕捉到了——在烛台的底座中，我们可以看到背景幕的光线反射到花瓶的侧面，后者朝向相机。

图8展示了使用环境遮罩技术对物体进行新颖的重照明。这里物体显示在背景图像前，但现在侧幕是从实际光源的照片中合成的。当这些光源在纹理映射中移动时，环境遮罩显示了该位置的光线如何反射到物体表面。图中展示了两种不同的照明配置，插图中显示了用于左侧和右侧侧幕的纹理映射。

图9展示了我们技术在某些情况下的失败。第一行展示了四个反射球堆叠的环境遮罩：（a）在每个环境面上显示蝴蝶图像的合成图像，（b）对应的照片。虽然合成图像很好地再现了照片，但它突显了我们当前设置只能捕捉物体环境遮罩的一小部分——相机和其他设备的反射是可见的，被捕捉在前景颜色项$F$中。这更多是工程上的失败，而不是理论上的失败；通过更复杂的系统在物体周围显示结构化图案，可以获得更完整的遮罩。图9的第二行展示了一个更根本的失败：当单个像素看到背景幕的两个不同区域时的结果。这里一个饮用玻璃被倾斜，使得背景幕通过玻璃和在其表面反射都可见。这在照片（d）中很明显，但在获取过程中条纹图案的两个图像相互干扰。获得的矩形几乎是任意选择的，这导致合成图像（c）中的噪声。

### 6 深度校正（Depth Correction）

到目前为止，我们已经介绍了一种方法，可以在固定距离的背景幕前获取物体的环境遮罩。因此，我们展示的所有合成图像也是在相同深度的背景幕前。然而，为了在任意深度的背景幕前进行合成，我们需要捕捉光线在空间中的传播信息。在本节中，我们概述了一种扩展我们方法的方法，它将前景物体的折射或反射光线建模为具有轴对齐矩形截面的3D光束。然后，通过在任意深度处截取光束的截面，可以构造该深度的背景区域。

为了为每个像素构建光束，我们在两个不同深度的背景幕前提取前景物体的环境遮罩。如前文所述，我们独立考虑每个矩形的水平范围和垂直范围。对于给定像素，设$[l, r]$和$[l', r']$是两个环境遮罩中矩形范围的左端点和右端点。有两种方法可以将这两个范围连接起来形成光束：要么将$l$连接到$l'$，$r$连接到$r'$；要么翻转端点，将$l$连接到$r'$，$r$连接到$l'$。后一种情况对应于两个背景平面之间存在焦点。为了区分这两种情况，我们可以提取一个位于两个原始深度之间的第三个环境遮罩，并测试该环境遮罩中的范围是否与直线连接或翻转连接最一致。然而，作为一种概念验证，到目前为止我们只使用了一个简单的标志，由用户设置，该标志控制所有像素的连接是否为“直线”或翻转。（注意，由于标志仅在每个像素内控制光束的翻转，即使标志设置不正确，折射图像也会以正确的方向出现；然而，由于在估计环境对每个像素的贡献时积分的区域过大或过小，得到的图像可能会显得过于锐利或过于模糊。）

最后，为了在任意深度的背景幕前合成这些物体，我们使用线性插值来截取光束与该深度处的平面的交点。得到的矩形用作正常环境合成操作中的区域范围。图10展示了具有深度的环境遮罩的一些初步结果，该环境遮罩是为一个放大镜捕捉的。

### 7 结论（Conclusion）

在本文中，我们引入了环境遮罩，并展示了它如何扩展传统的alpha遮罩，用于图像合成。环境遮罩能够模拟反射、折射、半透明、光泽和相互反射的效果。它还能够更接近现实地处理有色透明度。我们展示了一种新颖的方法，使用结构化漫反射光来获取真实物体的环境遮罩。为了从获取的图像中提取遮罩，我们开发了一个数学框架，用于分析摄影数据。这个框架允许我们以统计最优的方式识别反射和折射区间，并且对于镜面和近镜面表面证明是相当稳健的。通过使用求和面积表进行快速矩形积分，我们开发了一个软件渲染系统，可以在交互速度下将环境遮罩的物体合成到新的背景中，使用的硬件要求不高。通过将光源图像放入环境中，我们可以将环境合成器用作交互式照明工具。

这项研究引出了许多未来的工作领域。首先，我们在非线性校准和补偿颜色通道串扰方面还有更多工作要做。此外，在获取方面，我们希望探索减少捕捉环境遮罩所需图像数量的方法。此外，我们的方法假设每个纹理映射区域只有一个区域映射到像素上。当反射和折射光线方向突然变化并映射到同一背景幕时，这种情况并不成立。可以想象，通过增加最大似然问题的维度来识别多个区域，尽管可能需要更多的背景幕来正确关联水平和垂直区间。

如第5节所述，对于光泽表面，我们在估计轴对齐椭圆高斯时获得了更好的结果，而不是矩形。未来的一个研究领域是找到更好的函数，例如用于估计环境遮罩的定向、可能可操控的函数。同样，在一定的性能开销下，我们可以探索更准确的加权滤波方法来合成环境遮罩。

为了获取具有深度的环境遮罩，我们希望进一步研究提取折射和反射光束的3D方法。如前所述，我们可以在第三个深度获取信息，并使用这些信息决定每个像素是否在捕捉的深度极限之间存在焦点。与其独立地为每个深度提取矩形范围，我们可以同时优化所有三个深度。

为了渲染具有深度的图像，我们希望研究在具有变化深度的背景上进行合成的方法。允许背景旋转离开相机可能有助于模拟景深效果。此外，深度信息可以用于将物体合成到任意3D场景中，而不是2D背景中。我们还可以研究将多个环境“精灵”合成到一个场景中的方法。Miller和Mondesir[18]指出，尽管简单地将多个精灵层叠到背景中并不正确，但仍然给人一种引人注目的外观。然而，某种程度上的“深度”损失是不可避免的。也许我们的方法可以捕捉深度信息，从而实现更真实的合成。实际上，可能可以从多个视点获取或渲染我们的深度环境遮罩，从而形成一种新的渲染原语：镜面传输函数或镜面光场。对于每个入射光线，我们可以快速确定一组出射光线或棱镜。

在其他渲染领域，我们可以探索类似视图变形的方法[22]，在不同视点获取的环境遮罩之间进行过渡。此外，来自其他视点的图像可以用于通过物体投射光线到环境中，以创建焦散和阴影。

另一个有趣的研究领域是开发方法来创建和修改真实或合成物体的环境遮罩，无论是交互式还是算法式。例如，可以很容易地想象一个“绘画程序”，在其中通过各种绘画操作交互式地修改扫描物体的光散射属性；或者通过算法生成纹理，修改其反射属性。

### 8 致谢（Acknowledgements）

我们要感谢Steve Wolfman在项目初期的宝贵贡献。还要感谢Rick Szeliski提供了用于环境映射合成的全景数据集。该工作得到了NSF奖项9803226和9553199的支持，以及来自Intel、Microsoft和Pixar的工业捐赠和NSF研究生奖学金计划的支持。

### 参考文献（References）

1. Paul Besl. Active optical range imaging sensors. In Jorge L.C. Sanz, editor, Advances in Machine Vision, chapter 1, pages 1–63. Springer-Verlag, 1989.
2. J. F. Blinn and M. E. Newell. Texture and reflection in computer generated images. Communications of the ACM, 19:542–546, 1976.
3. Normand Brière and Pierre Poulin. Hierarchical view-dependent structures for interactive scene manipulation. In Proceedings of SIGGRAPH 96, pages 83–90, August 1996.
4. Peter J. Burt. Fast filter transforms for image processing. Computer Graphics and Image Processing, 16:20–51, 1981.
5. Brian Carrihill and Robert Hummel. Experiments with the intensity ratio depth sensor. In Computer Vision, Graphics, and Image Processing, volume 32, pages 337–358, 1985.
6. G. Chazan and N. Kiryati. Pyramidal intensity ratio depth sensor. Technical Report 121, Center for Communication and Information Technologies, Department of Electrical Engineering, Technion, Haifa, Israel, October 1995.
7. Robert L. Cook, Thomas Porter, and Loren Carpenter. Distributed ray tracing. Computer Graphics, 18(3):137–145, July 1984.
8. Franklin C. Crow. Summed-area tables for texture mapping. In Proceedings of SIGGRAPH 84, volume 18, pages 207–212, July 1984.
9. Paul E. Debevec and Jitendra Malik. Recovering high dynamic range radiance maps from photographs. In Proceedings of SIGGRAPH 97, pages 369–378, August 1997.
10. Julie Dorsey, James Arvo, and Donald Greenberg. Interactive design of complex time dependent lighting. IEEE Computer Graphics and Applications, 15
(2):26–36, March 1995.
11. R. Fielding. The Technique of Special Effects Cinematography, pages 220–243. Focal/Hastings House, London, third edition, 1972.
12. Reid Gershbein. Cinematic Lighting in Computer Graphics. PhD thesis, Princeton University, 1999. Expected.
13. Andrew S. Glassner. An overview of ray tracing. In Andrew S. Glassner, editor, An Introduction to Ray Tracing, chapter 1. Academic Press, 1989.
14. Ned Greene. Environment mapping and other applications of world projections. IEEE Computer Graphics and Applications, 6(11), November 1986.
15. Ned Greene and Paul S. Heckbert. Creating raster omnimax images from multiple perspective views using the elliptical weighted average filter. IEEE Computer Graphics and Applications, 6(6):21–27, June 1986.
16. Eli Horn and Nahum Kiryati. Toward optimal structured light patterns. In Proceedings of the International Conference on Recent Advances in Three-Dimensional Digital Imaging and Modeling, pages 28–35, 1997.
17. Douglas S. Kay and Donald P. Greenberg. Transparency for computer synthesized images. In Proceedings of SIGGRAPH 79, volume 13, pages 158–164, August 1979.
18. Gavin Miller and Marc Mondesir. Rendering hyper-sprites in real time. In G. Drettakis and N. Max, editors, Proceedings 1998 Eurographics Workshop on Rendering, pages 193–198, June 1998.
19. Jeffry S. Nimeroff, Eero Simoncelli, and Julie Dorsey. Efficient re-rendering of naturally illuminated environments. In Fifth Eurographics Workshop on Rendering, pages 359–373, June 1994.
20. Thomas Porter and Tom Duff. Compositing digital images. In Proceedings of SIGGRAPH 84, volume 18, pages 253–259, July 1984.
21. K. Sato and S. Inokuchi. Three-dimensional surface measurement by space encoding range imaging. Journal of Robotic Systems, 2:27–39, 1985.
22. Steven M. Seitz and Charles R. Dyer. View morphing: Synthesizing 3D metamorphoses using image transforms. In Proceedings of SIGGRAPH 96, pages 21–30, August 1996.
23. Carlo H. Séquin and Eliot K. Smyrl. Parameterized ray tracing. In Proceedings of SIGGRAPH 89, volume 23, pages 307–314, July 1989.
24. Jean Paul Serra. Image Analysis and Mathematical Morphology. Academic Press, 1982.
25. Alvy Ray Smith and James F. Blinn. Blue screen matting. In Proceedings of SIGGRAPH 96, pages 259–268, August 1996.
26. Douglas Voorhies and Jim Foran. Reflection vector shading hardware. In Andrew Glassner, editor, Proceedings of SIGGRAPH 94, pages 163–166, July 1994.
27. Gregory J. Ward. Measuring and modeling anisotropic reflection. In Edwin E. Catmull, editor, Computer Graphics (SIGGRAPH '92 Proceedings), volume 26, pages 265–272, July 1992.
