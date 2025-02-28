---
layout: mypost
title: 060, k023 NeUDF Learning Neural Unsigned Distance Fields with Volume Rendering
categories: [UDF, 表面重建]
---


# 文章信息

## 标题

NeUDF: Learning Neural Unsigned Distance Fields with Volume Rendering

NeUDF: 使用体渲染学习神经无符号距离场

## 作者

Yu-Tao Liu1,2 
Li Wang1,2 
Jie Yang1 
Weikai Chen3 
Xiaoxu Meng3 
Bo Yang3 
Lin Gao1,2* 

1Beijing Key Laboratory of Mobile Computing and Pervasive Device, Institute of Computing Technology, Chinese Academy of Sciences 

2University of Chinese Academy of Sciences 

3Digital Content Technology Center, Tencent Games

liuyutao17@mails.ucas.ac.cn {wangli20s, yangjie01}@ict.ac.cn chenwk891@gmail.com {xiaoxumeng, brandonyang}@global.tencent.com gaolin@ict.ac.cn


## 发表信息

文章收录于2023年的CVPR。


## 引用信息

```
@InProceedings{Liu_2023_CVPR,
    author    = {Liu, Yu-Tao and Wang, Li and Yang, Jie and Chen, Weikai and Meng, Xiaoxu and Yang, Bo and Gao, Lin},
    title     = {NeUDF: Leaning Neural Unsigned Distance Fields With Volume Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {237-247}
}
```

## 论文链接

[cvpr 2023 link](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_NeUDF_Leaning_Neural_Unsigned_Distance_Fields_With_Volume_Rendering_CVPR_2023_paper.html)

[Home Page](http://geometrylearning.com/neudf/)

[Github Link](https://github.com/IGLICT/NeUDF)

# 文章内容

## 介绍

17-231205. NeUDF: Learning Unsigned Distance Fields with Volume Rendering. 

本文是由计算所高林老师团队写成的，该文章发表于2023年的CVPR。文章将无符号距离函数融合进体渲染的流程中，使用无符号距离函数表示物体的几何信息，解决薄物体及非封闭物体的表面重建问题。

类似于NeuS，本文推导出基于无符号距离函数的密度函数、权值函数，该权值函数具有无偏性，在物体表面达到峰值，并考虑到物体的前后遮挡问题。为解决无符号距离函数在表面的0值附近不可导的问题，本文提出在表面前后做梯度的平滑梯度近似约束。为解决无符号距离函数的表面权值问题，文章提出权值映射函数需要保证在0点处函数值为0，无穷远处函数值为1，一阶导数大于0，二阶导数小于0。

作者探索比较几组符合要求的函数值，最终选定映射函数为x/(1+x)。在提取无符号距离函数表面的时候，作者提取空间中的零值附近点云，使用带屏蔽的泊松表面重建（Screened Poisson Surface Reconstruction, SPSR）形成三角网格，并过滤掉非零的UDF值。文章能够重建衣服、镂空木盒、立体切片场景，也可以重建实心物体表面，重建结果在开放物体表面中效果较好，在实拍场景、封闭物体表面重建效果不如NeuralWarp、HF-NeuS等文章，在实拍风扇场景中展现出过平滑、不完整等现象。


# 重点难点讲解

这篇论文的核心内容是介绍了一种名为 **NeUDF**（Neural Unsigned Distance Fields）的新型神经渲染框架，用于从多视角图像中重建具有任意拓扑结构的表面，包括开放表面和闭合表面。它解决了传统基于符号距离函数（SDF）的方法只能重建闭合表面的局限性，是多视角三维重建领域的一个重要进展。

### **论文重点**

1. **问题背景与动机**
   - **传统方法的局限性**：现有的基于符号距离函数（SDF）的神经渲染方法（如NeuS）只能重建闭合表面，无法处理现实世界中常见的开放表面结构（如衣物、3D扫描场景等）。这限制了这些方法在实际应用中的广泛性。
   - **研究目标**：提出一种能够从多视角图像中重建任意拓扑结构（包括开放表面和闭合表面）的神经渲染框架。

2. **NeUDF框架的核心贡献**
   - **使用无符号距离函数（UDF）**：UDF能够表示任意拓扑结构的表面，包括开放表面和闭合表面。这是NeUDF区别于传统SDF方法的关键。
   - **新的权重函数和采样策略**：为了适应UDF的特性，作者提出了两个新的公式：无偏权重函数和重要性采样策略。这些公式专门针对UDF的体积渲染进行了优化，解决了传统方法在开放表面渲染时出现的偏差问题。
   - **法线正则化策略**：为了解决UDF在零等值面附近梯度不稳定的问题，作者引入了法线正则化方法。通过在表面邻域内进行空间插值，提高了表面重建的质量。

3. **实验验证**
   - **数据集**：作者在多个具有挑战性的数据集上进行了实验，包括DTU、MGN和Deep Fashion 3D等，证明了NeUDF在开放表面重建任务上的优越性。
   - **定量评估**：使用Chamfer Distance（CD）作为评估指标，NeUDF在开放表面数据集上的表现显著优于现有的SOTA方法（如NeuS、IDR等）。
   - **定性评估**：通过可视化重建结果，展示了NeUDF在复杂开放结构（如衣物的袖子、领口等）上的高保真重建能力。

4. **创新点总结**
   - 提出了首个基于UDF的神经体积渲染框架，能够从2D图像中重建任意拓扑结构的表面。
   - 设计了新的无偏权重函数和重要性采样策略，解决了UDF渲染中的偏差问题。
   - 引入法线正则化方法，提高了开放表面重建的质量。
   - 在多个数据集上取得了SOTA性能，尤其是在开放表面重建任务上。

### **论文难点**

1. **UDF的渲染偏差问题**
   - **难点**：UDF在渲染时容易出现偏差，导致重建的表面出现多余的浮面或噪声。这是因为UDF的值域与SDF不同，直接将SDF的渲染权重函数应用于UDF会导致全局偏差。
   - **解决方法**：作者提出了新的无偏权重函数和重要性采样策略，专门针对UDF的特性进行了优化，解决了这一问题。

2. **UDF梯度不稳定问题**
   - **难点**：UDF的零等值面附近的梯度不稳定，导致重建的表面可能出现锯齿状或不光滑的现象。
   - **解决方法**：作者引入了法线正则化策略，通过在表面邻域内进行空间插值，平滑了梯度，提高了重建质量。

3. **开放表面的表示和重建**
   - **难点**：开放表面的表示和重建一直是三维重建领域的难题，因为它们没有明确的“内外”之分，传统的SDF方法无法处理。
   - **解决方法**：NeUDF通过UDF表示任意拓扑结构的表面，并结合新的渲染策略和正则化方法，成功地重建了开放表面。

4. **实验验证的复杂性**
   - **难点**：为了验证NeUDF的有效性，作者需要在多个具有挑战性的数据集上进行实验，并与现有的SOTA方法进行对比。这涉及到大量的实验设计和结果分析。
   - **解决方法**：作者在DTU、MGN、Deep Fashion 3D等多个数据集上进行了实验，并提供了详细的定量和定性评估结果，证明了NeUDF的优越性。

### **总结**

这篇论文的核心贡献在于提出了NeUDF框架，解决了传统基于SDF的神经渲染方法在开放表面重建上的局限性。通过引入UDF表示和新的渲染策略，NeUDF能够从多视角图像中重建具有任意拓扑结构的表面，包括复杂的开放表面。这一工作不仅在理论上具有创新性，而且在实际应用中具有重要的价值，为三维重建领域的发展提供了新的思路和方法。

## 法线正则化

### 3.3 法线正则化

由于UDF的零等值面上的点是尖点，这些点不可一阶求导，因此在学习到的曲面附近的采样点的梯度在数值上不稳定（抖动）。由于渲染权重函数以UDF梯度作为输入，不可靠的梯度会导致曲面重建不准确。为了解决这个问题，我们引入了法线正则化来进行空间插值。法线正则化用从邻域内插的法线替代简单采样的表面法线。图4提供了详细的说明。由于不稳定的法线仅存在于曲面附近，我们使用远离曲面的点的法线来近似不稳定的法线。我们在点$p(t_i)$处离散地表示为：

$$
n(p(t_i)) = \frac{\sum_{k=1}^{K} w_{i-k} \Psi'(p(t_{i-k}))}{\sum_{k=1}^{K} w_{i-k}} \quad (9)
$$

其中，$w_{i-k} = \|p(i) - p(i - k)\|_2^2$是点$p(i)$到点$p(i - k)$的距离，$\Psi'(\cdot)$是UDF $\Psi(\cdot)$的导数，用于返回UDF的梯度。通过利用法线正则化，我们的框架能够从二维图像中实现更平滑的开放曲面重建。我们可以通过调整法线正则化的权重来获得更详细的几何结构。实验表明，法线正则化可以防止二维图像中的高亮和暗区对高质量重建的影响，如图10所示。

### 法线正则化的作用

法线正则化的主要作用是通过在曲面邻域内进行空间插值来平滑UDF的梯度，从而提高曲面重建的准确性。具体来说：

1. **平滑梯度**：由于UDF的零等值面上的点不可一阶求导，导致在这些点附近的梯度数值不稳定。法线正则化通过使用邻域内点的法线的加权平均值来替代简单采样的法线，从而平滑这些不稳定的梯度。

2. **提高重建质量**：通过平滑梯度，法线正则化能够减少重建过程中出现的锯齿状或不光滑现象，使重建的曲面更加平滑和准确。

3. **防止高亮和暗区的影响**：在二维图像中，高亮和暗区可能会导致重建过程中出现误差。法线正则化通过平滑梯度，能够减少这些区域对重建结果的影响，从而提高重建质量。

### 法线正则化的运行机制

法线正则化的运行机制可以分为以下几个步骤：

1. **计算邻域内点的法线**：对于每个点$p(t_i)$，计算其邻域内点$p(t_{i-k})$的法线$\Psi'(p(t_{i-k}))$。

2. **计算权重**：计算每个邻域点的权重$w_{i-k}$，权重与点之间的距离的平方成正比，即$w_{i-k} = \|p(i) - p(i - k)\|_2^2$。

3. **加权平均**：对邻域内点的法线进行加权平均，得到点$p(t_i)$的法线$n(p(t_i))$，如公式(9)所示。

4. **应用法线正则化**：将计算得到的法线应用于渲染权重函数，从而平滑梯度，提高重建质量。

通过上述步骤，法线正则化能够在曲面邻域内进行空间插值，平滑UDF的梯度，从而提高曲面重建的准确性和质量。


# 详细讲解

## 一、研究背景与动机

### 1.1 多视角三维重建的挑战
多视角三维重建是计算机视觉和计算机图形学中的一个经典问题，目标是从多张二维图像中重建出三维物体或场景的形状和外观。传统的多视角立体（MVS）方法在输入图像稀疏或纹理缺乏时表现不佳。近年来，基于神经隐式表示的渲染技术取得了显著进展，能够通过最小化渲染结果与输入图像之间的差异来联合学习隐式几何和颜色场。

然而，现有基于符号距离函数（SDF）的方法存在一个关键限制：它们只能重建闭合表面，无法处理现实世界中广泛存在的开放表面结构（如衣物、3D扫描场景等）。这大大限制了这些方法的应用范围。

### 1.2 研究动机
为了解决上述问题，作者提出了 **NeUDF**（Neural Unsigned Distance Fields），这是一个基于无符号距离函数（UDF）的新型神经渲染框架，能够从多视角图像中重建具有任意拓扑结构的表面，包括开放表面和闭合表面。



## 二、NeUDF框架的核心思想

### 2.1 无符号距离函数（UDF）
无符号距离函数（UDF）是一种隐式表示方法，它返回查询点到目标表面的绝对距离。与SDF不同，UDF不依赖于“内外”划分，因此能够表示任意拓扑结构的表面，包括开放表面。UDF的定义如下：
$$
d = \Psi_O(x): \mathbb{R}^3 \rightarrow \mathbb{R}^+,
$$
其中 $d$ 是点 $x$ 到目标表面的距离。

### 2.2 NeUDF框架
NeUDF框架的核心是通过体积渲染技术从多视角图像中学习UDF表示。该框架包括以下几个关键部分：

#### 2.2.1 场景表示
NeUDF将场景表示为UDF的零等值面，即：
$$
S_O = \{x \in \mathbb{R}^3 | d = 0, (d, c) = \psi(v, x)\},
$$
其中 $\psi$ 是一个神经网络，用于预测UDF值 $d$ 和渲染颜色 $c$。

#### 2.2.2 无偏权重函数
为了实现UDF的无偏渲染，作者提出了一个新的权重函数 $w_r(t)$，其定义如下：
$$
w_r(t) = \tau_r(t) e^{-\int_0^t \tau_r(u) du},
$$
其中
$$
\tau_r(t) = \left| \frac{\partial (\sigma_r \circ \Psi \circ p)}{\partial t}(t) \right| / (\sigma_r \circ \Psi \circ p(t)),
$$
$\sigma_r(d)$ 是一个满足特定条件的累积分布函数。

#### 2.2.3 重要性采样策略
为了提高渲染质量和效率，作者设计了一个重要性采样策略，其权重函数 $w_s(t)$ 定义如下：
$$
w_s(t) = \tau_s(t) e^{-\int_0^t \tau_s(u) du},
$$
其中
$$
\tau_s(t) = \zeta_s \circ \Psi \circ p(t),
$$
$\zeta_s(d)$ 是一个单调递减函数。

#### 2.2.4 法线正则化
为了缓解UDF在零等值面附近梯度不稳定的问题，作者引入了法线正则化策略。具体来说，通过在表面邻域内进行空间插值来平滑梯度。


## 三、实验验证

### 3.1 数据集
作者在多个具有挑战性的数据集上进行了实验，包括：
- **DTU MVS数据集**：包含多个场景的49或64张图像，分辨率为1600×1200。
- **Deep Fashion 3D数据集**：包含真实扫描的衣物，渲染为200张彩色图像，分辨率为800×800。
- **Multi-Garment Net数据集**：包含真实扫描的衣物，渲染为200张彩色图像，分辨率为800×800。

### 3.2 评估指标
作者使用Chamfer Distance（CD）作为评估指标，用于衡量重建形状与真实形状之间的差异。

### 3.3 定量结果
| 数据集 | COLMAP | IDR | NeuS | NeuralWarp | HF-NeuS | NeUDF |
| --- | --- | --- | --- | --- | --- | --- |
| MGN-upper (6) | 12.32 | 19.68 | 11.65 | 15.40 | 9.16 | **6.78** |
| MGN-pants (4) | 30.62 | 23.70 | 17.95 | 22.26 | 24.02 | **16.43** |
| DF3D-upper (6) | 8.60 | 14.46 | 15.29 | 10.27 | 23.31 | **8.72** |
| DF3D-pants (4) | 25.91 | 16.91 | 16.00 | 7.99 | 12.29 | **5.77** |
| DF3D-dress (8) | 9.77 | 14.27 | 11.75 | 7.79 | 12.03 | **7.39** |
| DTU (15) | 3.75 | 4.92 | 4.46 | 3.78 | 5.60 | **4.98** |

从表中可以看出，NeUDF在开放表面数据集（MGN和DF3D）上的表现显著优于现有的SOTA方法。

### 3.4 定性结果
作者还通过可视化重建结果展示了NeUDF的优越性。例如，在DF3D数据集上，NeUDF能够准确重建衣物的袖子、领口等复杂开放结构，而其他基于SDF的方法（如NeuS、IDR等）则无法正确表示这些结构。


## 四、讨论与结论

### 4.1 优点
- NeUDF是首个基于UDF的神经体积渲染框架，能够从2D图像中重建任意拓扑结构的表面。
- 提出了新的无偏权重函数和重要性采样策略，解决了UDF渲染中的偏差问题。
- 引入法线正则化方法，提高了开放表面重建的质量。
- 在多个数据集上取得了SOTA性能，尤其是在开放表面重建任务上。

### 4.2 局限性
- NeUDF难以建模透明表面。
- 当输入图像信息不足（如视角稀疏或严重遮挡）时，重建质量会下降。
- 法线正则化可能会在平滑表面和高频细节之间产生权衡。

### 4.3 未来工作
- 扩展NeUDF以更好地重建透明表面。
- 提高对稀疏输入图像的支持能力。


## 五、总结

NeUDF通过引入UDF表示和新的渲染策略，成功地解决了传统基于SDF的神经渲染方法在开放表面重建上的局限性。它不仅在理论上具有创新性，而且在实际应用中具有重要的价值，为三维重建领域的发展提供了新的思路和方法。



# 原文翻译

**标题：NeUDF：基于体积渲染的神经无符号距离场学习**

**作者**：刘宇涛¹²、李旺¹²、杨杰¹、陈伟凯³、孟晓旭³、杨波³、高林¹²*

**单位**：¹北京移动计算与普适设备重点实验室，中国科学院计算技术研究所；²中国科学院大学；³腾讯游戏数字内容技术中心

**联系方式**：liuyutao17@mails.ucas.ac.cn；{wangli20s, yangjie01}@ict.ac.cn；chenwk891@gmail.com；{xiaoxumeng, brandonyang}@global.tencent.com；gaolin@ict.ac.cn

## **摘要**

多视角三维重建在神经隐式曲面渲染的最新进展下取得了显著进步。然而，现有的基于符号距离函数（SDF）的方法仅限于重建闭合表面，无法重建包含开放曲面结构的广泛现实世界物体。在这项工作中，我们引入了一个新的神经渲染框架，命名为NeUDF¹，它可以从多视角监督中重建具有任意拓扑结构的曲面。为了能够表示任意曲面，NeUDF采用了无符号距离函数（UDF）作为曲面表示方法。然而，直接扩展基于SDF的神经渲染器无法适用于UDF。我们提出了两个专门针对基于UDF的体积渲染的新权重函数公式。此外，为了解决开放曲面渲染中的内外测试失效问题，我们提出了一种专门的法线正则化策略来解决曲面方向的歧义。我们在多个具有挑战性的数据集（包括DTU[^21^]、MGN[^5^]和Deep Fashion 3D[^61^]）上广泛评估了我们的方法。实验结果表明，NeUDF在多视角曲面重建任务中显著优于现有最先进方法，尤其是在重建具有开放边界的复杂形状方面。

## **一、引言**  

多视角曲面重建是计算机视觉和计算机图形学中一个长期存在且基础性的问题。传统的多视角立体（MVS）方法[^43^][^44^]在输入图像稀疏或缺乏纹理时往往表现不佳。近年来，神经隐式表示[^9^][^30^][^32^][^38^]的最新进展在稀疏视角下实现了复杂几何形状的高质量重建。具体来说，它们[^13^][^17^][^28^][^53^][^54^][^57^][^62^]利用体积渲染方案，通过最小化渲染结果与输入图像之间的差异来联合学习隐式几何和颜色场。然而，由于这些方法使用符号距离函数（SDF）[^28^][^53^]或占据场[^37^]来表示曲面，它们只能重建闭合形状。这极大地限制了它们的应用，因为现实世界中存在许多具有开放曲面的形状，如服装、3D扫描场景等。最近的研究（如NDF[^11^]、3PSDF[^8^]和GIFS[^59^]）提出了新的神经隐式函数以表示具有任意拓扑结构的曲面。然而，这些方法与现有的神经渲染框架不兼容。因此，如何利用神经渲染重建非闭合形状（例如开放曲面）仍然是一个有待解决的问题。

我们通过引入NeUDF填补了这一空白，这是一个可以从多视角图像监督中重建任意拓扑形状的新体积渲染框架。NeUDF基于无符号距离函数（UDF），这是一种简单直接的隐式函数，返回查询点到目标曲面的绝对距离。尽管UDF的定义简单，但我们发现直接将基于SDF的神经渲染机制扩展到无符号距离场无法确保非闭合曲面的无偏差渲染。如图2所示，基于SDF的权重函数会在空洞区域产生虚假曲面，因为渲染权重会触发不期望的局部最大值。为了解决这一问题，我们提出了一个新的无偏权重范式，专门针对UDF且能够感知曲面遮挡。为了适应所提出的权重函数，我们进一步提出了一种定制的重要性采样策略，以确保高质量重建非闭合曲面。此外，为了解决UDF在零等值面附近梯度不一致的问题，我们引入了一种法线正则化方法，通过利用曲面邻域中的法线信息来增强梯度一致性。

据我们所知，NeUDF是首次尝试仅从二维图像监督中重建具有任意拓扑结构的曲面。我们在公共数据集（例如MGN[^5^]、Deep Fashion 3D[^61^]和BMVS[^56^]）上进行了广泛的实验，结果表明NeUDF在非闭合三维形状的多视角曲面重建任务中实现了超越最先进水平的新性能，同时在重建闭合曲面方面也取得了相当的结果。我们的贡献总结如下：
1. 提出了第一个基于UDF的神经体积渲染框架，命名为NeUDF，可用于重建具有任意拓扑结构（包括具有开放边界的复杂形状）的多视角形状。
2. 提出了一种专门针对UDF渲染的新型无偏权重函数和重要性采样策略。
3. 在多个具有挑战性的非闭合三维形状数据集上实现了多视角曲面重建任务中的最先进性能。


## **二、相关工作**

在本节中，我们首先讨论经典的隐式表示和神经渲染技术。接下来，我们对近期将这些技术结合以提升多视角重建任务性能的工作进行概述。

### （一）神经隐式表示
近年来，神经隐式表示方面的进展[9, 31, 38, 41, 55]突破了以往显式表示（例如点云、体素和网格）在拓扑结构和分辨率上的限制，为三维建模与重建树立了新的标杆。复杂的形状可以通过将查询点分类为位于形状内部或外部（二值占据）[10, 12, 15, 18, 30, 39, 42]，或者预测到表面的符号距离（SDF）[9, 22, 31, 38, 41]来进行隐式表示。由于依赖于三维空间的内外划分，这些方法仅能建模闭合物体。为了克服这一限制，基于无符号距离函数（UDF）[11, 50–52, 59]的方法被提出，使深度神经网络能够适当地表示并学习具有开放曲面的更广泛形状范围。例如，NDF[11]预测从输入查询点到其位置感知形状特征的无符号距离，该特征以多尺度方式编码。HSDF[52]同时预测UDF场和符号场，以实现更好的网格保真度。然而，这些方法[11, 50–52, 59]需要三维监督来进行网格重建。

### （二）神经渲染
除了几何信息外，为了真实地描绘场景，还需要外观信息，尤其是当输入观测数据以二维图像形式呈现时。基于神经隐式曲面渲染的方法[24, 26, 27, 36, 47, 49, 58]通过微分球面追踪[20]及其变体来寻找射线与曲面的交点，并使用另一个网络分支查询射线与曲面交点处的RGB颜色。由于反向传播的梯度受到整个空间的影响，像IDR[58]和DVR[36]这样的曲面渲染方法在没有额外二维掩码监督的情况下难以重建复杂形状。相比之下，基于神经体积渲染的方法[29, 32, 34, 35, 40, 48, 60]暗示射线不仅在二值交点处，而且在空间的每一个点都有机会与场景属性相互作用。对于依赖良好梯度行为进行优化的机器学习流程来说，这种连续模型作为可微渲染框架表现良好。

### （三）多视角重建
在深度学习出现之前，多视角立体（MVS）方法[1, 6, 7, 14, 25, 44–46]主要依赖于跨视角的图像特征匹配[6, 44]或体素网格等体积表示[1, 7, 14, 25, 46]。前者，如广泛使用的COLMAP[44]方法，高度依赖丰富的纹理信息和从点云生成的经典网格技术，因为它通过图像间的对应关系计算多视角深度图，并将其融合为密集点云；而后者由于体素表示的立方级内存增长，分辨率受限。近期结合隐式表示和神经渲染的工作[13, 17, 24, 27, 28, 36, 37, 53, 54, 57, 58]在重建闭合曲面时超越了以往的方法，能够实现高保真度重建。由于这些方法使用占据值[37]或符号距离函数[28, 53]（SDF）来表示曲面，因此它们的重建结果仅限于闭合形状。我们的NeUDF提出了一种针对无符号距离函数（UDF）的新型神经体积渲染算法，从而可以自然地将曲面提取为UDF的零等值面，能够表示具有开放曲面和薄结构的复杂形状。

## **三、方法**

给定一组标定后的图像$\{I_k | 1 \leq k \leq n\}$，我们的目标是仅使用二维图像监督来重建任意表面，包括闭合结构和开放结构。在本文中，表面被表示为无符号距离函数（UDFs）的零等值面。为了学习物体或场景的UDF表示，我们引入了一种新的神经渲染架构，该架构结合了用于渲染的无偏权重公式。我们首先基于UDF定义场景表示（第3.1节），然后介绍NeUDF及其两个专门针对基于UDF的体积渲染的权重函数公式（第3.2节）。最后，我们阐述了用于缓解二维图像歧义的法线正则化方法以及我们的损失配置（第3.4节）。

### 3.1 场景表示
与符号距离函数（SDF）不同，无符号距离函数（UDF）是无符号的，能够表示具有任意拓扑结构的开放曲面，以及闭合曲面。给定一个三维物体$O=\{V, F\}$，其中$V$和$F$分别是顶点和面的集合，物体$O$的UDF可以表示为一个函数$d = \Psi_O(x) : \mathbb{R}^3 \rightarrow \mathbb{R}^+$，它将点坐标$x$映射到曲面的欧几里得距离$d$。我们定义UDF的零等值面为：
$$
UDF_O = \{\Psi_O(x) | d < \epsilon, d = \arg\min_{f \in F} (\|x - f\|_2)\},
$$
其中$\epsilon$是一个小阈值，物体的表面可以通过UDF的零等值面来调节。我们引入了一个可微的体积渲染框架，用于从输入图像中预测UDF。该框架通过一个神经网络$\psi$来近似，它根据采样射线$v$上的空间位置$x$预测UDF值$d$和渲染颜色$c$：
$$
(d, c) = \psi(v, x) : S^2 \times \mathbb{R}^3 \rightarrow (\mathbb{R}^+, [0, 1]^3) \quad (1)
$$
借助体积渲染，权重通过最小化预测图像$I'_k$与真实图像$I_k$之间的距离来优化。学习到的表面$S_O$可以通过预测的UDF的零等值面来表示：
$$
S_O = \{x \in \mathbb{R}^3 | d = 0, (d, c) = \psi(v, x)\} \quad (2)
$$

### 3.2 NeUDF 渲染

渲染过程是学习准确UDF的关键，因为它通过沿射线$v$的积分将输出颜色和UDF值连接起来：
$$
C(o, v) = \int_{0}^{+\infty} w(t) c(p(t), v) \, dt, \quad (3)
$$
其中，$C(o, v)$是从相机原点$o$沿视图方向$v$的输出像素颜色，$w(t)$是点$p(t)$的权重函数，$c(p(t), v)$是沿视图方向$v$的点$p(t)$处的颜色。为了通过体积渲染重建UDF，我们首先引入了一个概率密度函数$\varsigma_r'(\Psi(x))$，称为U-density，其中$\Psi(x)$是点$x$的无符号距离。U-density函数$\varsigma_r'(\Psi(x))$将UDF场映射为一个概率密度分布，该分布在曲面附近具有显著的高值，从而实现精确重建。

受NeuS [53]的启发，我们利用U-density函数推导出一个无偏且考虑遮挡的权重函数$w_r(t)$及其不透明度$\tau_r(t)$：
$$
w_r(t) = \tau_r(t) e^{- \int_{0}^{t} \tau_r(u) \, du}, \quad (4)
$$
$$
\tau_r(t) = \frac{\partial (\varsigma_r \circ \Psi \circ p)}{\partial t}(t) \big/ (\varsigma_r \circ \Psi \circ p(t)), \quad (5)
$$
其中，$\circ$是函数复合运算符，$\varsigma_r(\cdot)$必须满足以下规则，以实现有效的UDF重建：
$$
\varsigma_r(0) = 0, \quad \lim_{d \to +\infty} \varsigma_r(d) = 1, \quad (6)
$$
$$
\varsigma_r'(d) > 0; \quad \varsigma_r''(d) < 0, \quad \forall d > 0. \quad (7)
$$
$\varsigma_r(d)$可以是任何满足上述形状的函数。由于$\varsigma_r(d)$是U-density的累积分布函数，$\varsigma_r(0) = 0$保证了不会从负距离的点累积密度。此外，$\varsigma_r'(d) > 0$和$\varsigma_r''(d) < 0$确保了U-density值为正，并且在曲面附近的值显著较高。参数$r$在$\varsigma_r(d)$中是可学习的，并控制密度的分布。这种函数结构弥合了体积渲染与曲面重建之间的体积-曲面差距，并保证了全局无偏性。更多详细讨论请参考我们的补充材料。

我们认为，基于SDF的神经渲染器的直接扩展将违反上述某些规则。例如，NeuS [53]中U-density的累积分布函数是$\Phi_s$（Sigmoid函数），而$\Phi_s(0) > 0$违反了公式(6)。这种违反将导致渲染权重出现偏差，从而产生图2中所示的冗余悬浮面和不规则噪声。需要注意的是，NeuS中提出的局部最大值约束无法解决UDF中的这种渲染偏差。有关无偏性以及全局/局部最大值约束的详细讨论，请参阅我们的补充材料。

在广泛的消融研究（第4.3节）中对$\varsigma_r(d)$的不同形式进行了评估后，我们最终选择了$\varsigma_r(d) = \frac{rd}{1 + rd}$，其中$r$初始化为0.05。此外，我们采用$\alpha$合成来离散化权重函数，该函数沿射线方向采样点，并根据权重积分累积颜色。有关公式(4)和公式(5)的无偏性和遮挡感知属性的详细离散化和证明，请参考我们的补充材料。


#### 重要点采样

适应渲染权重的点采样是体积渲染中的重要步骤。与SDF不同，为了实现UDF的无偏渲染，渲染函数应在交点之前分配更多权重（见图2(c)）。因此，如果渲染函数和采样函数使用相同的权重，UDF梯度的正则化（即Eikonal损失）将导致曲面两侧的梯度幅度严重不平衡。这可能会显著降低重建UDF场的质量。因此，我们提出了一个特别定制的采样权重函数（见图2(c)），以在整个空间中实现平衡的正则化。重要性采样$w_s(t)$的公式如下：

$$
w_s(t) = \tau_s(t) e^{- \int_{0}^{t} \tau_s(u) \, du}, \quad \tau_s(t) = \zeta_s \circ \Psi \circ p(t), \quad (8)
$$

其中，$\zeta_s(\cdot)$满足以下规则：$\zeta_s(d) > 0$且$\zeta_s'(d) < 0$，$\forall d > 0$。直观上，$\zeta_s(\cdot)$是第一象限中的单调递减函数。在本文中，我们使用：

$$
\zeta_s(d) = \frac{s e^{-sd}}{(1 + e^{-sd})^2},
$$

其中参数$s$控制$x = 0$处的强度。$s$从0.05开始，并在每个采样步$z$中以$2^{z-1}$的速率变化。任何能够与渲染函数实现平衡正则化的采样函数都与我们的框架兼容。有关上述规则的详细说明，请参阅我们的补充文档。此外，我们在消融研究（第4.3节）中定性和定量地评估了$\zeta_s(d)$的必要性。总体而言，在体积渲染过程中，权重函数在渲染（公式(4)）和采样（公式(8)）中协同使用，从而实现了高保真度的开放曲面重建。

### 3.3 法线正则化

由于UDF的零等值面上的点是不可微的尖点，因此在学习到的曲面附近采样点的梯度在数值上不稳定（抖动）。由于渲染权重函数以UDF梯度作为输入，不稳定的梯度会导致曲面重建不准确。为了缓解这一问题，我们引入了法线正则化以进行空间插值。图4展示了详细的说明。由于不稳定的法线仅存在于曲面附近，我们使用远离曲面的点的法线来近似不稳定的法线。我们在点$p(t_i)$处离散地表示为：

$$
n(p(t_i)) = \frac{\sum_{k=1}^{K} w_{i-k} \Psi'(p(t_{i-k}))}{\sum_{k=1}^{K} w_{i-k}} \quad (9)
$$

其中，$w_{i-k} = \|p(i) - p(i - k)\|_2^2$是点$p(i)$到点$p(i - k)$的距离，$\Psi'(\cdot)$是UDF $\Psi(\cdot)$的导数，用于返回UDF的梯度。通过利用法线正则化，我们的框架能够从二维图像中实现更平滑的开放曲面重建。我们可以通过调整法线正则化的权重来获得更详细的几何结构。实验表明，法线正则化可以防止二维图像中的高亮和暗区对高质量重建的影响，如图10所示。


### 3.4 训练

为了学习高保真度的开放曲面重建，我们通过最小化渲染图像与已知相机姿态的真实图像之间的差异来优化网络，而无需任何三维监督。参考NeuS [53]，我们同样应用了SDF体积渲染中使用的三种损失项：颜色损失$L_c$、Eikonal损失$L_e$ [58]和掩码损失$L_m$。颜色损失用于衡量渲染图像与输入图像之间的L1差异。Eikonal损失用于数值正则化采样点上的UDF梯度。如果提供了掩码，掩码损失还会在BCE度量下鼓励预测掩码接近真实掩码。总体而言，我们使用的损失由三部分组成：
$$
L = L_c + \alpha L_e + \beta L_m \quad (10)
$$
关于详细的实现和网络架构，请参考我们的补充文档。


## **四、实验与评估**

在本节中，我们通过定性和定量的方式对NeUDF在多视角重建任务中的表现进行验证，并进一步测试了我们的方法在真实场景中的应用。实验结果表明，NeUDF优于现有的最先进技术，并能够成功重建具有开放边界的复杂形状。最后，我们进行了消融研究和进一步讨论，以证明每个关键设计的重要性。

### 4.1 实验设置

**数据集**：由于我们的方法主要关注在多视角监督下的开放曲面重建，我们在三个常用的数据集上进行了实验，包括多服饰网络数据集（Multi-Garment Net，MGN）[4]、深度时尚3D数据集（Deep Fashion3D，DF3D）[61]和DTU多视角立体（MVS）数据集（DTU）[21]。对于DTU MVS数据集，每个场景包含49张或64张分辨率为1600×1200的图像，掩码来自IDR [58]。DF3D和MGN包含一些具有开放边界的实扫服饰，这些服饰被渲染为200张分辨率为800×800的彩色图像用于重建。我们分别从这两个数据集中采样了18个和10个来自不同类别的形状。详细的相机姿态请参考补充文档。此外，我们还收集了一些具有非闭合结构的复杂形状，并对它们进行了渲染以评估我们的框架。这些形状包含更复杂的结构，由具有开放边界的曲面组成，例如植物叶子和空心结构（见图1）。我们还测试了一些包含多样化形状的数据集（例如BMVS、Mixamo以及一些真实捕获的物体）。

**基线方法**：我们将NeUDF与多视角重建任务中的几种基线方法进行了比较，包括COLMAP [43, 44]、IDR [58]、NeuS [53]、NeuralWarp [13]和HF-NeuS [54]。COLMAP是一种广泛使用的MVS方法，它从多视角图像中重建点云，并通过球面旋转算法（Ball-Pivoting Algorithm，BPA）[3]提取显式的开放曲面。IDR是当前最先进的曲面渲染方法，能够在掩码监督下训练以重建高质量的网格。NeuS是通过基于SDF的体积渲染进行曲面重建的开创性工作，在曲面重建方面取得了令人印象深刻的结果。最新的工作，如NeuralWarp和HF-NeuS，在改进高频细节或几何一致性方面为闭合形状提供了更好的性能。然而，它们无法建模具有开放边界的任意曲面。我们在第1节中提到的直接解决方案也进行了评估，即通过在预测的SDF值上添加绝对值操作并保持其他配置不变来扩展NeuS渲染器，以实现UDF重建。

**评估指标**：为了衡量重建形状相对于真实形状的准确性，我们采用了常用的指标——Chamfer距离[2]（CD），用于与最先进方法进行定量比较。我们使用掩码泊松方法进行UDF网格提取，首先在UDF中采样一百万个点，然后采用SPSR [23]提取闭合网格，并通过掩码去除UDF值非零的虚假曲面。我们将不同数据集的所有网格缩放到单位球内，以确保公平比较。有关指标计算的详细信息，请参考Fan等人[16]。


### 4.2 多视角重建的对比

为了展示我们在多样化数据集（尤其是开放曲面）上的重建能力，我们在上述三个数据集上对最先进方法进行了定量和定性对比，包括拓扑和几何结构各异的开放曲面数据集，以及以往工作中使用的闭合曲面数据集。需要注意的是，IDR在DTU [21]、DF3D [61]和MGN [5]数据集上使用了掩码监督，而我们的方法仅在DTU [21]数据集上使用了掩码监督。

**定量结果**：我们在表1中报告了平均Chamfer距离。结果显示，我们的方法在两个开放曲面数据集（DF3D [61]和MGN [4]）上大幅优于这些基线方法。我们的方法是唯一能够重建高保真度开放曲面的方法，而基线方法则受限于闭合形状。对于闭合曲面数据集（DTU [21]），我们的方法与基线方法相当。我们还提供了对NeuS渲染器的直接扩展的评估。直接扩展在开放曲面样本上导致较大的Chamfer距离（直接扩展：9.53，我们的方法：1.49），这是由于表面噪声以及有时无法收敛（例如DTU的scan65）。

**定性结果**：DF3D和MGN数据集上的定量对比在图5中可视化。如图5（b）（c）（d）（e）所示，基于SDF的方法（IDR [58]、NeuS [53]、NeuralWarp [13]、HF-NeuS [54]）受限于闭合形状，在开放边界曲面上表现不佳。相比之下，NeUDF能够重建具有开放边界的高保真网格（例如袖子、领口和腰部），如图5（f）所示，且无需掩码。我们还在Mixamo [33]和BMVS [56]数据集上与NeuS [53]进行了进一步对比。如图6（mixamo-demon和bmvs-bear）所示，我们的方法能够重建具有开放边界的几何形状，例如熊手中的贺卡和单层斗篷。很明显，NeuS无法合成具有开放边界的曲面。相比之下，我们重建的形状几何结构准确，复杂的开放结构得以保留。更多定性结果见我们的补充文档。此外，我们在图1中展示了一些具有复杂结构的挑战性案例，例如空心盒子、植物叶子和基于补丁的鱼。从这些具有复杂开放边界的物体的结果中，我们可以清晰地看到详细的几何结构和复杂的开放曲面结构，这验证了NeUDF为多视角重建学习了更好的UDF。

**真实场景的捕获**：我们进一步在真实世界物体的捕获数据上评估我们的方法，包括书页、风扇叶片和植物叶子。对于每个场景，我们使用手机拍摄围绕物体的视频，并从视频中提取大约200帧。然后，我们使用COLMAP估计相机姿态，并将标定后的图像作为输入以优化网络参数，且不使用掩码监督。图7展示了书页的重建形状，更多真实场景的捕获结果见我们的补充文档。结果显示，NeuS倾向于将书页合并在一起，导致不真实的几何结构，而我们的方法实现了具有开放边界的准确曲面重建。


### 4.3 进一步讨论与分析

我们进行了三项消融研究，以验证我们方法中各个设计的有效性。首先，评估$\varsigma_r$在$\tau_r$中的不同选择，以证明其对UDF学习的有效性。接着，我们还验证了我们设计的**重要点采样**和**法线正则化**对于准确重建开放曲面的必要性。所有消融研究都在具有多样化形状的多个样本物体上进行。

**$\varsigma_r$在$\tau_r$中的选择**：尽管我们给出了$\varsigma_r$应满足的规则（公式(6)和公式(7)），但存在一系列满足这些规则的函数。这些函数都适用于UDF体积渲染，因此我们对几个不同的候选函数进行了验证，以检查每个函数在网络优化中的收敛能力，即在给定的训练迭代中，使用哪个函数可以使网络收敛到最佳结果。图9展示了遵循规则的三个候选函数（$1 - e^{-x}$、$\frac{2\arctan(x)}{\pi}$和$\frac{x}{1+x}$）的可视化结果。在给定的迭代次数（300k）后，使用函数$\frac{x}{1+x}$的网络在定性和定量方面都收敛到了最佳结果，而其他函数尚未完全收敛，导致曲面不完整且Chamfer距离略高。对多样化形状的评估也表明，所有函数都表现良好，而我们选择的函数（$\frac{x}{1+x}$）在我们的设置中表现最佳（我们的方法：1.11，候选函数：1.13/1.18）。

**重要点采样$w_s(t)$（公式(8)）的必要性**：为了证明公式(8)的必要性，我们设计了一个移除了重要点采样的消融版本，并使用公式(4)来对训练进行采样。图8（b）显示，使用相同权重进行渲染和采样的输出曲面不够平滑，且Chamfer距离误差较大，正如预期的那样。这些误差来自于采样点分布在曲面两侧不平衡，且网络未得到良好的正则化。图8（e）表明，使用重要点采样的网络得到了良好的正则化，并产生了更好的结果。

**法线正则化的必要性**：我们使用法线正则化（第3.3节）来解决UDF零等值面上的梯度不稳定问题，并对法线正则化的必要性进行了验证。如图8（c）所示，没有法线正则化的结果由于梯度计算不稳定，导致曲面粗糙且Chamfer距离较大。此外，法线正则化即使在极端情况下（例如极亮或极暗区域）也有利于曲面重建。图10展示了两种极亮和极暗光照条件下的情况。可视化结果表明，法线正则化对于缓解由光照条件模糊性（例如DTU头骨上的伪影和DTU金属兔子上的暗部）引起的几何误差至关重要。


## **五、讨论与结论**

### 限制
尽管我们的方法能够成功重建具有开放边界的任意曲面，但仍存在一些局限性。首先，使用我们的公式难以建模透明曲面。当输入图像中可见信息不足（例如视角稀疏或严重遮挡）时，重建质量会下降，图11给出了一个失败案例。由于法线正则化通过累积邻域信息来缓解曲面法线的歧义，因此在平滑度和高频细节之间存在权衡。此外，由于我们引入了UDF以提高表示能力，因此需要额外的网格化工具（如MeshUDF [19]或SPSR [23]），这可能会引入更多的重建误差。

### 结论
我们提出了NeUDF，这是一种基于UDF的新型体积渲染方法，能够从带或不带掩码的二维图像中实现具有开放和闭合曲面的任意形状的高保真度多视角重建。NeUDF在定性和定量方面均优于现有最先进方法，尤其是在具有开放边界的复杂曲面上。因此，我们的NeUDF在实际的三维应用中可以发挥关键作用。在未来工作中，我们可以扩展我们的公式以更好地重建透明曲面。增强我们的NeUDF以支持稀疏输入图像也是一个有趣的研究方向。

### 致谢
本研究得到了CCF-Tencent开放基金、北京市自然科学基金杰出青年学者项目（编号JQ21013）、国家自然科学基金（编号62061136007）以及中国科学院青年创新促进会的支持。




# 附录
## A. 概述
在主论文中，我们介绍了一种基于用户定义函数（UDF）的体绘制新方法，以实现对具有开放和封闭表面的任意形状的高保真多视图重建。本补充材料包含详细证明、实现细节以及多视图重建的额外结果。各部分组织如下：
- **B 节**：分析了基于有向距离函数（SDF）渲染器的简单 UDF 解决方案在颜色渲染中的固有偏差。
- **C 节**：详细证明了我们提出的 NeUDF 的无偏性和遮挡感知特性。
- **D 节**：提供了网络架构（D.1 节）、训练细节（D.2 节）和数据准备（D.3 节）的实现细节。
- **E 节**：提供了多视图重建的额外定性结果。

### 基于SDF渲染器的简单UDF解决方案中的偏差
在本节中，我们阐述了基于SDF渲染器的简单UDF解决方案所引入的颜色渲染偏差，该方案直接将NeuS的权重扩展到UDF。这种偏差会导致诸如冗余表面和浮动噪声等固有的几何误差。

为了应用基于NeuS的SDF渲染器的简单UDF解决方案，我们定义渲染颜色$C(o, v)$：
$$C(o, v) = \int_{0}^{+\infty} w_n(t)c(p(t), v)\mathrm{d}t \tag{11}$$
其中$(o, v)$是采样光线的原点和观察方向，$c(x, v)$是沿观察方向$v$在位置$x$处的颜色，$w_n(t)$是NeuS的渲染权重：
$$w_n(t) = \rho_s(t)e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u} \tag{12}$$
$$\rho_s(t) = \max\left\{-\frac{\frac{\partial(\Phi_s \circ \Psi \circ p)}{\partial t}(t)}{\Phi_s \circ \Psi \circ p(t)}, 0\right\} \tag{13}$$
其中$\rho_s(t)$表示NeuS的不透明度，$\Phi_s(d)$是Sigmoid函数，$\Psi(x)$是位置$x$处的UDF值。可学习参数$s$控制Sigmoid函数的分布，预计在训练过程中该参数会增加到无穷大。 

假设光线在其局部邻域内线性穿过开放表面，例如，存在一个区间$(t^l, t^r)$，交点$t^* \in (t^l, t^r)$，满足：
$$\Psi \circ p(t) = |\cos\theta| \cdot |t - t^*|, \forall t \in (t^l, t^r) \tag{14}$$ 

这里的公式表明在给定区间内，组合函数$\Psi \circ p(t)$（其中$\Psi$可能是有向距离函数相关，$p(t)$表示光线在$t$时刻的位置函数）与$t$和交点$t^*$的距离以及$\cos\theta$（$\theta$可能是光线与某个方向的夹角）相关，呈现出一种线性关系。 这通常用于在计算机图形学等领域中对光线与表面相交情况进行数学建模和分析。 

其中$\theta$是观察方向与表面法向量之间的夹角。

在有向距离函数（UDF）中，基于符号距离函数（SDF）渲染器（公式11）渲染的颜色$C(o, v)$，包含了几何形状的固有偏差和不一致性。设第一个交点为$t_0^*$及其对应的区间$(t_0^l, t_0^r)$，偏差可以用以下公式表示：
$$\lim_{s \to \infty} C(o, v) = 0.5c(p(t_0^*) , v) + \frac{2^k - 1}{2^{k + 1}}c_m + \frac{1}{2^{k + 1}}c_n \tag{15}$$
其中$k$是沿光线的交点数量，$c_m$是来自不可见表面的不需要的混合颜色，$c_n$是由渲染偏差引起的浮动噪声的颜色。参数$s$决定了沿光线的颜色权重分布，并且在训练过程中应该趋向于无穷大。 

请注意，与公式15对应的权重分布满足NeuS中讨论的局部最大值约束，即权重在每个交点处达到局部最大值（局部无偏差）。但是，由于体-面表示的差异，局部最大值约束对于开放表面的无偏差渲染是不够的。体渲染依赖于体级别的颜色融合来进行优化，而真实颜色恰好是采样光线与第一个相交表面的交点处的表面颜色。一个自洽的渲染过程应该能够解决这种体-面差异，也就是说，颜色融合范围应该尽可能限制在靠近第一个交点的位置（全局无偏差）。否则，网络无法通过体渲染收敛到一个表面表示。请注意，NeuS的权重对于符号距离函数（SDF）来说是全局和局部无偏差的，但对于有向距离函数（UDF）来说不是全局无偏差的，这种差异来自于SDF和UDF的值域差异。 

为了说明$c_m$和$c_n$的详细成因，我们首先证明：
$$\lim_{s \to \infty} \int_{0}^{t_0^l} w_n(t)\mathrm{d}t = 0 \tag{16}$$
$$\lim_{s \to \infty} \int_{t_0^l}^{t_0^r} w_n(t)\mathrm{d}t = 0.5 \tag{17}$$
这意味着输出颜色包含了不需要的偏差，其权重总和为0.5，并且这种偏差无法通过训练来纠正。然后我们展示偏差$c_m$和$c_n$的详细分布以作佐证。 

### 公式16的证明
具体来说，为了证明公式16，我们有：
$$
\begin{align*}
&\int_{0}^{t_0^l} w_n(t)\mathrm{d}t \\
=&\int_{0}^{t_0^l} \rho_s(t)e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u} \mathrm{d}t \\
=&\int_{0}^{t_0^l} -\frac{\partial }{\partial t}e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u} \mathrm{d}t \tag{18}\\
=& -e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u}\big|_{0}^{t_0^l}\\
=& -e^{-\int_{0}^{t_0^l} \rho_s(u)\mathrm{d}u} + 1\\
=& -e^{-\int_{0}^{t_0^l} \max\left\{\frac{\frac{\partial(\Phi_s \circ \Psi \circ p)}{\partial u}(u)}{\Phi_s \circ \Psi \circ p(u)}, 0\right\}\mathrm{d}u} + 1
\end{align*}
$$ 

由此可得：
$$
\begin{align*}
&\int_{0}^{t_0^l} w_n(t)\mathrm{d}t \\
&\leqslant -e^{-\int_{0}^{t_0^l} \left|\frac{\frac{\partial(\Phi_s \circ \Psi \circ p)}{\partial u}(u)}{\Phi_s \circ \Psi \circ p(u)}\right|\mathrm{d}u} + 1\\
&= -e^{-\int_{0}^{t_0^l} \left|\frac{\frac{\partial\Phi_s \circ \Psi \circ p(u)}{\partial \Psi \circ p(u)}\cdot\frac{\partial\Psi \circ p(u)}{\partial u}}{\Phi_s \circ \Psi \circ p(u)}\right|\mathrm{d}u} + 1 \tag{19}\\
&= -e^{-\int_{0}^{t_0^l} \left|\frac{\Phi_s' \circ \Psi \circ p(u)\cdot\frac{\partial\Psi \circ p(u)}{\partial u}}{\Phi_s \circ \Psi \circ p(u)}\right|\mathrm{d}u} + 1\\
&= -e^{-\int_{0}^{t_0^l} \frac{|\Phi_s' \circ \Psi \circ p(u)|\cdot\left|\frac{\partial\Psi \circ p(u)}{\partial u}\right|}{|\Phi_s \circ \Psi \circ p(u)|}\mathrm{d}u} + 1
\end{align*}
$$ 

定义：
$$
A = |\Phi_s' \circ \Psi \circ p(u)| \tag{20}
$$
$$
B = \left|\frac{\partial\Psi \circ p(u)}{\partial u}\right| \tag{21}
$$
$$
C = |\Phi_s \circ \Psi \circ p(u)| \tag{22}
$$

我们有：
$$
\int_{0}^{t_0^l} w_n(t)\mathrm{d}t = -e^{-\int_{0}^{t_0^l} \frac{A\cdot B}{C}\mathrm{d}u} + 1 \tag{23}
$$ 

因为$t_0^*$是$\Psi \circ p(t)$的第一个零点，且$\Psi(x)$是一个连续函数，所以有：
$$\exists\Psi_{min} > 0, s.t., \Psi \circ p(t) > \Psi_{min}, \forall t \in (0, t_0^l)$$

注意到$\Phi_s(x)$是Sigmoid函数$\Phi_s(x) = (1 + e^{-s*x})^{-1}$，并且$\frac{\partial\Psi\circ p(u)}{\partial u}$是有向距离函数（UDF）沿光线的梯度。我们有：
$$
\begin{align*}
C &= |\Phi_s \circ \Psi \circ p(u)| \tag{24}\\
&= (1 + e^{-s\cdot\Psi\circ p(u)})^{-1} \tag{25}\\
&> (1 + e^{-s\cdot\Psi_{min}})^{-1} \tag{26}\\
&> 0.5 \tag{27}\\
B&=\left|\frac{\partial\Psi \circ p(u)}{\partial u}\right| < 1 \tag{28}
\end{align*}
$$ 

并且对于任意$\epsilon > 0$，存在$S = \max\left\{1, \frac{-4t_0^l}{\ln(1 - \epsilon)\cdot\Psi_{min}^2}\right\}$， 使得对于任意$s > S$，有：
$$
\begin{align*}
A &= |\Phi_s' \circ \Psi \circ p(u)| \\
&= \frac{s\cdot e^{-s\cdot\Psi\circ p(t)}}{(1 + s\cdot e^{-s\cdot\Psi\circ p(t)})^2} \\
&\leqslant \frac{2}{\Psi^2 \circ p(t)\cdot s} \\
&\leqslant \frac{2}{\Psi_{min}^2\cdot s} \\
&\leqslant \frac{2}{\Psi_{min}^2\cdot \frac{-4t_0^l}{\ln(1 - \epsilon)\cdot\Psi_{min}^2}} \\
&= \frac{-0.5\ln(1 - \epsilon)}{t_0^l} \tag{29}
\end{align*}
$$ 

由此可得：
$$
\begin{align*}
\int_{0}^{t_0^l} w_n(t)\mathrm{d}t &= -e^{-\int_{0}^{t_0^l} \frac{A\cdot B}{C}\mathrm{d}u} + 1 \\
&< -e^{-\int_{0}^{t_0^l} \frac{\frac{0.5\ln(1 - \epsilon)}{t_0^l}\cdot 1}{0.5}\mathrm{d}u} + 1 \\
&= -e^{-\int_{0}^{t_0^l} \frac{\ln(1 - \epsilon)}{t_0^l}\mathrm{d}u} + 1 \tag{30}\\
&= -e^{-t_0^l\cdot\frac{\ln(1 - \epsilon)}{t_0^l}} + 1 \\
&= -e^{\ln(1 - \epsilon)} + 1 \\
&= -(1 - \epsilon) + 1 \\
&= \epsilon
\end{align*}
$$ 

由此可得：
$$
\begin{align*}
&\lim_{s \to \infty} \int_{0}^{t_0^l} w_n(t)\mathrm{d}t \\
=&\lim_{s \to \infty} (-e^{-\int_{0}^{t_0^l} \frac{A\cdot B}{C}\mathrm{d}u} + 1) \tag{31}\\
=& 0
\end{align*}
$$

公式31表明，在训练过程中，光线第一次相交之前的权重收敛到零， 因此在第一个相交表面之前，输出颜色不包含任何颜色信息。至此，完成了公式16的证明。 

### 公式17的证明
然后我们给出公式17的证明。与公式18的推导过程相同，我们有：
$$
\begin{align*}
&\int_{t_0^l}^{t_0^*} w_n(t)\mathrm{d}t \\
=&\int_{t_0^l}^{t_0^*} \rho_s(t)e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u} \mathrm{d}t \\
=&\int_{t_0^l}^{t_0^*} -\frac{\partial }{\partial t}e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u} \mathrm{d}t \tag{32}\\
=& -e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u}\big|_{t_0^l}^{t_0^*}\\
=& -e^{-\int_{0}^{t_0^*} \rho_s(u)\mathrm{d}u} + e^{-\int_{0}^{t_0^l} \rho_s(u)\mathrm{d}u}\\
=& -e^{-\int_{0}^{t_0^l} \rho_s(u)\mathrm{d}u - \int_{t_0^l}^{t_0^*} \rho_s(u)\mathrm{d}u} + e^{-\int_{0}^{t_0^l} \rho_s(u)\mathrm{d}u}\\
=&e^{-\int_{0}^{t_0^l} \rho_s(u)\mathrm{d}u} \left(-e^{-\int_{t_0^l}^{t_0^*} \rho_s(u)\mathrm{d}u} + 1\right)
\end{align*}
$$ 

注意到当$t \in (t_0^l, t_0^*)$时，我们有：
$$\frac{\partial\Psi \circ p(t)}{\partial t} = -|\cos\theta| \tag{33}$$

由此可得：
$$
\begin{align*}
&-e^{-\int_{t_0^l}^{t_0^*} \rho_s(u)\mathrm{d}u} + 1 \\
=& -e^{-\int_{t_0^l}^{t_0^*} \max\left\{\frac{\frac{\partial(\Phi_s \circ \Psi \circ p)}{\partial u}(u)}{\Phi_s \circ \Psi \circ p(u)}, 0\right\}\mathrm{d}u} + 1 \\
=& -e^{-\int_{t_0^l}^{t_0^*} \left|\frac{\frac{\partial(\Phi_s \circ \Psi \circ p)}{\partial u}(u)}{\Phi_s \circ \Psi \circ p(u)}\right|\mathrm{d}u} + 1 \\
=& -e^{-\int_{t_0^l}^{t_0^*} \left|\frac{\partial }{\partial u} \ln\Phi_s \circ \Psi \circ p(u)\mathrm{d}u\right|} + 1 \\
=& -e^{-\int_{t_0^l}^{t_0^*} -\frac{\partial }{\partial u} \ln\Phi_s \circ \Psi \circ p(u)\mathrm{d}u} + 1 \tag{34}\\
=& -e^{\ln\Phi_s \circ \Psi \circ p(t_0^*) - \ln\Phi_s \circ \Psi \circ p(t_0^l)} + 1 \\
=& - \frac{e^{\ln\Phi_s \circ \Psi \circ p(t_0^*)}}{e^{\ln\Phi_s \circ \Psi \circ p(t_0^l)}} + 1 \\
=& - \frac{\Phi_s \circ \Psi \circ p(t_0^*)}{\Phi_s \circ \Psi \circ p(t_0^l)} + 1 
\end{align*}
$$ 

由于$t_0^*$是交点，所以有$\Psi \circ p(t_0^*) = 0$且$\Phi_s \circ \Psi \circ p(t_0^*) = 0.5$。由此可得：
$$
\begin{align*}
&-e^{-\int_{t_0^l}^{t_0^*} \rho_s(u)\mathrm{d}u} + 1 \\
=& - \frac{\Phi_s \circ \Psi \circ p(t_0^*)}{\Phi_s \circ \Psi \circ p(t_0^l)} + 1 \tag{35}\\
=& - \frac{0.5}{\Phi_s \circ \Psi \circ p(t_0^l)} + 1 \\
\leqslant& - \frac{0.5}{1} + 1 = 0.5
\end{align*}
$$ 

对于任意$\epsilon > 0$，存在$S = \frac{-\ln 2\epsilon}{\Psi\circ p(t_0^l)}$，使得对于任意$s > S$，有：
$$
\begin{align*}
&-e^{-\int_{t_0^l}^{t_0^*} \rho_s(u)\mathrm{d}u} + 1 \\
=& - \frac{0.5}{\Phi_s \circ \Psi \circ p(t_0^l)} + 1 \\
=& - \frac{0.5}{(1 + e^{-s\cdot\Psi\circ p(t_0^l)})^{-1}} + 1 \\
\geqslant& - \frac{0.5}{(1 + e^{-\frac{-\ln 2\epsilon}{\Psi\circ p(t_0^l)}\cdot\Psi\circ p(t_0^l)})^{-1}} + 1 \tag{36}\\
=& - \frac{0.5}{(1 + e^{\ln 2\epsilon})^{-1}} + 1 \\
=& - \frac{0.5}{(1 + 2\epsilon)^{-1}} + 1 \\
=& -\epsilon + 0.5
\end{align*}
$$ 

公式35和36推导出：
$$\lim_{s \to \infty} \left(-e^{-\int_{t_0^l}^{t_0^*} \rho_s(u)\mathrm{d}u} + 1\right) = 0.5 \tag{37}$$

在公式30中已证明：
$$\lim_{s \to \infty} \left(-e^{-\int_{0}^{t_0^l} \rho(u)\mathrm{d}u} + 1\right) = 0，即\tag{38}$$
$$\lim_{s \to \infty} \left(e^{-\int_{0}^{t_0^l} \rho(u)\mathrm{d}u}\right) = 1 \tag{39}$$

公式32、37和39共同推导出：
$$
\begin{align*}
&\lim_{s \to \infty} \int_{t_0^l}^{t_0^r} w_n(t)\mathrm{d}t \\
=&\lim_{s \to \infty} \left(e^{-\int_{0}^{t_0^l} \rho_s(u)\mathrm{d}u} \left(-e^{-\int_{t_0^l}^{t_0^*} \rho_s(u)\mathrm{d}u} + 1\right)\right) \tag{40}\\
=& 0.5
\end{align*}
$$ 

公式40表明，在有向距离函数（UDF）中，NeuS的渲染颜色$C(o, v)$无法收敛到真实颜色$c(p(t_0^*), v)$，因为高达一半的权重没有受到约束，这会导致渲染颜色混合了不需要的偏差以及固有的几何误差。至此，完成了公式17的证明。

### 偏差的分布
此外，我们阐述偏差的组成部分，例如$c_m$和$c_n$，并展示相应的分布。

对于$t \in (t_0^*, t_1^*)$，其中$t_0^*$和$t_1^*$分别表示沿光线$p(t)$的第一个和第二个交点。考虑以下式子：
$$
\begin{align*}
w_n(t) &= \rho_s(t)e^{-\int_{0}^{t} \rho_s(u)\mathrm{d}u} \\
&= \rho_s(t)e^{-\int_{t_0^*}^{t} \rho_s(u)\mathrm{d}u} \cdot e^{-\int_{0}^{t_0^*} \rho_s(u)\mathrm{d}u} \tag{41}
\end{align*}
$$ 

如已证明的，$\lim_{s \to \infty} e^{-\int_{0}^{t_0^*} \rho_s(u)\mathrm{d}u} = 0.5$，则有：
$$w_n(t_1^*) = 0.5\rho_s(t)e^{-\int_{t_0^*}^{t_1^*} \rho_s(u)\mathrm{d}u} \tag{42}$$

根据假设$\exists(t_1^l, t_1^r) \supset t_1^*$，对于$t \in (t_1^l, t_1^r)$，沿光线的有向距离函数（UDF）值$\Psi(t)$是线性的。所以类似地，我们可以证明：
$$
\begin{align*}
&\lim_{s \to \infty} \int_{0}^{t_1^*} w_n(t)\mathrm{d}t \\
=& 0.5\lim_{s \to \infty} \int_{0}^{t_1^*} \rho_s(t)e^{-\int_{t_0^*}^{t} \rho_s(u)\mathrm{d}u} \mathrm{d}t \tag{43}\\
=& 0.25
\end{align*}
$$

因此，对于任意给定的$k > 0$，我们有：
$$\lim_{s \to \infty} \int_{0}^{t_k^*} w_n(t)\mathrm{d}t = \frac{1}{2^{k + 1}} \tag{44}$$ 

$k$个不可见表面的颜色混合到输出颜色$C(o, v)$中，其权重总和为$\frac{2^k - 1}{2^{k + 1}}$ 。混合颜色的积分$c_m$会导致不需要的偏差$\frac{2^k - 1}{2^{k + 1}}c_m$，在训练过程中无法纠正该偏差。最后一部分权重$1 - 0.5 - \frac{2^k - 1}{2^{k + 1}} = \frac{1}{2^{k + 1}}$ 来自于表面邻域之外的干扰，并在训练过程中产生新的冗余表面。偏差$c_m$和$c_n$会在不可见空间中导致诸如冗余表面和浮动噪声之类的固有几何误差。 

### C. 关于NeUDF无偏差和遮挡感知特性的证明
在本小节中，我们从三个方面阐述NeUDF在有向距离函数（UDF）学习中的能力。首先，我们表明与NeuS不同，NeUDF避免了在UDF中导致有偏差的渲染颜色和固有几何误差的$c_m$和$c_n$。然后，我们分别给出NeUDF的无偏差和遮挡感知特性的证明。

#### C.1 NeUDF中对$c_m$和$c_n$的避免
在给出NeUDF的无偏差和遮挡感知特性的详细证明之前，我们通过引入新的渲染权重函数，简要说明NeUDF不存在不需要的颜色$c_m$和$c_n$：
$$w_r(t) = \tau_r(t)e^{-\int_{0}^{t} \tau_r(u)\mathrm{d}u} \tag{45}$$
$$\tau_r(t) = \left|\frac{\frac{\partial\varsigma_r \circ \Psi \circ p}{\partial t}(t)}{\varsigma_r \circ \Psi \circ p(t)}\right| \tag{46}$$
其中$\varsigma_r(d)$满足：
$$\varsigma_r(0) = 0, \lim_{d \to \infty} = 1 \tag{47}$$
$$\forall d > 0, \varsigma_r'(d) > 0, \varsigma_r''(d) < 0 \tag{48}$$ 

与B部分的推导类似，有：
$$\lim_{r \to \infty} \int_{0}^{t_0^l} w_r(t)\mathrm{d}t = 0 \tag{49}$$

并且
$$
\begin{align*}
&\lim_{r \to \infty} \int_{t_0^l}^{t_0^*} w_n(t)\mathrm{d}t \\
=&\lim_{r \to \infty} e^{-\int_{0}^{t_0^l} \tau_r(u)\mathrm{d}u} \left(-e^{-\int_{t_0^l}^{t_0^*} \tau_r(u)\mathrm{d}u} + 1\right) \tag{50}\\
=&\lim_{r \to \infty} -e^{-\int_{t_0^l}^{t_0^*} \tau_r(u)\mathrm{d}u} + 1
\end{align*}
$$

当$t \in (t_0^l, t_0^r)$时，有：
$$\frac{\partial\Psi \circ p(t)}{\partial t} = -|\cos\theta| < 0 \tag{51}$$ 

我们有：
$$
\begin{align*}
& -e^{-\int_{t_0^l}^{t_0^*} \tau_r(u)\mathrm{d}u} + 1 \\
=& -e^{-\int_{t_0^l}^{t_0^*} \left|\frac{\frac{\partial\varsigma_r \circ \Psi \circ p}{\partial u}(u)}{\varsigma_r \circ \Psi \circ p(u)}\right|\mathrm{d}u} + 1 \\
=& -e^{-\int_{t_0^l}^{t_0^*} \left|\frac{\partial }{\partial u} \ln\varsigma_r \circ \Psi \circ p(u)\right|\mathrm{d}u} + 1 \\
=& -e^{\int_{t_0^l}^{t_0^*} \frac{\partial }{\partial u} \ln\varsigma_r \circ \Psi \circ p(u)\mathrm{d}u} + 1 \tag{52}\\
=& -e^{\ln\varsigma_r \circ \Psi \circ p(t_0^*) - \ln\varsigma \circ \Psi \circ p(t_l)} + 1 \\
=& - \frac{\varsigma_r \circ \Psi \circ p(t_0^*)}{\varsigma \circ \Psi \circ p(t_l)} + 1 \\
=& - 0 + 1 \\
=& 1
\end{align*}
$$ 

所以我们有：
$$\lim_{r \to \infty} \int_{t_0^l}^{t_0^*} w_n(t)\mathrm{d}t = 1 \tag{53}$$

由此可得：
$$
\begin{align*}
\lim_{r \to \infty} C(o, v) =& \lim_{r \to \infty} \int_{t_0^l}^{t_0^*} w_n(t)\mathrm{d}t \cdot c(p(t_0^*), v) \\
&+ (1 - \lim_{r \to \infty} \int_{t_0^l}^{t_0^*} w_n(t)\mathrm{d}t) \cdot c_m \tag{54}\\
=& c(p(t_0^*), v)
\end{align*}
$$

这表明NeUDF避免了由不需要的混合颜色$c_m$（和$c_n$）带来的限制。下一节将给出NeUDF无偏差特性的详细证明。 

#### C.2 NeUDF无偏差特性的证明
直观地说，渲染权重函数应该是无偏差的，也就是说，交点对结果的贡献应该比其邻域更大。在本小节中，我们证明NeUDF是无偏差的：
- 给定光线$p(t)$和有向距离函数（UDF）$\Psi(x)$，NeUDF中的渲染权重$w_r(t)$在交点$t^*$处达到局部最大值。

假设权重$w_r(t)$在零点$t^* \in (t^l, t^r)$的局部邻域$(t^l, t^r)$内是线性函数。我们分别考虑区间$(t^l, t^*)$和$(t^*, t^r)$。对于$t \in (t^l, t^*)$，我们有： 

\[
\begin{align*}
w_r(t) &= \tau_r(t)e^{-\int_{0}^{t} \tau(u)\mathrm{d}u}\\
&= \tau_r(t)e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}e^{-\int_{t^l}^{t} \tau(u)\mathrm{d}u}\\
&= \tau_r(t)e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}e^{-\int_{t^l}^{t} \left|\frac{\frac{\partial\varsigma_r \circ \Psi \circ p}{\partial u}(u)}{\varsigma_r \circ \Psi \circ p(t)}\right|\mathrm{d}u}\\
&= \tau_r(t)e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}e^{-\int_{t^l}^{t} \left|\frac{\partial }{\partial u} \ln\varsigma_r \circ \Psi \circ p(u)\right|\mathrm{d}u}\\
&= \tau_r(t)e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}e^{-\int_{t^l}^{t} \frac{\partial }{\partial u} \ln\varsigma_r \circ \Psi \circ p(u)\mathrm{d}u}\\
&= \tau_r(t)e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}e^{\ln\varsigma_r \circ \Psi \circ p(t)-\ln\varsigma_r \circ \Psi \circ p(t^l)}\\
&= \tau_r(t)e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}\frac{e^{\ln\varsigma_r \circ \Psi \circ p(t)}}{e^{\ln\varsigma_r \circ \Psi \circ p(t^l)}}\\
&= \tau_r(t)e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}\frac{\varsigma_r \circ \Psi \circ p(t)}{\varsigma_r \circ \Psi \circ p(t^l)}\\
&= \left|\frac{\frac{\partial\varsigma_r \circ \Psi \circ p}{\partial u}(u)}{\varsigma_r \circ \Psi \circ p(t)}\right|e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}\frac{\varsigma_r \circ \Psi \circ p(t)}{\varsigma_r \circ \Psi \circ p(t^l)}\\
&= \frac{\left|\frac{\partial\varsigma_r \circ \Psi \circ p(t)}{\partial\Psi \circ p(t)}\right|\cdot\left|\frac{\partial\Psi \circ p(t)}{\partial t}\right|}{|\varsigma_r \circ \Psi \circ p(t)|}e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}\frac{\varsigma_r \circ \Psi \circ p(t)}{\varsigma_r \circ \Psi \circ p(t^l)}\\
&= \frac{|\varsigma_r' \circ \Psi \circ p(t)|\cdot|\cos\theta|}{|\varsigma_r \circ \Psi \circ p(t)|}e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}\frac{\varsigma_r \circ \Psi \circ p(t)}{\varsigma_r \circ \Psi \circ p(t^l)}\\
&= \frac{\varsigma_r' \circ \Psi \circ p(t)\cdot|\cos\theta|}{\varsigma_r \circ \Psi \circ p(t)}e^{-\int_{0}^{t^l} \tau(u)\mathrm{d}u}\frac{\varsigma_r \circ \Psi \circ p(t)}{\varsigma_r \circ \Psi \circ p(t^l)}\\
&= \frac{\varsigma_r' \circ \Psi \circ p(t)\cdot|\cos\theta|\cdot e^{-\int_{0}^{t^l} \tau_r(u)\mathrm{d}u}}{\varsigma_r \circ \Psi \circ p(t^l)} \tag{55}
\end{align*}
\] 

对于给定的参数$r$，$\varsigma_r \circ \Psi \circ p(t^l)$，$e^{-\int_{0}^{t^l} \tau_r(u)\mathrm{d}u}$和$|\cos\theta|$都是常数。所以我们有：
$$w_r(t) = A\cdot\varsigma_r' \circ \Psi \circ p(t), A = \frac{|\cos\theta|\cdot e^{-\int_{0}^{t^l} \tau_r(u)\mathrm{d}u}}{\varsigma_r \circ \Psi \circ p(t^l)}, \tag{56}$$ 
其中对于任何给定的$r$，$A$是一个固定的正数。

注意到$\varsigma_r'(d) > 0$，$\varsigma_r''(d) < 0$，由此可得：
$$w_r(t_1) > w_r(t_2), \forall t_1 > t_2, t_1, t_2 \in (t^l, t^*). \tag{57}$$ 

对于$t \in (t^*, t^r)$，我们有：
$$\tau_r(t)=\left|\frac{\frac{\partial\varsigma_r \circ \Psi \circ p}{\partial t}(t)}{\varsigma_r \circ \Psi \circ p(t)}\right|=\frac{\varsigma_r' \circ \Psi \circ p(t)\cdot|\cos\theta|}{\varsigma_r \circ \Psi \circ p(t)} \tag{58}$$
对于任意$t_1 > t_2$，且$t_1, t_2 \in (t^*, t^r)$，有：
$$\tau_r(t_1) < \tau_r(t_2) \tag{59}$$
$$e^{-\int_{0}^{t_1} \tau_r(u)\mathrm{d}u} < e^{-\int_{0}^{t_2} \tau_r(u)\mathrm{d}u} \tag{60}$$ 

由此可得：
$$w_r(t_1) < w_r(t_2), \forall t_1 > t_2, t_1, t_2 \in (t^l, t^*). \tag{61}$$

公式(57)和(61)表明，离零点越近的点权重值越高。请注意，该证明并不要求严格的零点$t^*$，也就是说，当零点$t^*$存在一个小的扰动$\Delta$ ，即$\Psi \circ p(t^*) = \Delta > 0$ 时，该性质仍然成立。

根据经验，有向距离函数（UDF）的零点被编码为一个小的正数，所以权重函数$w_r(t)$ 沿光线是连续的。因此我们有：
$$w_r(t^*) > w_r(t), \forall t \in (t^l, t^r), t \neq t^* \tag{62}$$
证明完毕。 

#### C.3 NeUDF遮挡感知特性的证明
在本小节中，我们证明NeUDF具有遮挡感知能力。直观地说，对于采样光线中具有相同有向距离函数（UDF）值的两部分，我们希望输出颜色中更多的贡献来自离相机更近的部分。也就是说，离相机更近的表面更有可能具有更高的权重。

具体来说，给定两个表面$S_1$和$S_2$，其中$S_1$离相机更近，对于具有相同UDF值的两个对应点$p(t_1)$和$p(t_2)$，我们有：
$$\int_{t_1}^{t_1 + \delta} w_r(t)\mathrm{d}d_1(t) > \int_{t_2}^{t_2 + \delta} w_r(t)\mathrm{d}d_2(t), \tag{63}$$ 

其中$d_i(t)$表示位置$p(t)$与表面$S_i$之间的距离，$\delta$表示小的步长。
$$\tau_r(t)=\left|\frac{\frac{\partial\varsigma_r \circ \Psi \circ p}{\partial t}(t)}{\varsigma_r \circ \Psi \circ p(t)}\right|=\frac{|\varsigma_r' \circ \Psi \circ p(t)|\cdot|\cos\theta|}{\varsigma_r \circ \Psi \circ p(t)} \tag{64}$$

对于$t_1 < t_2$，$\Psi(t_1) = \Psi(t_2)$，$w_r(t_1), w_r(t_2) > 0$，我们有：
$$\frac{\tau_r(t_1)}{|\cos\theta_1|}=\frac{|\varsigma_r' \circ \Psi \circ p(t_1)|}{\varsigma_r \circ \Psi \circ p(t_1)}=\frac{|\varsigma_r' \circ \Psi \circ p(t_2)|}{\varsigma_r \circ \Psi \circ p(t_2)}=\frac{\tau_r(t_2)}{|\cos\theta_2|} \tag{65}$$ 

$$e^{-\int_{0}^{t_1} \tau_r(u)\mathrm{d}u} > e^{-\int_{0}^{t_2} \tau_r(u)\mathrm{d}u} \tag{66}$$
有：
$$
\begin{align*}
\frac{w_r(t_1)}{|\cos\theta|}&=\frac{\tau_r(t_1)e^{-\int_{0}^{t_1} \tau_r(t_u)\mathrm{d}u}}{|\cos\theta|}\\
&>\frac{\tau_r(t_2)e^{-\int_{0}^{t_2} \tau_r(t_u)\mathrm{d}u}}{|\cos\theta|}=\frac{w_r(t_2)}{|\cos\theta|}
\tag{67}
\end{align*}
$$ 

由此可得：
$$\int_{t_1}^{t_1 + \delta} w_r(t)\mathrm{d}d_1(t)=\int_{t_1}^{t_1 + \delta} \frac{w_r(t)}{|\cos\theta|}\mathrm{d}t \tag{68}$$
$$\int_{t_2}^{t_2 + \delta} w_r(t)\mathrm{d}d_2(t)=\int_{t_2}^{t_2 + \delta} \frac{w_r(t)}{|\cos\theta|}\mathrm{d}t \tag{69}$$
$$\int_{t_1}^{t_1 + \delta} w_r(t)\mathrm{d}d_1(t) > \int_{t_2}^{t_2 + \delta} w_r(t)\mathrm{d}d_2(t), \tag{70}$$
其中$d_i(t)$表示位置$p(t_i)$与表面$S_i$之间的距离。

公式(70)表明，第一个相交表面附近的累积权重高于第二个相交表面。这意味着更多的权重集中在前者表面上。请注意，这里不需要事先假设存在其他相交表面，也就是说，对于沿光线的两个以上的表面相交情况，遮挡感知特性同样成立。至此，完成了遮挡感知特性的证明。 

### D. 实现细节
#### D.1 网络架构
与IDR[53]和NeuS[53]类似，我们使用两个多层感知器（MLP）网络分别对有向距离函数（UDF）和颜色进行编码。UDF网络的输入是空间位置$p(t)$，输出是相应的UDF值以及一个256维的特征向量。UDF网络$\Psi(x)$由8个隐藏层组成，隐藏层大小为256，所有隐藏层和输出层的激活函数选择为$\beta = 100$的Softplus函数。还使用了跳跃连接将输入与第四层的输出相连。

颜色网络的输入是空间位置$p(t)$、视角方向$v$、UDF网络在空间位置$p(t)$处的梯度$n$ ，以及UDF网络导出的相应特征向量。颜色网络$c(x, v)$由4个隐藏层组成，隐藏层大小为256。在UDF网络的梯度$n$作为颜色网络的输入之前，先进行法向正则化处理。与NeuS中一样，采用相同的位置编码和权重归一化方法。 

#### D.2 训练细节
**离散化**：我们采用$\alpha$合成的方法对权重函数进行离散化，通过对$n$个点$p(t_i)=o + t_i\ (i = 1, ..., n, t_i < t_{i + 1})$进行采样，将采样光线划分为多个区间，并根据权重积分在每个区间内累积颜色：
$$
\begin{align*}
\alpha_i&=1 - e^{-\int_{t_i}^{t_{i + 1}} \tau_r(t)\mathrm{d}t}\\
&=\frac{|\varsigma_r \circ \Psi \circ p(t_i)-\varsigma_r \circ \Psi \circ p(t_{i + 1})|}{\varsigma_r \circ \Psi \circ p(t_i)} \tag{71}
\end{align*}
$$

我们对公式(71)进行了略微修改：
$$\alpha_i=\frac{\varsigma_i^{max}-\varsigma_i^{min}}{\varsigma_i^{max}} \tag{72}$$
其中$\varsigma_i^{max}$和$\varsigma_i^{min}$分别是集合$\{\varsigma_r \circ \Psi \circ p(t_i), \varsigma_r \circ \Psi \circ p(t_{i + 1})\}$中的最大值和最小值。 

**上采样**：我们首先对每条光线正式采样64个点，然后在采样权重$w_s(t)$的基础上分层进行重要性采样，再获取另外64个点：
$$w_s(t)=\tau_s(t)e^{-\int_{0}^{t} \tau_s(u)\mathrm{d}u},\tau_s(t)=\zeta_s \circ \Psi \circ p(t) \tag{73}$$

并且$\zeta_s(\cdot)$满足以下规则：对于任意$d > 0$，有$\zeta_s(d)>0$且$\zeta_s'(d)<0$。直观地说，由单调递减函数推导得到的$\tau_s(t)$是一个与视角无关的采样密度，并且该密度与有向距离函数（UDF）值呈正相关。为了推导采样权重$w_s(t)$，应用了经典的体渲染方案。

第$i$个采样点的权重$w_s(t_i)$进行了略微修改：
$$w_s'(t_i)=\max\{w_s(t_{i + k}),k = - 1,0,1\} \tag{74}$$

然后对权重$w_s'(t)$进行归一化处理，使其积分等于1：
$$w_s''(t)=\frac{w_s'(t)}{\sum_{i = 0}^{n - 1} w_s'(t_i)} \tag{75}$$ 

对于每次迭代，我们分层进行两次重要性采样，每次采样32个点，采样点总数为128个。如果没有提供掩码，每条光线会在单位球体外部额外随机采样32个点来表示外部场景。如NeuS[53]中那样，外部场景使用NeRF++[60]来表示。

**平台**：网络使用ADAM优化器进行训练，在前5000次迭代中，学习率逐渐提升至$2\times10^{-4}$，在训练结束时降至$1\times10^{-5}$。每次迭代从随机选择的8个输入相机位姿中采样512条随机光线。在有掩码的设置下，我们在单个英伟达3090 GPU上对每个模型总共训练40万次迭代，耗时9小时；在无掩码的设置下耗时11小时。 

#### D.3.数据准备 

**渲染数据**：为了生成定制化数据，我们使用`pyrender`软件包从真实物体渲染图像。对于每个带纹理的网格或彩色点云，我们渲染200个视角，图像分辨率为800×800像素。图12展示了相机位姿。可选择性地提供具有黑色背景的相应掩码。只有渲染图像和掩码用作网络的输入。

**采集数据**：我们还使用手机拍摄了一些现实世界中的物体。采集的图像是从围绕物体拍摄的视频中提取的。对于书本物体，我们采集了200张分辨率为1920×1440的图像。对于风扇物体，我们采集了59张分辨率为3456×4608的图像。对于植物物体，我们采集了200张分辨率为720×1280的图像。所有相机位姿由COLMAP[43,44]估计，且不提供掩码。

### E. 额外结果
我们展示了NeUDF在DF3D[61]、MGN[4]、DTU[21]、BMVS[56]数据集以及实际拍摄数据上更多的重建结果。图13展示了在无掩码监督的DF3D数据集上NeUDF与NeuS的对比。图14展示了在有掩码监督的DF3D数据集上NeUDF与NeuS的对比。图15展示了在无掩码监督的MGN数据集上NeUDF与NeuS的对比。图16展示了在有掩码监督的MGN数据集上NeUDF与NeuS的对比。图17展示了在有掩码监督的DTU和BMVS数据集上NeUDF与NeuS的对比。图18展示了具有开放表面的实际拍摄场景的额外结果。 