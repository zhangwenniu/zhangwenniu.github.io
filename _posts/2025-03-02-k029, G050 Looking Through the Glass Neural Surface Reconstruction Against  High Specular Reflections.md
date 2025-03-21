---
layout: mypost
title: k029, G050 Looking Through the Glass Neural Surface Reconstruction Against  High Specular Reflections
categories: [SDF, 透明, 反光, 表面重建]
---

如果有研究镜面反射或透明场景3D重建的研究者，可以通过邮箱zhangwenniu@163.com联系我，我目前创建了一个透明场景、镜面反射场景的研究者交流群，欢迎感兴趣的研究者加入。

# 论文链接

- [2023 CVPR Link](https://openaccess.thecvf.com/content/CVPR2023/html/Qiu_Looking_Through_the_Glass_Neural_Surface_Reconstruction_Against_High_Specular_CVPR_2023_paper.html)

- [Arxiv Link](https://arxiv.org/abs/2304.08706)

- [GitHub Link](https://github.com/JiaxiongQ/NeuS-HSR)

发表时间：Tue, 18 Apr 2023 02:34:58 UTC (38,603 KB)

# 重点难点讲解

这篇论文的重点和难点主要集中在如何有效地处理高镜面反射（HSR）对3D表面重建的干扰，以及如何通过神经隐式表示和渲染技术实现高质量的重建。以下是对重点和难点的详细讲解：

### **一、论文的重点**

#### **1.1 问题定义：高镜面反射（HSR）的挑战**

- **背景**：在通过玻璃拍摄目标物体时，玻璃表面的反射光会在目标物体前方形成虚拟图像。这种反射光（高镜面反射）会干扰3D重建过程，导致现有方法（如NeuS）无法正确重建目标物体的表面。

- **问题的核心**：高镜面反射引入了复杂的模糊性，违反了多视角一致性（multi-view consistency），使得现有的神经隐式方法容易混淆反射物体和目标物体。

#### **1.2 核心贡献：NeuS-HSR框架**

- **分解场景**：作者提出将场景分解为两部分：

  - **目标物体（Target Object）**：需要重建的物体本身。

  - **辅助平面（Auxiliary Plane）**：用于模拟镜面反射的部分。

- **增强目标物体的外观**：通过辅助平面模块，将镜面反射的干扰从目标物体中分离出来，从而更准确地重建目标物体的表面。

#### **1.3 方法的关键组成部分**

- **表面模块（Surface Module）**：基于NeuS方法，使用隐式符号距离函数（SDF）表示目标物体的表面，并通过体积渲染生成目标物体的外观。

- **辅助平面模块（Auxiliary Plane Module）**：通过神经网络生成辅助平面的属性（如位置、法线方向等），并利用反射变换模拟镜面反射的效果。

- **渲染过程（Rendering Process）**：将目标物体的外观和辅助平面的外观线性融合，生成最终的渲染图像，并通过与真实图像的对比进行优化。

#### **1.4 实验验证**

- **合成数据集和真实世界数据集**：作者通过合成数据集和真实世界数据集验证了NeuS-HSR的有效性。

- **量化评估**：使用Chamfer距离作为评估指标，NeuS-HSR在合成数据集上的平均Chamfer距离最小，表明其重建质量优于其他方法。

- **定性评估**：NeuS-HSR能够更准确地重建目标物体的表面，保留细节和薄结构，而其他方法（如NeuS和VolSDF）往往会将反射物体误认为是目标物体的一部分。

### **二、论文的难点**

#### **2.1 高镜面反射的复杂性**

- **难点**：高镜面反射会在目标物体前方形成虚拟图像，这些虚拟图像会干扰3D重建过程。现有的神经隐式方法（如NeuS）在这种场景下会失败，因为它们无法区分反射物体和目标物体。

- **解决方案**：NeuS-HSR通过将场景分解为目标物体和辅助平面两部分，利用辅助平面模块模拟镜面反射的效果，从而减少反射对目标物体重建的干扰。

#### **2.2 辅助平面模块的设计**

- **难点**：如何设计一个能够物理地模拟镜面反射的辅助平面模块，并将其与目标物体的重建过程结合。

- **解决方案**：

  - **神经网络生成属性**：通过神经网络`Fr`，将视图方向`v`映射到辅助平面的属性（如体积密度、位置和法线方向）。

  - **反射变换**：对于每个采样点，如果它在辅助平面的后面，将其投影到反射点，从而更真实地模拟镜面反射的效果。

#### **2.3 渲染过程的融合**

- **难点**：如何将目标物体的外观和辅助平面的外观融合在一起，生成最终的渲染图像，并确保模型能够专注于目标物体。

- **解决方案**：

  - **线性融合**：将目标物体的外观`Ct`和辅助平面的外观`Cr`线性组合，生成最终的渲染图像`C`。

  - **优化权重**：通过调整权重系数`φ1`和`φ2`，平衡两部分的贡献，使模型更专注于目标物体。

#### **2.4 模型的泛化能力**

- **难点**：如何确保NeuS-HSR在不同场景（包括合成数据集和真实世界数据集）上都能有效工作。

- **解决方案**：

  - **广泛的实验验证**：作者在合成数据集和真实世界数据集上进行了大量实验，验证了NeuS-HSR的泛化能力。

  - **无监督学习**：NeuS-HSR不需要额外的反射去除监督，能够直接从输入图像中学习目标物体的外观。

### **三、总结**

**重点**：

1. 提出了一种新的框架NeuS-HSR，通过分解场景为目标物体和辅助平面，解决了高镜面反射对3D重建的干扰问题。

2. 设计了辅助平面模块，利用反射变换和神经网络模拟镜面反射，增强目标物体的外观。

3. 在合成数据集和真实世界数据集上验证了NeuS-HSR的有效性，证明其在重建质量和鲁棒性方面优于现有方法。

**难点**：

1. 如何处理高镜面反射的复杂性，避免反射物体对目标物体重建的干扰。

2. 如何设计辅助平面模块，使其能够物理地模拟镜面反射。

3. 如何在渲染过程中平衡目标物体和辅助平面的贡献，确保模型专注于目标物体。

4. 如何确保模型在不同场景下的泛化能力。

通过这些重点和难点的突破，NeuS-HSR为在复杂场景下进行3D表面重建提供了一种有效的解决方案。

# 论文详解

### **1. 背景知识**

#### **1.1 研究问题**

这篇论文主要研究的是**3D物体表面重建**的问题，特别是在**高镜面反射（High Specular Reflections, HSR）**场景下的重建。简单来说，当我们通过玻璃拍摄物体时，玻璃表面的反射光会在目标物体前面形成一个虚拟的反射图像，这种反射光会干扰3D重建过程，导致现有的方法无法准确重建目标物体的表面。

#### **1.2 现有方法的局限性**

论文提到，现有的神经隐式方法（如NeuS）在处理轻微镜面反射时表现良好，但在高镜面反射场景下会失败。这是因为高镜面反射引入了复杂的模糊性，违反了多视角一致性（multi-view consistency），使得现有的方法容易混淆反射物体和目标物体。

#### **1.3 解决方案**

为了解决这个问题，作者提出了一个新的框架——**NeuS-HSR**。这个框架的核心思想是将场景分解为两部分：

1. **目标物体（Target Object）**：需要重建的物体本身。

2. **辅助平面（Auxiliary Plane）**：用于模拟镜面反射的部分。

通过这种分解，NeuS-HSR能够将目标物体的外观与镜面反射的干扰分开，从而更准确地重建目标物体的表面。

### **2. 方法**

#### **2.1 总体框架**

NeuS-HSR的总体框架如图4所示。它将场景分解为目标物体和辅助平面两部分，分别渲染这两部分的外观，最后将它们融合起来，生成最终的渲染图像。这个过程是端到端的，通过与真实图像的对比来优化模型。

#### **2.2 表面模块（Surface Module）**

表面模块基于NeuS方法，使用**隐式符号距离函数（SDF）**来表示目标物体的表面。SDF是一种数学方法，用于表示3D形状，它将空间中的每个点映射到一个标量值，表示该点到物体表面的距离。如果一个点在物体内部，SDF值为负；如果在物体外部，SDF值为正；如果在物体表面上，SDF值为零。

NeuS通过SDF和体积渲染（Volume Rendering）相结合的方式，生成目标物体的外观。具体来说，NeuS定义了一个无偏的、考虑遮挡的权重函数，用于计算每个相机光线上的渲染权重。这个权重函数将SDF与体积渲染连接起来，能够处理复杂的物体结构。

#### **2.3 辅助平面模块（Auxiliary Plane Module）**

辅助平面模块是NeuS-HSR的核心创新之一。它的目标是通过模拟镜面反射，增强目标物体的外观，从而减少镜面反射的干扰。

1. **辅助平面的定义**  

   作者假设场景中的镜面反射可以通过一个**辅助平面（Auxiliary Plane）**来建模。这个辅助平面代表了实际的平面反射器（如玻璃）。对于每个相机光线，作者设计了一个神经网络`Fr`，它将视图方向`v`映射到辅助平面的属性，包括：

   - **体积密度（Volume Density, σr）**：用于生成渲染权重。

   - **平面位置（Plane Position, dr）**：辅助平面与相机光线的交点。

   - **平面法线（Plane Normal, nr）**：辅助平面的法线方向。

2. **反射变换（Reflection Transformation）**  

   为了更真实地模拟镜面反射，作者使用了反射变换。对于每个采样点`p`，如果它在辅助平面的后面，作者会将其投影到反射点`pr`。这样，神经网络可以通过反射点来隐式地追踪入射光，从而更准确地模拟镜面反射的效果。

#### **2.4 渲染过程（Rendering Process）**

渲染过程是将目标物体的外观和辅助平面的外观融合起来，生成最终的渲染图像。具体步骤如下：

1. **目标物体外观（Target Object Appearance）**  

   使用NeuS的方法，通过采样点`p`、目标物体的法线`n`、视图方向`v`和SDF的特征`fp`，通过神经网络`Fc`生成目标物体的颜色值`ct`。

2. **辅助平面外观（Auxiliary Plane Appearance）**  

   对于辅助平面，作者使用了反射点`pr`和平面法线`nr`，同样通过神经网络`Fc`生成辅助平面的颜色值`cr`。

3. **融合（Fusion）**  

   最终的渲染图像`C`是目标物体外观`Ct`和辅助平面外观`Cr`的线性组合：

   $$
   C = \phi_1 C_t + \phi_2 C_r
   $$

   其中，`φ1`和`φ2`是权重系数，用于平衡两部分的贡献。

#### **2.5 损失函数（Loss Function）**

在训练过程中，作者通过最小化渲染图像`C`和真实图像`C̃`之间的差异来优化模型。损失函数由以下几部分组成：

1. **颜色损失（Color Loss, Lc）**：衡量渲染图像和真实图像之间的像素差异。

2. **SDF的正则化损失（Regularized Loss, Lr）**：确保SDF的梯度接近1，从而保证表面的连续性。

3. **平面法线的正则化损失（Ln）**：确保辅助平面的法线方向是单位向量。

最终的损失函数为：

$$
L = L_c + λ_1 (L_r + L_n)
$$

其中，`λ1`是一个超参数，用于平衡不同损失项的权重。

### **3. 实验**

#### **3.1 数据集**

作者使用了合成数据集和真实世界数据集来验证NeuS-HSR的有效性。

1. **合成数据集**  

   作者从DTU数据集中选择了10个场景，通过合成镜面反射来生成高镜面反射场景。具体方法是将目标物体的图像（传输图像`T`）和反射图像`R'`通过高斯卷积合成在一起，模拟镜面反射的效果。

2. **真实世界数据集**  

   作者从互联网上收集了6个高镜面反射场景，并使用COLMAP工具估计相机参数。

#### **3.2 对比方法**

作者将NeuS-HSR与以下几种方法进行了对比：

1. **NeuS**：一种基于SDF的神经隐式表面重建方法。

2. **VolSDF**：一种结合体积渲染和SDF的表面重建方法。

3. **UNISURF**：一种统一的神经隐式表面和辐射场重建方法。

4. **COLMAP**：一种经典的多视角立体重建方法。

#### **3.3 量化评估**

作者使用**Chamfer距离**作为评估指标，衡量重建物体表面与真实物体表面之间的差异。实验结果表明，NeuS-HSR在合成数据集上的平均Chamfer距离最小，表明其重建质量优于其他方法。

#### **3.4 定性评估**

通过可视化重建结果，作者展示了NeuS-HSR在处理高镜面反射场景时的优势。其他方法（如NeuS和VolSDF）往往会将反射物体误认为是目标物体的一部分，导致重建结果中出现噪声和不完整的表面。而NeuS-HSR能够更准确地重建目标物体的表面，保留细节和薄结构。

### **4. 讨论**

#### **4.1 模型组件**

NeuS-HSR的模型由两部分组成：目标物体和辅助平面。目标物体的外观被增强，而辅助平面的外观用于捕捉镜面反射。通过这种分解，模型能够更专注于目标物体，减少镜面反射的干扰。

#### **4.2 注意力分析**

在高镜面反射场景中，模型需要更多地关注目标物体路径，而不是辅助平面路径。实验结果表明，NeuS-HSR的目标物体渲染权重具有更高的峰值和更集中的分布，这表明模型能够更专注于目标物体，从而减少镜面反射的干扰。

#### **4.3 局限性**

尽管NeuS-HSR在处理高镜面反射场景时表现出色，但它仍然继承了神经隐式方法的局限性。例如，在没有先验信息的情况下，模型在未见区域的几何重建可能不准确。一个可能的解决方案是引入物体的对称性先验。

### **5. 结论**

NeuS-HSR提出了一种新的框架，用于在高镜面反射场景下重建目标物体的表面。通过将场景分解为目标物体和辅助平面两部分，NeuS-HSR能够更准确地重建目标物体的表面，减少镜面反射的干扰。实验结果表明，NeuS-HSR在合成数据集和真实世界数据集上的表现优于现有的最先进方法。

### **总结**

这篇论文的核心贡献在于提出了一种新的方法来解决高镜面反射

# 方法部分详解

### **3. 方法**

#### **3.1 问题背景**

在高镜面反射（HSR）场景下，目标物体前方会出现虚拟反射图像，这些反射图像会干扰3D重建过程。现有的神经隐式方法（如NeuS）在这种场景下会失败，因为它们无法区分反射物体和目标物体。为了解决这一问题，作者提出了NeuS-HSR框架，通过将场景分解为目标物体和辅助平面两部分，增强目标物体的外观，从而减少反射的干扰。

### **3.2 NeuS-HSR框架概述**

NeuS-HSR的整体流程如图4所示，包含以下三个主要部分：

1. **表面模块（Surface Module）**：基于NeuS方法，使用隐式符号距离函数（SDF）重建目标物体的表面。

2. **辅助平面模块（Auxiliary Plane Module）**：通过神经网络生成辅助平面的属性（位置、法线、体积密度），并利用反射变换模拟镜面反射。

3. **渲染过程（Rendering Process）**：将目标物体的外观和辅助平面的外观融合，生成最终的渲染图像，并通过与真实图像的对比进行优化。

### **3.3 表面模块（Surface Module）**

表面模块基于NeuS方法，使用隐式符号距离函数（SDF）来表示目标物体的表面。SDF是一种数学方法，用于表示3D形状，它将空间中的每个点映射到一个标量值，表示该点到物体表面的距离：

- 如果点在物体内部，SDF值为负；

- 如果点在物体外部，SDF值为正；

- 如果点在物体表面上，SDF值为零。

NeuS通过SDF和体积渲染相结合的方式，生成目标物体的外观。具体来说：

1. **SDF的定义**  

   SDF表示为$f: \mathbb{R}^3 \rightarrow \mathbb{R}$，目标物体表面$S$可以通过SDF的零水平集表示：

   $$
   S = \{p \in \mathbb{R}^3 \mid f(p) = 0\}.
   $$

2. **渲染权重的计算**  

   NeuS定义了一个无偏的、考虑遮挡的权重函数$w(t)$，用于计算每个相机光线上的渲染权重：

   $$
   w(t) = T(t) \rho(t), \quad T(t) = \exp\left(-\int_0^t \rho(u) du\right),
   $$

   其中$\rho(t)$表示密度函数，定义为：

   $$
   \rho(t) = \max\left(-\frac{d\Theta_s}{dt}(f(p(t))) \Theta_s(f(p(t))), 0\right).
   $$

   这里，$\Theta_s$是Sigmoid函数，$\theta_s$是其导数，用于表示物体表面附近的密度分布。

3. **目标物体的外观生成**  

   对于每个采样点$p_i$，通过神经网络（MLP）预测颜色值$c_i$，最终的目标物体外观$C_t$通过体积渲染公式计算：

   $$
   C_t = \sum_{i=1}^m w_i c_i,
   $$

   其中$m$是采样点的数量，$w_i$是渲染权重，$c_i$是颜色值。

### **3.4 辅助平面模块（Auxiliary Plane Module）**

辅助平面模块是NeuS-HSR的核心创新之一，用于模拟镜面反射，增强目标物体的外观。

1. **辅助平面的定义**  

   作者假设场景中的镜面反射可以通过一个**辅助平面（Auxiliary Plane）**来建模。对于每个相机光线，作者设计了一个神经网络$F_r: S^2 \rightarrow \mathbb{R} \times \mathbb{R} \times \mathbb{R}^3$，将视图方向$v$映射到辅助平面的属性：

   - **体积密度（Volume Density, $\sigma_r$）**：用于生成渲染权重。

   - **平面位置（Plane Position, $d_r$）**：辅助平面与相机光线的交点。

   - **平面法线（Plane Normal, $n_r$）**：辅助平面的法线方向。

   这些属性通过以下公式计算：

   $$
   \{\sigma_r, d_r, n_r\} = F_r(v).
   $$

2. **反射变换（Reflection Transformation）**  

   为了更真实地模拟镜面反射，作者使用了反射变换。对于每个采样点$p$，如果它在辅助平面的后面，作者会将其投影到反射点$p_r$，具体公式为：

   $$
   p_r = p - 2(n_r \cdot (p - p_d)) n_r,
   $$

   其中$p_d$是辅助平面与相机光线的交点。

3. **辅助平面的外观生成**  

   对于辅助平面，作者使用反射点$p_r$和平面法线$n_r$，通过神经网络（MLP）生成颜色值$c_r$。最终的辅助平面外观$C_r$通过体积渲染公式计算：

   $$
   C_r = \sum_{i=1}^m w_{r,i} c_{r,i},
   $$

   其中$w_{r,i}$是基于体积密度$\sigma_r$计算的渲染权重。

### **3.5 渲染过程（Rendering Process）**

渲染过程是将目标物体的外观和辅助平面的外观融合起来，生成最终的渲染图像。

1. **目标物体外观（Target Object Appearance）**  

   使用NeuS的方法，通过采样点$p$、目标物体的法线$n$、视图方向$v$和SDF的特征$f_p$，通过神经网络$F_c$生成目标物体的颜色值$c_t$：

   $$
   c_t = F_c(p, n, v, f_p).
   $$

2. **辅助平面外观（Auxiliary Plane Appearance）**  

   对于辅助平面，作者使用反射点$p_r$和平面法线$n_r$，通过相同的神经网络$F_c$生成颜色值$c_r$：

   $$
   c_r = F_c(p_r, n_r, v, f_p).
   $$

3. **融合（Fusion）**  

   最终的渲染图像$C$是目标物体外观$C_t$和辅助平面外观$C_r$的线性组合：

   $$
   C = \phi_1 C_t + \phi_2 C_r,
   $$

   其中$\phi_1 + \phi_2 = 1$，默认设置为$\phi_1 = 0.3$，$\phi_2 = 0.7$。

### **3.6 损失函数（Loss Function）**

在训练过程中，作者通过最小化渲染图像$C$和真实图像$\tilde{C}$之间的差异来优化模型。损失函数由以下几部分组成：

1. **颜色损失（Color Loss, $L_c$）**：衡量渲染图像和真实图像之间的像素差异：

   $$
   L_c = \frac{1}{b} \sum_{i} \|C_i - \tilde{C}_i\|_1,
   $$

   其中$b$是批量大小。

2. **SDF的正则化损失（Regularized Loss, $L_r$）**：确保SDF的梯度接近1，从而保证表面的连续性：

   $$
   L_r = \frac{1}{b m} \sum_{k,i} \left(\|\nabla f(p_i^k)\| - 1\right)^2,
   $$

   其中$m$是每条光线上的采样点数量。

3. **平面法线的正则化损失（$L_n$）**：确保辅助平面的法线方向是单位向量：

   $$
   L_n = \frac{1}{b} \sum_{i} \left(\|n_{r,i}\| - 1\right)^2.
   $$

最终的损失函数为：

$$
L = L_c + \lambda_1 (L_r + L_n),
$$

其中$\lambda_1$是一个超参数，用于平衡不同损失项的权重，作者默认设置为$\lambda_1 = 0.1$。

### **3.7 方法总结**

NeuS-HSR的核心思想是通过将场景分解为目标物体和辅助平面两部分，利用辅助平面模块模拟镜面反射，从而减少反射对目标物体重建的干扰。具体来说：

1. **表面模块**：基于NeuS方法，使用SDF和体积渲染生成目标物体的外观。

2. **辅助平面模块**：通过神经网络生成辅助平面的属性，并利用反射变换模拟

镜面反射。

3. **渲染过程**：将目标物体的外观和辅助平面的外观融合，生成最终的渲染图像。

4. **损失函数**：通过最小化渲染图像和真实图像之间的差异进行优化。

### **难点解析**

1. **如何模拟镜面反射？**  

   镜面反射的模拟是通过辅助平面模块实现的。作者通过神经网络生成辅助平面的位置、法线和体积密度，并利用反射变换将采样点投影到反射点，从而更真实地模拟镜面反射的效果。

2. **如何平衡目标物体和辅助平面的贡献？**  

   通过线性组合的方式将目标物体的外观和辅助平面的外观融合，并通过调整权重系数$\phi_1$和$\phi_2$来平衡两部分的贡献。此外，损失函数的设计也确保了模型更专注于目标物体。

3. **如何优化模型？**  

   通过最小化渲染图像和真实图像之间的差异进行优化。损失函数包括颜色损失、SDF的正则化损失和平面法线的正则化损失，这些损失项共同确保了模型的稳定性和准确性。

# 原文翻译

# 《透过玻璃看：对抗高镜面反射的神经表面重建》

## **作者**

- Jiaxiong Qiu<sup>1</sup>
- Peng-Tao Jiang<sup>2</sup>
- Yifan Zhu<sup>1</sup>
- Ze-Xin Yin<sup>1</sup>
- Ming-Ming Cheng<sup>1</sup>
- Bo Ren<sup>1</sup>（通讯作者）

<sup>1</sup>VCIP, CS, 南开大学  
<sup>2</sup>浙江大学

## **摘要**

神经隐式方法在轻微镜面高光下能够实现高质量的3D物体表面重建。然而，当通过玻璃拍摄目标物体时，目标物体前方常常出现高镜面反射（HSR）。这些场景中的复杂模糊性破坏了多视角一致性，从而给现有方法带来了正确重建目标物体的巨大挑战。为了解决这一问题，我们提出了一个基于隐式神经渲染的新型表面重建框架NeuS-HSR。在NeuS-HSR中，物体表面被参数化为隐式符号距离函数（SDF）。为了减少HSR的干扰，我们提出将渲染图像分解为两个部分：目标物体和辅助平面。我们设计了一个新颖的辅助平面模块，通过结合物理假设和神经网络生成辅助平面的外观。在合成数据集和真实世界数据集上的大量实验表明，NeuS-HSR在对抗HSR的准确性和鲁棒性方面优于现有的最先进方法。代码可在以下链接获取：[https://github.com/JiaxiongQ/NeuS-HSR](https://github.com/JiaxiongQ/NeuS-HSR)。

## 1. 引言

从多视角图像中重建3D物体表面是计算机视觉和图形学中的一个挑战性任务。最近，NeuS [45] 结合了表面渲染 [3, 12, 35, 52] 和体积渲染 [8, 29]，用于重建具有薄结构的物体，并在输入图像存在轻微镜面反射时表现出色。然而，当处理高镜面反射（HSR）场景时，NeuS无法恢复目标物体表面，如图1第二行所示。在通过玻璃拍摄目标物体时，高镜面反射是无处不在的。如图1第一行所示，在存在HSR的捕获视图中，我们可以识别出目标物体前方的虚拟图像。虚拟图像在物体表面上引入了视觉上的光度变化，这种变化破坏了多视角一致性，并为渲染带来了极大的模糊性，从而导致NeuS错误地重建了反射物体而不是目标物体。

为了适应HSR场景，一个直观的解决方案是首先应用反射去除方法来减少HSR，然后以增强后的目标物体外观作为监督来重建目标物体。然而，大多数最近的单图像反射去除工作 [4, 9, 23, 24, 26, 40] 需要真实背景或反射作为监督，这很难获取。此外，对于这些反射去除方法，测试场景应该出现在训练集中，这限制了它们的泛化能力。这些事实表明，显式使用反射去除方法来增强目标物体外观是不切实际的。一个最近的无监督反射去除方法NeRFReN [18] 通过隐式表示将渲染图像分解为反射和透射部分。然而，它受到受限的视图方向和简单平面反射器的限制。当我们将其应用于多视角重建场景时，如图3所示，它将目标物体视为反射图像中的内容，并未能为目标物体恢复正确的透射图像。如上所述，两阶段的直观解决方案在我们的任务中面临挑战。为了解决这一问题，我们考虑了一种比NeRFReN更有效的分解策略，以增强目标物体的外观，从而在单阶段中实现准确的表面重建。为了实现我们的目标，我们构建了以下假设：

**假设1**：受HSR影响的场景可以分解为目标物体和平面反射器组件。除了目标物体外，HSR和视图中的大多数其他内容都是通过平面反射器（即玻璃）反射和透射的。

**假设2**：平面反射器与相机视图方向相交，因为所有视图方向向量通常都指向目标物体并通过平面反射器。

基于上述物理假设，我们提出了NeuS-HSR，这是一个从RGB图像集中恢复目标物体表面以对抗HSR的新型物体重建框架。对于假设1，如图2所示，我们设计了一个辅助平面来表示平面反射器，因为我们希望通过它增强目标物体的外观。在辅助平面的帮助下，我们能够从监督信号中忠实地分离出目标物体和辅助平面的部分。对于目标物体部分，我们遵循NeuS [45] 来生成目标物体的外观。对于辅助平面部分，我们设计了一个辅助平面模块，以视图方向作为输入（假设2），利用神经网络生成视图依赖的辅助平面的属性（包括法线和位置）。当辅助平面确定后，我们基于反射变换 [16] 和神经网络获得辅助平面的外观。最后，我们将两个外观相加，得到渲染图像，该图像由捕获的图像监督进行单阶段训练。我们进行了一系列实验来评估NeuS-HSR。实验表明，NeuS-HSR在合成数据集上优于其他最先进方法，并且能够从真实世界场景中的HSR影响图像中恢复高质量的目标物体。

总结来说，我们的主要贡献如下：
- 我们提出通过分离场景中的目标物体和辅助平面部分来恢复受HSR影响的目标物体表面。
- 我们设计了一个辅助平面模块，用于物理地生成辅助平面部分的外观，以增强目标物体部分的外观。
- 在合成和真实世界场景上的大量实验表明，我们的方法在定量和定性方面都比其他最先进方法更准确地重建目标物体。


## 2. 相关工作

### 2.1 传统表面重建

经典的多视角表面重建方法主要分为两类：光度立体重建 [5, 6, 19, 50] 和多视角立体重建 [11, 13–15, 36–38]。光度立体重建受到严格实验环境的限制。对于多视角立体重建，输入图像通常围绕目标物体拍摄。早期的多视角立体方法 [11, 15, 36, 37] 主要关注具有漫反射材质的物体表面，它们都遵循Lambertian假设，即物体表面上的相同检测区域在所有视图中变化很小。然而，在真实世界的场景中，物体表面常常出现明显的镜面反射，例如高光，这使得Lambertian假设不再成立。广泛使用的结构光法（Structure From Motion, SFM）[34, 42, 49] 首先用于校准相机并生成每个视点的稀疏深度图。然后，通过泊松表面重建 [22] 与深度融合获得物体表面。然而，输出表面的质量容易受到特征点检测的影响，且目标物体表面缺乏丰富纹理的区域常常会出现伪影或空洞。在本工作中，我们专注于神经隐式方法，以在更真实的场景（即非Lambertian表面）中实现准确的3D物体表面重建。

### 2.2 神经隐式表面渲染

基于神经网络的隐式表示在新视图合成 [25, 27, 29, 39, 47] 和3D重建 [7, 10, 30, 31, 33, 41, 44–46, 48, 51, 52] 方面取得了令人鼓舞的结果。它们具有经典方法所不具备的特性，包括灵活的分辨率和自然的全局一致性。基于可微分光线投射的表面渲染被应用于不同形式的隐式形状表示的表面重建，例如占据函数 [32] 和符号距离函数（SDF）[52]。IDR [52] 通过SDF表示的零水平集提取物体表面上的点，并利用神经网络梯度求解可微分渲染公式。UNISURF [31]、VolSDF [51] 和 NeuS [45] 通过引入体积渲染方案 [29] 来学习隐式表面函数，从而从捕获的图像中提高表面重建质量。NeuralWarp [7] 是一种两阶段方法，用于优化基础模型（例如VolSDF）。NeRS [53] 专注于通过引入Phong模型 [20, 21, 43] 学习物体表面的外观。它使用一个规范球体来表示物体表面，并从稀疏图像集中学习物体纹理，但主要处理具有反射表面的物体，并且生成的物体表面缺乏细节。与这些工作相比，我们提出将物体表面重建扩展到更具挑战性的HSR场景中，并在单阶段中实现目标。我们的目标是正确恢复通过玻璃看到的物体表面，而不是反射表面。我们的方法在HSR场景中实现了比以往工作更好的重建精度和鲁棒性。

## 3. 方法

在本工作中，我们专注于在高镜面反射（HSR）场景下重建物体表面。如引言部分所述，HSR编码了非目标物体的信息，导致目标物体表面质量低下。为了应对HSR场景，我们引入了一种基于隐式神经渲染的新型物体表面重建方法NeuS-HSR。NeuS-HSR的流程如图4所示。具体来说，我们将HSR场景分解为两个组件：目标物体和辅助平面。为了渲染目标物体的外观，我们采用了NeuS的方案，并将其封装为表面模块。为了渲染辅助平面的外观，我们设计了一个基于反射变换 [16] 和多层感知机（MLPs）的辅助平面模块。最后，我们通过线性求和将两个外观融合，以获得渲染图像，该图像由视图中的捕获图像进行监督。以下，我们将分别介绍NeuS-HSR的三个部分，包括表面模块（第3.1节）、辅助平面模块（第3.2节）和渲染过程（第3.3节）。

### 3.1 表面模块

我们采用NeuS [45] 来渲染目标物体的外观。具体来说，NeuS基于隐式符号距离函数（SDF）$f: \mathbb{R}^3 \rightarrow \mathbb{R}$在每条相机光线$h_s$上构建了一个无偏且考虑遮挡的权重函数$w$。首先，$w$定义为：

$$
w(t) = T(t)\rho(t), \quad T(t) = \exp\left(-\int_0^t \rho(u) du\right). \quad (1)
$$

其中$t \in \mathbb{R}$是沿$h_s$的深度值，然后$\rho(t)$由下式构建：

$$
\rho(t) = \max\left(-\frac{d\Theta_s}{dt}(f(p(t))) \Theta_s(f(p(t))), 0\right). \quad (2)
$$

物体表面$S$可以通过点$p$的符号距离的零水平集来建模：

$$
S = \{p \in \mathbb{R}^3 \mid f(p) = 0\}.
$$

Logistic密度分布$\theta_S(p) = \frac{se^{-sp}}{(1 + e^{-sp})^2}$是Sigmoid函数$\Theta_s(p) = (1 + e^{-sp})^{-1}$的导数。其中，$1/s$是$\theta_S(p)$的标准差。构建$w$是NeuS的关键贡献，它正确地将隐式SDF和体积渲染结合起来，以处理复杂的物体结构。相机光线$h_s$在点$p$处可以表示为：

$$
h_s(t) = o + tv,
$$

其中$o$和$v$分别表示相机中心和视图方向。我们沿$h_s$采样$m$个点，然后通过以下公式获得像素颜色值$C$：

$$
C = \sum_{i=1}^m w_i c_i. \quad (3)
$$

其中$w_i = \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$，$c_i$是从MLPs学习到的颜色，而$\alpha_i$是公式(2)的离散化结果。在我们的任务中，$C$编码了HSR的内容，这导致了$w$的模糊性。这种模糊性使得SDF预测的MLPs倾向于建模HSR的内容，从而在目标物体表面周围产生过多的噪声。因此，我们提出了一个辅助平面模块，将MLPs的注意力转移到目标物体上，以处理HSR的干扰。

### 3.2 辅助平面模块

在高镜面反射（HSR）场景中，出现在平面反射器上且位于目标物体前方的虚拟图像为目标物体重建编码了极为模糊的信息。在没有先验信息的情况下，将渲染图像分解以增强目标物体的外观是一个不适定问题。NeRFReN [18] 采用深度平滑先验和双向深度一致性约束，通过隐式表示将渲染图像分解为两个部分：透射图像和反射图像。该方案在视图方向有限且平面反射器简单的情况下表现良好，但在HSR场景中无法在透射图像中保留目标物体。受NeRFReN的启发，我们提出一个辅助平面模块，以增强渲染图像中目标物体的外观。具体来说，我们使用一个辅助平面$R$来表示每条相机光线的实际平面反射器。为了物理地确定$R$，我们设计了一个新颖的神经网络$F_r: S^2 \rightarrow \mathbb{R} \times \mathbb{R} \times \mathbb{R}^3$，如图5（a）所示。$F_r$将视图方向$v$映射到用于生成渲染权重的体积密度$\sigma_r$以及$R$的属性（包括位置$d_r$和平面法线$n_r$），即：

$$
\{\sigma_r, d_r, n_r\} = F_r(v). \quad (4)
$$

我们假设$R$是在相机坐标系中构建的，3D点$p_d = d_r v$是$R$与$h_s$的交点。显然，$p_d$位于$R$上。给定$n_r = [A, B, C]$，$R$可以定义为：

$$
Ax + By + Cz + D = 0. \quad (5)
$$

其中$A^2 + B^2 + C^2 = 1$。我们将$p_d$代入公式(5)，得到$D = -d_r n_r \cdot v$。此外，沿相机光线采样的点是MLPs获取颜色值的输入的一部分。为了进一步物理地建模HSR，如图5（b）所示，对于沿$h_s$采样且位于$R$后方的点$p$，我们基于反射变换 [16] 将其投影到沿入射光路径的反射点$p_r$。然后，MLPs可以隐式地追踪入射光以渲染HSR。图6展示了这一操作在减少HSR干扰方面的有效性。反射点帮助MLPs物理地建模HSR，从而减少场景的模糊性并恢复更准确的目标物体表面。我们投影算法的详细信息在补充材料中进行了说明。

### 3.3 渲染

我们采用神经网络$F_c$分别预测目标路径$c_t$和平面路径$c_a$的颜色值。每条路径的输入不同。对于目标路径，我们遵循NeuS的方法，使用沿相机光线采样的点$p$、目标物体的表面法线$n$、视图方向$v$以及隐式SDF的特征$f_p$作为输入。于是，我们有：

$$
c_t = F_c(p, n, v, f_p).
$$

对于平面路径，相机坐标系中的采样点表示为$p' = p - o$。如图7所示，我们使用位于$R$前方的$p'$的部分点$p_t$以及反射点$p_a$作为输入点$p_r = p_t \cup p_a$。我们使用平面法线$n_r$作为输入法线。于是，对于平面路径，我们有：

$$
c_r = F_c(p_r, n_r, v, f_p).
$$

为了生成每条路径的渲染外观，我们还需要构建两个渲染权重。对于目标路径，我们按照第3.1节中定义的方案生成权重$w$。对于辅助平面路径，给定从平面网络$F_r$学到的体积密度$\sigma_r$，我们采用NeRFReN的方案生成权重$w_r$：

$$
w_i^r = \exp\left(-\sum_{j=1}^{i-1} \sigma_j^r \delta_j\right)(1 - \exp(-\sigma_i^r \delta_i)). \quad (6)
$$

其中$\delta_i = t_{i+1} - t_i$。最终，目标物体的外观$C_t(w, c_t)$和辅助平面的外观$C_r(w_r, c_r)$可以通过公式(3)生成。最终的渲染图像$C$通过$C_t$和$C_r$的线性组合获得，公式如下：

$$
C = \varphi_1 C_t + \varphi_2 C_r. \quad (7)
$$

其中$\varphi_1 + \varphi_2 = 1$。在实际应用中，默认设置为$\varphi_1 = 0.3$和$\varphi_2 = 0.7$。这一设置的详细信息在补充材料中进行了说明。

### 3.4 损失函数

在NeuS-HSR的训练过程中，我们优化渲染图像$C$与捕获图像$\tilde{C}$之间的差异。我们遵循NeuS中定义的损失函数，该函数由以下三个部分组成：颜色损失$L_c$ [45, 52]、隐式SDF的正则化损失$L_r$ [17]以及平面法线的损失$L_n$。损失函数的具体形式如下：

$$
\begin{cases}
L_c = \frac{1}{b} \sum\limits_{i} \text{L1}(C_i, \tilde{C}_i), \\
L_r = \frac{1}{bm} \sum\limits_{k,i} \left(\|\nabla f(p_i^k)\| - 1\right)^2, \\
L_n = \frac{1}{b} \sum\limits_{i} \left(\|n_i^r\| - 1\right)^2,
\end{cases}
$$

其中，$b$表示批量大小，$m$表示沿相机光线采样的点数。最终的损失函数可以定义为：

$$
L = L_c + \lambda_1 (L_r + L_n). \quad (9)
$$

其中，$\lambda_1$是一个常数。在实际应用中，我们默认设置$\lambda_1 = 0.1$。


## 4. 实验

我们进行了广泛的实验，结果表明我们的方法在定量（表1）和定性（图9、图10）方面均优于其他方法。我们还提供了几组消融实验，以揭示我们设计选择的必要性（图11）。

### 4.1 数据集

**合成数据集**  
为了定量评估NeuS-HSR及其他方法的性能，我们从DTU数据集 [1] 中合成了10个场景。我们遵循常见的单图像反射合成方法 [55] 来生成合成数据集。给定传输图像$T$（即包含目标物体的图像）和反射图像$R'$，带有反射的图像$I$可以定义为：

$$
I = T + K \otimes R'. \quad (10)
$$

其中，$K$是高斯核，$\otimes$表示卷积操作。我们随机选择一个场景作为反射部分，其他场景作为传输部分。然后我们采用公式(10)来获取HSR场景。合成数据集的示例如图8所示。

**真实世界数据集**  
为了验证我们的方法在真实世界场景中的有效性，我们从互联网上收集了6个HSR场景。我们利用广泛使用的工具COLMAP [36] 来估计相机参数。

### 4.2 设置

**实现细节**  
符号距离函数（SDF）$f$由多层感知机（MLP）参数化，包含8个线性层。随后，目标物体表面通过Marching Cubes算法 [28] 从隐式SDF中生成。辅助平面函数$F_r$由3层MLP组成，用于预测体积密度，以及2层MLP用于预测平面属性。渲染外观函数$F_c$由4层MLP建模。所有空间点均在单位球体内采样，单位球体外的场景由NeRF++ [54] 生成。采用位置编码 [29] 对沿相机光线采样的点$p$和视图方向$v$进行编码。通过几何初始化 [2] 对近似SDF进行预处理。光线的批量大小设置为512。我们在单个NVIDIA Tesla V100 GPU上训练NeuS-HSR 20万次迭代，耗时约12小时。

**对比方法**  
我们与其他相关方法进行了公平设置下的对比。相关方法包括：（i）最先进的神经隐式表面重建方法：NeuS [45]、VolSDF [51] 和 UNISURF [31]；（ii）经典的多视角立体方法：COLMAP [36]。对于COLMAP，我们使用Screened Poisson [22] 从估计的点云中重建其密集网格。本文中所有基于学习的模型均在没有真实掩码的情况下进行训练。

### 4.3 定量比较

在定量评估中，我们在合成数据集上进行了对比。按照 [31, 45, 51] 的方法，我们采用Chamfer距离作为评估指标，该指标代表目标物体的重建质量。我们在表1中报告了这些指标的分数。

### 4.4 定性比较

如图9所示，我们展示了由不同方法生成的重建结果。可以观察到，其他神经隐式方法生成的物体表面不完整且存在噪声，倾向于建模附着在目标物体上的虚假镜面反射。高镜面反射（HSR）对这些方法恢复目标几何形状造成了严重影响。相反，我们的方法能够获得更清晰的结果，并重建出具有正确几何细节的物体表面。这一事实表明，所提出的辅助平面模块能够减少HSR的干扰，并重建正确的目标物体表面。此外，我们还在更具挑战性的真实世界HSR场景中进一步评估了每种方法的鲁棒性。真实世界的HSR编码了比合成HSR更复杂、更多样的模糊信息，用于恢复目标物体表面。图10中的结果表明，与最先进的神经隐式方法相比，我们的方法能够更好地重建出目标物体的细薄结构。

### 4.5 消融研究

我们进行了几组消融实验，以研究不同设置对辅助平面模块的影响，包括体积密度$\sigma_r$和平面属性（包括位置$d_r$和平面法线$n_r$）。图11展示了每种设置的结果。

- **平面属性的影响**：对于视图中的每条相机光线，我们使用MLPs生成体积密度$\sigma_r$以及辅助平面的属性（$d_r$和$n_r$）。当我们移除辅助平面的属性，仅使用$\sigma_r$来生成权重时，与完整模型相比，性能显著下降。没有平面属性，MLPs无法隐式地追踪入射光以物理地分离目标物体的外观和其他部分。

- **体积密度的影响**：为了确定体积密度$\sigma_r$是否对恢复物体表面是必要的，我们禁用了MLPs的$\sigma_r$输出，并采用与目标路径相同的权重$w$来渲染两种外观。这一操作将来自两条路径的模糊性引入到预测SDF的MLPs中，从而产生了比完整模型更差的结果。然而，即使在这种设置下，我们的模型仍然比基线方法NeuS表现更好，这归功于鲁棒的辅助平面。

### 5. 讨论

**组件**  
我们的模型由两部分组成：目标物体和辅助平面。图12展示了每个部分的组件。目标物体的外观得到了忠实的增强，而HSR（高镜面反射）则由辅助平面模块捕获。辅助平面的表面法线和位置由MLPs自适应学习。在视图的所有相机光线中，平面的法线和位置趋于一致，这在物理上模拟了一个平面反射器。

**注意力分析**  
在HSR场景中，为了恢复准确的目标物体，我们的模型应该更多地关注目标路径，而不是平面路径。如图13所示，目标物体的渲染权重具有更高的峰值和更集中的分布，与辅助平面和NeuS的权重相比，这表明辅助平面模块使MLPs专注于目标物体，从而减少了HSR的干扰，实现了更准确的结果。

**局限性**  
所提出的方法继承了多视角重建中神经隐式方法的不适定性。由于缺乏先验信息，我们的模型在未见区域生成的目标物体几何形状不准确。一个可能的解决方案是引入物体的对称性。

### 6. 结论

在本工作中，我们提出了一个在HSR（高镜面反射）干扰下的多视角物体重建任务。为了解决这一任务，我们提出了NeuS-HSR，这是一个能够抵抗HSR的新型框架，能够恢复准确的3D物体表面。我们提出将通过玻璃捕获的场景分解为目标物体部分和辅助平面部分，通过辅助平面增强目标物体。我们设计了一个辅助平面模块，通过使用MLPs和反射变换物理地生成辅助平面的外观。在合成和真实世界场景上的综合实验表明，NeuS-HSR在定量重建质量和视觉检查方面优于先前的方法。此外，讨论部分探讨了我们在任务中的分解的有效性。

**致谢**  
本工作得到了中国国家重点研发计划（编号2018AAA0100400）、国家自然科学基金（编号61922046）和国家自然科学基金（编号62132012）的支持。