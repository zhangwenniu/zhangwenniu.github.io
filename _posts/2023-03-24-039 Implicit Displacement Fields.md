---
layout: mypost
title: 039 Geometry-consistent Neural Shape Representation with Implicit Displacement Fields
categories: [SDF]
---

# Geometry-consistent Neural Shape Representation with Implicit Displacement Fields

文章发表于2022年的ICLR，以下内容来自于GPT-3.5。

> 国际学习表示与表示会议（International Conference on Learning Representations，简称ICLR），是机器学习领域的一项重要学术会议之一，由国际学习表示和表示组织（International Conference on Learning Representations and Representation Learning，简称ICLRRL）主办。该会议成立于2013年，每年举办一次，会议内容涉及深度学习、表示学习、神经网络、计算机视觉、自然语言处理、强化学习等多个领域。
>
> ICLR旨在提供一个平台，让机器学习领域内的研究人员、学者和业界专家能够分享最新的研究成果、技术、应用和方法，并且可以互相交流、合作和建立联系。ICLR 的独特之处在于其对开放性的支持，它鼓励参与者提交未经发表的研究论文以及非正式的工作报告，以推动学科的快速进步。
>
> ICLR会议不仅是一个展示最新研究成果的平台，同时也是一个评估和推动机器学习领域研究方向的重要机构。ICLR每年都会邀请一批优秀的研究人员和学者担任大会主席、程序委员会主席和评审人员，他们共同组成了一支评估论文质量和决定接收论文的团队，确保会议的学术水准和研究价值。
>
> ICLR已经成为机器学习领域内最具影响力的学术会议之一，许多重要的深度学习算法和技术都是在此提出和发布的。同时，ICLR也为从事机器学习相关工作的学者、研究人员和业界专家提供了一个交流、合作和学习的平台。

> ICLR的网址是 https://iclr.cc/。
>
> 在ICLR官网上，您可以找到有关ICLR会议的详细信息，包括会议计划、注册、提交论文等。您还可以查看过去ICLR会议的记录，并从ICLR的开放访问存储库中获取大部分论文。您可以在https://openreview.net/group?id=ICLR.cc/2019/Conference 上找到2019年ICLR会议的论文，以此类推。此外，很多ICLR发表的论文也可以在arXiv（https://arxiv.org/）和其他学术数据库中找到。

这篇文章是HF-NeuS论文思路的主要提供者，在论文中提出了base function和displacement function的概念，这一概念在后续HF-NeuS中被沿用，用于提升表面重建的质量和效率。在阅读HF-NeuS的过程中，不理解论文里面的displacement function的含义，所以过来读这篇文章。

##  网站

代码：[yifita/idf: implicit displacement field (github.com)](https://github.com/yifita/idf)

论文：[2106.05187\] Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields (arxiv.org)](https://arxiv.org/abs/2106.05187)

论文在arxiv上有三个版本，第一个版本发表于2021年6月9号，第二个版本发表于2021年6月18日，第三个版本发表于2022年2月2日。

---

## Citation

@inproceedings{yifan2021geometry,
  title={Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields},
  author={Yifan, Wang and Rahmann, Lukas and Sorkine-hornung, Olga},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

这里的引用是不是有点问题，应该是ICLR 2022吧。



# 作者

> Wang Yifan ETH Zurich ywang@inf.ethz.ch 
>
> Lukas Rahmann ETH Zurich lukas.rahmann@gmail.com 
>
> Olga Sorkine-Hornung ETH Zurich sorkine@inf.ethz.ch

这几个作者来自于ETH Zurich，是苏黎世联邦理工学院。关于该学院，简介如下，来自于ChatGPT-3.5。

> ETH Zurich是瑞士苏黎世联邦理工学院（Eidgenössische Technische Hochschule Zürich）的简称，是全球知名的公立研究型大学之一。它成立于1855年，以科学和技术为重点，包括16个系和超过30个研究所，涵盖自然科学、工程、计算机科学、数学等领域。ETH Zurich拥有众多杰出的教授和研究团队，培养了无数优秀的科学家、工程师和企业家，如爱因斯坦、高锟等人。该校在全球大学排名中始终位居前列，尤其是在科学和工程领域享有盛誉。

# 摘要

几何表面重建可以分为两个步骤，一个步骤是表示平滑表面的基本函数，另一个是在平滑表面的基础上发生形变，产生表面上的细节，这种置换、形变的网络就是displacement network。

# 简介

文章引用了NeuS, VolSDF，同样是一篇使用神经隐式表达，构建三维几何表面的文章。

---

# 相关工作

---

我不禁要说，GPT确实在很大程度上帮助提升科研效率，以及信息获取的能力。以下是GPT对各向异性的高斯函数解释、公式。

anisotropic Gaussians（各向异性高斯函数）是一种二次型函数，用于在计算机图形学中对物体表面进行建模和渲染。与标准的高斯函数相比，anisotropic Gaussians 具有不同的协方差矩阵，可以控制其在不同方向上的拉伸和压缩程度，从而实现对表面细节的更精确和自然的刻画。这种函数常用于纹理合成、光泽度调整等应用中。

一个二维的各向异性高斯函数可以表示为：

f(x,y) = A * exp(-((a*(x-x0)^2 + 2b*(x-x0)*(y-y0) + c*(y-y0)^2)))

其中，A是幅度因子，a、b、c是协方差矩阵对应的系数，(x0, y0)是高斯函数中心点的坐标。在三维中，类似地，我们可以将其拓展到三个维度（x,y,z），其表达式为：

f(x,y,z) = A * exp(-((a*(x-x0)^2 + 2b*(x-x0)(y-y0)+ 2d*(x-x0)(z-z0) + c*(y-y0)^2 + 2e*(y-y0)*(z-z0) + f*(z-z0)^2)))

其中， a、b、c、d、e、f 是协方差矩阵对应的系数，(x0, y0, z0)是高斯函数中心点的坐标。

LaTeX语法：

一个二维的各向异性高斯函数可以表示为：

$$f(x,y) = A * \exp\left(-\left(\begin{matrix}(x-x_0)&(y-y_0)\end{matrix}\right) \cdot \left(\begin{matrix}a&b\\b&c\end{matrix}\right) \cdot \left(\begin{matrix}(x-x_0)\\(y-y_0)\end{matrix}\right)\right)$$

在三维中，类似地，我们可以将其拓展到三个维度（x,y,z），其表达式为：

$$f(x,y,z) = A * \exp\left(-\left(\begin{matrix}(x-x_0)&(y-y_0)&(z-z_0)\end{matrix}\right) \cdot \left(\begin{matrix}a&b&d\\b&c&e\\d&e&f\end{matrix}\right) \cdot \left(\begin{matrix}(x-x_0)\\(y-y_0)\\(z-z_0)\end{matrix}\right)\right)$$


Markdown语法：

一个二维的各向异性高斯函数可以表示为：

$$f(x,y) = A * exp\left(-\begin{pmatrix}(x-x_0)&(y-y_0)\end{pmatrix}\begin{pmatrix}a&b\\b&c\end{pmatrix}\begin{pmatrix}(x-x_0)\\(y-y_0)\end{pmatrix}\right)$$

在三维中，类似地，我们可以将其拓展到三个维度（x,y,z），其表达式为：

$$f(x,y,z) = A * exp\left(-\begin{pmatrix}(x-x_0)&(y-y_0)&(z-z_0)\end{pmatrix}\begin{pmatrix}a&b&d\\b&c&e\\d&e&f\end{pmatrix}\begin{pmatrix}(x-x_0)\\(y-y_0)\\(z-z_0)\end{pmatrix}\right)$$

---

# 方法

本文中的工作应用的是SIREN网络，该网络将激活函数中的ReLU替换为sin函数，据说通过这种替换，可以在函数的选择上就预先实现低频、高频的分离，是一种预先设置好期望的网络方法。SIREN论文有机会可以一看，不过目前的论文已经排到姥姥家了，很多内容没有来的及看，只能说是抓紧时间看，看完一篇有一篇的收获，看一篇是一篇。

> Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 7462–7473. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 53c04118df112c13a8c34b38343b9c10-Paper.pdf.

---

## 3.1 隐式偏移场（Implicit Displacement Fields, IDF) 

学习displacement function的方法是源于传统。在传统方法中，物体的平滑表面是已知的，在平滑表面上采样离散的点，通过对离散点增加偏移值，变为精细表面。但是本文中有两种不同，第一，表面点并不是离散的，而是处处有值的，这个值是通过sdf、MLP学习到的。第二，平滑的表面本身并不是已知的，而是通过学习得到的，平滑表面、偏移距离都是在线学习得到的。

---

关于近似$f\left(\hat{x}+\hat{d}(\hat{x})n\right)=\hat{f}(\hat{x})$的这一段，意思是，本来在图3里面，n方向是竖直着上去，但是\hat{n}的方向是斜着走的，这种走法会有一定方向上的偏差，总体而言，因为距离比较小，所以可以让两者近似。

由于作者又给出了一段理论上的证明，在误差界范围内，满足利普西斯连续以及Eikonal Loss的函数，是符合近似距离场要求的。

---

## 3.2 网络设计和训练

---

“Therefore, we adopt a progressive learning scheme, which first trains N ωB , and then gradually increase the impact of N ωD .” (Yifan 等, 2022, p. 5) 这一点很重要，如果有一个基函数，一个平移函数，这两者之间的偏移量是需要一种先后顺序的。应当先有基函数的收敛，再逐步训练平移函数。

---

## 3.3 可迁移的隐式位移场

> However, empirical studies (Chan et al., 2020) suggest that SIREN does not handle high-dimensional inputs well.

这里说的Chan et al., 2020指的是下面的文章。

> Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon  Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. arXiv preprint arXiv:2012.00926, 2020.

说是实验发现，SIREN的网络架构，对于高维的输入并不能够很好的处理。

# 4. 实验结果

## 实验细节

最大的可变化因子被设置为0.05了，数值很小。在自己实现的想法中，应当注意。

# 5. 结论和局限性

论文最后说，自己的两组模型需要预先配准好。





说实话，最后的实验结果，都是快速浏览过去的，留下一些印象是，论文中轮流固定基函数、置换函数，并评价他们互相迁移时候的网络性能表现。
