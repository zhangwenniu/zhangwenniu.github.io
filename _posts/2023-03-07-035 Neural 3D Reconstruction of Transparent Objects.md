---
layout: mypost
title: 035 Nerual 3D Reconstruction of Transparent Objects
categories: [论文阅读, 透明物体, 已读论文]
---

# Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes

这篇文章的工作引用大量的早期工作，2000-2010年期间，前深度学习时代的论文。这也说明透明物体表面重建的工作在过去几十年间的发展速度并不迅速。

# 作者

>  Zhengqin Li, Yu-Ying Yeh, Manmohan Chandraker
>  University of California, San Diego
>  {zhl378, yuyeh, mkchandraker}@eng.ucsd.edu

Project Link : [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes (ucsd.edu)](https://cseweb.ucsd.edu//~viscomp/projects/CVPR20Transparent/)

GitHub Link : [lzqsd/TransparentShapeReconstruction (github.com)](https://github.com/lzqsd/TransparentShapeReconstruction)

Bibtex: 

```
@inproceedings{li2020through,
    title={Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes},
    author={Li, Zhengqin and Yeh, Yu-Ying and Chandraker, Manmohan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={1262--1271},
	year={2020}
}
```

# 摘要

# 介绍

# 相关工作

## 多视角立体视觉

## 理论研究

理论研究提到，如果给定在玻璃体内部的折射与透射次数，是能够恢复出来透明物体的三维几何形状的。

*疑问*：形如论文中的有曲率的玻璃造物，折射只发生在物体表面，还是进入物体后会持续不断的发生折射呢？

## 受限条件下的获取

看到一篇文章，2018年的，也是进行透明物体重建的工作。

> [46] Bojian Wu, Yang Zhou, Yiming Qian, Minglun Cong, and Hui Huang. Full 3d reconstruction of transparent objects. *ACM ToG*, 37(4):103:1–103:11, July 2018. 1, 2, 3, 8

## 环境抠图（Environment Matting）

这一段的主要工作是做什么的不太清楚。主要不了解Environment matting是干啥的。

## 从自然图像中重建

本文的作者也能够从图像中恢复出透明物体表面的法向量。

## 除了玻璃以外的折射光材料

这类物体之前没有考虑到，比方说气流、火焰、流体，类似这些物体，如何进行表面重建呢？他们同样也让光路发生了折射。

# 3. 方法

## 设定和假设

工作中的折射率竟然是已知的。分割掩膜竟然也是已知的。

工作的输出是一个透明物体的点云。点云问题到Mesh表面，就涉及到Poison表面重建或者是Phase Transition论文里提到的内容了。

## 形状初始化

我怎么看着不对劲呢？首先有的掩膜构成三维视舱，从而初始化一个透明物体的几何形状。这……先验信息过于丰富了吧？

## 3.1 法向量重建

### 基础网络

看到一个多次折射的建模，我在想，如果是采用直射的方式建模，是不是能让重建的整体性虽然存在偏移，但是由于内部纹理的比较相似性，或者是同步的偏移量，让重建的效果是好的呢？

### 渲染网络

这里提到的双线性采样以得到入社辐射量的方法是从哪得来的？

### 代价量

这一段的大概意思应该是：

MVSNet预测深度图的时候，是在深度方向构建深度代价量，力图让某个深度处代价量最小，这个时候的预测、猜测的深度值和真实的深度值是相同的。

这里将深度变化为法向量，在周围某个空间采样法向量，法向量的采样如果从视舱的错误方向转为正确的方向，会让某个代价量变小，从而导致整体的法向量预测是正确的。如果预测正确，就是想要的非视舱的，存在凹陷区域的形状法向量了。

在这个过程中，需要对三维方向的视角进行变换，而三维空间中角度变化只需要两个角度的自由量就可以了。

前表面和后表面分别需要两次法向量的方向变换，每个变换有两个自由度，这样自由度的空间就是四维了。

### 后处理

## 3.2 点云重建

### 特征映射

### 点云微调

这一篇论文满篇都是问号。。。