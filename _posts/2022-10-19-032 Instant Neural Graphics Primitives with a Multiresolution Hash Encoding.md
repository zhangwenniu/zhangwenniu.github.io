---
layout: mypost
title: 032 Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
categories: [NeRF, 表面重建]
---

- [标题](#标题)
  - [链接](#链接)
  - [作者](#作者)
  - [贡献](#贡献)
- [方法](#方法)

# 标题

Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

SIGGRAPH, 2022, Best Paper

## 链接

[Arxiv Page Link](https://arxiv.org/abs/2201.05989); [Arxiv PDF Link](https://arxiv.org/pdf/2201.05989.pdf)

[Project Page](https://nvlabs.github.io/instant-ngp/)

## 作者

>  THOMAS MÜLLER, NVIDIA, 
>
> Switzerland ALEX EVANS, NVIDIA, 
>
> United Kingdom CHRISTOPH SCHIED, NVIDIA, 
>
> USA ALEXANDER KELLER, NVIDIA, Germany

英伟达2022年的力作，极大的提高了模型的效果。

## 贡献

贡献是加速了NERF的训练速度、渲染速度，加速加疯了。

之前我看的论文，要16个小时训练一个NERF；这篇论文，几秒钟就能训练一个NERF，渲染1920x1080的图像只要几微秒。

# 方法

从底层降低带宽，解决哈希冲突的方向入手，还提高了一些并行性。

