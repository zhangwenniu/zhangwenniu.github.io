---
layout: mypost
title: 058 Mip-NeRF 360 Unbounded Anti-Aliased Neural Radiance Fields
categories: [NeRF]
---


# 文章信息

## 标题

Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

Mip-NeRF360：无界抗锯齿的神经辐射网络

## 作者

Jonathan T. Barron 1
Ben Mildenhall 1
Dor Verbin 1,2
Pratul P. Srinivasan 1 
Peter Hedman 1 

1 Google Research
2 Harvard University

主力作者都是谷歌研究院的，其中，本文的第一作者Jonathan T. Barron是NeRF原论文的第四作者。本文的第二作者Ben Mildenhall是NeRF原论文的第一作者。MipNeRF360的第四作者Pratul P. Srinivasan的作者是NeRF原论文的第二作者，也是共同一作。

Ben Midenhall, Pratul P. Srinivasan在2020年ECCV的时候还是UC Berkeley的学生，等到2022CVPR的时候就已经在谷歌研究院了。为未来三到五年开辟了一条新道路的人。

## 发表信息




## 引用信息

该文章收录于2022年的cvpr，论文截稿时间是2021年的年底，当时Instant-NGP还没有出来。

```
@InProceedings{Barron_2022_CVPR,
    author    = {Barron, Jonathan T. and Mildenhall, Ben and Verbin, Dor and Srinivasan, Pratul P. and Hedman, Peter},
    title     = {Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5470-5479}
}
```

## 论文链接

在ieee上收录的链接[https://ieeexplore.ieee.org/document/9878829](https://ieeexplore.ieee.org/document/9878829)

Arxiv上的链接[https://arxiv.org/abs/2111.12077](https://arxiv.org/abs/2111.12077)

GitHub上，Mipnerf360与ref-nerf, RawNeRF合在一起，放成同一个代码仓库里面了。[Github Link](https://github.com/google-research/multinerf)

cvpr 2022的链接[https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html](https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html)


# 文章内容

## 介绍

​16-231204. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. 

该文章发表于2022年的CVPR，本文的1、2、4作分别是NeRF文章中的4、1、2作，入职谷歌研究院工作。

本文主要解决无界场景下的新视角合成问题，与NeRF++相似，将远距离的位置信息采用距离的逆表示，映射到标准化设备坐标下，思想是近距离的多采样，远距离的采样间隔更大一些。

文章基于MipNeRF的圆锥截锥体内的高斯表示，将积分位置编码拓展到远距离处。能够做到抗锯齿、减少浮点伪影、深度图预测更准确的效果。

文章同时提出利用小规模的MLP预测采样权重信息，利用大容量MLP预测场景中的颜色细节信息，对NeRF的粗糙-精细网络架构做了调整，以希望不增大太多训练代价的情况下，提高模型表示的容量，尤其是在无界场景中，更多的场景信息需要被表示。

此外，文章还提出一种优化函数，促使采样区间集中的同时，减少半透明的体密度值，约束体密度接近零或者接近1。在一段时间内，MipNeRF-360用于作为新视角合成中的基准线。
