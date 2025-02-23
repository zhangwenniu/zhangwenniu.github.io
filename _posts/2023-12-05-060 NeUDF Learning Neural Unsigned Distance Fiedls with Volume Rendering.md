---
layout: mypost
title: 060 NeUDF Learning Neural Unsigned Distance Fields with Volume Rendering
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
