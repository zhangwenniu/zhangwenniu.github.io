---
layout: mypost
title: 014 Pixel2Mesh - Generating 3D Mesh Models from Single RGB Images
categories: [论文阅读, GNN, Mesh重建, 已读论文]
---

- [标题](#标题)
  - [期刊](#期刊)
  - [作者](#作者)
- [笔记](#笔记)

# 标题

Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images

从单张RGB图像，生成3D Mesh模型，使用的方式是GCN图卷积网络与CNN卷及网络的融合。

## 期刊

ECCV, 2018, 866 citations

## 作者
Nanyang Wang1 ⋆, Yinda Zhang2 ⋆, Zhuwen Li3 ⋆, Yanwei Fu4, Wei Liu5, Yu-Gang Jiang1 †
1Shanghai Key Lab of Intelligent Information Processing, School of Computer Science, Fudan University 2Princeton University 3Intel Labs 4School of Data Science, Fudan University 5Tencent AI Lab

这篇论文的作者成分有点复杂啊。复旦大学为主力，普林斯顿大学，英特尔实验室，腾讯AI实验室。

# 笔记

-   ECCV的论文。竟然能让写14页啊。CVPR好像只能写8页的样子。·

-   每个顶点有坐标信息，这个坐标信息会用相机内参投影到成像平面上，得到成像平面的坐标。
    
    CNN卷积能够将图像缩小尺度，得到不同通道上的一个个缩小宽度、高度的图像，将这个图像双线性插值到成像平面相同大小，在上述的投影位置得到该位置处的CNN特征。
    
    每个Channel中的小特征图都会有该顶点的特征信息，将所有channel的特征信息拼接起来，就得到了顶点的CNN特征。
    

![image](46HP4D76.png)

-   2022年10月8日15:12:46，中午午觉醒了之后的想法。
    
    pixel2mesh用的是gcn，他用于重建图像中的物体。
    
    但是只能重建仅包含一个物体的图像，不能重建复杂场景的图像，比如多个物体。
    
    我的想法是，如果图中有多个物体，就用图像检测算法，找出图中所有可能的物体，对每个物体分别进行重建。
    
    图中的各个物体分别作为一张graph，这样，一个图就变成了超图，图中的每个节点是一个物体，代表一张图。学习的是各个子图之间的位置关系、遮挡关系等。
    
-   ![image](IJTV5GHD.png)
    
    -   添加采样点的一个直观方法是，在所有三角面片的mesh网格中，加上三角面片的中心点。并且将这个中心点与周围的三角形三个顶点连接起来。这叫做基于平面、表面的方法。
    -   基于边的上采样，其实就是在每条边的中点加上一个顶点。
    -   基于表面的上采样方法，会导致顶点的度是不平衡的。基于边的上采样方法，上采样的结果依然是规则的。
-   Pixel2Mesh论文采用4种损失函数，分别为：
    
    1、Chamfer Loss，约束顶点的位置；
    
    2、法向量loss，保证表面法向量一致；
    
    3、laplacian正则化，在形变过程中保持邻居节点相对位置关系；
    
    4、边长正则化，防止出现外点。
    
-   在解释normal loss的时候，原文这样说：
    
    “optimizing this loss is equivalent as forcing the normal of a locally fitted tangent plane to be consistent with the observation, which works practically well in our experiment” (Wang 等。, 2018, p. 8)
    
    然而，优化这个损失函数，就等价于强制让一个局部适应的切平面的法向量，与观测得到的法向量是一致的。这在实验中的效果很好（pratically well）。
    
    一个主要的点在于，tangent plane是什么。
    
    我搜了tangent，就是曲线中的切线。tangent plane就是曲面中的切平面。考虑一个平面、面片是足够精细的，这样在一个顶点周围的点，就可以近似代替一个顶点的切平面上的一条线，与曲面相切。既然是相切，就表示与法向量垂直，所以这个损失函数是有意义的。
    
-   其实这种方法是由一个问题的，就是它的重建结果是闭包的、包络的、整体的。问题在于，真实重建的场景存在许多空心结构、不连续结构。简单的物体堆叠就会让这种方式无法重建出来两个物体。也就是说，没有图切割的方法。
