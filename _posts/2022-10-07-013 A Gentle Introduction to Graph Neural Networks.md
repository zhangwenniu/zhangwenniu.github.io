---
layout: mypost
title: 013 A Gentle Introduction to Graph Neural Networks
categories: [GNN]
---

- [标题](#标题)
- [期刊](#期刊)
- [笔记](#笔记)

# 标题
[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
介绍图神经网络的一篇Blog

# 期刊
Distill, 2021.09, 24 citations

[Distill](https://distill.pub/)意思是蒸馏，该网站的特点是所有的论文、博客都尽全力使用大量的交互式图表、精美的图表引导读者逐步学习论文内涵。用到大量的javascript做前端。截止2022年10月7日，上一次Blog发表还是在2021年9月，一年多没有更新了。感觉网站的作者也觉得劳心费力，不再更新了。


# 笔记

笔记

- 将稀疏矩阵用在GPU上面其实一直是比较难的基础性问题。

- 不管有多少个顶点，都只有一个全连接层。也就是所有的顶点都共享一个全连接层和参数。

    ![image](F2LUUXHD.png)

    所有的顶点共享一个MLP，所有的边共享一个MLP，全局信息有一个MLP。一共三个MLP。

-   李沐说，
    
    GNN这里使用的卷积，是将邻域节点的信息加起来，每个顶点的信息比重是完全相等的。
    
    但是CNN窗口里面的权重是变化的，每个像素的权重是不同的。
    
    另一方面，GNN保留了CNN的通道数的情况啊，通过MLP的时候是有通道的感觉的。
    
    我觉得，GNN的通道，是反映在顶点、边、全局这三个通道的。或者说，MLP可以有多个MLP，对应的就是多个channel了。
    
-   通过堆叠信息传递GNN网络层，一个节点就可以最终获得整个图的所有信息。  
    也就是说，在三层之后，一个顶点就距离自己三步的顶点信息了。
    
-   顶点到边的信息，边到顶点的信息。传递有两种方式。
    
    1、把周围边的信息加起来，投影到顶点的维度上。
    
    2、把边、顶点的信息concat在一起，然后再投影。
    
    聚合完周围的信息，最后是进入各自的MLP，学习新的向量、更新。![image](RSGA7ZQM.png)
    
-   李沐说，
    
    对于GNN来说，一个顶点可能跟周围的几个顶点都有关系，在信息传递到最后的时候，每个顶点可能能看到图中的很多信息。
    
    但是神经网络里面，计算梯度的时候，是要记录下来所有涉及到的顶点的。如果传递了所有的图，所有的顶点信息都被记录下来，会导致梯度的计算非常困难，内存占用很大。
    
    因此需要将模型进行采样，原始图很大，可以通过不同的采样方法，记录一个个小子图在传递信息过程中的数值和梯度，便于计算梯度、使计算存储量可以接受。
    
-   噢哟！那GNN应用到三维重建，重建mesh面片的时候，不得费死劲了。很多很多的顶点，信息可怎么传递呢。跟知识图谱是一样的吗，也有很多很多的点，矩阵很大？怎么把三维重建的信息嵌入到图里面呢？
-   李沐说，如果不做假设，什么东西都学不出来。
    
    对于CNN，假设是空间变换的不变性。对于RNN，假设的是时序的连续性。对于GNN，假设的是图的交换不变性。
    
-   GCN是图卷积神经网络；MPNN是Message Passing Neural Network，信息传递神经网络，其实这篇Blog引用过这篇文章。
-   李沐对GNN的评价：1、图本身是一个比较难以优化的东西，在GPU、CPU上加速比较困难。2、GNN是对超参数比较敏感的，架构、采样、优化都很敏感。3、门槛高，导致在工业界的应用比较困难。
