---
layout: mypost
title: 058 A trip down the graphics pipeline pixel coordinates
categories: [图形学]
---


# 文章信息

## 标题

A trip down the graphics pipeline: pixel coordinates

计算机图形学管线中的旅行：像素坐标

## 作者

James F. Blinn, Caltech

这哥们挺有个性的，论文标题就挂着一个自己的照片，而且叫做Jim Blinn's Corner. 

## 发表信息

文章于1991年收录于IEEE Computer Graphics and Applications。

## 引用信息



```
@ARTICLE{126885,
  author={Blinn, J.F.},
  journal={IEEE Computer Graphics and Applications}, 
  title={A trip down the graphics pipeline: pixel coordinates}, 
  year={1991},
  volume={11},
  number={4},
  pages={81-85},
  doi={10.1109/38.126885}}

```

## 论文链接

doi: 10.1109/38.126885

[ieee link](https://ieeexplore.ieee.org/abstract/document/126885)

[sci-hub link](https://sci-hub.hkvisa.net/10.1109/38.126885)


# 文章内容

## 摘要

> It turns out that the obvious way to transform NDC to pixel space is wrong. To show you why, I'll have to go through several intermediate stages before I get to the correct way. 
>
> 事实证明，将标准化设备坐标系转换到像素空间的明显方法是错误的。为了说明这一点，我必须通过几个中间阶段，随后说明正确的方式是怎样的。
>
> There are some problems in graphics that I've spent a lot of time trying to figure out even though they seem real simple. I don't know if this is because I'm dumb or because these problems are really profound. I've never seen any of them described anywhere in the literature; maybe the people who write books just haven't come across them. 
> One set of such problems comes from the use of tranformations in the graphics pipeline. No, I'm not going to beleaguer you with yet more ravings about how wonderful homogeneous coordinates are. You already know that. (And if you don't, you might want to refer to the Big White Book.) What I am going to do is discuss a few of the odd little quirks of the transformation process that I've discovered over the years and how I learned to deal with them. Some of this may look like I'm making a big deal out of nothing, but it's important to get it right. 
> More specifically, over the next few issues I am going to talk about
> 1. a detailed examination of what it means to be a pixel. 
> 2. a nifty way to merge the old concept of the window-to-viewport transformation with clipping space that makes it easy to clip viewports, and 
> 3. what's really going on with the homogeneous perspective transform. 
> These are part of a series begun with my column in the January 1991 issue ("A trip down the graphics pipeline: Line Clipping"). For variety, I probably will intersperse these discussions with columns on other topics.
>
> 在计算机图形学当中，我花了很多时间试图搞清楚一些问题，尽管这些问题可能看起来很简单。我不确定是因为我太笨了还是因为这些问题太过深奥了。这些问题我从没有在其他文献中看到过，可能是这些写书的人并没有遇到这些问题。
> 其中有一些问题来自于图形学管线中的变换。不，我没有要使用这些更乱七八糟的东西来围攻你，让你知道齐次坐标系是多么美妙。你肯定已经知道齐次坐标的好处了。要是你不知道，你可以参考大白皮书。【注：这里指的是J.D. Foley et al., Computer Graphics: Principles and Practice, Addison-Wesley, Reading, Mass., 1990.】我将要讨论的是一些奇怪的问题，我这些年在使用变换的过程中遇到的问题，以及我学到的如何处理这些问题的方法。其中有些可能看起来我干了一些事情但是也没什么，但是这对于把事情弄对很重要。
> 更加具体而言，在接下来的一些问题中，我将要讨论：
> 1. 对于像素而言，一个详细的检查意味着什么。
> 2. 一个精巧的方法，用于将传统的窗口到十点的变化与裁切空间融合起来，能够让裁切视角更加容易。
> 3. 在齐次投影变换的时候会发生什么事情。
> 这些都是我写专栏的时候的系列部分，我的专栏从1991年的1月开始写起（名字叫做，在计算机管线上的旅程：裁剪线）。为了多样性，我可能会用其他的专栏话题来点缀这些讨论。

## 介绍

14-231202. A Trip Down the Graphics Pipeline: Pixel Coordinates. 

文章发表于1991年的IEEE Computer Graphics and Applications，该文章的写作方式接近博客的写作方式，作者的英文辞藻比较丰富，用到许多比喻含义的词汇，以求使读者更好理解作者的表达意思。

作者提出一种将像素为单位的坐标映射到-1及+1之间，文中将宽度映射到-1到+1之间，高度的映射是按照宽高比计算得到的-a到+a区间。

文章同时讨论了最后半个像素的映射方式及使用较小的浮点数差额对应于+1的整数像素映射方法。映射后的坐标系空间为标准化设备坐标（Normalized device coordinates, NDC），应该是NDC映射空间的首次提出者。在NeRF等变种问题中，世界坐标有时候也被映射到NDC空间中。
