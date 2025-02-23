---
layout: mypost
title: 066 GlassLoc, Plenoptic Grasp Pose Detection in Transparent Clutter
categories: [透明, 抓取]
---


# 文章信息

## 标题

GlassLoc: Plenoptic Grasp Pose Detection in Transparent Clutter


## 作者

Zheming Zhou, Tianyang Pan, Shiyu Wu, Haonan Chang, Odest Chadwicke, Jenkins


## 发表信息



## 引用信息



## 论文链接


## 后人对此文章的评价


# 文章内容

## 摘要

> 

## 介绍



## 本文的组织结构


- Abstract
- 1. Introduction
- II. Related Work
  - A. Grasp Perception In Clutter
  - B. Light Field Photography
- III. Problem Formulation and Approach
- IV. Pleoptic Grasp Pose Detection Methods
  - A. Multi-view Depth Likelihood Volume
  - B. Reflection Suppression
  - C. Grasp Representation and Classification
  - D. Training Data Generation
  - E. Grasp Search
- V. Results
  - A. Experimental Setup
  - B. Evaluation
- VI. Conclusion
- 


# Key Points

# Abstract 

# 原文

## 摘要
 
透明物体在需要灵巧机器人操作的许多环境中普遍存在。这种透明材料会给机器人感知和操作带来相当大的不确定性，仍然是机器人技术面临的一个开放性挑战。当多个透明物体聚集成杂乱的堆积时，这个问题会变得更加严重。例如，在家庭环境中，厨房、餐厅和接待区经常会出现玻璃器皿堆积的情况，这些物体对现代机器人来说基本上是不可见的。我们提出了GlassLoc算法，该算法使用光场感知来检测透明杂物中透明物体的抓取姿态。GlassLoc基于深度似然体积（DLV）描述符对空间中可抓取位置进行分类。我们扩展了DLV，从多个光场视点推断给定空间中透明物体的占用情况。我们在装有第一代Lytro相机的Michigan Progress Fetch机器人上演示和评估了GlassLoc算法。通过对各种透明玻璃器皿在轻微杂乱环境中的抓取检测和执行实验，评估了我们算法的有效性。


## I. 引言  

在家庭环境中进行机器人抓取具有挑战性，这是由于传感器不确定性、场景复杂性和执行器不精确性造成的。最近的研究结果表明，使用点云局部特征[27]和手动标记的抓取置信度[17]的抓取姿态检测(GPD)可以应用于生成各种物体的可行抓取姿态。然而，家庭环境中包含大量透明物体，从厨房用具（如酒杯和容器）到房屋装饰（如窗户和桌子）。这些物体上的反射和透明材料会导致深度相机产生无效读数。这个问题在现实世界中变得更加显著，因为堆积的透明物体会导致机器人在试图与物体交互时产生意外的操作行为。正确估计透明度对于防止机器人执行危险动作和将机器人应用扩展到更具挑战性的场景是必要的。

在透明杂物中执行抓取的问题因机器人无法正确感知和描述透明表面而变得复杂。之前的几种方法[14]，[15]试图通过在深度观测中寻找无效值来解决这个问题，但它们仅限于自上而下的抓取，并假设目标物体在深度图中形成可区分的轮廓（由无效点形成）。最近，几种方法采用光场相机来观察透明度并显示出有希望的结果。Zhou等人[30]使用单次光场图像形成了一种名为深度似然体积(DLV)的新型光场描述符。通过给定相应的物体CAD模型，他们成功估计了单个透明物体或半透明表面后面物体的姿态。基于此，我们将这个想法扩展到更通用的透明物体杂乱环境下的抓取检测场景。

我们在本文中做出了几项贡献。首先，我们提出了GlassLoc算法，用于在分离和轻微重叠的杂乱环境中检测透明物体的六自由度抓取姿态。其次，我们提出了一个通用模型，用于通过多射线融合和反射抑制从多视角光场观测构建深度似然体积。最后，我们将我们的算法与机器人操作流程集成，在八个场景和五种不同的透明物体上执行桌面抓取和放置任务。我们的结果显示，在220次抓取试验中，所有测试物体的抓取成功率为81%。

## VI. 结论

在本文中，我们提出了用于透明杂物机器人操作的GlassLoc算法。我们使用多视角光场观测来构建深度似然体积作为光场描述符，以表征具有多个透明物体的环境。我们证明，通过我们的算法，机器人能够在桌面透明杂乱环境中执行准确的抓取。
