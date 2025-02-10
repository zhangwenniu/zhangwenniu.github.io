---
layout: mypost
title: 062 Dense Reconstruction of Transparent Objects by Altering Incident Light Paths Through Refraction
categories: [论文阅读, 表面重建, 读完论文]
---


# 文章信息

## 标题

Dense Reconstruction of Transparent Objects by Altering Incident Light Paths Through Refraction

## 作者

Kai Han1 · Kwan-Yee K. Wong1 · Miaomiao Liu2

## 发表信息



## 引用信息


## 论文链接


## 后人对此文章的评价


# 文章内容

## 摘要

> 

## 介绍

​19-231207. Dense Reconstruction of Transparent Objects by Altering Incident Light Paths Through Refraction. 本文于2017年被收录于视觉方向的顶刊IJCV。本文解决在设定场景下的透明物体单表面重建问题，并使用单次折射近似重建较薄的透明物体表面。实验将透明物体放在格雷模式码或标定板的上方，在透明物与标定码之间不注入水、注入水的情况下分别拍摄图像，通过比对两次拍摄情况下像素的对应情况，定位光路的变化。作者假设从相机到透明物体第一个表面再到第二个表面的光路并不发生变化，但是从透明物体的第二个表面到参考平面的光路发生偏折，通过计算这两组偏折光路的交点，能够得到远离相机的物体表面。具体而言，作者将参考平面放在两组不同的高度上，分别得到相同像素在参考平面上的对应，因而可以在一组介质中得到一条射线的表达，注入液体之后，又能得到另一条射线，计算两条射线的最小距离所在的两个点，取平均之后作为重建表面的点，文章中同时讨论了精度问题、噪声的剔除方法。计算点云的位置不需要知道折射率的先验知识，完全依赖标定板的标定准确性。如果希望计算点云的法向量，需要提供两次拍摄介质的折射率，比如空气的折射率是1.0，水的折射率是1.33。作者分析平整平面透明物体的光线折射情况，本文由于需要光线的偏折，因而不适用玻璃板的重建。对薄物体而言，本文采用单次折射的假设用于近似实际上的两次折射，作者分析认为这种近似方法的误差与物体的厚度呈线性增长的关系，但是如果物体很薄，会退化为平行透明板的情况，重建误差会增加。做薄物体表面重建时，因为采用单次折射近似，此时不需要将物体放置在液体中，只需要两次参考平面的位置移动即可。作者最后在模拟圆球、模拟锥形面上做仿真重建实验，并实拍玻璃圆盘、鱼形玻璃盘，做透明物体重建。

## 本文的组织结构


- Abstract
- 1 Introduction
- 2 Related Work
- 3 Shape Recovery of Transparent Objects
  - 3.1 Notations and Problem Formulation
  - 3.2 Setup and Assumption
  - 3.3 Dense Refraction Correspondeces
  - 3.4 Light Path Triangulation
  - 3.5 Surface Normal Reconstruction
- 4 Recovery of Thin Transparent Objects
  - 4.1 Setup and Assumptions
  - 4.2 Surface Reconstruction
- 5 Discussion
  - 5.1 Total Internal Reflection
  - 5.2 Object Analysis
  - 5.3 Single Refraction Approximation
- 6 Experimental Evaluation
  - 6.1 Synthetic Data
    - First method on a convex object
    - First method on a concave object
    - Second method on a thin convex cone
    - Second method on a spherical shell
  - 6.2 Real Data
- 7 Conslusions


# Key Points

# Abstract 

