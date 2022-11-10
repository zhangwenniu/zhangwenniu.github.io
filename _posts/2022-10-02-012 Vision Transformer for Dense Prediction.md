---
layout: mypost
title: 012 Vision Transformer for Dense Prediction
categories: [论文阅读, Transformer, Vision Transformer, Dense Prediction, 已读论文]
---

- [解决问题](#解决问题)
- [解决方法](#解决方法)
- [输入输出](#输入输出)
- [优点](#优点)
- [缺点](#缺点)

> Vision Transformer for Dense Prediction
> Proceedings of the IEEE/CVF International Conference on Computer Vision; 2021; Citation 284 (Till 2022年10月4日)

## 解决问题

对图像稠密预测。论文中有两个分支，深度图预测和语义分割。

## 解决方法

用Vision Transformer解决稠密预测的问题，此前多使用卷积神经网络做此类问题。

## 输入输出

输入是图像。深度图预测问题时候，输出是每个像素的深度，以及可视化的深度图；语义分割问题的时候，输出是每个像素的语义。

## 优点

* 使用Transformer，时间、参数与CNN架构的sota规模近似，性能更佳。
* 可以处理不同分辨率的图像，只要将其转换成不同的token即可。

## 缺点

* 我看代码，似乎仍然用的是缩放到指定图像大小上进行预测的？我以为是针对任意分辨率，直接预测。
* 训练时候用的encoder，没有用到Transformer里面的decoder。这是什么原因？
