---
layout: mypost
title: Design Space for Graph Neural Networks
categories: [GNN]
---

- [标题](#标题)
  - [任务](#任务)
  - [刊物](#刊物)
  - [作者](#作者)
- [摘要](#摘要)
- [结论](#结论)

# 标题
Design Space for Graph Neural Networks

## 任务
设计构建GNN的度量空间。

## 刊物
NIPS, 2020, 110 citations(till 2022年10月5日22:23:32), [Design Space for Graph Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/c5c3d4fe6b2cc463c7d7ecba17cc9de7-Abstract.html)

## 作者
Jiaxuan You Rex Ying Jure Leskovec Department of Computer Science, Stanford University {jiaxuan, rexy, jure}@cs.stanford.edu

Stanford的大佬。昨天还看到Stanford大佬怼Harford大佬的rap，今天就看到Stanford的论文了。确实立意很高，有大志向。

# 摘要

贡献点：构建比较不同GNN任务和框架的度量标准，给出不同任务可能适宜的GNN网络框架。开源了GraphGym，探索不同GNN任务和设计架构的平台。

结论：不同的GNN任务之间是可以迁移的。


# 结论

- 独到之处在于，此前的任务都是单个任务设计单个GNN框架；当前的研究是系统性的，研究不同GNN设计的可迁移性。
- 最好的GNN设计在不同的任务上相差很大。可能有些GNN模型在一个任务上表现出最佳的性能，在另一些任务上就会表现出很差的性能来了。
- 为了证明一个算法的优势是否存在，关键在于采样随机的模型-任务的组合，接下来探索在什么样的场景下，算法的优越性确实能够提高性能。
- 这篇文章看来是用秩的方法衡量不同任务之间的相似性了。只要任务是可以用秩设计出来，就能用他们的方法进行度量。也算是一种不错的选择。


