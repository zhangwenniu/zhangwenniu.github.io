---
layout: mypost
title: 043 NeRF++ - ANALYZING AND IMPROVINGNEURAL RADIANCE FIELDS
categories: [论文阅读, 透明模型, 读完论文]
---


# 文章信息

## 标题

NeRF++ : ANALYZING AND IMPROVINGNEURAL RADIANCE FIELDS

NeRF++: 对辐射场的分析和提高

## 作者

作者的主页是：[https://kai-46.github.io/website/](https://kai-46.github.io/website/)

教育经历（来自领英的主页[https://www.linkedin.com/in/kai-zhang-53910214a/](https://www.linkedin.com/in/kai-zhang-53910214a/)）：

美国康奈尔大学 - 博士 - 5年
Cornell University
Doctor of Philosophy - PhD Electrical and Computer Engineering 4.0
2017年 - 2022年
美国康奈尔大学 - 硕士 - 4年（不太清楚这是怎么算的）
Cornell University
Master of Science - MS Electrical and Computer Engineering 4.0
2017年 - 2021年
清华大学 - 本科 - 4年
Tsinghua University
Bachelor of Engineering - BE Electronic Engineering 3.8
2013年 - 2017年
社团活动:Minor in business administration

我感觉，做科研想要做出成果，一个是持续学习的能力跟进最新的工作前沿，另一个是快速跟进、快速学习、快速部署的能力，再后就是分析问题的能力、判断力、决策力，找准未来的解决方向。

## 发表信息



## 引用信息

```bash
@article{DBLP:journals/corr/abs-2010-07492,
  author       = {Kai Zhang and
                  Gernot Riegler and
                  Noah Snavely and
                  Vladlen Koltun},
  title        = {NeRF++: Analyzing and Improving Neural Radiance Fields},
  journal      = {CoRR},
  volume       = {abs/2010.07492},
  year         = {2020},
  url          = {https://arxiv.org/abs/2010.07492},
  eprinttype    = {arXiv},
  eprint       = {2010.07492},
  timestamp    = {Thu, 24 Aug 2023 14:55:38 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2010-07492.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


@ARTICLE{2020arXiv201007492Z,
注：这里的第一个姓都是用花括号括起来的，因为jelly不支持这种格式，所以就删掉了。
       author = {Zhang, Kai and Riegler, Gernot and Snavely, Noah and Koltun, Vladlen},
        title = "{NeRF++: Analyzing and Improving Neural Radiance Fields}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2020,
        month = oct,
          eid = {arXiv:2010.07492},
        pages = {arXiv:2010.07492},
          doi = {10.48550/arXiv.2010.07492},
archivePrefix = {arXiv},
       eprint = {2010.07492},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201007492Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@article{Zhang2020NeRFAA,
  title={NeRF++: Analyzing and Improving Neural Radiance Fields},
  author={Kai Zhang and Gernot Riegler and Noah Snavely and Vladlen Koltun},
  journal={ArXiv},
  year={2020},
  volume={abs/2010.07492},
  url={https://api.semanticscholar.org/CorpusID:222380037}
}

```

## 论文链接

[https://arxiv.org/abs/2010.07492](https://arxiv.org/abs/2010.07492)

## 后人对此文章的评价

我认为，这篇文章中的想法比较适用于360度场景的视角合成与预测。应用的实用性比较高。

# 文章内容

## 摘要

神经辐射场在新视角合成方面取得令人惊异的结果，一系列拍摄的场景包括360°拍摄的有界场景、前向拍摄有界场景或者无界场景。NeRF让MLP调整学习表示与视角方向无关的不透明度以及依赖视角的颜色量，输入是一系列图像的集合，在新视角上使用体渲染技术采样。在这篇技术报告中，我们首先讨论辐射场和其中的潜在歧义性，也就是所谓的形状-辐射的歧义性（shape-radiance ambiguity），我们一并分析了NeRF在避免这样的歧义性问题中的成功之处。其次，我们解决了一个参数化的问题，包括将NeRF应用到一个360°拍摄物体的场景中，场景是一个大尺度、无界的场景。我们的方法提升了在这个具有挑战性场景中的成像可信性。代码可以在[https://github.com/Kai-46/nerfplusplus](https://github.com/Kai-46/nerfplusplus)中查阅。