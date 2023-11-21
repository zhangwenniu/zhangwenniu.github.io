---
layout: mypost
title: 043 ANALYZING AND IMPROVINGNEURAL RADIANCE FIELDS
categories: [论文阅读, NeRF, 读完论文]
---
# 文章信息

## 标题

A Reflectance Model for Computer Graphics

计算机图形学的反射模型

## 作者

Robert L. Cook, Lucasfilm Ltd.

and Kenneth E. Torrance, Cornell University

## 发表信息

论文发表时间：1982年

## 引用信息

这篇文章能下载到两种版本，一种是Computer Graphics的，分两栏：[https://dl.acm.org/doi/pdf/10.1145/965161.806819](https://dl.acm.org/doi/pdf/10.1145/965161.806819). Page: [https://dl.acm.org/doi/abs/10.1145/965161.806819](https://dl.acm.org/doi/abs/10.1145/965161.806819)

```
@article{10.1145/965161.806819,
author = {Cook, Robert L. and Torrance, Kenneth E.},
title = {A Reflectance Model for Computer Graphics},
year = {1981},
issue_date = {August 1981},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {15},
number = {3},
issn = {0097-8930},
url = {https://doi.org/10.1145/965161.806819},
doi = {10.1145/965161.806819},
abstract = {This paper presents a new reflectance model for rendering computer synthesized images. The model accounts for the relative brightness of different materials and light sources in the same scene. It describes the directional distribution of the reflected light and a color shift that occurs as the reflectance changes with incidence angle. The paper presents a method for obtaining the spectral energy distribution of the light reflected from an object made of a specific real material and discusses a procedure for accurately reproducing the color associated with the spectral energy distribution. The model is applied to the simulation of a metal and a plastic.},
journal = {SIGGRAPH Comput. Graph.},
month = {aug},
pages = {307–316},
numpages = {10},
keywords = {Shading, Computer graphics, Reflectance, Image synthesis}
}

@inproceedings{10.1145/800224.806819,
author = {Cook, Robert L. and Torrance, Kenneth E.},
title = {A Reflectance Model for Computer Graphics},
year = {1981},
isbn = {0897910451},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/800224.806819},
doi = {10.1145/800224.806819},
abstract = {This paper presents a new reflectance model for rendering computer synthesized images. The model accounts for the relative brightness of different materials and light sources in the same scene. It describes the directional distribution of the reflected light and a color shift that occurs as the reflectance changes with incidence angle. The paper presents a method for obtaining the spectral energy distribution of the light reflected from an object made of a specific real material and discusses a procedure for accurately reproducing the color associated with the spectral energy distribution. The model is applied to the simulation of a metal and a plastic.},
booktitle = {Proceedings of the 8th Annual Conference on Computer Graphics and Interactive Techniques},
pages = {307–316},
numpages = {10},
keywords = {Reflectance, Computer graphics, Shading, Image synthesis},
location = {Dallas, Texas, USA},
series = {SIGGRAPH '81}
}
```

另一种是ACM Transactions on Graphics, Vol. 1, No. 1, January 1982, Pages 7-24. 不分栏的[https://dl.acm.org/doi/pdf/10.1145/357290.357293](https://dl.acm.org/doi/pdf/10.1145/357290.357293).

Page: [https://dl.acm.org/doi/10.1145/357290.357293](https://dl.acm.org/doi/10.1145/357290.357293)

```
@article{cook1982reflectance,
  title={A reflectance model for computer graphics},
  author={Cook, Robert L and Torrance, Kenneth E.},
  journal={ACM Transactions on Graphics (ToG)},
  volume={1},
  number={1},
  pages={7--24},
  year={1982},
  publisher={ACM New York, NY, USA}
}

@article{10.1145/357290.357293,
author = {Cook, R. L. and Torrance, K. E.},
title = {A Reflectance Model for Computer Graphics},
year = {1982},
issue_date = {Jan. 1982},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {1},
number = {1},
issn = {0730-0301},
url = {https://doi.org/10.1145/357290.357293},
doi = {10.1145/357290.357293},
journal = {ACM Trans. Graph.},
month = {jan},
pages = {7–24},
numpages = {18},
keywords = {image synthesis, reflectance}
}
```



## 论文链接

[A reflectance model for computer graphics](https://dl.acm.org/doi/abs/10.1145/965161.806819)


## 后人对此文章的评价

微表面模型。

对有一定粗糙度的表面，对表面高光项目建模。


# 文章内容

## 摘要

提出了一个新的计算机合成图像的反射模型。本模型考虑在同一场景下不同材质和光源的相对亮度。这描述了反射光的方向分布和一个颜色偏移，因为反射光随着入射角度发生变化。提出一种获取反射光谱能量分布的模型，需要指定由某种材质制成的物体。讨论了一种准确获取光能量分布的过程。本模型用于模拟金属和塑料的材质。

类别：计算机图形学；三维图形学；真实；颜色，光照，阴影，纹理。

一般项：算法。

其他关键词：图像合成，反射。

## 介绍

在计算机图形学中渲染真实图像，需要考虑物体是如何对光进行反射的。反射模型需要考虑反射光的空间分布和颜色分布。反射模型是与其他几何部分是独立开的，与表面的几何表示方法、隐式表面算法是不同的。

多数的真实物体既不是纯镜面反射的，也不是纯粹漫反射的。纯粹的镜面反射就像镜子一样，理想的漫反射又称作Lambertian反射。Phong提出了计算机图形学中的反射模型，是一种镜面反射和漫反射的组合项。Phong模型中的镜面分量是沿着反射方向散布的，这是通过余弦函数的指数次项来得到的。

```
关于Phong模型，可以参见博客：[图形学基础 | Phong光照模型](https://blog.csdn.net/qjh5606/article/details/89761955)

这里引用的两篇文章，分别是Phong模型的博士毕业论文，最初发表时候的文章。

[14] PHONG,B.T. "Illumination for Computer Generated Images." Ph.D. dissertation, Univ. of Utah, Salt Lake City, 1973.

@article{phong1973illumination,
  title={Illumination for computer-generated images [PhD thesis]},
  author={Phong, Bui-Tuong},
  journal={The University of Utah},
  year={1973}
}

[15] PHONG, B.T. Illumination for computer generated pictures. Commun. ACM 18, 6 (June 1975), 311-317.

@article{phong1975illumination,
  title={Illumination for computer generated pictures},
  author={Phong, Bui Tuong},
  journal={Communications of the ACM},
  volume={18},
  number={6},
  pages={311--317},
  year={1975},
  publisher={ACM New York, NY, USA}
}

```
