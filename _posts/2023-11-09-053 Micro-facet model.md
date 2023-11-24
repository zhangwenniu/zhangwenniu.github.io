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
在该版本首页的页尾有说明，此版本是Computer Graphics版本的修改版。 

> This paper is a revision of a paper that appeared in Computer Graphics, vol. 15, no. 3, 1981, ACM.

也就是说，两栏的那版论文是初次发表的版本，没有分两栏的版本是后面修改后的版本。

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

# Key Points

# Introduction

- The reflectance model must describe both the color and the spatial distribution of the reflected light. 反射模型需要考虑颜色和反射光的空间分布。
- The model is independent of the other aspects of image synthesis, such as the surface geometry representation and the hidden surface algorithm. 反射模型与物体的几何表达方式无关，无论是用Mesh还是隐式场，反射模型都不发生改变。
- Most real surfaces are neither ideal specular (mirrorlike) reflectors nor ideal diffuse (Lambertian) reflectors. 理想镜面叫做ideal specular, 理想漫反射是Lambertian.
- Phong [14, 15] proposed a reflectance model for computer graphics that was a linear combination of specular and diffuse reflection. Phong的方法提出将反射模型解耦成镜面反射和漫反射的线性组合。
- Whitted [24] extended these models by adding a term for ideal specular reflection from perfectly smooth surfaces. 加上完美的镜面反射项。
- The foregoing models treat reflection as consisting of three components: ambient, diffuse, and specular. 反射由三个分量组成，环境光、漫反射、镜面光。
- The ambient component represents light that is assumed to be uniformly incident from the environment and that is reflected equally in all directions by the surface. 环境光表示光从环境中均匀的发射出来，在物体的表面上均匀的反射。
- The diffuse component represents light that is scattered equally in all directions. The specular component represents highlights, light that is concentrated around the mirror direction. 漫反射分量表示光线在各个方向均匀散射，镜面分量表示高光，集中在镜面的方向上。
- The specular component was assumed to be the color of the light source; the Fresnel equation was used to obtain the angular variation of the intensity, but not the color, of the specular component. 镜面分量假设颜色属于光源，菲涅尔方程用于计算角度的变化，但不考虑颜色的变化。
- The ambient and diffuse components were assumed to be the color of the material. 环境光和漫反射分量假设是材质的颜色。
- The new reflectance model is then applied to the simulation of a metal and a plastic, with an explanation of why images rendered with previous models often look plastic, and how this plastic appearance can be avoided. Cook的模型用于模拟金属和塑料的材质，能够揭示为什么之前的模型渲染图像效果看起来像塑料，以及塑料的观感将如何避免。

# The Reflactance Model

- The spectral composition of the reflected light is determined by the spectral composition of the light source and the wavelength-selective reflection of the surface. 物体的材质对光线是选择性反射的，对不同光谱的分量，反射程度不保持一致。


|Symbol | Description | Chinese Description |
|:---:|:---:|:---:|
| $\alpha$ | Angle between N and H | N和H之间的角度 |
| $\theta$ | Angle between L and H or V and H | L和H之间的角度，或者说是V和H之间的角度 |
| $\lambda$ | Wavelength | 波长 |
| $D$ | Facet slope distribution function | 面片斜率分布函数 |
| $d$ | Fraction of reflactance that is diffuse | 反射中属于漫反射的比例 |
| $d\omega_{i}$ | Solid angle of a beam of incident light | 入射光束的立体角 |
| $E_{i}$ | Energy of incident light | 入射光的能量 |
| $F$ | Reflectance of a perfectly smooth surface | 完美光滑平面的反射率 |
| $f$ | Unblocked fraction of hemisphere | 半球未受到遮挡的比例 |
| $G$ | Geometrical attenuation factor | 几何衰减因子 |
| $H$ | Unit angular bisector of V and L | V和L的单位角平分方向 |
| $I_{i}$ | Average intensity of the incident light | 入射光的平均强度 |
| $I_{ia}$ | Intensity of the incident ambient light | 入射环境光的强度 |
| $I_{r}$ | Intensity of the reflected light | 反射光的强度 |
| $I_{ra}$ | Intensity of the reflected ambient light | 反射光的环境光强度 |
| $k$ | Extinction coefficient | 消光系数 |
| $L$ | Unit vector in the direction of a light | 光的单位方向向量 |
| $m$ | Root mean square slope of faces | 面斜率的均方根 |
| $N$ | Unit surface normal | 单位表面法向量 |
| $n$ | Index of refraction  | 折射率 |
| $R_{a}$ | Ambient reflectance | 环境光反射 |
| $R$ | Total bidirectional reflectance | 全部双向反射 |
| $R_{d}$ | Diffuse bidirectional reflectance | 双向漫反射 |
| $R_{a}$ | Specular bidirectional reflectance | 镜面双向反射 |
| $s$ | Fraction of reflectance that is specular | 反射中镜面的比例 |
| $V$ | Unit vector in direction of the viewer | 观察者方向的单位向量 |
| $\omega$ | Relative weight of a facet slope | 面斜率的相对权重 |


- H = (V+L)/length(V + L) which is the unit normal to a hypothetical surface that would reflect light specularly from the light source to the viewer. H是一个假想中的法向量，沿着该法向量，会将入射光反射到观察者的方向上。
- The energy of the incident light is expressed as energy per unit time and per unit area of the reflecting surface. 入射光的能量（energy）表示为单位时间、单位反射表面上的能量。
- The intensity of the incident light is similar, but is expressed per unit projected area and, in addition, per unit solid angle [20, 8]. 入射光的强度（intensity）是相似的，表示为单位投影面积、单位立体角上。加上了角度的限制，以及投影面积的投影二字的限制。
- Solid angle is the projected area of the light source divided by the square of the distance to the light source and can be treated as a constant for a distant light source. 立体角是光源的投影面积除以到光源距离的平方，对远距离的光源，立体角可以被视作常量。
- $E_{i} = I_{i}(N\cdot L) d\omega{i}$。入射光束的能量。
- Except for mirrors or near-mirrors, the incoming beam is reflected over a wide range of angles. 除了镜子接近镜面的情况，输入的光束会在一个大范围的角度内反射。
- For this reason, the reflected intensity in any given direction depends on the incident energy, not just on the incident intensity. 反射光在指定方向上的强度不止依赖于入射光的强度（包含单位立体角度），主要是依赖于入射的能量。
- The ratio of the reflected intensity in a given direction to the incident energy from another direction (within a small solid angle) is called the bidirectional reflectance. 指定方向上的反射到的强度与入射光在很小的立体角范围内的指定方向上的能量的比值，叫做双向反射值(bidirectional reflectance). 
- The diffuse component originates from internal scattering (in which the incident light penetrates beneath the surface of the material) or from multiple surface reflections (which occur if the surface is sufficiently rough). 漫反射分量主要源自内部散射（当入射光穿过物体表面之后，在物体内部发生散射），或者是物体在表面上发生多次反射（这需要当物体的表面足够粗糙）。
- The specular and diffuse components can have different colors if the material is not homogeneous. 如果物体的材质不够齐次，镜面分量和漫反射分量可以是不同的。
- In addition to direct illumination by individual light sources, an object may be illuminated by background or ambient illumination. All light that is not direct illumination from a specific light source is lumped together into ambient illumination. 除了独立的光源对物体直接照射，物体也可以被背景或者环境光照亮。如果不是从特定光源照过来的直接光，其他的光都可以被归类于环境光照。
- The amount of light reflected toward the viewer from any particular direction of ambient illumination is small, but the effect is significant when integrated over the entire hemisphere of illuminating angles. 从特定的角度反射至指定方向的环境光是很小的，但是在整个半球面上积分之后，光照的效果就很显著了。
- For simplicity, we assume that $R_{a}$, is independent of viewing direction. 在近似将反射量分解为漫反射分量和镜面反射分量的时候，假定漫反射的反射率是与视角方向无关的。
- In addition we assume that the ambient illumination is uniformly incident. 第二个假设，环境光的光照是在各个入射方向上都均匀分布的。
- The term f is the fraction of the illuminating hemisphere that is not blocked by nearby objects (such as a corner) [25]. It is given by $f = \frac{1}{\pi}\int (N\cdot L) d\omega_{i}$, where the integration is done over the unblocked part of illuminating hemisphere. 该数值f表示没有没被周围物体遮挡的概率。
- $I_{r} = I_{ia}R_{a} + \sum_{l} I_{il}(N\cdot L_{l}) d\omega_{il} (sR_{s} + dR_{d})$
- This formulation accounts for the effect of light sources with different intensities and different projected areas which may illuminate a scene. For example, an illuminating beam with the same intensity (Ii) and angle of illumination (N. L) as another beam, but with twice the solid angle (do,i) of that beam, will make a surface appear twice as bright. An illuminating beam with twice the intensity of another beam, but with the same angle of illumination and solid angle, will also make a surface appear twice as bright. 主要关注$d\omega_{il}$，不同的立体角会带来不同的光照效果。相同的角度、相同的亮度，更大范围的立体角，会让照明效果更亮。
- This paper does not consider the reflection of light from other objects in the environment. This reflection can be calculated as in [24] or [6] if the surface is perfectly smooth, but even this pure specular reflection should be wavelength dependent. 即时是完全镜面的反射，都要考虑到不同波长的影响。
- The above reflectance model implicitly depends on several variables. For example, the intensities depend on wavelength, s and d depend on the material, and the reflectances depend on these variables plus the reflection geometry and the surface roughness. 上文中的反射模型隐式的依赖于许多变量。比如说，强度依赖于波长，s和d作为镜面比例和漫反射比例是依赖于材质的，反射依赖于波长、材质，还依赖于反射物体的几何属性和粗糙程度。
- The ambient and diffuse components reflect light equally in all directions. 环境光和漫反射光是在各个方向上均等分布的。环境光是从周围环境影响到观测点，而漫反射是从观测点均匀发射到四周环境中。
- Thus Ra and Ro do not depend on the location of the observer. On the other hand, the specular component reflects more light in some directions than in others, so that Rs does depend on the location of the observer. 环境光反射分量和漫反射分量不依赖于观测者的位置。但是镜面反射分量在某些方向上的分量比较大，会依赖于观测者的位置。
- The angular spread of the specular component can be described by assuming that the surface consists of microfacets, each of which reflects specularly [23]. Only facets whose normal is in the direction H contribute to the specular component of reflection from L to V. 假定表面是由许多微表面组成的，其中法向量是H的表面，将光线反射到观察者的位置。
- 镜面分量可以表示为$R_{s} = \dfrac{F}{\pi} \dfrac{DG}{(N\cdot L)(N \cdot V)}$
- The Fresnel term F describes how light is reflected from each smooth microfacet. It is a function of incidence angle and wavelength and is discussed in the next section. F是菲尼尔项，表示从每个平滑的微表面上是如何反射光线的。F是入射角度和波长的函数。
- The geometrical attenuation factor G accounts for the shadowing and masking of one facet by another and is discussed in detail in [5, 6, 23]. Briefly, it is $G = min\{1, \dfrac{2(N\cdot H)(N\cdot V)}{(V\cdot H)}, \dfrac{2(N\cdot H)(N\cdot L)}{(V\cdot H)} \}$. G是几何衰减项，表示阴影和掩膜，考虑面片被其他面片所遮挡的情况。
- The facet slope distribution function D represents the fraction of the facets that are oriented in the direction H. D表示有多少面片朝向H方向。
- One of the formulations he described is the Gaussian model [23]: $D = ce^{-(\alpha/m)^2}$. 是一种单峰的分布情况。
- Beckmann [2] provided a comprehensive theory that encompasses all of these materials and is applicable to a wide range of surface conditions ranging from smooth to very rough. 关于斜率分布函数，首先是从雷达、红外光的研究开始的。一些人研究物理特性，做过复杂的理论分析，认为有一些函数可以表示从粗糙到光滑的表面斜率分布函数。
- For rough surfaces, the Beckmann distribution function is $D = \dfrac{1}{m^2 \cos^{4}\alpha} e^{-[(\tan \alpha)/m]^2}$ 
- The advantage of the Beckmann function is that it gives the absolute magnitude of the reflectance without introducing arbitrary constants; the disadvantage is that it requires more computation. 讨论了Beckmann函数的优势，是没有引入任意的常量，给出绝对的幅度值；缺点是增大计算量。这已经是四十年前的文章了。
- The wavelength dependence of the reflectance is not affected by the surface roughness except for surfaces that are almost completely smooth, which are described by physical optics (wave theory) and which have a distribution function D that is wavelength dependent. 对反射对光的波长的依赖并没有收到表面粗糙度的影响，除了表面是完全光滑的情况下。这个时候，表面分布函数是与光的波长有依赖关系的。
- The Beckmann distribution model accounts for this wavelength dependence and for the transition region between physical and geometrical optics (i.e., between very smooth surfaces and rough surfaces). For simplicity, we ignore the cases in which D is wavelength dependent. (For a further discussion, see [2] and [9].) Beckmann函数考虑波长的依赖效果，简单起见，我们认为D是与光的波长无关的。
- Some surfaces have two or more scales of roughness, or slope m, and can be modeled by using two or more distribution functions [16]. 不同粗糙度的情况下，使用多组分布函数，加权求和。

# Spectral Composition of the Reflected Light
- The ambient, diffuse, and specular reflectances all depend on wavelength. Ra, Ro, and the F term of Rs may be obtained from the appropriate reflectance spectra for the material. A nonhomogeneous material may have different reflectance spectra for each of the three reflectances, though Ra is restricted to being a linear combination of Rs and Rd. 环境光、漫反射、镜面反射都依赖于光的波长，一个非同质的材质可能对三个反射分量都有不同的数值。
- The reflectance data are usually for illumination at normal incidence. These values are normally measured for polished surfaces and must be multiplied by 1/$\pi$ to obtain the bidirectional reflectance for a rough surface [20]. 讨论材质的反射值，反射值一般是在抛光表面上计算得到的，通过垂直入射的方法度量。在计算粗糙表面的双向反射值的时候，需要对数值除以$\pi$.
- For example, some metals develop an oxide layer with time which can drastically alter the color [1]. 如果金属表面出现金属膜之后，很可能导致材质的颜色发生变化。这是由于不同的材质的双向反射分布函数对不同光谱的吸收程度、反光程度不同。
- The spectral energy distribution of the reflected light is found by multiplying the spectral energy distribution of the incident light by the reflectance spectrum of the surface. 入射光的光谱能量乘以反射的比率，就得到反射光的能量分布。
- In general, Rd and F vary with the geometry of reflection. For convenience, we subsequently take Rd to be the bidirectional reflectance for illumination in a direction normal to the reflecting surface. This is reasonable because the reflectance varies only slightly for incidence angles within about 70° of the surface normal [21]. 一般情况下，漫反射值和菲涅尔项随着反射物体的几何发生变化。方便起见，我们接下来使用Rd在法向量入射时候的光照来计算。这很合理，因为反射值只有在70°倾角的时候才有微小的变化。
- We specifically allow for the directional dependence of F, however, as this leads to a color shift when the directions of incidence and reflection are near grazing. 当入射光和反射光接近相切入射的时候，会有颜色的偏移，所以对菲涅尔项加上了方向的依赖。
- To obtain the spectral and angular variation of F, we have adopted a practical compromise. If n and k are known, we use the Fresnel equation. If not, but the normal reflectance is known, we fit the Fresnel equation to the measured normal reflectance for a polished surface. 为了获取F的光谱和角度上的变化，使用一种折中的方法。如果折射率和消隐系数已知，我们就使用菲涅尔方程。如果折射率和消隐系数未知，我们就调整菲涅尔方程，以适用于度量到的法向量反射值，在一个抛光表面上度量。
- For nonmetals, for which k = 0, this immediately gives us an estimate of the index of refraction n. For metals, for which k is generally not 0, we set k equal to 0 and get an effective value for n from the normal reflectance. The angular dependence of F is then available from the Fresnel equation. The foregoing procedure yields the correct value of F for normal incidence and a good estimate of its angular dependence, which is only weakly dependent on the extinction coefficient k. 对于非金属而言，消隐系数为0的情况下，这就立即给出我们一个折射率的数值。对于金属而言，如果消隐系数不为零，我们人为将消隐系数设置为0，这样能够从垂直入射时候的方程中求解出折射率。将该折射率带回到菲涅尔方程中，就能够得到对角度的依赖关系。菲涅尔项只对于消隐系数有较小的依赖关系。
- The procedure may be repeated at other wavelengths to obtain the spectral and directional dependence of the reflectance. 上述过程可以重复进行，以得到反射值对光谱和方向的依赖关系。

# Applications
- Any color alterations are a result of the reflectance of the surface material. Light that penetrates into the material interacts with the pigments. 这样思考问题：反射出来的颜色变化，是由于物体本身的材质所导致的。
- A plastic may thus be simulated by using a colored diffuse component and a white specular component. 塑料可以使用一个有颜色的漫反射分量和一个白色的镜面分量来模拟。
- Metals conduct electricity. An impinging electromagnetic wave can stimulate the motion of electrons near the surface, which in turn leads to reemission (reflection) of a wave. 金属是导电的。侵入性的电磁波能够促进接近表面的电子的运动，结果是重新发射波，也就是反射。这是金属反光的一种粒子性质解释。
- Thus internal reflections are not present to contribute to a diffuse component, which can be important for a nonmetal. 因此，内部的反射并不作用于漫反射分量，金属内部的漫反射分量并不重要；对于非金属材质而言，漫反射是很重要的。


# Conclusions

CONCLUSIONS The following conclusions can be stated. 
- 1. The specular component is usually the color of the material, not the color of the light source. The ambient, diffuse, and specular components may have different colors if the material is not homogeneous. 镜面分量通常是物体材质本身的，而不是光源的。环境光、漫反射、镜面分量，如果材质并不是同质的，三个分量可能有不同的颜色。
- 2. The concept of bidirectional reflectance is necessary to simulate different light sources and materials in the same scene. 双向反射值对于模拟不同的光源、材质，即使在相同场景中，也是很重要的。
- 3. The facet slope distribution models used by Blinn are easy to calculate and are very similar to others in the optics literature. More than one facet slope distribution function can be combined to represent a surface. Blinn的面斜率分布模型很容易计算，与其他光学文献中的效果是很像的。可以使用不止一种的面斜率分布函数用于组合表示物体的表面。
- 4. The Fresnel equation predicts a color shift of the specular component at grazing angles. Calculating this color shift is computationally expensive unless an approximation procedure or a lookup table is used. 菲涅尔方程用于预测镜面分量在入射角上的颜色偏移数值。计算这种颜色偏移的计算开销很大，要么使用一个近似的算法，要么使用查表方法。
- 5. The spectral energy distribution of light reflected from a specific material can be obtained by using the reflectance model together with the spectral energy distribution of the light source and the reflectance spectrum of the material. The laws of trichromatic color reproduction can be used to convert this spectral energy distribution to the appropriate RGB values for the particular monitor being used. 从一个特定材质上反射的光谱能量，可以通过使用反射模型、光源的光谱能量分布、材质的反射光谱量获得。使用三通道光谱颜色重建，可以用于将光谱能量分布转换成RGB的颜色近似值，用于特定的显示器展示。
- 6. Certain types of materials, notably some painted objects and plastics, have specular and diffuse components that do not have the same color. Metals have a specular component with a color determined by the light source and the reflectance of the metal. The diffuse component is often negligible for metals.对于特定的材质，特别是一些染色的物体和塑料，有不同颜色的镜面分量和漫反射分量。金属有镜面分量，颜色被光源和反射模型所决定。漫反射分量经常不用于金属的表示。