---
layout: book
title: 08 反射模型
book_id: book2
book_title: 机器学习入门
book_description: 这本书介绍了机器学习的基本概念、算法和应用，适合对人工智能感兴趣的读者。
chapter_number: 3
previous_chapter: /books/book2/ch2
date: 2023-02-02
---




# 08 反射模型 

本章定义了一组用于描述光线在表面散射方式的类。回顾一下，在5.6.1节中，我们引入了双向反射分布函数（BRDF）抽象来描述光线在表面的反射，双向透射分布函数（BTDF）来描述光线在表面的透射，以及双向散射分布函数（BSDF）来涵盖这两种效应。在本章中，我们将首先定义这些表面反射和透射函数的通用接口。

许多表面的散射通常最好描述为多个BRDF和BTDF的空间变化混合；在第9章中，我们将引入一个BSDF对象，它结合了多个BRDF和BTDF来表示表面的整体散射。本章暂时回避了反射和透射属性在表面上变化的问题；第10章的纹理类将解决这个问题。BRDF和BTDF明确地只模拟光线在表面单个点进入和离开的散射。对于表现出有意义的次表面光传输的表面，我们将在第11章引入一些相关理论后，在第11.4节中引入BSSRDF类，该类模拟次表面散射。

表面反射模型有多种来源：

• **测量数据**：许多现实世界表面的反射分布特性已在实验室中测量。这些数据可以直接以表格形式使用，或用于计算一组基函数的系数。

• **现象学模型**：试图描述现实世界表面定性特性的方程在模仿这些特性方面可能非常有效。这些类型的BSDF特别容易使用，因为它们通常具有直观的参数来修改其行为（例如，“粗糙度”）。

• **模拟**：有时，已知表面的组成信息的低层次细节。例如，我们可能知道某种涂料是由某种平均大小的彩色颗粒悬浮在介质中，或者某种织物由两种类型的线组成，每种线都有已知的反射特性。在这些情况下，可以模拟微几何结构的光散射以生成反射数据。这种模拟可以在渲染期间进行，也可以作为预处理，之后可以将其拟合到一组基函数中以供渲染期间使用。

• **物理（波动）光学**：一些反射模型是通过详细的光模型推导出来的，将光视为波，并求解麦克斯韦方程以找到它如何从已知特性的表面散射。然而，这些模型往往计算成本高昂，并且在渲染应用中通常并不比基于几何光学的模型更准确。

• **几何光学**：与模拟方法一样，如果已知表面的低层次散射和几何特性，则有时可以直接从这些描述中推导出闭合形式的反射模型。几何光学使得模拟光与表面的相互作用更加容易，因为可以忽略复杂的波效应，如偏振。

本章末尾的“进一步阅读”部分提供了各种此类反射模型的参考资料。

在定义相关接口之前，简要回顾一下它们如何融入整个系统是有必要的。如果使用SamplerIntegrator，则对每条射线调用SamplerIntegrator::Li()方法实现。在找到与几何原语的最接近交点后，它调用与该原语关联的表面着色器。表面着色器作为Material子类的方法实现，负责决定表面上某一点的BSDF是什么；它返回一个BSDF对象，该对象保存了它已分配并初始化的BRDF和BTDF，以表示该点的散射。然后，积分器使用BSDF根据该点的入射照明计算散射光。（使用BDPTIntegrator、MLTIntegrator或SPPMIntegrator而不是SamplerIntegrator的过程大致相似。）

### 基本术语  

为了能够比较不同反射模型的视觉外观，我们将引入一些基本术语来描述表面的反射。  

表面的反射可以分为四大类：**漫反射**、**光泽镜面反射**、**完美镜面反射**和**逆反射**（图8.1）。大多数真实表面的反射是这四种类型的混合。**漫反射表面**将光线均匀地散射到所有方向。虽然完全漫反射表面在物理上无法实现，但接近漫反射表面的例子包括哑光黑板和哑光涂料。**光泽镜面反射表面**（如塑料或高光涂料）优先将光线散射到一组反射方向——它们显示出其他物体的模糊反射。**完美镜面反射表面**将入射光线散射到单一出射方向。镜子和玻璃是完美镜面反射表面的例子。最后，**逆反射表面**（如天鹅绒或月球）主要将光线沿入射方向散射回去。本章中的图像将展示这些不同类型反射在渲染场景中的差异。  

![fig.8.1](pbr.fig.8.1.png)

> 图8.1：表面的反射通常可以根据入射方向（粗线）的反射光分布进行分类：(a) 漫反射，(b) 光泽镜面反射，(c) 完美镜面反射，以及(d) 逆反射分布。



给定特定的反射类别，反射分布函数可以是**各向同性**或**各向异性**的。大多数物体是各向同性的：如果你选择表面上的一个点并围绕该点的法线轴旋转，反射的光线分布不会改变。相比之下，**各向异性材料**在旋转时会反射不同量的光。各向异性表面的例子包括拉丝金属、许多类型的布料和光盘。

### 几何设置  

![fig.8.2](pbr.fig.8.2.png)

> 图8.2：基本BSDF接口设置。着色坐标系由正交基向量$(s, t, n)$定义。我们将这些向量定向为在该坐标系中沿x、y和z轴。在世界空间中的方向向量$\omega$在调用任何BRDF或BTDF方法之前被转换到着色坐标系。  




pbrt中的反射计算在反射坐标系中进行评估，其中被着色点的两个切向量和法向量分别与x、y和z轴对齐（图8.2）。传递给BRDF和BTDF例程并从中返回的所有方向向量都将相对于该坐标系定义。理解这个坐标系对于理解本章中的BRDF和BTDF实现非常重要。  

着色坐标系还为用球坐标$(\theta, \phi)$表达方向提供了一个框架；角度$\theta$是从给定方向到z轴的夹角，$\phi$是方向投影到xy平面后与x轴形成的角度。给定该坐标系中的方向向量$\omega$，可以轻松计算其与法线方向形成的角度的余弦值：  

$$
\cos \theta = (\mathbf{n} \cdot \omega) = ((0, 0, 1) \cdot \omega) = \omega_z.
$$  

我们将提供实用函数来计算该值及其一些有用的变体；它们的使用有助于澄清BRDF和BTDF的实现。  

```cpp
〈BSDF 内联函数〉 ≡  
inline Float CosTheta(const Vector3f &w) { return w.z; }  
inline Float Cos2Theta(const Vector3f &w) { return w.z * w.z; }  
inline Float AbsCosTheta(const Vector3f &w) { return std::abs(w.z); }  
```  

$\sin^2 \theta$的值可以使用三角恒等式$\sin^2 \theta + \cos^2 \theta = 1$计算，尽管我们需要小心避免在极少数情况下由于浮点舍入误差导致$1 - \text{Cos2Theta(w)}$小于零时对负数取平方根。  

```cpp
〈BSDF 内联函数〉 +≡  
inline Float Sin2Theta(const Vector3f &w) {  
    return std::max((Float)0, (Float)1 - Cos2Theta(w));  
}  
inline Float SinTheta(const Vector3f &w) {  
    return std::sqrt(Sin2Theta(w));  
}  
```  

角度$\theta$的正切值可以通过恒等式$\tan \theta = \sin \theta / \cos \theta$计算。  

```cpp
〈BSDF 内联函数〉 +≡  
inline Float TanTheta(const Vector3f &w) {  
    return SinTheta(w) / CosTheta(w);  
}  
inline Float Tan2Theta(const Vector3f &w) {  
    return Sin2Theta(w) / Cos2Theta(w);  
}  
```

![fig.8.3](pbr.fig.8.3.png)

> 图8.3：$\sin \phi$和$\cos \phi$的值可以使用圆坐标方程$x = r \cos \phi$和$y = r \sin \phi$计算，其中$r$（虚线长度）等于$\sin \theta$。  


我们同样可以使用着色坐标系来简化$\phi$角的正弦和余弦计算（图8.3）。在被着色点的平面中，向量$\omega$的坐标为$(x, y)$，分别由$r \cos \phi$和$r \sin \phi$给出。半径$r$为$\sin \theta$，因此：  
$$
\cos \phi = \frac{x}{r} = \frac{x}{\sin \theta}, \quad \sin \phi = \frac{y}{r} = \frac{y}{\sin \theta}.
$$  

```cpp
〈BSDF 内联函数〉 +≡  
inline Float CosPhi(const Vector3f &w) {  
    Float sinTheta = SinTheta(w);  
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);  
}  
inline Float SinPhi(const Vector3f &w) {  
    Float sinTheta = SinTheta(w);  
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);  
}  
```  

```cpp
〈BSDF 内联函数〉 +≡  
inline Float Cos2Phi(const Vector3f &w) {  
    return CosPhi(w) * CosPhi(w);  
}  
inline Float Sin2Phi(const Vector3f &w) {  
    return SinPhi(w) * SinPhi(w);  
}  
```  

在着色坐标系中，两个向量之间角度$\Delta\phi$的余弦可以通过将两个向量的z坐标归零得到2D向量，然后对它们进行归一化来找到。这两个向量的点积给出了它们之间角度的余弦。下面的实现重新排列了一些项以提高效率，因此只需进行一次平方根运算。  

```cpp
〈BSDF 内联函数〉 +≡  
inline Float CosDPhi(const Vector3f &wa, const Vector3f &wb) {  
    return Clamp((wa.x * wb.x + wa.y * wb.y) /  
        std::sqrt((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)), -1, 1);  
}  
```  

在阅读本章代码以及向pbrt添加BRDF和BTDF时，需要记住一些重要的约定和实现细节。 


在阅读本章代码和添加BRDF和BTDF时，需要记住一些重要的约定和实现细节：

- 入射光方向$\omega_i$和出射观察方向$\omega_o$在变换到表面局部坐标系后都会被归一化并朝外。

- 在pbrt中，按照惯例，表面法线n总是指向物体的“外部”，这使得判断光是进入还是离开透射物体变得容易：如果入射光方向$\omega_i$与n在同一半球，则光正在进入；否则，光正在离开。因此，需要记住的一个细节是，法线可能在表面的相反侧，而不是$\omega_i$和$\omega_o$方向向量的一侧或两侧。与许多其他渲染器不同，pbrt不会将法线翻转以位于$\omega_o$的同一侧。

- 用于阴影的局部坐标系可能与Chapter 3中Shape::Intersect()例程返回的坐标系不完全相同；它们可以在相交和阴影之间进行修改，以实现诸如凹凸映射等效果。参见第9章了解这种修改的示例。

- 最后，BRDF和BTDF的实现不应关心$\omega_i$和$\omega_o$是否在同一半球。例如，虽然反射BRDF原则上应该检测入射方向是否在表面上方，出射方向是否在表面下方，并在这种情况下始终返回无反射，但在这里我们期望反射函数计算并返回使用适当公式反射的光量，忽略它们不在同一半球的细节。pbrt中的高级代码将确保仅评估适当的反射或透射散射例程。这种约定的价值将在第9.1节中解释。




































































































































































































































































































































































































































































































































































