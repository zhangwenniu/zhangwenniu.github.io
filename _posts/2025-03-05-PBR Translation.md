---
layout: mypost
title: k035 PBR Translation
categories: [Physically Based Rendering]
---




# 6.4 真实相机

薄透镜模型使得渲染具有景深模糊效果的图像成为可能，但它只是对实际相机镜头系统的粗略近似。实际相机镜头系统由一系列多个镜头元件组成，每个元件都会改变通过它的辐射分布。（图6.16展示了一个22毫米焦距广角镜头的横截面，该镜头包含八个元件。）即使是基本的手机相机通常也有大约五个独立的镜头元件，而DSLR镜头可能有十个或更多。一般来说，具有更多镜头元件的复杂镜头系统可以创建比简单镜头系统更高质量的图像。

![figure.6.16](figure.6.16.png)

> 图6.16：广角镜头系统的横截面（pbrt发行版中的scenes/lenses/wide.22.dat）。镜头坐标系中，胶片平面垂直于z轴并位于z = 0处。镜头位于左侧，沿着负z轴方向，然后场景位于镜头的左侧。光圈光阑，由镜头系统中间的粗黑线表示，会阻挡击中它的光线。在许多镜头系统中，光圈光阑的大小可以调整，以在较短的曝光时间（使用较大光圈）和更大的景深（使用较小光圈）之间进行权衡。

![figure.6.17](figure.6.17.png)

> 图6.17：使用具有非常宽视场的鱼眼镜头渲染的图像。注意边缘的暗化，这是由于准确模拟了图像形成的辐射度量学（第6.4.7节）；以及直线扭曲为曲线的现象，这是许多广角镜头的特征，但在使用投影矩阵表示镜头投影模型时并未考虑到这一点。


本节讨论RealisticCamera的实现，它模拟光线通过镜头系统（如图6.16所示）的聚焦过程，以渲染如图6.17所示的图像。其实现基于光线追踪，相机沿着光线路径穿过镜头元件，考虑不同折射率介质（空气、不同类型的玻璃）之间界面的折射，直到光线路径离开光学系统或者被光圈或镜头外壳吸收。从前镜头元件离开的光线代表相机响应曲线的样本，可以与估计任意光线入射辐射的积分器（如SamplerIntegrator）一起使用。RealisticCamera的实现在文件cameras/realistic.h和cameras/realistic.cpp中。

〈RealisticCamera声明〉≡
```
class RealisticCamera : public Camera {
public:
  〈RealisticCamera公共方法〉
private:
  〈RealisticCamera私有声明 381〉
  〈RealisticCamera私有数据 379〉
  〈RealisticCamera私有方法 381〉
};
```

除了将相机放置在场景中的常规变换、Film以及快门打开和关闭时间外，RealisticCamera构造函数还接受镜头系统描述文件的文件名、到所需焦平面的距离以及光圈的直径。simpleWeighting参数的效果将在第13章关于蒙特卡洛积分的初步知识和第6.4.7节关于图像形成辐射度量学之后的第13.6.6节中描述。

〈RealisticCamera方法定义〉≡
```
RealisticCamera::RealisticCamera(const AnimatedTransform &CameraToWorld,
                                Float shutterOpen, Float shutterClose,
                                Float apertureDiameter, Float focusDistance,
                                bool simpleWeighting, const char *lensFile,
                                Film *film, const Medium *medium)
    : Camera(CameraToWorld, shutterOpen, shutterClose, film, medium),
      simpleWeighting(simpleWeighting) {
  〈从镜头描述文件加载元件数据〉
  〈为给定的焦距计算镜头-胶片距离 389〉
  〈在胶片上采样点处计算出瞳孔边界 390〉
}
```

〈RealisticCamera私有数据〉≡ 378
```
const bool simpleWeighting;
```

从磁盘加载镜头描述文件后，构造函数会调整镜头和胶片之间的间距，使焦平面位于所需深度focusDistance处，然后预计算一些关于最接近胶片的镜头元件哪些区域从胶片平面的各个点看来会传递场景光线到胶片的信息。在介绍了背景材料之后，片段〈为给定的焦距计算镜头-胶片距离〉和〈在胶片上采样点处计算出瞳孔边界〉将分别在第6.4.4和6.4.5节中定义。


# 6.4.1 镜头系统表示

镜头系统由一系列镜头元件组成，每个元件通常是某种形式的玻璃。镜头系统设计师的挑战是设计一系列元件，在空间限制（例如，为了保持手机的薄度，手机相机的厚度非常有限）、成本和制造便利性的约束下，在胶片或传感器上形成高质量的图像。

制造横截面为球形的镜头最为简单，而且镜头系统通常围绕光轴对称，光轴按惯例用z表示。在本节的其余部分中，我们将假设这两个特性都成立。与6.2.3节一样，镜头系统的定义使用一个坐标系统，其中胶片与z = 0平面对齐，镜头位于胶片的左侧，沿着-z轴。

![figure.6.18](figure.6.18.png)

> 图6.18：镜头界面（实心曲线）在位置z处与光轴相交。界面几何形状由界面的光圈半径（描述其在光轴上下的范围）和元件的曲率半径r描述。如果元件具有球形横截面，则其轮廓由一个球体给出，该球体的中心在光轴上距离r处，并且该球体也通过z点。如果r为负值，则元件界面从场景看是凹面的（如图所示）；否则是凸面的。镜头的厚度给出了到右侧下一个界面的距离，或者对于最后面的界面，给出了到胶片平面的距离。


镜头系统通常以各个镜头元件（或空气）之间的一系列界面来表示，而不是对每个元件进行显式表示。表6.1显示了定义每个界面的量。表中的最后一项定义了最右侧的界面，如图6.18所示：它是一个球面的截面，其半径等于曲率半径。元件的厚度是沿z轴到右侧下一个元件（或到胶片平面）的距离，折射率是界面右侧介质的折射率。元件在z轴上下的范围由光圈直径设定。

LensElementInterface结构表示单个镜头元件界面。

〈RealisticCamera私有声明〉≡ 378
```
struct LensElementInterface {
    Float curvatureRadius;
    Float thickness;
    Float eta;
    Float apertureRadius;
};
```

片段〈从镜头描述文件加载元件数据〉（此处未包含）读取镜头元件并初始化RealisticCamera::elementInterfaces数组。有关文件格式的详细信息，请参阅源代码中的注释，该格式与表6.1的结构平行，pbrt发行版中的目录scenes/lenses中有许多示例镜头描述。

![Table 6.1](table.6.1.png)

> 表6.1：图6.16中镜头系统的表格描述。每行描述两个镜头元件之间的界面、元件与空气之间的界面或光圈光阑。第一行描述最左侧的界面。半径为0的元件对应于光圈光阑。距离以毫米为单位测量。


对从文件读取的值进行了两项调整：首先，镜头系统传统上以毫米为单位描述，但pbrt假设场景以米为单位测量。因此，除折射率外的字段均按1/1000缩放。其次，元件的直径除以二；在接下来的代码中，半径是一个更便于使用的量。

〈RealisticCamera私有数据〉+≡ 378
```
std::vector<LensElementInterface> elementInterfaces;
```

一旦加载了元件界面描述，方便地获取与镜头系统相关的一些值是很有用的。LensRearZ()和LensFrontZ()分别返回镜头系统后部和前部元件的z深度。注意，返回的z深度是在相机空间中，而不是镜头空间，因此具有正值。

〈RealisticCamera私有方法〉≡ 378
```
Float LensRearZ() const { return elementInterfaces.back().thickness; }
```

![figure.6.19](figure.6.19.png)

> 图6.19：元件厚度与光轴上位置的关系。胶片平面位于z = 0处，后部元件的厚度t3给出了从胶片到其界面的距离；后部界面在z = −t3处与光轴相交。下一个元件具有厚度t2并位于z = −t3 − t2处，依此类推。前部元件在$\sum_i -t_i$处与z轴相交。


找到前部元件的z位置需要对所有元件厚度求和（见图6.19）。这个值不需要在系统性能敏感的部分中使用，因此在需要时重新计算是可以的。如果此方法的性能是个问题，最好在RealisticCamera中缓存此值。

〈RealisticCamera私有方法〉+≡ 378
```
Float LensFrontZ() const {
    Float zSum = 0;
    for (const LensElementInterface &element : elementInterfaces)
        zSum += element.thickness;
    return zSum;
}
```

RearElementRadius()返回后部元件的光圈半径（以米为单位）。

〈RealisticCamera私有方法〉+≡ 378
```
Float RearElementRadius() const {
    return elementInterfaces.back().apertureRadius;
}
```

# 6.4.2 通过镜头追踪光线

给定一条从镜头系统胶片侧开始的光线，TraceLensesFromFilm()依次计算与每个元件的交点，如果光线在穿过镜头系统的过程中被阻挡，则终止光线并返回false。否则返回true并用相机空间中的出射光线初始化*rOut。在遍历过程中，elementZ跟踪当前镜头元件的z轴截距。由于光线从胶片开始，与elementInterfaces中存储的顺序相比，镜头是按相反顺序遍历的。

〈RealisticCamera方法定义〉+≡
```
bool RealisticCamera::TraceLensesFromFilm(const Ray &rCamera, Ray *rOut) const {
    Float elementZ = 0;
    〈将rCamera从相机空间转换到镜头系统空间 383〉
    for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = elementInterfaces[i];
        〈更新来自胶片的光线，考虑与元件的交互 383〉
    }
    〈将rLens从镜头系统空间转换回相机空间 385〉
    return true;
}
```

因为在pbrt的相机空间中相机指向+z轴，但镜头沿着-z轴，所以光线原点和方向的z分量需要取反。虽然这是一个足够简单的变换，可以直接应用，但我们更喜欢使用显式Transform来使意图清晰。

〈将rCamera从相机空间转换到镜头系统空间〉≡ 382
```
static const Transform CameraToLens = Scale(1, 1, -1);
Ray rLens = CameraToLens(rCamera);
```

回想图6.19中元件z轴截距的计算方式：因为我们是从后到前访问元件，所以在考虑元件交互之前，必须从elementZ中减去元件的厚度来计算其z轴截距。

〈更新来自胶片的光线，考虑与元件的交互〉≡ 382
```
elementZ -= element.thickness;
〈计算光线与镜头元件的交点 383〉
〈根据元件光圈测试交点 384〉
〈根据元件界面交互更新光线路径 385〉
```

给定元件的z轴截距，下一步是计算光线与元件界面（或光圈平面）相交的参数t值。对于光圈光阑，使用光线-平面测试（遵循3.1.2节）。对于球面界面，IntersectSphericalElement()执行此测试，并在找到交点时返回表面法线；计算折射光线方向时需要法线。

〈计算光线与镜头元件的交点〉≡ 383
```
Float t;
Normal3f n;
bool isStop = (element.curvatureRadius == 0);
if (isStop)
    t = (elementZ - rLens.o.z) / rLens.d.z;
else {
    Float radius = element.curvatureRadius;
    Float zCenter = elementZ + element.curvatureRadius;
    if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
        return false;
}
```

IntersectSphericalElement()方法通常与Sphere::Intersect()类似，不过它专门针对元件中心位于z轴上的事实（因此，中心的x和y分量为零）。由于片段〈计算光线-元件交点的t0和t1〉和〈计算光线交点处元件的表面法线〉与Sphere::Intersect()实现相似，此处文本中未包含它们。

〈RealisticCamera方法定义〉+≡
```
bool RealisticCamera::IntersectSphericalElement(Float radius, Float zCenter,
                                              const Ray &ray, Float *t,
                                              Normal3f *n) {
    〈计算光线-元件交点的t0和t1〉
    〈根据光线方向和元件曲率选择交点t 384〉
    〈计算光线交点处元件的表面法线〉
    return true;
}
```

然而，在选择要返回哪个交点时存在一个微妙之处：t > 0的最近交点不一定在元件界面上；参见图6.20.2。例如，对于从场景接近并与凹透镜（具有负曲率半径）相交的光线，应该返回两个交点中较远的一个，无论较近的一个是否有t > 0。幸运的是，基于光线方向和曲率半径的简单逻辑可以指示使用哪个t值。

〈根据光线方向和元件曲率选择交点t〉≡ 383
```
bool useCloserT = (ray.d.z > 0) ^ (radius < 0);
*t = useCloserT ? std::min(t0, t1) : std::max(t0, t1);
if (*t < 0) return false;
```

每个镜头元件在光轴周围延伸一定半径；如果与元件的交点在此半径之外，则光线实际上会与镜头外壳相交并终止。类似地，如果光线与光圈光阑相交，它也会终止。因此，这里我们根据当前元件的适当限制测试交点，如果光线存活，则终止光线或将其原点更新为当前交点。

〈根据元件光圈测试交点〉≡ 383
```
Point3f pHit = rLens(t);
Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
if (r2 > element.apertureRadius * element.apertureRadius)
    return false;
rLens.o = pHit;
```

如果当前元件是光圈，光线的路径不会受到穿过元件界面的影响。对于玻璃（或者，禁止使用塑料）镜头元件，当光线从一个折射率的介质进入另一个折射率的介质时，光线的方向在界面处发生变化。（光线可能从空气进入玻璃，从玻璃进入空气，或者从一种折射率的玻璃进入具有不同折射率的另一种玻璃。）





第8.2节讨论了两种介质边界处的折射率变化如何改变光线的方向和光线携带的辐射量。（在这种情况下，我们可以忽略辐射量的变化，因为如果光线进入镜头系统和离开时处于相同的介质中——这里两者都是空气——这种变化会相互抵消。）Refract()函数在第8.2.3节中定义；注意，它期望入射方向指向远离表面，所以在传递给它之前，光线方向被取反。在发生全内反射的情况下，此函数返回false，此时光线路径终止。否则，折射方向在w中返回。

一般来说，通过这样的界面的一些光被传输，一些被反射。这里我们忽略反射并假设完全传输。虽然这是一个近似，但它是合理的：镜头通常制造时带有涂层，设计用于将反射减少到光线携带的辐射量的约0.25%。（然而，模拟这种少量的反射对于捕捉镜头眩光等效果可能很重要。）

〈根据元件界面交互更新光线路径〉≡ 383
```
if (!isStop) {
    Vector3f w;
    Float etaI = element.eta;
    Float etaT = (i > 0 && elementInterfaces[i - 1].eta != 0) ?
        elementInterfaces[i - 1].eta : 1;
    if (!Refract(Normalize(-rLens.d), n, etaI / etaT, &w))
        return false;
    rLens.d = w;
}
```


如果光线成功地从前镜头元件出来，它只需要从镜头空间转换到相机空间。

〈将rLens从镜头系统空间转换回相机空间〉≡ 382
```
if (rOut != nullptr) {
    static const Transform LensToCamera = Scale(1, 1, -1);
    *rOut = LensToCamera(rLens);
}
```


TraceLensesFromScene()方法与TraceLensesFromFilm()非常相似，此处未包含。主要区别是它从前到后而不是从后到前遍历元件。注意，它假设传递给它的光线已经在相机空间中；如果光线从世界空间开始，调用者负责执行变换。返回的光线在相机空间中，从后镜头元件朝向胶片。

〈RealisticCamera私有方法〉+≡ 378
```
bool TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const;
```


