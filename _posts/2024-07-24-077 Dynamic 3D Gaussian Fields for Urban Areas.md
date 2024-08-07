---
layout: mypost
title: Dynamic 3D Gaussian Fields for Urban Areas
categories: [论文, 自动驾驶, 3DGS]
---


# 原文

Abstract

We present an efficient neural 3D scene representation for novel-view synthesis (NVS) in large-scale, dynamic urban areas.

Existing works are not well suited for applications like mixed-reality or closed-loop simulation due to their limited visual quality and non-interactive rendering speeds.

Recently, rasterization-based approaches have achieved high-quality NVS at impressive speeds.

However, these methods are limited to small-scale, homogeneous data, i.e. they cannot handle severe appearance and geometry variations due to weather, season, and lighting and do not scale to larger, dynamic areas with thousands of images.

We propose 4DGF, a neural scene representation that scales to large-scale dynamic urban areas, handles heterogeneous input data, and substantially improves rendering speeds.

We use 3D Gaussians as an efficient geometry scaffold while relying on neural fields as a compact and flexible appearance model.

We integrate scene dynamics via a scene graph at global scale while modeling articulated motions on a local level via deformations.

This decomposed approach enables flexible scene composition suitable for real-world applications.

In experiments, we surpass the state-of-the-art by over 3 dB in PSNR and more than 200× in rendering speed.

1 Introduction

The problem of synthesizing novel views from a set of images has received widespread attention in recent years due to its importance for technologies like AR/VR and robotics.

In particular, obtaining interactive, high-quality renderings of large-scale, dynamic urban areas under varying weather, lightning, and seasonal conditions is a key requirement for closed-loop robotic simulation and immersive VR experiences.

To achieve this goal, sensor-equipped vehicles act as a frequent data source that is becoming widely available in city-scale mapping and autonomous driving, creating the possibility of building up-to-date digital twins of entire cities.

However, modeling these scenarios is extremely challenging as heterogeneous data sources have to be processed and combined: different weather, lighting, seasons, and distinct dynamic and transient objects pose significant challenges to the reconstruction and rendering of dynamic urban areas.

In recent years, neural radiance fields have shown great promise in achieving realistic novel view synthesis of static and dynamic scenes.

While earlier methods were limited to controlled environments, several recent works have explored large-scale, dynamic areas.

Among these, many works resort to removing dynamic regions and thus produce partial reconstructions.

In contrast, fewer works model scene dynamics.

These methods exhibit clear limitations, such as rendering speed which can be attributed to the high cost of ray traversal in volumetric rendering.

Therefore, rasterization-based techniques have recently emerged as a viable alternative.

Most notably, Kerbl et al. propose a scene representation based on 3D Gaussian primitives that can be efficiently rendered with a tile-based rasterizer at a high visual quality.

While demonstrating impressive rendering speeds, it requires millions of Gaussian primitives with high-dimensional spherical harmonics coefficients as color representation to achieve good view synthesis results.

This limits its applicability to large-scale urban areas due to high memory requirements.

Furthermore, due to its explicit color representation, it cannot model transient geometry and appearance variations commonly encountered in city-scale mapping and autonomous driving use cases such as seasonal and weather changes.

Lastly, the approach is limited to static scenes which complicates representing dynamic objects such as moving vehicles or pedestrians commonly encountered in urban areas.

To this end, we propose 4DGF, a method that takes a hybrid approach to modeling dynamic urban areas.

In particular, we use 3D Gaussian primitives as an efficient geometry scaffold.

However, we do not store appearance as a per-primitive attribute, thus avoiding more than 80% of its memory footprint.

Instead, we use fixed-size neural fields as a compact and flexible alternative.

This allows us to model drastically different appearances and transient geometry which is essential to reconstructing urban areas from heterogeneous data.

Finally, we model scene dynamics with a graph-based representation that maps dynamic objects to canonical space for reconstruction.

We model non-rigid deformations in this canonical space with our neural fields to cope with articulated dynamic objects common in urban areas such as pedestrians and cyclists.

This decomposed approach further enables a flexible scene composition suitable to downstream applications.

The key contributions of this work are:

We introduce 4DGF, a hybrid neural scene representation for dynamic urban areas that leverages 3D Gaussians as an efficient geometry scaffold and neural fields as a compact and flexible appearance representation.

We use neural fields to incorporate scene-specific transient geometry and appearances into the rendering process of 3D Gaussian splatting, overcoming its limitation to static, homogeneous data sources while benefitting from its efficient rendering.

We integrate scene dynamics via i) a graph-based representation, mapping dynamic objects to canonical space, and ii) modeling non-rigid deformations in this canonical space.

This enables effective reconstruction of dynamic objects from in-the-wild captures.

We show that 4DGF effectively reconstructs large-scale, dynamic urban areas with over ten thousand images, achieves state-of-the-art results across four dynamic outdoor benchmarks, and is more than 200× faster to render than the previous state-of-the-art.

2 Related Work

Dynamic scene representations.

Scene representations are a pillar of computer vision and graphics research.

Over decades, researchers have studied various static and dynamic scene representations for numerous problem setups.

Recently, neural rendering has given rise to a new class of scene representations for photo-realistic image synthesis.

While earlier methods in this scope were limited to static scenes, dynamic scene representations have emerged quickly.

These scene representations can be broadly classified into implicit and explicit representations.

Implicit representations encode the scene as a parametric function modeled as neural network, while explicit representations use a collection of low-level primitives.

In both cases, scene dynamics are simulated as i) deformations of a canonical volume, ii) particle-level motion such as scene flow, or iii) rigid transformations of local geometric primitives.

On the contrary, traditional computer graphics literature uses scene graphs to compose entities into complex scenes.

Therefore, another area of research explores decomposing scenes into higher-level elements, where entities and their spatial relations are expressed as a directed graph.

This concept was recently revisited for view synthesis.

In this work, we take a hybrid approach that uses i) explicit geometric primitives for fast rendering, ii) implicit neural fields to model appearance and geometry variation, and iii) a scene graph to decompose individual dynamic and static components.


Efficient rendering and 3D Gaussian splatting.

Aside from accuracy, the rendering speed of a scene representation is equally important.

While rendering speed highly depends on the representation efficiency itself, it also varies with the form of rendering that is coupled with it to generate an image.

Traditionally, neural radiance fields use implicit functions and volumetric rendering which produce accurate renderings but suffer from costly function evaluation and ray traversal.

To remedy these issues, many techniques for caching and efficient sampling have been developed.

However, these approaches often suffer from excessive GPU memory requirements and are still limited in rendering speed.

Therefore, researchers have opted to exploit more efficient forms of rendering, baking neural scene representations into meshes for efficient rasterization.

This area of research has recently been disrupted by 3D Gaussian splatting, which i) represents the scene as a set of anisotropic 3D Gaussian primitives ii) uses an efficient tile-based, differentiable rasterizer, and iii) enables effective optimization by adaptive density control (ADC), which facilitates primitive growth and pruning.

This led to a paradigm shift from baking neural scene representations to a more streamlined approach.


However, the method of Kerbl. et al. exhibits clear limitations, which has sparked a very active field of research with many concurrent works.

For instance, several works tackle dynamic scenes by adapting approaches described in the paragraph above.

Another line of work focuses on modeling larger-scale scenes.

Lastly, several concurrent works investigate the reconstruction of dynamic street scenes.

These methods are generally limited to homogeneous data and in scale.

In contrast, our method scales to tens of thousands of images and effectively reconstructs large, dynamic urban areas from heterogeneous data while also providing orders of magnitude faster rendering than traditional approaches.

Reconstructing urban areas.

Dynamic urban areas are particularly challenging to reconstruct due to the complexity of both the scenes and the capturing process.

Hence, significant research efforts have focused on adapting view synthesis approaches from controlled, small-scale environments to larger, real-world scenes.

In particular, researchers have investigated the use of depth priors from e.g.

LiDAR, providing additional information such as camera exposure, jointly optimizing camera parameters, and developing specialized sky and light modeling approaches.

However, since scene dynamics are challenging to approach, many works simply remove dynamic areas, providing only a partial reconstruction.

Fewer works explicitly model scene dynamics and consequently, these works have clear limitations: they do not scale beyond single, short video clip, struggle to accurately represent dynamic objects, or are costly to render.

Instead, we present a fast, scalable method that faithfully represents scene dynamics.

3 Method

3.1 Problem setup

We are given a set of heterogeneous sequences S that capture a common geographic area from a moving vehicle.

The vehicle is equipped with calibrated cameras mounted in a surround-view setup.

We denote with Cs the set of cameras of sequence s ∈ S and with C the total set of cameras, i.e.

C := S s∈S Cs.

For each camera c ∈ C, we assume to know the intrinsic Kc parameters and the pose Pc ∈ SE(3), expressed in the ego-vehicle reference frame.

Ego-vehicle poses Pts ∈ SE(3) are provided for each sequence s ∈ S and timesteps t ∈ Ts and are expressed in the world reference frame that is shared across all sequences.

Here, Ts denotes a set of timestamps relative to s.

Indeed, we assume that timestamps cannot be compared across sequences because we lack a mapping to a global timeline, which is often the case with benchmark datasets due to privacy reasons.

For each sequence s ∈ S, camera c ∈ Cs and timestamp t ∈ Ts we have an RGB image It (s,c) ∈H×W ×3.

Each sequence has additionally an associated set of dynamic objects Os.

Dynamic objects o ∈ Os are associated with a 3D bounding box track that holds its (stationary) 3D object dimensions so ∈ R3+ and poses {ξt0 o , ..., ξtn o } ⊂ SE(3) w.r.t. the ego-vehicle frame, where ti ∈ To ⊂ Ts.

Our goal is to estimate the plenoptic function for the shared geographic area spanned by the training sequences, i.e. a function f (P, K, t, s), which outputs a rendered RGB image of size (H, W ) for a given camera pose P with calibration K in the conditions of sequence s ∈ S at time t ∈ Ts.

3.2 Representation

We model a parameterized, plenoptic function fθ, which depends on the following components: i) a scene graph G that provides the scene configuration and latent conditioning signals ω for each sequence s, object o, and time t, ii) sets of 3D Gaussians that serve as a geometry scaffold for the scene and objects, and iii) implicit neural fields that model appearance and modulate the geometry scaffold according to the conditioning signals.

See Figure 2 for an overview of our method.

Scene configuration.

Inspired by, we factorize the scene with a graph representation G = (V, E), holding latent conditioning signals at the nodes V and coordinate system transformations along the edges E.

The nodes V consist of a root node vr defining the global coordinate system, camera nodes {vc}c∈C , and for each sequence s ∈ S, sequence nodes {vt s}t∈Ts and dynamic object nodes {vo}o∈Os .

We associate latent vectors ω to sequence and object nodes representing local appearance and geometry.

Specifically, we model the time-varying sequence appearance and geometry via ωt s := (1)

where As and Gs are appearance and geometry modulation matrices, respectively, and γ(·) is a 1D basis function of sines and cosines with linearly increasing frequencies at log-scale.

Time t is normalized to via the maximum sequence length maxs∈S |Ts|.

For objects, we use both an object code and a time encoding ωt o := . (2) Nodes in the graph G are connected by oriented edges that define rigid transformations between the canonical frames of the nodes.

We have Pts for sequence to root edges, Pc for camera to sequence edges, and ξt o for object to sequence edges.

3D Gaussians.

We represent the scene geometry with sets of anisotropic 3D Gaussians primitives G = {Gr} ∪ {Go : o ∈ Os, s ∈ S}.

Each 3D Gaussian primitive gk is parameterized by its mean μk, covariance matrix Σk, and a base opacity αk.

The covariance matrix is decomposed into a rotation matrix represented as a unit quaternion qk and a scaling vector ak ∈ R3+.

The geometry of gk is represented by gk(x) = exp  −1 2⊤Σ−1 k  . (3)

The common scene geometry scaffold is modeled with a single set of 3D Gaussians Gr, while we have a separate set Go of 3D Gaussians for each dynamic object o.

Indeed, scene geometry is largely consistent across sequences while object geometries are distinct.

The 3D Gaussians Gr are represented in world frame, while each set Go is represented in a canonical, object-centric coordinate frame, which can be mapped to the world frame by traversing G.

Differently from, our 3D Gaussians do not hold any appearance information, reducing the memory footprint of the representation by more than 80%.

Instead, we leverage neural networks to regress a color information cs,t k and an updated opacity αs,t k for each sequence s ∈ S and time t ∈ Ts.

For 3D Gaussians in Gr modeling the scene scaffolding, we predict an opacity attenuation term νs,t k that is used to model transient geometry by downscaling αk Instead, for 3D Gaussians in Go modeling objects the base opacity is left invariant.

Hence αs,t k := ν s,t k αk if gk ∈ Gr αk else. (4)

Pruning decisions in ADC are obtained by thresholding the base opacity αk, which is directly accessible without computational overhead.

Finally, in the presence of non-rigid objects o, we predict deformation terms δμt k to the position of 3D primitives in Go via a neural network, for each time t ∈ To.

In this case, the final position of the primitive is given by μt k := μk + δμt k . (5)

Appearance and transient geometry.

Given the scene graph G and the 3D Gaussians G, we use two efficient neural fields to decode the appearance parameters of each primitive.

For 3D Gaussians in Gr modeling the static scene, the neural field is denoted by φ and regresses the opacity attenuation term νs,t k and a color ck, given the 3D Gaussian primitive’s position μk, a viewing direction d, the base opacity αk and the latent code of the node ωt s, i.e. (ν s,t k , cs,t k ) := φ(μk, d, αk, ωt s) . (6)

where s ∈ S and t ∈ Ts.

The opacity attenuation contributes to modeling transient geometry, for it potentially enables removing parts of the scene encoded in the original set of Gaussians.

Moreover, it does not depend on the viewing direction d.

For 3D Gaussians in Go modeling dynamic objects, the neural field is denoted by ψ and regresses a color ct k.

Besides the primitive’s position and viewing direction, we condition ψ on latent vectors ωt s and ωt o to model both local object texture and global sequence appearance such as illumination.

Here, the sequence s is the one where o belongs to, i.e. satisfying o ∈ Os, and t ∈ To.

Accordingly, the color ct k for a 3D Gaussian in Go is given by cs,t k := ψ(μk, d, ωt s, ωt o) . (7) Both μk and d are expressed in the canonical, object-centric space of object o.

Using neural fields has three key advantages for our purpose.

First, by sharing the parameters of φ and ψ across all 3D Gaussians G, we achieve a significantly more compact representation than in when scaling to large-scale urban areas.

Second, it allows us to model sequence-dependent appearance and transient geometry which is fundamental to learning a scene representation from heterogeneous data.

Third, information sharing between nodes enables an interaction of sequence and object appearance.

Non-rigid objects.

Street scenes are occupied not only by rigidly moving vehicles but also by, e.g., pedestrians and cyclists that move in a non-rigid manner.

These pose a significant challenge due to their unconstrained motion under limited visual coverage.

Therefore, we take a decomposed approach to modeling non-rigid objects that uses the scene graph G to model the global, rigid object motion while using a deformation head χ to model the local, articulated motion.

The deformation head predicts a local position offset δμt k via δμt k := χ(fψ, γ(t)) (8) given an intermediate feature representation fψ of ψ conditioned on μk.

This way, we can deform the position of μk over time in canonical space as per Equation (5).

Background modeling.

To achieve a faithful rendering of far-away objects and the sky, it is important to have a background model.

Inspired by, where points are sampled along a ray at increasing distance outside the scene bounds, we place 3D Gaussians on spheres around the scene with radius r2i+1 for i ∈ {1, 2, 3} where r is half of the scene bound diameter.

To avoid ambiguity with foreground scene geometry and to increase efficiency, we remove all points that are i) below the ground plane, ii) occluded by foreground scene points, or iii) outside of the view frustum of any training view.

To uniformly distribute points on each sphere, we utilize the Fibonacci sphere sampling algorithm, which arranges points in a spiral pattern using a golden ratio-based formula.

Even though this sampling is not optimal, it serves as a faster approximation of the optimal sampling.

3.3 Composition and Rendering Scene composition.

To render our representation from the perspective of camera c at time t in sequence s, we traverse the graph G to obtain the latent vector ωt s and the latent vector ωt o of each visible object o ∈ Os, i.e. such that t ∈ To.

Moreover, for each 3D Gaussian primitive gk in G, we use the collected camera parameters, object scale, and pose information to determine the transformation Πc k mapping points from the primitive’s reference frame (e.g. world for Gr, object-space for Go) to the image space of camera c.

Opacities αs,t k are computed as per Equation (4), while colors cs,t k are computed for primitives in Gr and in Go via Equations (6) and (7), respectively.

Rasterization.

To render the scene from camera c, we follow and splat the 3D Gaussians to the image plane.

Practically, for each primitive, we compute a 2D Gaussian kernel denoted by gc k with mean μc k given by the projection of the primitive’s position to the image plane, i.e. μc k := Πc k (μk ), and with covariance given by Σc k := Jc k Σk Jc⊤ k , where Jc k is the Jacobian of Πc k evaluated at μk.

Finally, we apply traditional alpha compositing of the 3D Gaussians to render pixels p of camera c: cs,t(p) := K X k=0 cs,t k wk k−1 Y j=0 (1 − wj) with wk := αs,t k gc k(p) . (9)

3.4 Optimization To optimize parameters θ of fθ, i.e. 3D Gaussian parameters μk, αk, qk and ak, sequence latent vectors ωt s and implicit neural fields ψ and φ, we use an end-to-end differentiable rendering pipeline.

We render both an RGB color image ˆ I and a depth image ˆ D and apply the following loss function: L(ˆ I, I, ˆ D, D) = λrgbLrgb(ˆ I, I) + λssimLssim(ˆ I, I) + λdepLdep( ˆ D, D) (10) where Lrgb is the L1 norm, Lssim is the structural similarity index measure, and Ldep is the L2 norm.

We use the posed training images and LiDAR measurements as the ground truth.

If no depth ground-truth is available, we drop the depth-related loss from L.

Pose optimization.

Next to optimizing scene geometry, it is crucial to refine the pose parameters of the reconstruction for in-the-wild scenarios since provided poses often have limited accuracy.

Thus, we optimize the residuals δPts ∈ se(3), δPc ∈ se(3) and δξt o ∈ se(2) jointly with parameters θ.

We constrain object pose residuals to se(2) to incorporate the prior that objects move on the ground plane and are oriented upright.

See our supp. mat. for details on camera pose gradients.

Adaptive density control.

To facilitate the growth and pruning of 3D Gaussian primitives, the optimization of the parameters θ is interleaved by an ADC mechanism.

This mechanism is essential to achieve photo-realistic rendering.

However, it was not designed for training on tens of thousands of images, and thus we develop a streamlined multi-GPU variant.

First, accumulating statistics across processes is essential.

Then, instead of running ADC on GPU 0 and synchronizing the results, we synchronize only non-deterministic parts of ADC, i.e. the random samples drawn from the 3D Gaussians that are being split.

These are usually much fewer than the total number of 3D Gaussians and thus avoids communication overhead.

Next, the 3D Gaussian parameters are replaced by their updated replicas.

However, this will impair the synchronization of the gradients because, in PyTorch DDP, parameters are only registered once at model initialization.

Therefore, we re-initialize the Reducer upon finishing the ADC mechanism in the low-level API provided in.

Furthermore, urban street scenes pose some unique challenges to ADC, such as a large variation in scale, e.g. extreme close-ups of nearby cars mixed with far-away buildings and sky.

This can lead to blurry renderings for close-ups due to insufficient densification.

We address this by using maximum 2D screen size as a splitting criterion.1 In addition, ADC considers the world-space scale ak of a 3D Gaussian to prune large primitives which hurts background regions far from the camera.

Hence, we first test if a 3D Gaussian is inside the scene bounds before pruning it according to ak.

Finally, the scale of urban areas leads to memory issues when the growth of 3D Gaussian primitives is unconstrained.

Therefore, we introduce a threshold that limits primitive growth while keeping pruning in place.

See our supp. mat. for more details and analysis.

4 Experiments Datasets and Metrics.

We evaluate our approach across three dynamic outdoor benchmarks.

First, we utilize the recently proposed NVS benchmark on Argoverse 2 to compare against the state-of-the-art in the multi-sequence scenario and to showcase the scalability of our method.

Second, we use the established Waymo Open, KITTI and VKITTI2 benchmarks to compare to existing approaches in single-sequence scenarios.

For Waymo, we use the dynamic-32 split of, while for KITTI and VKITTI2 we follow.

We apply commonly used metrics to measure view synthesis quality: PSNR, SSIM, and LPIPS (AlexNet).

Implementation details.

We use λrgb := 0.8, λssim := 0.2 and λdepth := 0.05.

We use the LiDAR point clouds as initialization for the 3D Gaussians.

We first filter the points of dynamic objects using the 3D bounding box annotations and subsequently initialize the static scene with the remaining points while using the filtered points to initialize each dynamic object.

We use mean voxelization with voxel size τ to remove redundant points.

See our supp. mat. for more details.

4.1 Comparison to State-of-the-Art We compare with prior art across two experimental settings: single-sequence and multi-sequence.

In the former, we are given a single input sequence and aim to synthesize hold-out viewpoints from that same sequence.

In the latter, we are given multiple, heterogeneous input sequences and aim to synthesize hold-out viewpoints across all of these sequences from a single model.

Multi-sequence setting.

In Table 1, we show results on the Argoverse 2 NVS benchmark proposed in.

We compare to state-of-the-art approaches and the baselines introduced in.

The results highlight that our approach scales well to large-scale dynamic urban scenes, outperforming previous work in performance and rendering speed by a significant margin.

Specifically, we outperform by more than 3 points in PSNR while rendering more than 200× faster.

To examine these results more closely, we show a qualitative comparison in Figure 3.

We see that while SUDS struggles with dynamic objects and ML-NSG produces blurry renderings, our work provides sharp renderings and accurately represented dynamic objects, in both RGB color and depth images.

Overall, the results highlight that our model can faithfully represent heterogeneous data at high visual quality in a single 3D representation while being much faster to render than previous work.

Single-sequence setting.

In Table 2, we show results on the KITTI and VKITTI benchmarks at varying training split fractions.

Our approach outperforms previous work as well as concurrent 3D Gaussian-based approaches.

In Table 3, we show results on Waymo Open, specifically on the Dynamic-32 split proposed in.

We outperform previous work by a large margin while our rendering speed is 700× faster than the best alternative.

Note that our rendering speed increases for smaller-scale scenes.

Furthermore, we show that, contrary to previous approaches, our method does not suffer from lower view quality in dynamic areas.

This corroborates the strength of our contributions, showing that our method is not only scalable to large-scale, heterogeneous street data but also demonstrates superior performance in smaller-scale, homogeneous street data.

4.2 Ablation Studies

We verify our design choices in both the multi- and single-sequence setting.

For a fair comparison, we set the global maximum of 3D Gaussians to 8 and 4.1 million, respectively.

We perform these ablation studies on the residential split of.

We use the full overlap in the multi-sequence setting, while using a single sequence of this split for the singlesequence setting.

In Table 4a, we verify the components that are not specific to the multi-sequence setting.

In particular, we show that our approach to modeling scene dynamics is highly effective, evident from the large disparity in performance between the static and the dynamic variants.

Next, we show that modeling appearance with a neural field is on par with the proposed solution in, while being more memory efficient.

In particular, when modeling view-dependent color as a per-Gaussian attribute as in the model uses 8.6 GB of GPU memory during training, while it uses only 4.5 GB with fixed-size neural fields.

Similarly, storing the parameters of the former takes 922 MB, while the latter takes only 203 MB.

Note that this disparity increases with the number of 3D Gaussians per scene.

Finally, we achieve the best performance when adding the generated 3D Gaussian background.

We now scrutinize components specific to multi-sequence data in Table 4b.

We compare the view synthesis performance of our model when i) not modeling sequence appearance or transient geometry, ii) only modeling sequence appearance, iii) modeling both sequence appearance and transient geometry.

Naturally, we observe a large gap in performance between i) and ii), since the appearance changes between sequences are drastic (see Figure 3).

However, there is still a significant gap between ii) and iii), demonstrating that modeling both sequence appearance and transient geometry is important for view synthesis from heterogeneous data sources.

Finally, we provide qualitative results for non-rigid object view synthesis in Figure 4, and show that our approach can model articulate motion without the use of domain priors.

In our supp. mat., we provide further analysis.

5 Conclusion We presented 4DGF, a neural scene representation for dynamic urban areas. 4DGF models highly dynamic, large-scale urban areas with 3D Gaussians as efficient geometry scaffold and compact but flexible neural fields modeling large appearance and geometry variations across captures.

We use a scene graph to model dynamic object motion and flexibly compose the representation at arbitrary configurations and conditions.

We jointly optimize the 3D Gaussians, the neural fields, and the scene graph, showing state-of-the-art view synthesis quality and interactive rendering speeds.

Limitations.

While 4DGF improves novel view synthesis in dynamic urban areas, the challenging nature of the problem leaves room for further exploration.

Although we model scene dynamics, appearance, and geometry variations, other factors influence image renderings in real-world captures.

First, in-the-wild captures often exhibit distortions caused by the physical image formation process.

Thus, modeling phenomena like rolling shutter, white balance, motion and defocus blur, and chromatic aberrations is necessary to avoid reconstruction artifacts.

Second, the assumption of a pinhole camera model in persists in our work and thus our method falls short of modeling more complex camera models like equirectangular cameras which may be suboptimal for certain capturing settings.

Broader Impact.

We expect our work to positively impact real-world use cases like robotic simulation and mixed reality by improving the underlying technology.

While we do not expect malicious uses of our method, we note that an inaccurate simulation, i.e. a failure of our system, could misrepresent the robotic system performance, possibly affecting real-world deployment.

A Appendix We provide further details on our method and the experimental setting, as well as additional experimental results.

We accompany this supplemental material with a demonstration video.

A.1 Demonstration Video We showcase the robustness of our method by rendering a complete free-form trajectory across five highly diverse sequences using the same model.

Specifically, we chose the model trained on the residential split in Argoverse 2.

To obtain the trajectory, we interpolate keyframes selected throughout the total geographic area of the residential split into a single, smooth trajectory that encompasses most of its spatial extent.

We also apply periodical translations and rotations to this trajectory to increase the variety of synthesized viewpoints.

We use a constant speed of 10 meters per second.

We choose five different sequences in the data split as the references, spanning sunny daylight conditions in summer to near sunset in winter.

Consequently, the appearance of the sequences changes drastically, e.g. from green, fully-leafed trees to empty branches and snow or from bright sunlight to dark clouds.

Furthermore, we render each sequence with its unique set of dynamic objects, simulating various distinct traffic scenarios.

We show that our model is able to perform dynamic view synthesis in all of these conditions at high quality, faithfully representing scene appearance, transient geometry, and dynamic objects in each of the conditions.

We highlight that this scenario is extremely difficult, as it requires the model to generalize well beyond the training trajectories, represent totally different appearances and geometry, and model hundreds of dynamic, fast-moving objects.

Despite this fact, our method produces realistic renderings, showing its potential for real-world applications.

A.2 Method Neural field architectures.

To maximize efficiency, we model φ and ψ with hash grids and tiny MLPs.

The hash grids interpolate feature vectors at the nearest voxel vertices at multiple levels.

The feature vectors are obtained by indexing a feature table with a hash function.

Both neural fields are given input conditioning signals ωt s ∈ R64 and ωt o := and output a color c among the other outputs defined in Section 3.2.

For φ, we use the 3D Gaussian mean μk to query the hash function at a certain 3D position yielding an intermediate feature representation fφ.

We input the feature fφ, the sequence latent code ωt s, and the base opacity αk into MLPα which outputs the opacity attenuation νs,t k .

In a parallel branch, we input fφ, ωt s, and the viewing direction d encoded by a spherical harmonics encoding of degree 4 into the color head MLPc of φ that will define the final color of the 3D Gaussian.

For ψ, we use a 4D hash function while using only three dimensions for interpolation of the feature vectors, effectively modeling a 4D hash grid.

We use both the position μk and the object code ωo, i.e. the object identity, as the fourth dimension of the hash grid to model an arbitrarily large number of objects with a single hash table without a linear increase in memory.

We input the intermediate feature fψ and the time encoding γ(t) into the deformation head MLPχ which will output the non-rigid deformation of the object at time t, if applicable.

In parallel, we input ωt s, fψ, γ(t), and the encoded relative viewing direction d into the color head MLPc to output the final color.

Note that relative viewing direction refers to the viewing direction in canonical, object-centric space.

As noted in Section 3.2, the MLP heads are shared across all objects.

We list a detailed overview of the architectures in Table 5.

Note that, i) we decrease the hash table size of ψ in single-sequence experiments to 215 as we find this to be sufficient, and ii) we use two identical networks for ψ to separate rigid from non-rigid object instances.

Color prediction.

The kernel function gk prevents a full saturation of the rendered color within the support of the primitive as long as the primitive’s RGB color is bounded in the range.

This can be a problem for background and other uniformly textured regions that contain large 3D Gaussians, specifically larger than a single pixel.

Therefore, inspired by, we use a scaled sigmoid activation function for the color head MLPc: f (x) := 1 c sigmoid(cx) (11) where c := 0.9 is a constant scaling factor.

This allows the color prediction to slightly exceed the valid RGB color space.

After alpha compositing, we clamp the rendered RGB to the valid range following.

Time-dependent appearance.

In addition to conditioning the object appearance on the sequence at hand, we model the appearance of dynamic objects as a function of time by inputting γ(t) to MLPc as described above.

This way, our method adapts to changes in scene lighting that are more intricate than the general scene appearance.

This could be specular reflections, dynamic indicators such as brake lights, or shadows cast onto the object as it moves through the environment.

Space contraction.

We use space contraction to query unbounded 3D Gaussian locations from the neural fields.

In particular, we use the following function for space contraction: ζ(x) := (x, ∥x∥ ≤ 1  2− 1 ∥x∥ x ∥x∥ , ∥x∥ > 1 . (12) For φ, we use ∥ · ∥∞ as the norm to contract the space, while for ψ we use the Frobenius norm ∥ · ∥F .

Note that we use space contraction for ψ because 3D Gaussians may extend beyond the 3D object dimensions to represent e.g. shadows, however, most of the representation capacity should be allocated to the object itself.

Continuous-time object poses.

Both Argoverse 2 and Waymo Open provide precise timing information for both the LiDAR pointclouds to which the 3D bounding boxes are synchronized, and the camera images.

Thus, we treat the dynamic object poses {ξt0 o , ..., ξtn o } as a continuous function of time ξo(t), i.e. we interpolate between at ta ≤ t < tb to time t to compute ξo(t).

This also allows us to render videos at arbitrary frame rates with realistic, smooth object trajectories.

Anti-aliased rendering.

Inspired by, we compensate for the screen space dilation introduced in when evaluating gc k multiplying by a compensation factor: gc k(p) := s |Σc k| |Σc k + bI| exp  −1 2 (p − μc k )⊤ (Σc k + bI)−1(p − μc k)  , (13) where b is chosen to cover a single pixel in screen space.

This helps us to render views at different sampling rates.

Gradients of camera parameters.

Different from, we not only optimize the scene geometry but also the parameters of the camera poses.

This greatly improves view quality in scenarios with imperfect camera calibration which is frequently the case in street scene datasets.

In particular, we approximate the gradients w.r.t. a camera pose as: ∂L ∂t ≈ − X k ∂L ∂μk , ∂L ∂R ≈ − " X k ∂L ∂μk (μk − t)⊤ # R . (14) This formulation was concurrently proposed in, so we refer to them for a detailed derivation.

We obtain the gradients w.r.t. the vehicle poses ξ via automatic differentiation.


Adaptive density control.

We elaborate on the modifications described in Section 3.4.

Specifically, we observe that the same 3D Gaussian will be rendered at varying but dominantly small scales.

This biases the distribution of positional gradients towards views where the object is relatively small in view space, leading to blurry renderings for close-ups due to insufficient densification.

This motivates us to use maximum 2D screen size as an additional splitting criterion.

In addition to the adjustments described above and inspired by recent findings, we adapt the criterion of densification during ADC.

In particular, Kerbl et al. use the average absolute value of positional gradient ∂L μk across multiple iterations.

The positional gradient of a projected 3D Gaussian is the sum of the positional gradients across the pixels it covers: ∂L μk = X i ∂L ∂pi ∂pi ∂μk . (15) However, this criterion is suboptimal when a 3D Gaussian spans more than a single pixel, a scenario that is particularly relevant for large-scale urban scenes.

Specifically, since the positional gradient is composed of a sum of per-pixel gradients, these can point in different directions and thus cancel each other out.

Therefore, we threshold X i 
 
 
 
 ∂L ∂pi ∂pi ∂μk 
 
 
 
1 (16) as the criterion to drive densification decisions.

This ensures that the overall magnitude of the gradients is considered, independent of the direction.

However, this leads to an increased expected value, and therefore we increase the densification threshold to 0.0006.

Hyperparameters.

We describe the hyperparameters used for our method, while training details can be found in Appendix A.3.

For ADC, we use an opacity threshold of 0.005 to cull transparent 3D Gaussians.

To maximize view quality, we do not cull 3D Gaussians after densification stops.

We use a near clip plane at a 1.0m distance, scaled by the global scene scaling factor.

We set this threshold to avoid numerical instability in the projection of 3D Gaussians.

Indeed, the Jacobian Jc k used in gc k scales inversely with the depth of the primitive, which causes numerical instabilities as the depth of a 3D Gaussian approaches zero.

For γ(t), we use 6 frequencies to encode time t.

A.3 Experiments Data preprocessing.

For each dataset, we obtain the initialization of the 3D Gaussians from a point cloud of the scene obtained from the provided LiDAR measurements.

To avoid redundant points slowing down training, we voxelize this initial pointcloud with voxel sizes of τ := 0.1m and τ := 0.15m for the single- and multi-sequence experiments, respectively.

We use the provided 3D bounding box annotations to filter points belonging to dynamic objects, to initialize the 3D Gaussians for each object, and as our object poses ξ.

For KITTI and VKITTI, we follow the established benchmark used in.

We use the full resolution 375 × 1242 images for training and evaluation and evaluate at varying training set fractions.

For Argoverse 2, we follow the experimental setup of.

In particular, we use the full resolution 1550 × 2080 images for training and evaluation and use all cameras of every 10th temporal frame as the testing split.

Note that we used the provided masks from to mask out parts of the ego-vehicle for both training and evaluation.

For Waymo Open, we follow the experimental setup of EmerNeRF.

We use the three front cameras (FRONT, FRONT_LEFT, FRONT_RIGHT) and resize the images to 640 × 960 for both training and evaluation.

We use only the first LiDAR return as initial points for our reconstruction.

We follow and evaluate the cameras of every 10th temporal frame.

For separate evaluation of dynamic objects, we compute masks from the 2D ground truth camera bounding boxes.

We keep only objects exceeding a velocity of 1 m/s to filter for potential sensor and annotation noise.

We determine the velocities from the corresponding 3D bounding box annotations.

Note also that do not undistort the input images, and we follow this setup for a fair comparison.

Implementation details.

For Ldep, we use only the LiDAR measurements at the time of the camera sensor recording as ground truth to ensure dynamic objects receive valid depth supervision.

We implement our method in PyTorch with tools from nerfstudio.

For visualization of the depth, we use the inferno_r colormap and linear scaling in the 1-82.5 meters range.

During training, we use the Adam optimizer with β1 := 0.9, β2 := 0.999.

We use separate learning rates for each 3D Gaussian attribute, the neural fields, and the sequence latent codes ωt s.

In particular, for means μ, we use an exponential decay learning rate schedule from 1.6 · 10−5 to 1.6 · 10−6, for opacity α, we use a learning rate of 5 · 10−2, for scales a and rotations q, we use a learning rate of 10−3.

The neural fields are trained with an exponential decay learning rate schedule from 2.5 · 10−3 to 2.5 · 10−4.

The sequence latent vectors ωt s are optimized with a learning rate of 5 · 10−4.

We optimize camera and object pose parameters with an exponential decay learning rate schedule from 10−5 to 10−6.

To counter pose drift, we apply weight decay with a factor 10−2.

Note that we also optimize the height of object poses ξ.

We follow previous works and optimize the evaluation camera poses when optimizing training poses to compensate for pose errors introduced by drifting geometry through optimized training poses that may contaminate the view synthesis quality measurement.

In our multi-sequence experiments in Table 1 and Table 4, we train our model on 8 NVIDIA A100 40GB GPUs for 125,000 steps, taking approximately 2.5 days.

In our single-sequence experiments, we train our model on a single RTX 4090 GPU for several hours.

On Waymo Open, we train our model for 60,000 steps while for KITTI and VKITTI2 we train the model for 30,000 steps.

For our singl-sequence experiments in Table 4 we use a schedule of 100,000 steps.

We chose a longer schedule for Waymo Open and Argoverse 2 since the scenes are more complex and contain about 5 − 10× as many images as the sequences in KITTI and VKITTI2.

We linearly scale the number warm-up steps, the steps per ADC, and the maximum step to invoke ADC with the number of training steps.

For multi-GPU training, we reduce these parameters linearly with the number of GPUs.

However, we observed that scaling the learning rates linearly does perform subpar to the initial learning rates in the multi-GPU setup, and therefore we keep the learning rates the same across all experiments.

Additional ablation studies.

In Table 6a, we show that while our approach benefits from highquality 3D bounding boxes, it is robust to noise and achieves a high view synthesis quality even with noisy predictions acquired from a 3D tracking algorithm.

In Table 6b, we demonstrate that the deformation head yields a small, albeit noticeable improvement in quantitative rendering results.

This corroborates the utility of deformation head χ beyond the qualitative examples shown in Figures 4 and 8.

Note that the threshold to distinguish between dynamic and static areas is 1m/s following so that some instances like slow-moving pedestrians will be classified as static.

Also, since non-rigid entities usually cover only a small portion of the scene, expected improvements are inherently small.

In Table 7, we show that our modified ADC increases view quality in general, and perceptual quality in particular as it avoids blurry close-up renderings.

Note that our ADC leads to roughly twice the number of 3D Gaussians belonging to objects compared to vanilla ADC, thus avoiding insufficient densification.

We also show a qualitative example in Figure 5, illustrating this effect.

The close-up car rendering is significantly sharper using the modified ADC.

Note that for both variants, we use the absolute gradient criterion (see Appendix A.2) for a fair comparison.

Qualitative results.

We provide an additional qualitative comparison of the variants iii) and ii) introduced in Section 4.2, i.e. our model with and without transient geometry modeling.

In Figure 6, we show multiple examples confirming that iii) indeed models transient geometry such as tree leaves or temporary advertisement banners (bottom left), and effectively mitigates the severe artifacts present in the RGB renderings of ii).

Furthermore, the depth maps show that iii) faithfully represents the true geometry, while ii) lacks geometric variability across sequences.

In addition, we show qualitative comparisons to the state-of-the-art in Figure 7.

Our method continues to produce sharper renderings than the previous best-performing method, while also handling articulated objects such as pedestrians which are missing in the reconstruction of previous works (bottom two rows).

Finally, we show another temporal sequence of evaluation frames in Figure 8.

Our method handles unconstrained motions and can also reconstruct more complicated scenarios such as a pedestrian carrying a stroller (right), or a grocery bag (left).

