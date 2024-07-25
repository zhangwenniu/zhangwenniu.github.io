---
layout: mypost
title: Scaffold Gaussian
categories: [论文, 3DGS]
---

Figure 1.

Scaffold-GS represents the scene using a set of 3D Gaussians structured in a dual-layered hierarchy.

Anchored on a sparse grid of initial points, a modest set of neural Gaussians are spawned from each anchor to dynamically adapt to various viewing angles and distances.

Our method achieves rendering quality and speed comparable to 3D-GS with a more compact model (last row metrics: PSNR/storage size/FPS).

Across multiple datasets, Scaffold-GS demonstrates more robustness in large outdoor scenes and intricate indoor environments with challenging observing views e.g. transparency, specularity, reflection, texture-less regions and fine-scale details.

Abstract 

Neural rendering methods have significantly advanced photo-realistic 3D scene rendering in various academic and industrial applications.

The recent 3D Gaussian Splatting method has achieved the state-of-the-art rendering quality and speed combining the benefits of both primitive-based representations and volumetric representations.

However, it often leads to heavily redundant Gaussians that try to fit every training view, neglecting the underlying scene geometry.

Consequently, the resulting model becomes less robust to significant view changes, texture-less area and lighting effects.

We introduce Scaffold-GS, which uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum.

Anchor growing and pruning strategies are developed based on the importance of neural Gaussians to reliably improve the scene coverage.

We show that our method effectively reduces redundant Gaussians while delivering high-quality rendering.

We also demonstrates an enhanced capability to accommodate scenes with varying levels-of-detail and view-dependent ob* denotes equal contribution. servations, without sacrificing the rendering speed.

Project page: https://city-super.github.io/scaffold-gs/.

1.

Introduction Photo-realistic and real-time rendering of 3D scenes has always been a pivotal interest in both academic research and industrial domains, with applications spanning virtual reality, media generation, and large-scale scene visualization.

Traditional primitive-based representations like meshes and points are faster due to the use of rasterization techniques tailored for modern GPUs.

However, they often yield low-quality renderings, exhibiting discontinuity and blurry artifacts.

In contrast, volumetric representations and neural radiance fields utilize learning-based parametric models, hence can produce continuous rendering results with more details preserved.

Nevertheless, they come with the cost of timeconsuming stochastic sampling, leading to slower performance and potential noise.

In recent times, 3D Gaussian Splatting (3D-GS) has achieved state-of-the-art rendering quality and speed.

Initialized from point clouds derived from Structure from Motion (SfM), this method optimizes a set of 3D Gaussians to represent the scene.

It preserves the inherent continuity found in volumetric representations, whilst facilitating rapid rasterization by splatting 3D Gaussians onto 2D image planes.

While this approach offers several advantages, it tends to excessively expand Gaussian balls to accommodate every training view, thereby neglecting scene structure.

This results in significant redundancy and limits its scalability, particularly in the context of complex large-scale scenes.

Furthermore, view-dependent effects are baked into individual Gaussian parameters with little interpolation capabilities, making it less robust to substantial view changes and lighting effects.

We present Scaffold-GS, a Gaussian-based approach that utilizes anchor points to establish a hierarchical and regionaware 3D scene representation.

We construct a sparse grid of anchor points initiated from SfM points.

Each of these anchors tethers a set of neural Gaussians with learnable offsets, whose attributes (i.e. opacity, color, rotation, scale) are dynamically predicted based on the anchor feature and the viewing position.

Unlike the vanilla 3D-GS which allows 3D Gaussians to freely drift and split, our strategy exploits scene structure to guide and constrain the distribution of 3D Gaussians, whilst allowing them to locally adaptive to varying viewing angles and distances.

We further develop the corresponding growing and pruning operations for anchors to enhance the scene coverage.

Through extensive experiments, we show that our method delivers rendering quality on par with or even surpassing the original 3D-GS.

At inference time, we limit the prediction of neural Gaussians to anchors within the view frustum, and filter out trivial neural Gaussians based on their opacity with a filtering step (i.e. learnable selector).

As a result, our approach can render at a similar speed (around 100 FPS at 1K resolution) as the original 3D-GS with little computational overhead.

Moreover, our storage requirements are significantly reduced as we only need to store anchor points and MLP predictors for each scene.

In conclusion, our contributions are: 1) Leveraging scene structure, we initiate anchor points from a sparse voxel grid to guide the distribution of local 3D Gaussians, forming a hierarchical and region-aware scene representation; 2) Within the view frustum, we predict neural Gaussians from each anchor on-the-fly to accommodate diverse viewing directions and distances, resulting in more robust novel view synthesis; 3) We develop a more reliable anchor growing and pruning strategy utilizing the predicted neural Gaussians for better scene coverage. 2.

Related work MLP-based Neural Fields and Rendering.

Early neural fields typically adopt a multi-layer perceptron (MLP) as the global approximator of 3D scene geometry and appearance.

They directly use spatial coordinates (and viewing direction) as input to the MLP and predict point-wise attribute, e.g. signed distance to scene surface (SDF), or density and color of that point.

Because of its volumetric nature and inductive bias of MLPs, this stream of methods achieves the SOTA performance in novel view synthesis.

The major challenge of this scene representation is that the MLP need to be evaluated on a large number of sampled points along each camera ray.

Consequently, rendering becomes extremely slow, with limited scalability towards complex and large-scale scenes.

Despite several works have been proposed to accelerate or mitigate the intensive volumetric ray-marching, e.g. using proposal network, baking technique, and surface rendering.

They either incorporated more MLPs or traded rendering quality for speed.

Grid-based Neural Fields and Rendering.

This type of scene representations are usually based on a dense uniform grid of voxels.

They have been greatly used in 3D shape and geometry modeling.

Some recent methods have also focused on faster training and inference of radiance field by exploiting spatial data structure to store scene features, which were interpolated and queried by sampled points during ray-marching.

For instance, Plenoxel adopted a sparse voxel grid to interpolate a continuous density field, and represented viewdependent visual effects with Spherical Harmonics.

The idea of tensor factorization has been studied in multiple works to further reduce data redundancy and speed-up rendering.

K-planes used neural planes to parameterize a 3D scene, optionally with an additional temporal plane to accommodate dynamics.

Several generative works also capitalized on triplane structure to model a 3D latent space for better geometry consistency.

InstantNGP used a hash grid and achieved drastically faster feature query, enabling real-time rendering of neural radiance field.

Although these approaches can produce highquality results and are more efficient than global MLP representation, they still need to query many samples to render a pixel, and struggle to represent empty space effectively.

Point-based Neural Fields and Rendering.

Point-based representations utilize the geometric primitive (i.e. point clouds) for scene rendering.

A typical procedure is to rasterize an unstructured set of points using a fixed size, and exploits specialized modules on GPU and graphics APIs for rendering.

In spite of its fast speed and flexibil-ity to solve topological changes, they usually suffer from holes and outliers that lead to artifacts in rendering.

To alleviate the discontinuity issue, differentiable point-based rendering has been extensively studied to model objects geometry.

In particular, used differentiable surface splatting that treats point primitives as discs, ellipsoids or surfels that are larger than a pixel. augmented points with neural features and rendered using 2D CNNs.

As a comparison, Point-NeRF achieved high-quality novel view synthesis utilizing 3D volume rendering, along with region growing and point pruning during optimization.

However, they resorted to volumetric raymarching, hence hindered display rate.

Notably, the recent work 3D-GS employed anisotropic 3D Gaussians initialized from structure from motion (SfM) to represent 3D scenes, where a 3D Gaussian was optimized as a volume and projected to 2D to be rasterized as a primitive.

Since it integrated pixel color using α-blender, 3D-GS produced high-quality results with fine-scale detail, and rendered at real-time frame rate. 3.

Methods The original 3D-GS optimizes Gaussians to reconstruct every training view, with heuristic splitting and pruning operations but in general neglects the underlying scene structure.

This often leads to highly redundant Gaussians and makes the model less robust to novel viewing angles and distances.

To address this issue, we propose a hierarchical 3D Gaussian scene representation that respects the scene geometric structure, with anchor points initialized from SfM to encode local scene information and spawn local neural Gaussians.

The physical properties of neural Gaussians are decoded from the learned anchor features in a viewdependent manner on-the-fly.

Fig. 2 illustrates our framework.

We start with a brief background of 3D-GS then unfold our proposed method in details.

Sec. 3.2.1 introduces how to initialize the scene with a regular sparse grid of anchor points from the irregular SfM point clouds.

Sec. 3.2.2 explains how we predict neural Gaussians properties based on anchor points and view-dependent information.

To make our method more robust to the noisy initialization, Sec. 3.3 introduces a neural Gaussian based “growing” and “pruning” operations to refine the anchor points.

Sec. 3.4 elaborates training details. 3.1.

Preliminaries 3D-GS represents the scene with a set of anisotropic 3D Gaussians that inherit the differential properties of volumetric representation while be efficiently rendered via a tile-based rasterization.

Starting from a set of Structure-from-Motion (SfM) points, each point is designated as the position (mean) μ of a 3D Gaussian: G(x) = e− 1 2 (x−μ)T Σ−1(x−μ), (1) where x is an arbitrary position within the 3D scene and Σ denotes the covariance matrix of the 3D Gaussian. Σ is formulated using a scaling matrix S and rotation matrix R to maintain its positive semi-definite: Σ = RSST RT , (2) In addition to color c modeled by Spherical harmonics, each 3D Gaussian is associated with an opacity α which is multiplied by G(x) during the blending process.

Distinct from conventional volumetric representations, 3D-GS efficiently renders the scene via tile-based rasterization instead of resource-intensive ray-marching.

The 3D Gaussian G(x) are first transformed to 2D Gaussians G′(x) on the image plane following the projection process as described in.

Then a tile-based rasterizer is designed to efficiently sort the 2D Gaussians and employ α-blending: C(x′) = X i∈N ciσi i−1 Y j=1 (1 − σj), σi = αiG′ i(x′), (3) where x′ is the queried pixel position and N denotes the number of sorted 2D Gaussians associated with the queried pixel.

Leveraging the differentiable rasterizer, all attributes of the 3D Gaussians are learnable and directly optimized end-to-end via training view reconstruction. 3.2.

Scaffold-GS 3.2.1 Anchor Point Initialization Consistent with existing methods, we use the sparse point cloud from COLMAP as our initial input.

We then voxelize the scene from this point cloud P ∈ RM×3 as: V=  P ε  · ε, (4) where V ∈ RN×3 denotes voxel centers, and ε is the voxel size.

We then remove duplicate entries, denoted by {·} to reduce the redundancy and irregularity in P.

The center of each voxel v ∈ V is treated as an anchor point, equipped with a local context feature fv ∈ R32, a scaling factor lv ∈ R3, and k learnable offsets Ov ∈ Rk×3.

In a slight abuse of terminology, we will denote the anchor point as v in the following context.

We further enhance fv to be multi-resolution and view-dependent.

For each anchor v, we 1) create a features bank {fv, fv↓1 , fv↓2 }, where ↓n denotes fv being down-sampled by 2n factors; 2) blend the feature bank with view-dependent weights to form an integrated anchor feature ˆ fv.

Specifically, Given a camera at position xc and an anchor at position xv, we calculate their relative distance and viewing direction with: δvc = ∥xv − xc∥2, ⃗dvc = xv − xc ∥xv − xc∥2 , (5) then weighted sum the feature bank with weights predicted from a tiny MLP Fw: {w, w1, w2} = Softmax(Fw(δvc, ⃗dvc)), (6) ˆ fv = w · fv + w1 · fv↓1 + w2 · fv↓2 , (7) 3.2.2 Neural Gaussian Derivation In this section, we elaborate on how our approach derives neural Gaussians from anchor points.

Unless specified otherwise, F∗ represents a particular MLP throughout the section.

Moreover, we introduce two efficient pre-filtering strategies to reduce MLP overhead.

We parameterize a neural Gaussian with its position μ ∈ R3, opacity α ∈ R, covariance-related quaternion q ∈ R4 and scaling s ∈ R3, and color c ∈ R3.

As shown in Fig. 2(b), for each visible anchor point within the viewing frustum, we spawn k neural Gaussians and predict their attributes.

Specifically, given an anchor point located at xv, the positions of its neural Gaussians are calculated as: {μ0, ..., μk−1} = xv + {O0, . . . , Ok−1} · lv, (8) where {O0, O1, ..., Ok−1} ∈ Rk×3 are the learnable offsets and lv is the scaling factor associated with that anchor, as described in Sec. 3.2.1.

The attributes of k neural Gaussians are directly decoded from the anchor feature ˆ fv, the relative viewing distance δvc and direction ⃗dvc between the camera and the anchor point (Eq. 5) through individual MLPs, denoted as Fα, Fc, Fq and Fs.

Note that attributes are decoded in one-pass.

For example, opacity values of neural Gaussians spawned from an anchor point are given by: {α0, ..., αk−1} = Fα( ˆ fv, δvc, ⃗dvc), (9) their colors {ci}, quaternions {qi} and scales {si} are similarly derived.

Implementation details are in supplementary.

Note that the prediction of neural Gaussian attributes are on-the-fly, meaning that only anchors visible within the frustum are activated to spawn neural Gaussians.

To make the rasterization more efficient, we only keep neural Gaussians whose opacity values are larger than a predefined threshold τα.

This substantially cuts down the computational load and helps our method maintain a high rendering speed on-par with the original 3D-GS. 3.3.

Anchor Points Refinement Growing Operation.

Since neural Gaussians are closely tied to their anchor points which are initialized from SfM points, their modeling power is limited to a local region, as has been pointed out in.

This poses challenges to the initial placement of anchor points, especially in textureless and less observed areas.

We therefore propose an errorbased anchor growing policy that grows new anchors where neural Gaussians find significant.

To determine a significant area, we first spatially quantize the neural Gaussians by constructing voxels of size εg.

For each voxel, we compute the averaged gradients of the included neural Gaussians over N training iterations, denoted as ∇g.

Then, voxels with ∇g > τg is deemed as significant, where τg is a pre-defined threshold; and a new anchor point is thereby deployed at the center of that voxel if there was no anchor point established.

Fig. 3 illustrates this growing operation.

In practice, we quantize the space into multi-resolution voxel grid to al-low new anchors to be added at different granularity, where ε(m) g = εg/4m−1, τ (m) g = τg ∗ 2m−1, (10) where m denotes the level of quantization.

To further regulate the addition of new anchors, we apply a random elimination to these candidates.

This cautious approach to adding points effectively curbs the rapid expansion of anchors.

Pruning Operation To eliminate trivial anchors, we accumulate the opacity values of their associated neural Gaussians over N training iterations.

If an anchor fails to produce neural Gaussians with a satisfactory level of opacity, we then remove it from the scene. 3.4.

Losses Design We optimize the learnable parameters and MLPs with respect to the L1 loss over rendered pixel colors, with SSIM term LSSIM and volume regularization Lvol.

The total supervision is given by: L = L1 + λSSIMLSSIM + λvolLvol, (11) where the volume regularization Lvol is: Lvol = Nng X i=1 Prod(si). (12) Here, Nng denotes the number of neural Gaussians in the scene and Prod(·) is the product of the values of a vector, e.g., in our case the scale si of each neural Gaussian.

The volume regularization term encourages the neural Gaussians to be small with minimal overlapping. 4.

Experiments 4.1.

Experimental Setup Dataset and Metrics.

We conducted a comprehensive evaluation across 27 scenes from publicly available datasets.

Specifically, we tested our approach on all available scenes tested in the 3D-GS, including seven scenes from Mip-NeRF360, two scenes from Tanks&Temples, two scenes from DeepBlending and synthetic Blender dataset.

We additionally evaluated on datasets with contents captured at multiple LODs to demonstrate our advantages in view-adaptive rendering.

Six scenes from BungeeNeRF and two scenes from VR-NeRF are selected.

The former provides multiscale outdoor observations and the latter captures intricate indoor environments.

Apart from the commonly used metrics (PSNR, SSIM, and LPIPS), we additionally report the storage size (MB) and the rendering speed (FPS) for model compactness and performance efficiency.

We provide the averaged metrics over all scenes of each dataset in the main paper and leave the full quantitative results on each scene in the supplementary.

Baseline and Implementation. 3D-GS is selected as our main baseline for its established SOTA performance in novel view synthesis.

Both 3D-GS and our method were trained for 30k iterations.

We also record the results of MipNeRF360, iNGP and Plenoxels as in.

For our method, we set k = 10 for all experiments.

All the MLPs employed in our approach are 2-layer MLPs with ReLU activation; the dimensions of the hidden units are all 32.

For anchor points refinement, we average gradients over N = 100 iterations, and by default use τg = 64ε.

On intricate scenes and the ones with dominant texture-less regions, we use τg = 16ε.

An anchor is pruned if the accumulated opacity of its neural Gaussians is less than 0.5 at each round of refinement.

The two loss weights λSSIM and λvol are set to 0.2 and 0.001 in our experiments.

Please check the supplementary material for more details. 4.2.

Results Analysis Our evaluation was conducted on diverse datasets, ranging from synthetic object-level scenes, indoor and outdoor environments, to large-scale urban scenes and landscapes.

A variety of improvements can be observed especially on challenging cases, such as texture-less area, insufficient observations, fine-scale details and view-dependent light effects.

See Fig. 1 and Fig. 4 for examples.

Comparisons.

In assessing the quality of our approach, we compared with 3D-GS, Mip-NeRF360, iNGP and Plenoxels on real-world datasets.

Qualitative results are presented in Tab. 1.

The quality metrics for Mip-NeRF360, iNGP and Plenoxels align with those reported in the 3D-GS study.

It can be noticed that our approach achieves comparable results with the SOTA algorithms on Mip-NeRF360 dataset, and surpassed the SOTA on Tanks&Temples and DeepBlending, which captures more challenging environments with the presence of e.g. changing lighting, texture-less regions and reflections.

In terms of efficiency, we evaluated rendering speed and storage size of our method and 3D-GS, as shown in Tab. 2.

Our method achieved real-time rendering while using less storage, indicating that our model is more compact than 3D-GS without sacrificing rendering quality and speed.

Additionally, akin to prior grid-based methods, our approach converged faster than 3D-GS.

See supplementary material for more analysis.

We also examined our method on the synthetic Blender dataset, which provides an exhaustive set of views capturing objects at 360◦.

A good set of initial SfM points is not readily available in this dataset, thus we start from 100k grid points and learn to grow and prune points with our anchor refinement operations.

After 30k iterations, we used the re-mained points as initialized anchors and re-run our framework.

The PSNR score and storage size compared with 3DGS are presented in Tab. 3.

Fig. 1 also demonstrates that our method can achieve better visual quality with more reliable geometry and texture details.

Multi-scale Scene Contents.

We examined our model’s capability in handling multi-scale scene details on the BungeeNeRF and VR-NeRF datasets.

As shown in Tab. 3, our method achieved superior quality whilst using fewer storage size to store the model compared to 3D-GS.

As illustrated in Fig. 4 and Fig. 5, our method was superior in accommodating varying levels of detail in the scene.

In contrast, images rendered from 3D-GS often suffered from noticeable blurry and needle-shaped artifacts.

This is likely because that Gaussian attributes are optimized to overfit multi-scale training views, creating excessive Gaussians that work for each observing distance.

However, it can easily lead to ambiguity and uncertainty when synthesizing novel views, since it lacks the ability to reason about viewing angle and distance.

On contrary, our method efficiently encoded local structures into compact neural features, enhancing both rendering quality and convergence speed.

Details are provided in the supplementary material.

Feature Analysis.

We further perform an analysis of the learnable anchor features and the selector mechanism.

As depicted in Fig. 6, the clustered pattern suggests that the compact anchor feature spaces adeptly capture regions with similar visual attributes and geometries, as evidenced by their proximity in the encoded feature space.

View Adaptability.

To support that our neural Gaussians are view-adaptive, we explore how the values of attributes change when the same Gaussian is observed from different positions.

Fig. 7 demonstrates a varying distribution of attributes intensity at different viewing positions, while maintaining a degree of local continuity.

This accounts for the superior view adaptability of our method compared to the static attributes of 3D-GS, as well as its enhanced generalizability to novel views.

Selection Process by Opacity.

We examine the decoded opacity from neural Gaussians and our opacity-based selection process (Fig. 2(b)) from two aspects.

First, without the anchor point refinement module, we filter neural Gaussian using their decoded opacity values to extract geometry from a random point cloud.

Fig. 8 demonstrates that the remained neural Gaussians effectively reconstruct the coarse structure of the bulldozer model from random points, highlighting its capability for implicit geometry modeling under mainly rendering-based supervision.

We found this is conceptually similar to the proposal network utilized in, serving as the geometry proxy estimator for efficient sampling.

Second, we apply different k values in our method. 4.3.

Ablation Studies Efficacy of Filtering Strategies.

We evaluated our filtering strategies (Sec. 3.2.2), which we found crucial for speeding up our method.

As Tab. 4 shows, while these strategies had no notable effect on fidelity, they significantly enhanced inference speed.

However, there was a risk of masking pertinent neural Gaussians, which we aim to address in future works.

Efficacy of Anchor Points Refinement Policy.

We evaluated our growing and pruning operations described in Sec. 3.3.

Tab. 5 shows the results of disabling each operation in isolation and maintaining the rest of the method.

We found that the addition operation is crucial for accurately reconstructing details and texture-less areas, while the pruning operation plays an important role in eliminating trivial Gaussians and maintaining the efficiency of our approach. 4.4.

Discussions and Limitations Through our experiments, we found that the initial points play a crucial role for high-fidelity results.

Initializing our framework from SfM point clouds is a swift and viable solution, considering these point clouds usually arise as a byproduct of image calibration processes.

However, this approach may be suboptimal for scenarios dominated by large texture-less regions.

Despite our anchor point refinement strategy can remedy this issue to some extent, it still suffers from extremely sparse points.

We expect that our algorithm will progressively improve as the field advances, yielding more accurate results.

Further details are discussed in the supplementary material. 5.

Conclusion In this work, we introduce Scaffold-GS, a novel 3D neural scene representation for efficient view-adaptive rendering.

The core of Scaffold-GS lies in its structural arrangement of 3D Gaussians guided by anchor points from SfM, whose attributes are on-the-fly decoded from view-dependent MLPs.

We show that our approach leverages a much more compact set of Gaussians to achieve comparable or even better results than the SOTA algorithms.

The advantage of our view-adaptive neural Gaussians is particularly evident in challenging cases where 3D-GS usually fails.

We further show that our anchor points encode local features in a meaningful way that exhibits semantic patterns to some degree, suggesting its potential applicability in a range of versatile tasks such as large-scale modeling, manipulation and interpretation in the future. 6.

Overview This supplementary is organized as follows: (1) In the first section, we elaborate implementation details of our Scaffold-GS, including anchor point feature enhancement (Sec.3.2.1), structure of MLPs (Sec.3.2.2) and anchor point refinement strategies (Sec.3.3); (2) The second part describes our dataset preparation steps.

We then show additional experimental results and analysis based on our training observations. 7.

Implementation details.

Feature Bank.

To enhance the view-adaptability, we update the anchor feature through a view-dependent encoding.

Following calculating the relative distance δvc and viewing direction ⃗dvc of a camera and an anchor, we predict a weight vector w ∈ R3 as follows: (w, w1, w2) = Softmax(Fw(δvc, ⃗dvc)), (13) where Fw is a tiny MLP that serves as a view encoding function.

We then encode the view direction information to the anchor feature fv by compositing a feature bank containing information with different resolutions as follows: ˆ fv = w · fv + w1 · fv↓1 + w2 · fv↓2 , (14) In practice, we implement the feature bank via slicing and repeating, as illustrated in Fig. 10.

We found this slicing and mixture operation improves Scaffold-GS’s ability to capture different scene granularity.

The distribution of feature bank’s weights is illustrated in Fig. 11.

MLPs as feature decoders.

The core MLPs include the opacity MLP Fα, the color MLP Fc and the covariance MLP Fs and Fq.

All of these F∗ are implemented in a LINEAR → RELU → LINEAR style with the hidden dimension of 32, as illustrated in Fig. 12.

Each branch’s output is activated with a head layer.

For opacity, the output is activated by Tanh, where value 0 serves as a natural threshold for selecting valid samples and the final valid values can cover the full range of and activate it with a normalization to obtain a valid quaternion. • For scaling, we adjust the base scaling sv of each anchor with the MLP output as follows: {s0, ..., sk−1} = Sigmoid(Fs) · sv, (16) Voxel Size.

The voxel size ε sets the finest anchor resolution.

We employ two strategies: 1) Use the median of the nearest-neighbor distances among all initial points: ε is adapted to point cloud density, yielding denser anchors with enhanced rendering quality but might introduce more computational overhead; 2) Set ε manually to either 0.005 or 0.01: this is effective in most scenarios but might lead to missing details in texture-less regions.

We found these two strategies adequately accommodate various scene complexities in our experiments.

Anchor Refinement.

As briefly discussed in the main paper, the voxelization process suggests that our method may behave sensitive to initial SfM results.

We illustrate the effect of the anchor refinement process in Fig. 13, where new anchors enhance scene details and fill gaps in large textureless regions and less observed areas. 8.

Experiments and Results Additional Data Preprocessing.

We used COLMAP to estimate camera poses and generate SfM points for VR-NeRF and BungeeNeRF datasets.

Both two datasets are challenging in terms of varying levels of details presented in the captures.

The VR-NeRF dataset was tested using its eye-level subset with 3 cameras.

For all other datasets, we adhered to the original 3D-GS method, sourcing them from public resources.

Per-scene Results.

Here we list the error metrics used in our evaluation in Sec.4 across all considered methods and scenes, as shown in Tab. 6-17.

Training Process Analysis.

Figure 14 illustrates the variations in PSNR during the training process for both training and testing views.

Our method demonstrates quicker convergence, enhanced robustness, and better generalization compared to 3D-GS, as evidenced by the rapid increase in training PSNR and higher testing PSNR.

Specifically, for the Amsterdam and Pompidou scenes in BungeeNeRF, we trained them with images at three coarser scales and evaluated them at a novel finer scale.

The fact that 3D-GS achieved higher training PSNR but lower testing PSNR indicates its tendency to overfit at training scales.
