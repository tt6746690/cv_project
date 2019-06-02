# Summary



From reading some recent papers on demosaicing. It is obvious to see the following trends

- utilizing neural networks in some capacity in solving the problem. This can range from learning image prior from data in some classical convex optimization frameowork such as ADMM (2017_learned_proximal_operators) to end-to-end training on large datasets for demosaicing [2018_deepdemosaicking, 2019_deepisp]. 
- joint optimization, bundling demosaicing with deblurring, superresolution, denoising [2016_deepjoint, 2017_jointadmm, 2018_jointgan_perceptual, 2018_iterative_resnet_joint], etc. To the extreme, the entire image processing pipeline is under one single framework [2014_flexISP, 2019_deepisp]

I think [2017_learned_proximal_operators, 2017_RED, 2018_iterative_resnet_joint] are most applicable for our project. The idea of these methods is to formulate demosaicing as an inverse problems. The image priors are learnt from data and acts as regularizers in the optimization problem. The benefit of such method is given as follows

- does not require large datasets that deep neural network methods rely on. Usually, small dataset of at most several hundred images is enough to give promising results
- the image priors, i.e. total variation, cross-channel correlation, are learnt from data. 
    - they are not ad hoc, and heuristic based
    - the priors can be learnt for different downstream reconstruction tasks, i.e. multispectral imaging and structured light

There are however problems, for example

- convergence is not guaranteed for nonconvex regularizers.

    - there is some work by [2017_RED] that under some assumptions that most denoising method satisfies, the method is guaranteed to converge.
    - there is some theory used by [this paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chang_One_Network_to_ICCV_2017_paper.pdf) which states that under certain assumptions, nonconvex regularizers converges to stationary points

- runtime is slower than interpolation based or purely neural network based methods
 




# Demosaicing Method Performance



The following tables keep benchmarked the performance of demosaicing methods over a few datasets

- `kodak`: 24 photos of size 768x512 or 512x768 (http://r0k.us/graphics/kodak/)
- `mcm`: (https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm)
- `msr` (Microsoft dataset): 500 bayer images in both linear RGB and sRGB space (https://www.microsoft.com/en-us/download/details.aspx?id=52535)
- `{kodak,mcm}_sigma20`: gaussian white noise of `\sigma=20` is applied to test images

Row `i` and column `j` of each table indicates the psnr of `j`-th method in `i`-th paper evaluated on a particular dataset
    
## kodak


|  | 2014_flexISP| 2016_sequential_energy_minimization| 2016_deepjoint| 2017_jointadmm| 2018_deepdemosaicking| 2018_iterative_resnet_joint| 2018_deepjoint_design| 2019_multiframe_superres|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 2016_deepjoint| 40.0| 0.0| 41.2| 0.0| 0.0| 0.0| 0.0| 0.0|
| 2017_jointadmm| 34.98| 0.0| 33.88| 31.63| 0.0| 0.0| 0.0| 0.0|
| 2018_deepdemosaicking| 0.0| 0.0| 41.45| 0.0| 42.12| 0.0| 0.0| 0.0|
| 2018_iterative_resnet_joint| 40.0| 35.3| 41.2| 0.0| 42.0| 42.0| 0.0| 0.0|
| 2018_deepjoint_design| 0.0| 0.0| 41.79| 0.0| 0.0| 0.0| 41.86| 0.0|
| 2019_multiframe_superres| 35.08| 0.0| 39.67| 0.0| 0.0| 0.0| 0.0| 42.86|




## kodak_sigma20


|  | 2014_flexISP| 2016_sequential_energy_minimization| 2016_deepjoint| 2017_jointadmm| 2018_jointgan_perceptual| 2018_deepjoint_design|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 2018_jointgan_perceptual| 25.15| 23.0| 29.17| 29.26| 30.74| 0.0|
| 2018_deepjoint_design| 0.0| 0.0| 30.0| 0.0| 0.0| 31.2|




## mcm


|  | 2014_flexISP| 2016_sequential_energy_minimization| 2016_deepjoint| 2017_jointadmm| 2017_learned_proximal_operators| 2018_deepdemosaicking| 2018_iterative_resnet_joint| 2018_deepjoint_design| 2019_multiframe_superres|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 2016_deepjoint| 38.6| 0.0| 39.5| 0.0| 0.0| 0.0| 0.0| 0.0| 0.0|
| 2017_jointadmm| 35.18| 0.0| 32.49| 32.66| 0.0| 0.0| 0.0| 0.0| 0.0|
| 2017_learned_proximal_operators| 36.12| 0.0| 39.5| 0.0| 37.12| 0.0| 0.0| 0.0| 0.0|
| 2018_iterative_resnet_joint| 38.6| 30.8| 39.5| 0.0| 0.0| 39.0| 39.7| 0.0| 0.0|
| 2018_deepjoint_design| 0.0| 0.0| 39.14| 0.0| 0.0| 0.0| 0.0| 39.51| 0.0|
| 2019_multiframe_superres| 35.15| 0.0| 37.58| 0.0| 0.0| 0.0| 0.0| 0.0| 41.26|




## mcm_sigma20


|  | 2014_flexISP| 2016_sequential_energy_minimization| 2016_deepjoint| 2017_jointadmm| 2018_jointgan_perceptual| 2018_deepjoint_design|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 2018_jointgan_perceptual| 25.01| 22.99| 28.79| 29.31| 30.77| 0.0|
| 2018_deepjoint_design| 0.0| 0.0| 30.15| 0.0| 0.0| 30.87|




## msr_linear


|  | 2014_flexISP| 2016_sequential_energy_minimization| 2016_deepjoint| 2018_cascade_residual_denoising| 2018_iterative_resnet_joint| 2019_deepisp|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 2016_sequential_energy_minimization| 40.0| 40.92| 0.0| 0.0| 0.0| 0.0|
| 2016_deepjoint| 40.0| 0.0| 42.7| 0.0| 0.0| 0.0|
| 2018_cascade_residual_denoising| 0.0| 40.9| 0.0| 41.0| 0.0| 0.0|
| 2018_iterative_resnet_joint| 40.0| 40.9| 42.7| 0.0| 42.8| 0.0|
| 2019_deepisp| 38.28| 38.93| 38.6| 0.0| 0.0| 39.31|




## msr_linear_noisy


|  | 2014_flexISP| 2016_sequential_energy_minimization| 2016_deepjoint| 2018_cascade_residual_denoising| 2018_iterative_resnet_joint|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 2016_sequential_energy_minimization| 38.28| 38.93| 0.0| 0.0| 0.0|
| 2018_cascade_residual_denoising| 0.0| 38.8| 0.0| 39.2| 0.0|
| 2018_iterative_resnet_joint| 0.0| 38.8| 38.6| 0.0| 40.1|




# Paper summaries


- [FlexISP:A Flexible Camera Image Processing Framework (2014_flexISP)](http://www.cs.ubc.ca/labs/imager/tr/2014/FlexISP/FlexISP_Heide2014_lowres.pdf)
    - notes
        - end-to-end image processing that enforce image priors as proximal operators and use ADMM/primal-dual for optimization
        - end-to-end reduces error introduced in each steps of image processing, as each stage is not independent
        - applied to demosaicing, denoising, deconvolution, and a variety of reconstruction tasks
    - remark
        - recontruction based, no need for large datasets
        - classical regularizers that have proven to work well
        - choice of prior is shown to influence performance, so the choice of prior is important but ad hoc
        - no principled ways to pick solver parameters, i.e. the weight of regularizers
        - nonconvexity of the regularizers makes the optimization not guaranteed to converge to global optimum




- [Learning Joint Demosaicing and Denoising Based on Sequential Energy Minimization (2016_sequential_energy_minimization)](https://pure.tugraz.at/ws/portalfiles/portal/3625282/0004.pdf)
    - notes
        - demosacing+denoising as image restoration problem
        - method learngs efficient regularization by a variational energy minimization
        - demosaicing performed in linear RGB space
        - adaptable to different CFA pattern
        - regularization is learnt from data
        - noise level is more realistic, i.e. Poisson + Gaussian
        - data fidelity term and image priors are learnt from data
    - remark
        - 200 training images, which seems reasonable




- [Deep Joint Demosaicking and Denoising (2016_deepjoint)](https://groups.csail.mit.edu/graphics/demosaicnet/data/demosaicnet_slides.pdf)
    - notes
        - data-driven (instead of hand-crafted priors) approach to demosaicing and denoising using deep neural nets
        - goal is to reduce computation speed and reduce artifacts
        - generated millions of sRGB images, mosaicked, and injected noise.
        - came up with new metrics for identifying hard patches
    - remark
        - requires 2.5M images, not really feasible




- [Joint demosaicing and denoising of noisy bayer images with ADMM (2017_jointadmm)](https://www.researchgate.net/profile/Hanlin_Tan/publication/317058420_Joint_demosaicing_and_denoising_of_noisy_bayer_images_with_ADMM/links/59479f95a6fdccfa5949fc82/Joint-demosaicing-and-denoising-of-noisy-bayer-images-with-ADMM.pdf?origin=publication_detail)
    - notes
        - unified objective function with hidden priors, optimized wth ADMM for demosaicing noisy bayer input
        - included 4 prior terms, i.e. smoothness TV, denoising CBM3D, cross-channel, interpolation-based priors
        - results showed ADMM based more robust to noise
    - remark
        - purely optimization based
        - however regularizers are pretty ad hoc




- [The Little Engine that Could:Regularization by Denoising (RED) (2017_RED)](https://arxiv.org/pdf/1611.02862.pdf)
    - notes
        - use denoiser as regularization of inverse problems
        - image adaptive laplacian-based function that motivates the use of denoisers
        - assumptions of denoisers and proved convergence of results
        - achieved state of art image superresolution




- [Learning Proximal Operators:Using Denoising Networks for Regularizing Inverse Imaging Problems (2017_learned_proximal_operators)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Meinhardt_Learning_Proximal_Operators_ICCV_2017_paper.pdf)
    - notes
        - replace regularizer in energy minimization methods (primal-dual hybrid gradient PDHG) with a denoising neural network - residual denoising network DnCNN (https://arxiv.org/pdf/1608.03981.pdf)
        - reduce problem-specific training (i.e. different images, different noise levels)
        - related work section detailed some theory, proofs on convergence of custom proximal algorithms
        - RED is motivated by the observation that proximal operator is equivalent to image denoising
        - method implemented in a DSL for proximal methods in processing images (http://people.csail.mit.edu/jrk/proximal.pdf)
    - remark
        - DnCNN trained with 400 images of size 180x180, larger dataset negligible performance increase
        - seem to claim that using neuralnets for regularizers is better




- [Joint Demosaicing and Denoising with Perceptual Optimization on a Generative Adversarial Network (2018_jointgan_perceptual)](https://arxiv.org/pdf/1802.04723.pdf)
    - notes
        - used discriminator instead of PSNR/SSIM to remark perceptual quality of the reconstructed image
        - jointly optimize for demosaic and denoise
    - remark
        - 1400 high quality color photo, into 100x100 patches, data-augmentation x8 -> 320000 iamges
        - maybe manageable




- [DeepDemosaicking:Adaptive Image Demosaicking via Multiple Deep Fully Convolutional Networks (2018_deepdemosaicking)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8283772)
    - notes
        - initial demosaic combined with demosaic refinement reduces color artifacts using deep residual estimation
    - remark
        - 838 high res image from several sources -> 100000 64x64 patches




- [Deep Image Demosaicking using a Cascade of Convolutional Residual Denoising Networks (2018_cascade_residual_denoising)](https://arxiv.org/pdf/1803.05215.pdf)
    - notes
        - majorization maximiztion network
        - claimed to be more interpretable than deep learning approaches, and generalize well on small dataset
        - deep residual denoiser
    - remark
        - a previous paper of iterative resnet paper




- [Iterative Residual Network for Deep Joint Image Demosaicking and Denoising (2018_iterative_resnet_joint)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8668795)
    - notes
        - iterative neural networks for efficient optimization (majorization minimization)
        - use resdnet to learn the regularizing term
        - the algorithm is equivalent to Inexact Proximal Gradient Descent
        - used more realistic affine noise model instead of gaussian noise model - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.1943&rep=rep1&type=pdf
    - remark
        - compared to RED, P3, this method does not require setting of hyperparameters
        - resdnet pretrained on berkeley dataset (500 color images)
        - this method trained on 200 linear noise free RGB images (`msr`) outperforms 2016_deepjoint




- [Deep Joint Design of Color Filter Arrays and Demosaicing (2018_deepjoint_design)](https://inf.ufrgs.br/~bhenz/projects/joint_cfa_demosaicing/deep_joint_design_of_color_filter_arrays_and_demosaicing_pre-print.pdf)
    - notes
        - uses autoencoder to encode input to `multispectral mosaic`, which is then decoded to full resolution image
        - finds mosaic pattern and demosaicing algorithm that minimizes color-reconstruction error
        - achieved state of arts result, while allowing for cfa mosaic design
    - remark
        - has some good background references on cfa design, maybe applicable in our case
        - a followup https://evanfletcher42.com/2018/09/23/arbitrary-learned-mosaic/




- [DeepISP:Towards Learning an End-to-End Image Processing Pipeline (2019_deepisp)](https://arxiv.org/pdf/1801.06724.pdf)
    - notes
        - learns mapping from raw low-light mosaiced image to output image - low-level tasks (demosaicing+denoising) - high-level (color correction, image adjustment)
    - remark
        - masking patterns are fixed, the color of each filter is tuned
        - training dataset too large to be feasible, 5 days of training time




- [Handheld Multi-Frame Super-Resolution (2019_multiframe_superres)](https://arxiv.org/pdf/1905.03277.pdf)
    - notes
        - the method creates a complete RGB without demosaicing from a burst of CFA raw images
        - the method uses natural hand tremor to acquire images with small offsets
        - the frames are aligned and merged to form a single image
        - solve demosaicing + superresolution as an image reconstruction problem
        - does not rely on cross-channel correlation
    - remark
        - hand tremor not feasible for c2b camera



