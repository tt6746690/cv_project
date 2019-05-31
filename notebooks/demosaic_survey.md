- [FlexISP:A Flexible Camera Image Processing Framework](http://www.cs.ubc.ca/labs/imager/tr/2014/FlexISP/FlexISP_Heide2014_lowres.pdf)
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




- [Learning Joint Demosaicing and Denoising Based on Sequential Energy Minimization](https://pure.tugraz.at/ws/portalfiles/portal/3625282/0004.pdf)
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




- [Deep Joint Demosaicking and Denoising](https://groups.csail.mit.edu/graphics/demosaicnet/data/demosaicnet_slides.pdf)
    - notes
        - data-driven (instead of hand-crafted priors) approach to demosaicing and denoising using deep neural nets
        - goal is to reduce computation speed and reduce artifacts
        - generated millions of sRGB images, mosaicked, and injected noise.
        - came up with new metrics for identifying hard patches
    - remark
        - requires 2.5M images, not really feasible




- [Joint demosaicing and denoising of noisy bayer images with ADMM](https://www.researchgate.net/profile/Hanlin_Tan/publication/317058420_Joint_demosaicing_and_denoising_of_noisy_bayer_images_with_ADMM/links/59479f95a6fdccfa5949fc82/Joint-demosaicing-and-denoising-of-noisy-bayer-images-with-ADMM.pdf?origin=publication_detail)
    - notes
        - unified objective function with hidden priors, optimized wth ADMM for demosaicing noisy bayer input
        - included 4 prior terms, i.e. smoothness TV, denoising CBM3D, cross-channel, interpolation-based priors
        - results showed ADMM based more robust to noise
    - remark
        - purely optimization based
        - however regularizers are pretty ad hoc




- [The Little Engine that Could:Regularization by Denoising (RED)](https://arxiv.org/pdf/1611.02862.pdf)
    - notes
        - use denoiser as regularization of inverse problems
        - image adaptive laplacian-based function that motivates the use of denoisers
        - assumptions of denoisers and proved convergence of results
        - achieved state of art image superresolution




- [Learning Proximal Operators:Using Denoising Networks for Regularizing Inverse Imaging Problems](http://openaccess.thecvf.com/content_ICCV_2017/papers/Meinhardt_Learning_Proximal_Operators_ICCV_2017_paper.pdf)
    - notes
        - replace regularizer in energy minimization methods (primal-dual hybrid gradient PDHG) with a denoising neural network - residual denoising network DnCNN (https://arxiv.org/pdf/1608.03981.pdf)
        - reduce problem-specific training (i.e. different images, different noise levels)
        - related work section detailed some theory, proofs on convergence of custom proximal algorithms
        - RED is motivated by the observation that proximal operator is equivalent to image denoising
        - method implemented in a DSL for proximal methods in processing images (http://people.csail.mit.edu/jrk/proximal.pdf)
    - remark
        - DnCNN trained with 400 images of size 180x180, larger dataset negligible performance increase
        - seem to claim that using neuralnets for regularizers is better




- [Joint Demosaicing and Denoising with Perceptual Optimization on a Generative Adversarial Network](https://arxiv.org/pdf/1802.04723.pdf)
    - notes
        - used discriminator instead of PSNR/SSIM to remark perceptual quality of the reconstructed image
        - jointly optimize for demosaic and denoise
    - remark
        - 1400 high quality color photo, into 100x100 patches, data-augmentation x8 -> 320000 iamges
        - maybe manageable




- [DeepDemosaicking:Adaptive Image Demosaicking via Multiple Deep Fully Convolutional Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8283772)
    - notes
        - initial demosaic combined with demosaic refinement reduces color artifacts using deep residual estimation
    - remark
        - 838 high res image from several sources -> 100000 64x64 patches




- [Deep Image Demosaicking using a Cascade of Convolutional Residual Denoising Networks](https://arxiv.org/pdf/1803.05215.pdf)
    - notes
        - majorization maximiztion network
        - claimed to be more interpretable than deep learning approaches, and generalize well on small dataset
        - deep residual denoiser
    - remark
        - a previous paper of iterative resnet paper




- [Iterative Residual Network for Deep Joint Image Demosaicking and Denoising](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8668795)
    - notes
        - iterative neural networks for efficient optimization (majorization minimization)
        - use resdnet to learn the regularizing term
        - the algorithm is equivalent to Inexact Proximal Gradient Descent
        - used more realistic affine noise model instead of gaussian noise model - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.1943&rep=rep1&type=pdf
    - remark
        - compared to RED, P3, this method does not require setting of hyperparameters
        - resdnet pretrained on berkeley dataset (500 color images)
        - microsoft dataset, 500 natural images with panasonic dmc-lx3 ccd with bayer cfa - 200 training images




- [Deep Joint Design of Color Filter Arrays and Demosaicing](https://inf.ufrgs.br/~bhenz/projects/joint_cfa_demosaicing/deep_joint_design_of_color_filter_arrays_and_demosaicing_pre-print.pdf)
    - notes
        - uses autoencoder to encode input to `multispectral mosaic`, which is then decoded to full resolution image
        - finds mosaic pattern and demosaicing algorithm that minimizes color-reconstruction error
        - achieved state of arts result, while allowing for cfa mosaic design
    - remark
        - has some good background references on cfa design, maybe applicable in our case
        - a followup https://evanfletcher42.com/2018/09/23/arbitrary-learned-mosaic/




- [DeepISP:Towards Learning an End-to-End Image Processing Pipeline](https://arxiv.org/pdf/1801.06724.pdf)
    - notes
        - learns mapping from raw low-light mosaiced image to output image - low-level tasks (demosaicing+denoising) - high-level (color correction, image adjustment)
    - remark
        - masking patterns are differentiable weights inside encoder
        - training dataset too large to be feasible, 5 days of training time




- [Handheld Multi-Frame Super-Resolution](https://arxiv.org/pdf/1905.03277.pdf)
    - notes
        - the method creates a complete RGB without demosaicing from a burst of CFA raw images
        - the method uses natural hand tremor to acquire images with small offsets
        - the frames are aligned and merged to form a single image
        - solve demosaicing + superresolution as an image reconstruction problem
        - does not rely on cross-channel correlation
    - remark
        - hand tremor not feasible for c2b camera



