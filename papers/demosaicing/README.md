
### Reports 

+ [part7.3_demosaicing](part7.3_demosaicing.pdf)
    + outlines some basic demosaicing methods


### Interpolation-based demosaicing

+ [2002_demosaicking_methods_for_bayer_color_arrays](2002_demosaicking_methods_for_bayer_color_arrays.pdf)
    + original method that 
        + bilinear interpolate G channel
        + constant-hue assumption: interpolate R,B to maintain constant chrominance
    + intro
        + green is representative of luminance (response curve peaks ~550nm)
        + linear interpolation fails at edges
        + performance metrics
            + MSE in RBB color space
            + delta E error in CIELAB color space
        + imaging model 
            + locally, red/blue perfectly correlated with green over a small neighborhood and differ from green by only an offset
                + `G_ij = R_ij + k`
        + constant hue assumption
            + hue: the he degree to which a stimulus can be described as similar to or different from stimuli that are described as red, green, blue, and yellow
            + chrominance: red, blue
            + lumimnance: green
            + hue
                + `R/G, B/G`
                + allowed to change gradually (reduce color fringe)
            + if constant hue
                + `R_ij / R_kl = G_ij / G_kl` 
                    + ratio `log(A/B)=logA-logB`, since exposure is logarithmic
                + known R_ij, G_ij as measured values, G_lk is interpolated
                    + the missing chrominance `R_kl = G_kl (R_ij / G_ij)`
        + idea
            + interpolate G using bilinear interpolation
            + compute hue in R,B (`R<-R/G, B<-B/G`)
            + interpolate hue in R,B with bilinear interpolation
            + determine chrominance (B,R) from hue (`R = hue[0]*G`)



+ [2004_high_quality_linear_interpolation_for_demosaicing_of_bayer_patterned_color_images](2004_high_quality_linear_interpolation_for_demosaicing_of_bayer_patterned_color_images.pdf)
    + a summary: [2011_malvar_he_cutler_lienar_image_demosaicking](2011_malvar_he_cutler_lienar_image_demosaicking.pdf)
        + method used by opencv `demosaic`: https://docs.opencv.org/3.4/db/d8c/group__cudaimgproc__color.html
    + a new 5x5 demosacing linear filter 


+ [2006_color_demosaicing_using_variance_of_color_differences](2006_color_demosaicing_using_variance_of_color_differences.pdf)
    + abstract
        + missing green first estimated based on variances of color dfferences alone edge directions, missing red/blue estimated based on interpolated green plane
        + goodness
            + preserves textures details, reduce color artifacts
    + previously 
        + (ACPI) adaptive color plane interpolation
            + a base method
            + extensions
                + (PCSD) same interpolation direction for each color component of a pixel reduces color artifacts 
                + (AHDDA) local homogeneity as indicator to pick direction for interpolation
        + ACPI
            + steps
                + interpolate green along direction of smallest gradient


+ [2014_rethinking_color_cameras](2014_rethinking_color_cameras.pdf)
    + traditionally
        + camera sub-sample measurements at alternating pixel locations and then demosaick measurements to create full color image by up-sampling
        + but blocks majority of incident light
            + each color filter blocks roughly 2/3 of incident light, color camera 3x slower than grayscale counterpart
        + prone to artifact during reconstruction
    + idea
        + new computational approach for co-designed sampling pattern and reconstruction leading to reduction in noise and aliasing artifact
        + pattern panchromatic
            + no filter for majority of pixels -> measures luminance without subsampling
            + avoid loss of light, and aliasing and high spatial-frequency variations
        + color sampled sparsely
            + 2x2 bayer block sparsely placed (reconsturction prevents loss of color-fidelity )
            + then propagated via guidance from un-aliased luminance channel
        + reconstruction
            + infer missing luminance samples by hole-filling
            + chromaticity propagated by colorization with a spatio-spectral image model
        + faster and sharper color camera
    + why it works
        + redundancies in spatio-spectral content of natural images
        + boundaries are sparse, exists contiguous regions where variations in colors primarily due to shading
            + luminance and chrominance vary
            + chromaticities, relative ratios between different channels, stays constant
        + at boundaries, both luminance and chromaticities change abruptly
    + steps
        + Recovering luminance (at bayers blocks)
            + wavelet transform, minimization problem
        + estimating chromaticities at bayer blocks
            + chromaticity (for R,G,B)
                + ratio between a color channel value and the luminance value at the same pixel
                + `c =  m / l`
                    + `c` chromaticity
                    + `m` measured value of r,g,b color channel
                    + `l` luminance
            + assumption
                + bayer block does not span material boundary
                    + so chromaticity assumed to be same for all 4 pixels in a block
                + colro and luminance related 
                    + luminance is a weighted sum of color r,g,b channels
             +formulation
                + least squared minimization chromaticity of r,g,b separately
                    + minimizing `lc - m`, and
                    + regularized by a noise variance, 
                        + biasing chromaticity at dark pixels toward gray
            + remove outliers, using median filtering, on estimates on blocks that occur at     boundaries or in dark regions
        + propagating chromaticities
            + partition
                + KxK patches s.t. 4 corners of each patch includes a bayer block. 
                + chromaticity inin each patch as a linear combination of chromaticity at the 4 corners, where the weights based on luminance
            + material affinity `alpha`
                + encode affinity between a bayer block and pixels around it
                + for each bayer block `j` and sites `n` around it, compute affinity
                    + `alpha_j[n]` within overlapping `(2K+1) x (2K+1)` regions around the block
                + optimization
                    + minimize a 
                    + constrained with `alpha=1` at center and `alpha=0` at other 8 bayer blocks
            + combination weights
                + `weight_j[n] \propsto alpha^2 * l`
                    + `j \in {1,2,3,4}`, i.e. the 4 corners
                    + gives initial estimates
                + the apply non-local kernel to refine chromaticity estimates
        + experiments
            + full-color image, then added gaussian noise
            + color-sampling frequency K, trade-off between light-efficiency and sharpness.
        + performance
            + the good
                + better reconstruction even in not-noisy scenarios
                + a lot better in images with lots of noise
            + the bad
                + not able to reconstruct hue with very fine image structures
                    + changes in chromaticity happen at a rate faster than color sampling frequency K
                + but still able to recover texture info
    + questions
        + recovering luminance
            + confused with in-painting with wavelet-based approaches
            + use wavelet decomposition and l1 minimization to infer luminance values
            + not entirely sure of the inner workings
        + propagating chromaticities
            + confused about formulation of c as well, 
            + what does e or c represent.
        + why so many convolutions...
        + does not really relate to the project?
            + more similar to spatially-varying exposure (SVE) images. 
            + no color captured in sparse locations?
            + so what is the take-away idea from this paper?


## Data-driven learning based


+ [2003_enhancing_resolution_along_multiple_imaging_dimensions_using_assorted_pixels](2003_enhancing_resolution_along_multiple_imaging_dimensions_using_assorted_pixels.pdf)
    + goal
        + enhance resolution using local structure models (polynomial funtion of intensities) learnt offline from images
    + multisampling 
        + a framework for using pixels to sample simultaneously multiple dimension of an image (space, time, spectrum - color, brightness - dynamic range, and polarization)
    + learn structured model
        + https://en.wikipedia.org/wiki/Nyquist_frequency
            + interpolation methods enhances spatial resolution but introduce errors in blurring and aliasing
        + different dimension of imaging highly correlated with reach other (due to reflectance and illumination of scene)
        + local structured model
            + learn a local mapping function from data, 
                + inputs: low-res multisampled images
                + labels: correct high-resolution images
            + model is a polynomial function of brightness measured within a local neighborhood
                + learn local mapping `f`
                + `H(i, j) = f(M(X, Y))` where
                    + `H` is high quality value at pixel i, j
                    + `M` is measured low-res value at pixel `x,y` in a neighbhorhood `X,Y` around `i,j`
            + use a single structured model for each type of local sampling pattern
        + previous methods
            + Markov model, bayesian, kernel
        + novel
            + resolution enhancement over multiple dimensions
    + SVC and models
        + SVC (spatially varying color)
            + bayes color mosaic
                + mosaic of R, G, B in digital color sensors
                + have 4 different sampling pattern for a neighborhood of 3x3
        + goal  
            + compute value of (R,G,B) at each pixel
        + model 
            + `M_p(\lambda) = A_p C_p(\lambda)`
                + `p` represent pattern, `\lambda` represent color
                + `A_P` are measurements of neighborhood having pattern `P`
                    + where each row has all relevant powers of measured data `M_P` within a neighborhood
                + `C_P(\lambda)` coefficients of polynomial mapping function
            + optimize with least squared
                + `C = (A^TA)^{-1} A^T H`
        + performance
            + visual and luminance error
            + stability of learnt model
                + good on training data, or test data similar to training data
                + if trainnig data is large, then random test data has larger error but is stable
    + SVEC (spatially varying exposure and color)
        + simutaneous sampling of space, color, and exposure
            + given 8 bit single color -> construct 12 bit for each 3 color values at each pixel
        + pattern
            + base pattern
        + still the same model,
    + questions 
        + SVC model
            + why is the polynomial model for SVC accounts for correlation between different color channels, seems the coefficients for the model is computed independently
            + to get high quality value from low-res value in the neighborhood, why the model multiplies measurement for each pixel and every other pixel
                + to model pairwise correlation between pixels?
            + compute number of coefficients, why + P not * P
        + is the project currently using bicubic interpolation?

+ [2013_joint_demosacing_and_denoising_via_learned_nonparametric_random_fields](2013_joint_demosacing_and_denoising_via_learned_nonparametric_random_fields.pdf)
    + 

+ [2016_klatzer_learning_joint_demosaicing_and_denoising_based_on_sequential_energy_minimization](2016_klatzer_learning_joint_demosaicing_and_denoising_based_on_sequential_energy_minimization.pdf)
    + abstract
        + demosacing+denoising as image restoration problem
        + method learngs efficient regularization by a variational energy minimization 
    + introduction
        + challenges
            + interpolating edges/corners
            + error propagation with interpolating channels separately/sequentially
            + noise maybe non-Gaussian, maybe a complex distribution
        + dataset
            + images already processed, 
            + [Khashbi] introduces the Microsoft Demosaicing dataset
    + related work
        + demosaicing 
            + interpolation 
                + heuristic-based, 
                + some jointly demosaic+denoise but only in Gaussian settings
            + learning based
                + [Khashbi] regression tree fields
            + inverse problem 
                + priors for regularization: TV, color difference, hue smoothness, denoiser (BM3D)
                + hand-crafted demosaicing not able to capture image statistics 
        + this method
            + learning+reconstrction based
            + demosaicing in linRBG space
            + adaptable to different CFA pattern and camera types
            + jointly denoise and demosaic under non-Gaussian noise
            + regularization does not rely on handcrafted correlation, but learns 
    + method
        + upper level 
            + L2 loss between ground truth `g` and a sequence of reconstructed images `u^s`
        + lower level quadratic energy `Q`
            + gradient descent given `grad f`
            + backpropagation
        + energy function `f`
            + `f(u) = R(u) + D(u)`
                + `R` fields of experts prior
                    + nonlinearity with radial basis function (to learn image priors from data)
                + `D` data fidelity 
                    + nonlinearity with radial basis function (to learn non-Gaussian noise)
            + can compute `grad(f)` 
        + noise model 
            + mixture of poisson+gaussian (Microsoft dataset)
        + optimization 
            + lbfgs-b
    + experiments & results
        + mean of psnr for 200 images
        + slightly better (1dB) over FlexISP
            

+ [2016_deep_joint_demosaicking_and_denoising](2016_deep_joint_demosaicking_and_denoising.pdf)
    + slides (https://groups.csail.mit.edu/graphics/demosaicnet/data/demosaicnet_slides.pdf)
    + abstract
        + data-driven (instead of hand-crafted priors) approach to demosaicing and denoising using deep neural nets
        + goal 
            + reduce computation speed
            + reduce artifacts
        + dataset
            + millions of sRGB images, mosaicked, and added noise
            + metrics to identify difficult patches
            + inject noise
    + introduction 
        + compare to flexISP (2014) 
            + non-local natural image priors is still handcrafted, and incoporation of optimization and non-local priors leads to increase in computation cost
        + contribution
            + `demosaicnet` uses data-driven local filtering approach for efficiency
            + capable of handling wide range of noise
            + method of building training set rich in challenging artifacts (moire, etc)
            + SOTA results
            + runs faster
    + related work 
        + demosaicing
            + filters, smooth hue priors
            + replace hand-crafted filters with deep learning
        + self-similarity
            + some methods uses neural nets
            + but trained on small datasets
            + this work builds a larger dataset
        + joint denoising and demosaicking
            + Heide [2014] use a global primal-dual optimization with self-similarity prior but is too slow
            + Khashabi [2014] (joint via learned nonparametric random fields) learning approach
            + Klatzer [2016] (learning joint demosaicing and denoising baesd on sequential energy minimization) sequential energy minimization
            + this work expose noise level as a parameter
    + network
        + convert Bayer input to quarter-resolution multi-channel image
            + each 2x2 patch -> 4 channel feature 
            + makes spatial pattern invariant with a period of 1 pixel
        + D conv layer, with ReLU nonlinearity
            + first D-1
                + W^2 weights per layer
            + last filter 
                + 12W weights
                + last output has 12 channels
        + upsample 12 channel -> 3 channel full res image, then concatenate with masked input, totals to 6 channels
        + 1 last convolution at full resolution
        + hyperparameter
    + joint denoising with multiple noise levels
        + motivation
            + want 1 network for different noise levels
        + idea
            + train a single network on a range of noise levels and explicitly add noise level as an input parameter to the network
        + training
            + for each input image, sample a noise level in a range `sigma \in [a,b]`,
            + corrupt image with a centered additive Gaussian noise with `sigma^2` before feeding into the network
            + also feed noise level `sigma` as the 5th channel
    + training 
        + D = 15
        + W = 64
        + 3x3 filters
        + optimize normalized L2 loss
        + 64 batch size 
        + learning rate 10^-4
        + weight decay 10^-8
        + ~600,000 trainable parameters
        + training on Titan X, taking 2-3 weeks...
    + training data
        + problem
            + hard cases are rare and diluted by vastly more common easy areas. 
            + L2 norm fail to notice demosaicking artifacts
        + solution 
            + detect challenging cases, and tune network to learn the hard examples
            + just use the network trained on standard network and look for classes of images that the network prone to produce artifacts,
                + luminance around thin structures
                    + detected via HDR-VDP2 ... PSNR does not capture this artifact properly
                + color moire
                    + transform input/output to Lab space and compute 2D fourier transform
                    + quantify gain in frequencies
            + then fine-tune or train network from scratch to improve performance on difficult cases
                + i.e. training done on problematic patches (with high HDR-VDP2), not the entire image
    + results
        + metric
            + PSNR, error averaged over pixels and color channels before taking the average
        + test set
            + the HDR-VDP mined dataset and the color moire dataset
        + demosaicking noise-free images
            + better results ...
        +joint denoising and demosaicking on noisy images
            + better results ...
        + also non-Bayer mosaicks
            + generalizes
    + novelty
        + model tailered to zippering and moire artifact
        + metrics for finding hard dataset and fine-tune network on those
    + questions
        + https://github.com/mgharbi/demosaicnet
            + code available
            + took a look, seems pretty straightforward
            + although training would take days and weeks, and input dataset is not as abundant. 
            + so might not be the most feasible ....
        + relating to the project


+ [2018_deep_image_demosaicking_using_a_cascade_of_convolutional_residual_denoising_networks](2018_deep_image_demosaicking_using_a_cascade_of_convolutional_residual_denoising_networks.pdf)
    + basically 2016 paper but extended with majority maximization ...

+ [2018_deep_residual_network_for_joint_demosaicing_and_superresolution](2018_deep_residual_network_for_joint_demosaicing_and_superresolution.pdf)


## Inverse problem

+ [2014_flexISP_a_flexible_camera_image_processing_framework](2014_flexISP_a_flexible_camera_image_processing_framework.pdf)
    + abstract
        + end-to-end image processing that
            + enforce image priors as proximal operators
            + ADMM/primaldual optimization
        + idea: end-to-end reduces error introduecd in each step of image processing
            + each stage is ill-posed
            + individual stage not independent
        + applied to demosaicking, denoising, deconvolution, etc.
    + related work
        + denoising
            + priors: self-similarity and sparsity
            + BM3D and TV (total variations)
        + joint optimization 
            + addressing subproblems does not yield best-quality reconstructions
            + demosaiking+denoising
            + demosaiking+denoising
            + ...
            + ADMM applied to image processing literature
    + optimization 
        + inverse problem
            + `z = Ax + \eta`
            + `min_x 1/2 ||z-Ax||^2 + R(x)`
            + converted to standard constrained optimization 
        + solver (primal dual faster than ADMM)
        + regularization 
            + image gradient sparsity (TV)
            + denoising (NLM,BM3D)
            + cross channel gradient correlation for edge consistency
        + combination of regularizers
            + weighted sum
    + design choices
        + optimization 
            + priors as proximal operators
        + prior choices
            + external priors
                + tested TV, curvelet, EPLL
                + TV is most cost-effective
            + internal priors (Denoiser)
                + tested BM3D, NLM, DCT
                + BM3D best
    + applications & results
        + demosaiking
            + 2.2 dB over best previous method
        + deblurring
        + interlased HDR
        + color array camera
        + burst denoising and demosaicking
        + process JPEG compressed image
    + discussion 
        + priors
            + choice of prior importants
        + failure cases
            + inputs with no self-similarity (i.e. random noie)
            + misconfiguring solver parameter
    + observation 
        + regularizer weights influence reconstruction
        + did not prove convexity / guaranteed convergence

+ [2017_joint_demosaicing_and_denoising_of_nioisy_bayer_images_with_ADMM](2017_joint_demosaicing_and_denoising_of_nioisy_bayer_images_with_ADMM.pdf)
    + codebase: https://github.com/TomHeaven/Joint-Demosaic-and-Denoising-with-ADMM
    + abstract
        + unified objective function with hidden priors and optimize with ADMM for recovering color image from noisy Bayer input
        + perform better in PSNR and human vision
        + more robust to variations of noise levels
    + intro 
        + recent research in jointly demosaic and denoise (good place to see what is SOTA)
    + problem formulation
        + image formation 
            + `b = Ax + \eta`
                + `b` bayer iamge
                + `A` downsampling operator
                + `x \in \R^{3n}`
                + `\eta` noise vector
        + image recovery 
            + inverse problem 
            + `\min || Ax-b || + T(x)`
                + where `T` are prior functions
        + priors
            + flat areas: smoothness prior
            + edge: edge-preserving denoising prior
            + bayer mosaic pattern: structural information
            + total variation 
            + (CBM3D) attenuation of additive white Gaussian noise from 
            + cross-channel prior (basically 2004 paper that MATLAB `demosaic` uses)
        + handwavy about parts of objective function as _hidden function_, i.e. did not establish convexity of BM3D/demosaocing prior (ability to do follow up work)
    + optimization with ADMM
        + standard algo i think
    + experiments
        + Kodak + McM
        + inputs
            + downsample -> Gaussian white noise
            + noise level is known
        + methods compared 
            + FlexISP
            + deep joint 2016
        + results
            + better result for noisy inputs
            + slow (900s vs 4s for deepjoint)
    + follow up
        + faster algo
        + convergence of ADMM not determined


## Reviews

+ [2005_demosaicking_color_filter_array_interpolation](2005_demosaicking_color_filter_array_interpolation.pdf)
    + a survey of heuristic-based, reconstruction-based, image model based interpolation techniques for bayer color filter array (CFA)
    + metrics
        + MSE
        + S-CIELAB, measures color reconstruction accuracy in a uniform color space
        + subjective assessment of aliasing

+ [2008_image_demosaicing_a_systematic_survey.pdf](2008_image_demosaicing_a_systematic_survey.pdf)
    + points
        + have problem understanding signal processing stuff.
        + dont really understand frequency domain formulation, i.e. figure 3
    + relating to the project
        + sequential demosaicing methods: interpolate G channel first then R,B channels. Reduced aliasing because G is sampled more frequently. 
            + not applicable to our project, since there is no channel that is sampled more. So cannot exploit this particular aspect.
        + 12 test images only, might be too small...

+ [2011_color_image_demosaicking_an_overview](2011_color_image_demosaicking_an_overview.pdf)
    + picked some good algorithms

+ [2018_colour_filter_array_demosaicking_a_brief_survey](https://www.tandfonline.com/doi/full/10.1080/13682199.2018.1534388)

