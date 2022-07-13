
# Meeting Notes



## How to use the camera system

+ bitfile 

+ lightcrafter gui
    + display mode: stored pattern sequence
    + connection
        + first connect usb to projector
        + check ip address in network settings
        + fill in ip to connection tab
    + stored pattern sequence
        + bit depth: 1
        + pattern count: # images
        + pattern type: normal 
        + input trigger: external (positive)
        + LED select: Green
        + exposure: 2000

+ imagegui
    + bit file: fixedFPN
    + pattern file: 
        + the code tensor
        + for groundtruth stuff: the 5black
    + exposure: 60
    + masks: 1
    + trigger num: 12



+ camera setup 
    + physical size of the field
    + object to image distance
+ textured objects
    + 


## 2018.11.09

+ what was tried before
    + linear interpolation from matlab `demosaic`
        + 1/N resolution 
        + interpolate linearly
+ approaches
    + cast existing demosaicing algorithm on tiles we have right now
        + adapt the best demosaicing algorithm and adapt to our problem
    + learn-based algorithm
        + given a large collection of scenes, try to learn the codes!
        + existing learning based algorithm
            + train from scratch
+ todo
    + literature search on state-of-the-arts algorithm
        + take note of how micro-phasing affect how the model is set up
        + shoud not expect the same kind of correlation from existing spatially varying algorithms
    + look for work that cite papers the _joint denoising_ or _assorted pixels_
        + finding data...
            + rendering in game engine?
            + synthesizing dataset without looking at 3D geometry ...
                + feasible, pipeline...
        + or take 2D image and multiply with pattern, as training data
            + but would not have depth information
    + identify top performing algorithms
        + non-learning based
            + probably have to implement them ourselves
        + re-train using data
    + paper reading
        + take look at both, performance on metrics
        + take look at insights on how they exploited correlation in natural images
+ endgoal
    + general demosaicing for micro-phasing, stereo-imaging
    + (mian) just determine the layout of mosaics/patterns, then use general purpose algorithms on them
+ tricky part 
    + at what level / how to model correlation for stereo-imaging


## 2019.05.07

+ demosaicing 
    + _assorted pixels_: 
        + exploits: {intra,inter}-channel redundancies
        + model: lower order polynomial of nearby pixel/channel intensities
        + parameter estimation: least squared with pseudo-inverse
    + _2005,2008 review_
        + _non-adaptive methods_ various kinds of interpolations
        + _heuristic-based methods_
            + edge-directed, avoid interpolation across edges 
        + _reconstruction-based methods_
            + assumptions about interchannel correlation or priors, and solve a minimization problem
            + RGB imaging: minimize a function that captures 1) spatial smoothness 2) color correlation (constant hue assumption)
                + sequential demosaicing: as G sampled 2x than R,B, interpolate G first, then compute R,B with constant hue assumption
        + _learning-based methods_
+ constant hue assumption 
        + is valid for multispectral imaging, since all C2B camera doing is spatially sampling view of the scene under different illumination conditions. For the setting of {R,G,B} imaging, methods like assorted pixels, etc. (that exploit the constant hue assumption) definitely an improvement over the existing interpolation schemes. 
        + for downstream reconstruction tasks like, structured light, photometric stereo, etc. would not expect methods that exploits this assumption to work. Inter-channel correlation is low locally, so methods like bicubic interpolation (that works independently on each channel) might be best it can be. However, for structured light, illumination patterns are shifted spatially, so color intensities might correlate cross-channel but at some spatial offset. 
+ feasibility
    + datasets (some demosaicing methods are data-driven, i.e. assorted pixels)
        + full-res ground truth image, downsampled to fit the resolution of spatial sampling, train a model to minimize some reconstruction error
        + this is indeed possible for C2B camera
            + take S number of photos, fixed illumination for 1 frame, light all go to bucket1 ... 
            + each of S frames is ground truth for that channel
        + size:
            + simpler models: number of images relatively small
            + deep nets: mined from Flickr, not really applicable in this case (simulated images? )
+ todo
    + might be good to generate some datasets ...



## 2019.05.09 (slack)

In summary, there are several categories of methods

__non-adaptive methods__, which includes various kinds of interpolation, mostly working on a single channel at once and so would not be able to exploit inter-channel relationship.

__heuristic-based methods__. One heuristic would be to avoid interpolation cross edges; This might not work well with structured light illumination, the gradient estimates most likely perpendicular to the spatial sinusoidals, but this probably works fine for multispectral imaging. Another heuristic would be to interpolate channels that is sampled more frequently (Green in Bayer tile) to achieve better SNR. This is not readily adaptable to C2B camera, although in some cases this heuristic is applicable (i.e. RGGB tilling the ECCV paper figures). This could work though, if we add a constraint that 1 type of code tensor is shared by say 50% of the pixels. A heuristic that often use in conjunction with the previous heuristic exploits the constant hue assumption: ratio/difference between channels are approximately constant within boundary of objects. The method would use the full-res green channel to determine values for other channels. I still need to check if this applies to IR images. But again, if the aim is a general framework that works for any type of subsequent reconstruction, then this heuristic would not be applicable for structured light.

__learning-based methods__
the methods are data-driven. Older methods like assorted pixels to construct an over-determined system of linear equations relating polynomial coefficients with pixel intensities. The model is optimized with least squared via pseudo-inverse. The model is trained over ~20 images (a lot more patches within ~20 images) and could capture intra- and inter-channel dependencies. By the virtual of its small size, the dataset could be easily created by taking `S` frames of photos, each frame has a fixed illumination, where each pixel in each `s=1,...,S` frames collects light on bucket-1. This would result in `S` full-res ground-truth images for training. More recent methods uses deep networks that requires   larger datasets (millions of images) and perhaps not feasible for C2B camera. Or it might work with _transfer learning_ , not sure ? But there is lot's of *joint-* methods that combines demosaicing with superresolution and denoising. `demosaicnet` achieved state of arts performance, that surpasses traditional methods by sizable margin (see benchmark from 2016 paper's slides).

I think from last meeting with Kyros and you, it seems that a more general framework would be more desirable. Even if the performance might not be better than those _deep_ models, Ideally, the goal would be a general framework for demosaicing/upsampling for the C2B camera, where we could inject different priors for different downstream reconstruction tasks.

## 2019.05.11


+ project
    + demosaicing ... 
        + shape prior instead of color prio?
        + does it perform better than normal color demosaicing?
        + Random code matrix or tiled (tiled for now because using interpolation methods for demosaicing)?
        + demosaicing might taylor to downstream 3d reconstruction ...


## 2019.05.28


+ so far
    + background learning on 
        + convex optimization (ongoing)
    + implemented a few (interpolation) method in python and did some comparison 
        + bilinear filtering
        + smooth hue
        + median hue
        + laplacian corrected
    + experiments with additive white noise, performance drops with higher noise level
    + literature search on 
        + demosaicing method
            + read more carelly some early papers on interpolation based methods
            + did not read too carefully on some of the more sophisticated methods
            + deep learning methods outperforms classical methods by 5dB (PSNR) 
            + there is one paper that uses ADMM for optimization, but did not prove convexity of some prior terms
        + regularization as denoising (redner 2017)
            + denoising residual as regularizer, optimization with ADMM
            + energy minimization, seems pretty easy to generalize
            + proofs on convexity of objective (denoiser has some constraint) and guarantees global convergence
            + seems pretty good performance, for a variety of tasks
        + image denoising
            + classical method (not implemented yet)
                + total variation
                + bilateral filter
                + nonlocal mean filter
            + methods that uses laplacian as a regularizer for denoising
+ plans
    + create a benchmark dataset to evaluate the performance of image processing methods for c2b camera
        + each frame captures 
    + try Redner/PnP on demosaicing/denoising tasks
        + since they didn't use demosaicing as a reconstruction tasks, so need to verify if this is indeed a performant method
            + a ee367 course project indicates ADMM+TV performed worse than bilinear/matlab `demosaic`
            + but the results seem a bit off, so need to reproduce this as soon as possible
+ questions
    + what is goal of different reconstruction tasks
        + rgb: look as natural to human as possible ?
        + multispectral imaging 
        + structured light
    + cfa design, i.e. tile as a free variable in optimization 
        + maybe looking into it



+ todo
    + a complete picture for state of art for the problem
        + primarily demosaicing and maybe denoising
        + implementation easy/hard
            + data sufficiency, flexibility
            + a blog for what i have done
        + pro/con of methods ...
        + state of arts 
        + cvpr 2018, see paper that cited that...
        + flexisp (read)
    + dataset generation for benchmarking purposes
        + structured light: shapenet dataset, project light with mitsuba (ask parsa/wenjian)
        + multispectral imaging: not for now ... 
    + start using the cameras
        + camera noise characterization, optimize for specific camera
    + question
        + specific priors for different reconstuction 
            + maybe denoising prior is enough 
        + arbitrary tiling (cfa design)

+ read papers 
    + michael brown
        + noise characterization of cellphone
    + http://people.csail.mit.edu/billf/publications/Noise-Optimal_Capture.pdf
        + explanation for noise model
    + 


## 2019.06.05

+ create a good dataset
    + take dataset wenjian and parsa created from simulation
        + shapenet geometry, project structured light, simulation with mitsuba
        + realistic 
            + noise model (michael brown) of actual sensor
            + lens/projector defocus
        + some problem with mitsuba
            + if we need to do the rendering ourselves
        + randomly textured plane
    + camera dataset
        + the image should be demultiplexed version in the image formation pipeline
        + average multiple frames, for each pattern
            + lots of pattern (ask wenjian)
                + note patterns are specific to number of patterns provided 
            + long exposure
            + good reconstruction ...
        + setup T2, get optimized mask from wenjian
        + good to capture videos (not for testing dataset)
+ think about video (>1 frame) superresolution, by changing patterns (masks) in each frame 
+ think more about structured light upsampling thinking about albedo and projector pattern, etc. 
+ think about computing ratio
+ 2 pixel neighborhood,
    + 2 frame -> full res image (minimal reconstruction), is reconstruction better than 4 pixel neighborhood?
    + what is optimal # of patterns given a exposure time ?
+ think about integrating demultiplexing into the framework
    + demultiplexing fits quite well into admm framework
        + `y = DWx + e`
        + where `D` is demosaicing matrix, `W` is demultiplexing matrix
        + but note `e` should be applied to the intermediate images ...
+ end goal
    + different paramters that migth give better reconstruction 
    + reconstruction results
    + low-level and high-level goals are trade-offs
        + wenjian / parsa
            + pattern, number of pattern, optimal algos to recover depth 
        + me
            + reconstruct full-res less noisy image



#### 06.26

+ experiments answering _how well RED works_
    + see arbitrary masks performance (T2 max 7 patterns per frame)
    + metrics 
        + convergence time 
        + psnr
    + initial guess [groundtruth image, demosaiced, zero, zero for unknown, random]
        + see convergence
        + psnr
        + convergence speed
    + masks
        + bayer
        + 3-4 random 
        + other masks
    + noise
        + 25,30,35,40
+ learnt prior
    + for different reconstruction ...
+ arange tiling/mosaicing


+ another question: jointly optimize for projector pattern and sensor mask
    + not that much of a priority now, as long as arbitrary mask works for the method
+ rotate the camera setup, to verify that `horz3` is better than `vert3` for structured light where projector pattern are horizontal lines
+ talk to john with hardware+software


#### 06.28

+ how to choose masks
    + given a set of groundtruth images, idea is to choose mask `M` which minimizes the MSE of reconstructed image to the groundtruth image
        + motivated by the fact that the choice of mask should be dependent on applications (i.e. structured light patterns)
        + not sure if this is a good idea, since this requires, for every different pattern, acquire test dataset, and then optimize for `M`
    + formulated as a bilevel optimization mixed-integer problem
        + `M` is a large vector indicating 1s' or 0s', or takes values from set {1,2,...,F}
        + relaxed problem `M` is floating, but rounded to integers 
        + bilevel optimization methods
            + replace lower level problem with the corresponding KKT condition, not feasible because
                + lower level problem is not convex
                + even if lower level problem is convex, constraint quantification is not satisfied, need some nontrivial derivation of a good constraint quantification
            + use gradient based methods
                + seems promising, since even if the lower level problem is nonconvex, the gradient is shown in the RED paper (assuming mild assumptions to the denoiser)
                + might need to do some math, and more readings on the topic
    + formulated as a combinatorial optimization problem
        + assumptions
            + (tesselation) periodic tiling of repeated patterns (generalizable to different `F`)
            + tiling algorithms might tile local patterns in different ways to fill the rectangle
        + idea is to try different combinations (#combinations small, so time-wise feasible)
    + formulated as a problem with some prior assumptions
        + say minimize distance (at each pixel) to pixels of different color (or frame#) for all pixels
        + may be a good enough heuristic to a good mask `M`

+ feedback from mian
    + should nail down the reconstruction method first
    + focus on learnt priors first
    + might just be one model that jointly optimize for 
        + reconstruction of image
        + masks




#### 2019.07.05

+ prepare
    + show briefly
        + S=4, show
            + reconstructed video sequence (compare to whats been done previously)
        + S=7, show
            + reconstructed image
            + convergence plot
    + questions
        + should we use RED as reconstruction ?
            + if it is good enough
        + how to do mask optimization ?
            + necessity ? 
                + convergence to fixed point
                + random seems to be good enough
            + if so, 
                + jointly determines reconstruction algo and mask pattern
                + or settle on fixed algo (RED) and do 
                    + data driven bilevel optimization
                    + heuristic (e.g. minimize distance per pix)
    + todos
        + learnt prior (hear back from GPU)

+ what is tradeoff between spatial resolution / # patterns ?
    + experiment with T4 ?
    + 5-8 optimized patterns
        + good result from 4 patterns from wenzheng
        + zncc ?
    + ...
        + 4 patterns in 1 shot (fixed total exposure time)
        + 8 patterns in same time/subframe (then use 1-128)
+ evaluate performance 
    + demosaic/red on noisy image ...
        + challenging scene, lots of edges, 
            + texture discontinuity
            + depth discontinuity
            + neighborhood size is important for this cases
            + the ball that grow small&large
            + resolution charts ... planar
                + print resolution charts and drew holes with laser cutter?
+ ratio images ?
    + albedo invariant ?
    + red with ratio images?
+ do denoising in another domain
    + over albedo/correspondance map ?
    + disparity/albedo -> infer pixel
+ denoising
    + in image domain, ignore correlation
    + in albedo/disparity where all correspondence are captured
    + do them in both ...
        + if there is correspondence ...
        + proximal methods 
        + auxillary variable that is also optimized
    + optimization variables are 
        + denoised image
        + albedo
        + disparity
    + already have an estimate of depth/albedo from currently method, where denoiser is applied independently to eash channel ...
+ sensor mask
    + tiling arrangement ...
+ what could be complished in the amount of time here
    + talk to mian about it ...
    + 2 things
        + more exploration of optimization method
        + gpu version of method

+ next meeting
    + 4,5PM eastern time. 
    + email ...



+ TODO
    + check performance of red in ratio space and intensity space
        + motivation ratio space does not have textures
        + define bucket 1 ratio as 
            + `r_1 = I_1 / (I_1+I_2)`
        + do reconstruction on ratio images, get `S` ratio images back 
        + compare with reconstruction with red in intensity space where the output image is converted to ratio images in the end
    + (intensity space) check performance of RED on images with realistic noise characteristics and compare to RED on images with artificially added noise
        + groundtruth noiseless images, stacked with noisy images (light goto bucket 1)
        + run RED/demosaic on input that is multiplexed and spatially subsampled (according to a chosen mask) from `S` noisy input
        + compare PSNR/SSIM
    


## 07.24

+ summary 
    + red performs a lot better in ratio space, maintains spatial resolution
    + using images captured with c2b camera, ~1dB increase in performance than previous method
    + spatial spectral trade-off
        + 41 -> 40 from S=4 -> S=7 on resolution chart image
+ what to do next?
    + retraining denoiser on c2b images
        + probably not that necessary, since almost no visible noise
    + end-to-end training of an optimization based network optimizing for mse or psnr
        + probably increase in performance, but 
        + might not give more insights to the problem
    + investigate more on how to regulate depth/disparity?
        + depth/disparity a nonlinear function of ratio images, do regularization on disparity
        + is there a nonlinear mapping from ratio to disparity, analytical?
        + can we regularize on ratio images instead? 
            + not sure how to start on this


+ todo  
    + noise characteristic of different sensor masks
        + alphabet or more objects
        + sensor masks to use (`512x(176*4=704)`)
            + all goto bucket 1
            + bayer
            + cant do because code wrong
                + toeplitz
                + rand
                + vert
                + horz 
        + projector pattern 
            + sinusoids
    + spatial spectral 
        + experiments
            + given a fixed total exposure in terms of clock cycles 420
            + divide by S to get subframe time 105/84/70/60
            + sensor mask: all_black
            + projector pattern: optimized, sinusoids
            + image with S=4,5,6,7
            + triggernumber=24,16,12
        + aquisition time should be same (constant exposure) for different subframes ..
            + ask Xia
            + `(int((205+exp)/trig_value)*trig_value + trig_value - 205)*n_subframes * 10ns` or 0.01ms
                + `exp` for 1 frame
                + `trig_value` number of frames to make a trigger
        + S increase, get noisier and projector quantization error...
        + use parsa/wenjian's projector patterns for ZNCC decoder
        + structured light solver also changed ... 
            + not solving linear system -> ZNCC decoder (3pixel decoder?)
            + column disparity
    + make sure performance carry over to reconstruction
        + end-to-end performance carry over to disparity reconstruction?
            + albedo, disparity, or both
            + albedo affects depth?
            + performance dominated at depth discontiuities
        + if ratio images are indeed carry over to depth estimation
    + training a denoiser on c2b images help?
        + temporal noise ... 
        + positional pecularity? fixed pattern noise
        + positional dependency?
        + even ratio images alter the images
    + precise noise model should not be pursued..
    + ratio/disparity
        + idea of using ratio as proxy for disparity is to maintain convexity
    + discuss next steps
    + check if ratio images are promising
    + write some report before leave ...


## 09.04

+ what has been done
    + spatial spectral trade-off
    + wrote first part of the report, still needs to fill in experiments
+ what needs to be done
    + images acquired with constant total exposure for sinusoidal/optimized pattern, check if depth corresponds to ground-truth. 
    + verify that the subframe exposure time is indeed the exposure time excluding mask upload time ..


## 10.01


+ think more about how multiplexing matrix is constructed 
    + why the resulting demultiplexed images are out of order
    + making assumptions?
+ do some experiment 
    + given S=7, we can take 4 images, and do corresponding reconstructions
    + only difference is in trying to do reconstruction, change the shifts from uniformly sample to sample according to what images are taken in the first place ...
    + compare different S for scene with varying depth
+ think about how chinese reminder theorem can be used to determine the amount of circular shift required for input images

## 10.17 (Mian)

+ regularization 
    + be cautious about adding arbitrary regularizers

+ todo 
    + do zncc on optimized code first
        + do zncc reconstruction on 
            + spatial sinusoids 
            + optimized pattern on `alphabet` scene
        + take a look at ratio images with zncc code 
    + compare ratio and intensity images for disparity estimation 
        + look at fourier transform of intensity/ratio
        + reduce frequency bandwidth in fourier domain -> easier to demosaic
    + why texture in ratio images
    + need to do capture more images with total exposure fixed


+ think about 
    + __read more on adrian dalca's papers on using VI/VAE etc. for data imputation__
        + perhaps there is an equivalent probabilistic formulation
        + mechanistically incorporates neural network in the setup
        + https://arxiv.org/pdf/1903.03503.pdf
    + what patterns to use ?
        + cosine vs. optimized, debrujn
    + arrange illuminations mosaics ?
        + toeplitz, etc.
    + but these are algorithm dependent
    + __reglarizer that takes into the pattern into account__
        + take information of pattern 
        + pattern-specific prior 
        + lookup table
            + ~600 proejctor pattens 
            + enforce prior that colinear with one of 600 pixels, at each pixel
                + every pixel, NCC with 600 projector pixel, larger the better
            + ratio images do this
                + each pixel scaled the same
            + for now, disregard interaction between pixels, think about each pixel independently


## 11.26 Mian & Parsa 

+ ask Parsa
    + where could ZNCC decoding go wrong? 
        + setup
            + optimized_patterns_pt01-608-4-64
                + 608 padded to 684 with zeros, since the projector Mian has more pixels (vertically)
    + solution 
        + try low frequency
        + get good groundtruth
            + project 100 cosines, compute ground truth
            + get pixel wise error
        + new zncc patterns,
            + 
    + old zncc
        + error = number of pixels that are more tha epsilon away from groundtruth
    + new zncc can define new loss functions 
        + error = sum of l1 norm pixel wise 
    + neighborhood
        + along the epipolar line
        + performance does not matter much
    + no improvement with increasing neighborhood
    + Hamiltonian pattern !
        + very good performance
        + ECCV 2018
    + a single frequency code 
        + heuristic for number of patterns, ...
        + 7 patterns
            + 5 pattern
            + 2 different frequencies, coprime with prev
    + neural network inspired
        + extract features from 5 neighbors, before going to zncc layer
        + adapt to different noise levels


## 2020.05.27

+ read more on how ratio images are important
    + read proofs in c2b paper
+ how to choose the subsampling mask 
+ what is trade-off between spatial resolution / #patterns
    + challenging casses
    + depth discontinuity
    + texture discontinuity
    + resolution
+ ratio image vs. intensity images
    + motivation: ratio images do not have texture
    + RED does a lot better in ratio space ~dB increase in perf
    + yet to test: performance carry over to disparity reconstruction
+ denoiser      
    + re-train on noise characteristic of c2b camera
+ do denoising in another domain!
    + optimized variable should be albedo, disparity, denoised image etc.
+ end-to-end optimization 
    + think about ways to regularize disparity etc.
    + relationship between ratio images and disparity/phase
+ do zncc on optimized code
+ think about fast algorithm for video decoding
+ matrix inversion lemma on quadratic update! to simply 
    + See if can use simplification in DeSCI paper here
+ optimization 
    + decreasing noise level 
    + adaptive rate, lr, gamma, etc. 
    + termination condition (insufficient update terminates the optimization)