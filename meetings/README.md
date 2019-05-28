
# Meeting Notes

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



+ todo
    + a complete picture for state of art for the problem
        + primarily demosaicing and maybe denoising
        + implementation easy/hard
            + data sufficiency, flexibility
            + a blog for what i have done
        + pro/con of methods ...
        + state of arts 
        + cvpr 2018, see paper that cited that...
        + flexisp
    + dataset generation for benchmarking purposes
        + structured light: shapenet dataset, project light with mitsuba (ask parsa/wenjian)
        + multispectral imaging: not for now ... 
    + start using the cameras
        + camera noise characterization, optimize for specific camera
    + question
        + specific priors for different reconstuction 
        + arbitrary tiling (cfa design)

+ read papers 
    + michael brown
        + noise characterization of cellphone
    + http://people.csail.mit.edu/billf/publications/Noise-Optimal_Capture.pdf
        + explanation for noise model
    + 