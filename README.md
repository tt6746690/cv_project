


```
# python-opencv tutorial
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

# conda env support in jupyter notebook
# conda install nb_conda // already in yml

# select conda env for jupyter notebook
conda env create -f cv_project.yml
conda activate cv_project

# download the datasets 
python3 demosaicing/download_kodak.py --output-dir=data/kodak

# matlab bug: duplicate libomp loaded
export KMP_DUPLICATE_LIB_OK=True
```


## Ideas

+ multi-convex programming for estimating multiple latent factors
    + used in: Non-Line-of-Sight Imaging with Partial Occluders and Surface Normals
+ graphical model approach
    + somehow take into account geometry (depth/disparity) is causal factor to illuminated images (S demutiplexed images)
    + define a sensible potential
    + look more into graphical model approach to image processing, inference of geometry, etc.


## Learning

+ light transport, lambertian reflectance, albedo, image formation model (Forsyth 2503 multiview geometry)
+ photometric stereo (2503 slides+a1)
+ structured light
    + https://www.osapublishing.org/DirectPDFAccess/FBC163A9-EDFF-9962-3464BD70B5AC1546_211561/aop-3-2-128.pdf?da=1&id=211561&seq=0&mobile=no
+ compressive sensing 
    + http://www.cs.toronto.edu/~kyros/courses/2530/papers/Lecture-10/Hitomi2011.pdf
    + https://ieeexplore.ieee.org/document/7442841?tp=&arnumber=7442841
+ computational photography
    + http://www.cs.toronto.edu/~kyros/courses/2530/
    + https://stanford.edu/class/ee367/
+ discrete differential geometry 
    + https://graphics.stanford.edu/courses/cs468-13-spring/
+ deep learning refresher
    + http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/


## Random

+ can do hrdr with 2-bit camera as well
+ for c2b, rgb packed into rggb, g channel does not have additional information since they are simply the same subsampled image duplicated.
+ regarding goals for different reconstruction tasks
    + spectral imaging: constant-hue, no abrupt hue change over edges
    + structured light, stereo: devoid of artifacts
+ arbitrary mosaic tiling: 
    + adaptive graph laplacian
    + combinatorial optimization, branch bound, simulated annealing, etc.
    + see if can be converted to graph problems
+ use dark frame subtraction for fixed-pattern noise
    + take several frames of image with lens capped,
    + subtract the dark frame from subsequent images




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


## Idea


+ flexisp   
    + mask matrix `M` to mask pixels' contribution to data term when having low confidence in them, i.e.too saturated, or noisy
    + denoiser + TV provides better regularization to the problem
        + self-similarity prior of NLM, BM3D ... can fill in values even if lots of data are missing
            + but only works if image exhibits self-similarity
+ denoiser 
    + can be trained on T2 images
        + groundtruth, and denoisy inputs (lots of them, since can record 1000 images, each with different noise characteristics)
    + simply applying denoiser, is 
        + unstable for median filter
        + for tnrd denoiser, the performance is not as good as RED ...
    + unet/resnet
+ 

