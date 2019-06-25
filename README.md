


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
```


## Questions 

+ is R,G,B in exposure space or log of that ...
    + not sure
+ (psnr comparison) numbers are different (even for bilinear) and the results are often times conflicting, i.e. 2016 Klatz is a lot worse than 2014 flexISP in `demosaicnet` but the reverse is true in Klatz's paper
    + keep track of a table, report psnr of methods in each paper, -> wholistic view
+ (test dataset) for now, given `S` stacked (noise-less) / scene, multiplex to `2F` bucket images which serve as groudtruth images. These images are then downsampled (according to bayer pattern) and different demosaicing methods are tested. However, this might not capture the realistic noise characteristic of the camera, since the low-res image is a spatial downsampling of a _noiseless_ grounth truth image. Could we do better than this?


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
    + 
