


```
# python-opencv tutorial
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

# conda env support in jupyter notebook
conda install nb_conda

# select conda env for jupyter notebook
conda env create -f cv_project.yml
conda activate cv_project

# download the datasets 
python3 demosaicing/download_kodak.py --output-dir=data/kodak
```


## Questions 


+ is R,G,B in exposure space or log of that ...


## Todo

+ project
    + demosaicing ... 
        + shape prior instead of color prio?
        + does it perform better than normal color demosaicing?
        + Random code matrix or tiled (tiled for now because using interpolation methods for demosaicing)?
        + demosaicing might taylor to downstream 3d reconstruction ...
+ learning
    + light transport, lambertian reflectance, albedo, image formation model (Forsyth 2503 multiview geometry)
    + photometric stereo (2503 slides+a1)
    + structured light
        + https://www.osapublishing.org/DirectPDFAccess/FBC163A9-EDFF-9962-3464BD70B5AC1546_211561/aop-3-2-128.pdf?da=1&id=211561&seq=0&mobile=no
    + compressive sensing 
        + http://www.cs.toronto.edu/~kyros/courses/2530/papers/Lecture-10/Hitomi2011.pdf
        + https://ieeexplore.ieee.org/document/7442841?tp=&arnumber=7442841
    + computational photography
        + https://stanford.edu/class/ee367/
+ reading demosaicing algorithms ...
    + think about substituting color prior with other prior 


## Random

+ can do hrdr with 2-bit camera as well
+ for c2b, rgb packed into rggb, g channel does not have additional information since they are simply the same subsampled image duplicated.
+ regarding goals for different reconstruction tasks
    + spectral imaging: constant-hue, no abrupt hue change over edges
    + structured light, stereo: devoid of artifacts



## Weekly


### May 7

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



### May 14