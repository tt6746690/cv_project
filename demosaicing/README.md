
## Datset 


```sh
python3 demosaicing/download_kodak.py --output-dir=data/kodak
```


## Implementations 

+ opencv
    + https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#gga4e0972be5de079fed4e3a10e24ef5ef0a8945844ab075687f4d4196abe1ce0db4
        + variable number of gradients
        + Edge-Aware Demosaicing.



+ bilinear interpolation 
    + https://en.wikipedia.org/wiki/Bilinear_interpolation
    + https://github.com/csrjtan/CDM-CNN/blob/master/src/utillities/bilinear.m

+ bayer 
    + tutorial: https://github.com/codeplaysoftware/visioncpp/wiki/Example:-Bayer-Filter-Demosaic
    + opencv: https://github.com/opencv/opencv/blob/master/modules/imgproc/src/demosaicing.cpp
    + homework: https://github.com/Shmeve/bayer-demosaicing
    + https://github.com/colour-science/colour-demosaicing
    + https://github.com/eric612/BayerToRGB

+ nearest-neighbor interpolation 
    + https://github.com/scivision/pysumix/blob/master/pysumix/demosaic.py

+ Joint Demosaicing and Denoising of Noisy Bayer Images with ADMM
    + https://github.com/TomHeaven/Joint-Demosaic-and-Denoising-with-ADMM

+ 2016 demosaicnet
    + https://github.com/mgharbi/demosaicnet_torch/

+ 2018 Deep Joint Design of Color Filter Arrays and Demosaicing
    + https://github.com/bernardohenz/deep_joint_design_cfa_demosaicing