


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


## Todos


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
    + probabilistic formulation or alternating optimization 
+ do zncc on optimized code
+ think about fast algorithm for video decoding
+ matrix inversion lemma on quadratic update! to simply 
    + See if can use simplification in DeSCI paper here
+ optimization 
    + decreasing noise level 
    + adaptive rate, lr, gamma, etc. 
    + termination condition (insufficient update terminates the optimization)
+ do hdr, hyperspectral imaging with c2b as well, joint optimization etc.
+ denoiser
    + trained on c2b images, might not be that important as Gaussian seems to be an OK noise model.
