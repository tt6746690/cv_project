


### Camera

+ [2018_coded_two_bucket_cameras_for_computer_vision](2018_coded_two_bucket_cameras_for_computer_vision.pdf)
    + abstract 
        + coded two-bucket imaging (C2B) for 3D shape estimation.  the sensor modulates light arriving at each pixel and outputs 2 images per pixel. When coupled with dynamic light sources acquire illumination mosaics, which can be processsed to acquire live disparity/normal maps of dynamic objects.
    + questions
        + spatial multiplexing (F frames -> 1 frame that is spatially multiplex)
            + achieved with a tiling and a correspondence of frame to pixel sampling scheme
            + what is the motivation for this
                + photometric stereo needs several photos, needs a sequence of this ...
                + but scene always move in a video, adapt traditional algoritm (stereo) work under this framework (reconstruction in 1 shot)
                + similar to color imaging (several spectrum 1 in shot)
                + potential for different tiles (4x4 -> recover 16 frames)



### Points

+ What metrics used in demosaicking papers ?
    + color
        + MSE/PSNR: error between reference and reconstructed images
        + S-CIELAB: perceptual color fidelity (http://scarlet.stanford.edu/~brian/scielab/introduction.html)
    + artifacts
        + a measure of zipper effect near sharp edges(https://pdfs.semanticscholar.org/9309/59339e8b69b90d18b479dbfa06049f4a5182.pdf)
        + a measure of aliasing (false color)
+ How confident are we in adapting bayer CFA interpolation methods to structured light images ?
    + assumptions that cannot be assumed 
        + constant/smooth hue assumption (spectrally, hue is constant/smooth inside boundaries of objects) since the image is grayscale
        + each pixel has 
    + assumptions that can be exploited 
        + homogeneity assumption (spatially, neighboring pixels morel likely to have similar colors)
    + some demosaicing algorithm exploit structure of bayer color filter array, which is not the case in our case.
        + sequential demosaicing methods (interpolate G channel first, then interpolate R,G) relies on the rationale tht since G channel is sampled more it is less aliased. This is not true for the two bucket camera


+ http://people.duke.edu/~sf59/TIP_Demos_Final_Color.pdf
    + considers super-resolution and demosaicing as the same problem