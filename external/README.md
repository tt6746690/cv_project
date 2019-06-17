


+ pca_cfa_denoising
    + http://www4.comp.polyu.edu.hk/~cslzhang/PCA-CFA-Denoising.htm

+ [Joint-Demosaic-and-Denoising-with-ADMM]
    + https://github.com/TomHeaven/Joint-Demosaic-and-Denoising-with-ADMM

+ [RED]
    + https://github.com/google/RED.git
    + tndr denoising 
        + https://drive.google.com/file/d/0B9L0NyTobx_3NHdJLUtPQWJmc2c/view 
        + need to download the dropbox file ...
        + mex compile `lut_eval.c`
            + https://stackoverflow.com/questions/37362414/openmp-with-mex-in-matlab-on-mac
            + rename to `lut_eval.cpp`
            + `mex lut_eval.cpp`
            + `export KMP_DUPLICATE_LIB_OK=True`  https://github.com/dmlc/xgboost/issues/1715