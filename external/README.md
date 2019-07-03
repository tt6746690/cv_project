


+ pca_cfa_denoising
    + http://www4.comp.polyu.edu.hk/~cslzhang/PCA-CFA-Denoising.htm

+ [demosaicnet]
    + installation 
        + caffe http://caffe.berkeleyvision.org/install_osx.html
        + https://medium.com/@shardulgo/installing-caffe-1-0-on-mac-os-x-part-1-4-64cf479be75d

+ [Joint-Demosaic-and-Denoising-with-ADMM]
    + https://github.com/TomHeaven/Joint-Demosaic-and-Denoising-with-ADMM
    + setup
        + caffe
        + BM3D not working, code too old
            + `otool -L bm3d_thr_color.mexmaci64`
            + `patch < CBM3D.patch` for BM3D
            + substitute denoiser to channel-wise tnrd denoising
    + cannot make this work, even on Kodak images ...

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

+ [learn_prox_ops]
    + instal
        + git lfs https://github.com/git-lfs/git-lfs
        + https://github.com/facebook/prophet/issues/418
        + requirements.txt: `matplotlib>=2.0.2`
        + also `setuptools`
        + `ln -s /usr/local/opt/llvm/bin/clang /usr/local/bin/clang-omp`
        + have a hard time installing this as well...
        + `conda create --name learn_prox_op --file requirements.txt`
            ```
            pip3 install \
                colour-demosaicing==0.1.2 \
                sacred==0.7.0 \
                git+https://github.com/timmeinhardt/pybm3d@learn_prox_ops \
                tensorflow==1.3.0 \
                opencv-python==3.3.0.10 \
                pymongo \
                pybm3d

            conda install opencv-python

            ```



+ [deep_demosaick]
    + https://github.com/cig-skoltech/deep_demosaick
    + `conda create -c prigoyal -c menpo -c pytorch -c soumith --name deep_demosaicking --file requirements.txt`
        + conflicting deps
            + `numexpr`
            + `openblas-devel` remove this as dependencies ...
        + `conda init` not working
            + `source ~/miniconda3/etc/profile.d/conda.sh`
    + ssh tunneling
        ```
        ssh -L 8000:localhost:8888 wpq@comps0.cs.toronto.edu
        ssh -L 8000:localhost:8888 wpq@scheduler.cs.toronto.edu
        // RTX2080 
        srun --partition=gpunodes --nodelist=gpunode16 --mail-type=ALL,TIME_LIMIT_90 --mail-user=wpq@cs.toronto.edu  --pty bash --login

        ```
    + extra deps
        ```
        conda install -c conda-forge scikit-image tqdm
        ```

+ [argmin-TR-2016-code]
    + http://users.cecs.anu.edu.au/~sgould/
    + examples that solve bilevel problems with gradient descent

+ [On_RED]
    + https://github.com/edward-reehorst/On_RED/
    + new interpretation on red, proposed a faster algo