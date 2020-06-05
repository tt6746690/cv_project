function ProjectPaths()
    addpath(genpath('../external/RED/tnrd_denoising/'));
    addpath(genpath('../external/RED/minimizers/'));
    addpath(genpath('../external/RED/parameters/'));
    addpath(genpath('../external/RED/helper_functions/'));
    addpath(genpath("../external/mian/helperFunctions/"));
    addpath(genpath("../external/BM3D"));
    % setting this env variable to ensure tnrd does not crash MATLAB
    setenv('KMP_DUPLICATE_LIB_OK','True');
end
