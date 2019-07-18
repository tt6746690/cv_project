function params = GetDemosaicDemultiplexParams(light_mode)

    % regularization factor
    params.lambda = 0.008;
    
    % number of outer iterations
    if light_mode
        params.outer_iters = 50;
    else
        params.outer_iters = 100;
    end

    % number of inner iterations
    params.inner_iters = 200;
    
    % level of noise assumed in the regularization-denoiser
    params.effective_sigma = 3;

    % admm parameter
    params.beta = 1e-3;

    % number of denoising applications
    params.inner_denoiser_iters = 1;

    % relaxation parameter of ADMM
    params.alpha = 1;

    % denoiser type: {medfilter, tnrd}
    params.denoiser_type = "medfilter";
    
return