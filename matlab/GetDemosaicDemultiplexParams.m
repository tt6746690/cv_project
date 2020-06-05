function params = GetDemosaicDemultiplexParams(light_mode)

    % regularization factor
    params.lambda = 0.008;

    % for ADMMSmooth
    params.lambda2 = params.lambda;
    params.lambda3 = params.lambda;
    
    % number of outer iterations
    if light_mode
        params.outer_iters = 1;
        params.denoiser_type = "medfilter";
    else
        params.outer_iters = 100;
        params.denoiser_type = "medfilter";
    end

    % number of inner iterations
    params.inner_iters = 200;

    % number of denoising applications
    params.inner_denoiser_iters = 1;
    
    % level of noise assumed in the regularization-denoiser
    params.effective_sigma = 5;

    % admm parameter
    params.rho  = 1e-3;

    % z-update step method {fixed_point,denoiser}
    params.z_update_method = "fixed_point";

end