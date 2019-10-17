function params = GetDemosaicDemultiplexParams(light_mode)

    % regularization factor
    params.lambda = 0.008;

    % for ADMMSmooth
    params.lambda2 = params.lambda;
    params.lambda3 = params.lambda;
    
    % number of outer iterations
    if light_mode
        params.outer_iters = 5;
        params.denoiser_type = "medfilter";
    else
        params.outer_iters = 100;
        params.denoiser_type = "tnrd";
    end

    % number of inner iterations
    params.inner_iters = 200;
    
    % level of noise assumed in the regularization-denoiser
    params.effective_sigma = 3;

    % admm parameter
    params.beta = 1e-3;
    params.rho  = 1e-3;

    % number of denoising applications
    params.inner_denoiser_iters = 1;

    % relaxation parameter of ADMM
    params.alpha = 1;

    % v-update step method {fixed_point,denoiser}
    params.v_update_method = "fixed_point";

end