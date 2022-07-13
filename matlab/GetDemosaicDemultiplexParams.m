function params = GetDemosaicDemultiplexParams(varargin)

    save_iterates = '';

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'SaveIterateDirectory'}, ...
        {'save_iterates'});
    v = 1;
    while v <= numel(varargin)
        param_name = varargin{v};
        if isKey(params_to_variables,param_name)
            assert(v+1<=numel(varargin));
            v = v+1;
            % Trick: use feval on anonymous function to use assignin to this workspace
            feval(@()assignin('caller',params_to_variables(param_name),varargin{v}));
        else
            error('Unsupported parameter: %s',varargin{v});
        end
        v=v+1;
    end 

    % regularization factor
    params.lambda = 0.008;

    % for ADMMSmooth
    params.lambda2 = params.lambda;
    params.lambda3 = params.lambda;
    
    % number of outer iterations
    params.outer_iters = 50;

    % denoiser type {'mf','tnrd','bm3d'}
    params.denoiser_type = "tnrd";

    % number of denoising applications
    params.inner_denoiser_iters = 1;
    
    % level of noise assumed in the regularization-denoiser
    params.effective_sigma = 5;

    % admm parameter
    params.rho  = 1e-3;

    % z-update step method {fixed_point,denoiser}
    params.z_update_method = "fixed_point";

    % print states 
    params.verbose = true;

    % saves iterates to file, {'', or '/path/to/directory/'}
    params.save_iterates = save_iterates;

    % saves psnrs/ssims
    params.compute_psnr_ssim = true;
end