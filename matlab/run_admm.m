function demul_im = run_admm(y,A,InitEstFunc,params,orig_im,varargin)
    % Run ADMM
    %

    RatioIntensity = 'intensity';
    ADMMFunc = @ADMM;

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'RatioIntensity','ADMMFunc'}, ...
        {'RatioIntensity','ADMMFunc'});
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

    switch RatioIntensity
    case 'ratio'
        input_ratio_im = IntensityToRatio(y);
        [admm_ratio_im,~,~,~] = ADMMFunc(input_ratio_im,A,InitEstFunc,params,orig_im);
        demul_im = admm_ratio_im/255;
        demul_im = RatioToIntensity(demul_im,sum(y,3))*255;
    case 'intensity'
        [demul_im,~,~,~] = ADMMFunc(y,A,InitEstFunc,params,orig_im);
    otherwise
        warning('not ratio/intensity');
    end
end