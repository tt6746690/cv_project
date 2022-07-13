function Y = Denoiser(X,effective_sigma,denoiser_type,varargin)
    % Denoiser `X` using `denoiser_type`  
    %
    %
    MaxNumWorkers = 100;

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'MaxNumWorkers'}, ...
        {'MaxNumWorkers'});
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

    if size(X,3) ~= 1
        Y = zeros(size(X));
        parfor (i = 1:size(X,3), MaxNumWorkers)
            Y(:,:,i) = DenoiserOnOneImage(X(:,:,i),effective_sigma,denoiser_type);
        end
    else
        Y = DenoiserOnOneImage(X,effective_sigma,denoiser_type);
    end
end


function Y = DenoiserOnOneImage(X,sigma,denoiser_type)
    switch denoiser_type
    case "mf"
        fsize = [3 3];
        Y = medfilt2(X,fsize);
    case "bm3d"
        Y = BM3D(1,X,sigma);
        Y = Y*sigma/5;
    case "tnrd"
        Y = ReactionDiffusion(5/sigma*X);
        Y = Y*sigma/5;
    otherwise
        warning("no such denoiser")
    end
end