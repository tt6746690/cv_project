function M = SubsamplingMask(mask_type,h,w,F,varargin)
% Get spatial subsampling mask for image of size hxw
%       where pixel (i,j) has value from frame M_{ij}'s intensity value
%
%     random
%     bayer
%         1 2
%         2 3
%     horz
%         1 2 3 ... F
%         1 2 3 ... F
%     vert
%         1 1
%         2 2
%         3 3
%         ...
%         F F
%     toeplitz
%         1 2 3 ... F 1 2
%         2 1 2 3 ... F 1
%         3 2 1 2 3 ... F
%     tiles
%         tile tile ...
%         tile tile ...
%
%
%   >> M = SubsamplingMask("bayer",4,4,3)
%   M = 
%       1     2     1     2
%       2     3     2     3
%       1     2     1     2
%       2     3     2     3
%

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'Tile'}, ...
        {'tile'});
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


    switch mask_type
    case 'bayer'
        assert(F == 3);
        M = BayerMask(h,w);
    case 'random'
        M = floor(rand(h,w)/(1/F))+1;
    case 'horz'
        M = zeros(h,w);
        for f = 1:F
            M(:,f:F:end) = f;
        end
    case 'vert'
        M = zeros(h,w);
        for f = 1:F
            M(f:F:end,:) = f;
        end
    case 'toeplitz'
        p = repmat(full(1:F),1,ceil(max(w,h)/F));
        c = p(1,1:h);
        r = p(1,1:w);
        M = toeplitz(c,r);
    case 'tiles'
        M = repmat(tile,ceil([h w]./size(tile)));
        M = M(1:h,1:w);
    otherwise
        warning('mask not set properly');
    end
end