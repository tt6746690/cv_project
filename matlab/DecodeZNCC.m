function phase = DecodeZNCC(X,P,lb,ub,varargin)
    % Decode phase 
    %       - maximizing ZNCC between observed vs. projected pattern along epipolar line
    %           http://www.dgp.toronto.edu/optimalsl/
    %       - depth bounds
    %
    % Inputs:
    %       X       hxwxS
    %           S demultiplexed images
    %       P       [0,1]^hprojxS
    %           S projector patterns along a column
    %       depthbounds     [0,2pi]^(h*w)
    %           lb  hxw         pixel-wise phase lower bound
    %           ub  hxw         pixel-wise phase upper bound
    %
    % Outputs:
    %       phase   [0,2pi]^(h*w)
    %           unwrapped phase 
    %

    normalize_by = 'std';

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'NormalizeBy'}, ...
        {'normalize_by'});
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

    switch normalize_by
    case 'none'
        normalize_ = @(x) normalize(x,2,'center','mean');
    case 'std'
        normalize_ = @(x) normalize(normalize(x,2,'center','mean'),2,'norm',2);
    end

    assert(numel(find(ub<0))==0);
    
    [hproj,~] = size(P);

    % maintain same range for depth bounds as SLTriangulation
    %       [0,2pi] => [0,hproj]
    lb = lb*hproj/(2*pi);
    ub = ub*hproj/(2*pi);

    % prevents ceil(ub(i)) yielding invalid index to zncc
    ub(ub<=0) = eps;

    [h,w,S] = size(X);
    X = reshape(X,[],S);

    X = normalize_(X);
    P = normalize_(P);

    zncc = X*P';

    for i = 1:size(zncc,1)
        zncc(i, 1 : floor(lb(i)) ) = -inf;
        zncc(i, ceil(ub(i)) : end) = -inf; 
    end

    [~,I] = max(zncc,[],2);

    phase = reshape(I,h,w);
    phase = (2*pi)*phase/hproj;
end