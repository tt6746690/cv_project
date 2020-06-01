function [phase,zncc,I] = DecodeZNCC(X,P,lb,ub,varargin)
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
    %       depthbounds     [0,hproj-1]^(h*w)
    %           lb  hxw         pixel-wise phase lower bound
    %           ub  hxw         pixel-wise phase upper bound
    %
    % Outputs:
    %       phase   [0,hproj-1]^(h*w)
    %           unwrapped phase 
    %

    NormalizeBy = 'std';
    NPixelNeighbors = 1;

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'NormalizeBy','NPixelNeighbors'}, ...
        {'NormalizeBy','NPixelNeighbors'});
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

    switch NormalizeBy
    case 'none'
        normalize_ = @(x) normalize(x,2,'center','mean');
    case 'std'
        normalize_ = @(x) normalize(normalize(x,2,'center','mean'),2,'norm',2);
    end

    assert(numel(find(ub<0))==0);
    
    [hproj,~] = size(P);

    % \to [1,hproj]
    lb = lb + 1;
    ub = ub + 1;

    [h,w,S] = size(X);
    X = reshape(X,[],S);

    switch NPixelNeighbors
    case 1
    case 3
        X = [circshift(X,1,1) X circshift(X,-1,1)];
        P = [circshift(P,1,1) P circshift(P,-1,1)];
    case 5
        X = [circshift(X,2,1) circshift(X,1,1) X circshift(X,-1,1) circshift(X,-2,1)];
        P = [circshift(P,2,1) circshift(P,1,1) P circshift(P,-1,1) circshift(P,-2,1)];
    otherwise
        warning("invalid npixelneighbors");
    end

    X = normalize_(X);
    P = normalize_(P);

    zncc = X*P'; % h*w x hproj

    for i = 1:size(zncc,1)
        zncc(i, 1 : floor(lb(i)) ) = -inf;
        zncc(i, ceil(ub(i)) : end) = -inf; 
    end

    [~,I] = max(zncc,[],2);

    phase = reshape(I,h,w);

    % cellfun(@(v) assignin('base',v,evalin('caller',v)),who);
end