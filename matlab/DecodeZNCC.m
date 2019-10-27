function phase = DecodeZNCC(X,P,lb,ub)
    % Decode phase 
    %       - maximizing ZNCC between observed vs. projected pattern along epipolar line
    %           http://www.dgp.toronto.edu/optimalsl/
    %       - depth bounds
    %
    % Inputs:
    %       X       hxwxS
    %           S demultiplexed images
    %       P       hprojxS
    %           S projector patterns along a column
    %       depthbounds \in [0,1]^(h*w)
    %           lb  hxw         pixel-wise phase lower bound
    %           ub  hxw         pixel-wise phase upper bound
    %

    assert(numel(find(ub<0))==0);
    ub(ub<=0) = eps;
    
    [hproj,~] = size(P);
    [h,w,S] = size(X);
    X = reshape(X,[],S);

    % normalize_ = @(x) normalize(normalize(x,2,'center','mean'),2,'norm',2);
    normalize_ = @(x) normalize(x,2,'center','mean');
    X = normalize_(X);
    P = normalize_(P);

    zncc = X*P';

    for i = 1:size(zncc,1)
        zncc(i, 1 : floor(lb(i)) ) = -inf;
        if ub(i) == 0
            continue
        end
        zncc(i, ceil(ub(i)) : end) = -inf; 
    end

    [~,I] = max(zncc,[],2);

    phase = reshape(I,h,w);
    phase = (2*pi)*phase/hproj;
end


