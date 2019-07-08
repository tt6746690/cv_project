function M = SubsamplingMask(mask_type,h,w,F)
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
%
%
%   >> M = SubsamplingMask("bayer",4,4,3)
%   M = 
%       1     2     1     2
%       2     3     2     3
%       1     2     1     2
%       2     3     2     3
%
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
    otherwise
        warning('mask not set properly');
    end
end