function im = RatioToIntensity(ratio_im,denom)
% Converts image in intensity space to ratio space
%
%   `ratio_im`  h x w x S
%   `denom`     h x w
%   `im`        h x w x S
%
    im = ratio_im.*repmat(denom,1,1,size(ratio_im,3));
end