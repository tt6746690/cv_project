function ratio_im = IntensityToRatio(im)
% Converts image in intensity space to ratio space
%
%   `im`        h x w x 2
%   `ratio_im`  h x w x S
%
    ratio_im = im ./ repmat(sum(im,3),1,1,size(im,3));
    ratio_im(isnan(ratio_im)) = 0;
end
