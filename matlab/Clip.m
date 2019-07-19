function im = Clip(im,l,r)
% Clips an image `im` to be pixel-wise withint [l,r]
%
    assert(l<r,"lower bound should be less than upper bound");
    im = min(max(im,l),r);
end 