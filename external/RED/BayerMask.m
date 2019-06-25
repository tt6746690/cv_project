function M = BayerMask(h,w)
%   Get bggr bayer mask for an image of size hxw
%
    assert(mod(h,2)==0 && mod(w,2)==0, 'image size multiple of 2');

    M = zeros(h,w);
    M(1:2:end,1:2:end) = 1;
    M(2:2:end,1:2:end) = 2;
    M(1:2:end,2:2:end) = 2;
    M(2:2:end,2:2:end) = 3;
end