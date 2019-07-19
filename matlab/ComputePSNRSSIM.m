function [psnr,ssim] = ComputePSNRSSIM(im1,im2)
    % Computes psnr and ssim between two images 
    %
    psnr = ComputePSNR(im1,im2);
    ssim = ComputeSSIM(im1,im2);
end