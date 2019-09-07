
% want to know if noise characteristics and subsequent 
% reconstruction performance differ for 
% - all goto bucket 1 sensor mask pattern (used for experiments)
% - bayer, toeplitz, etc. (used in realistic image acquisition process)
% by comparing
% - (groundtruth) stacked images of allblack sensor mask
% - (1) subsampled images of allblack sensor mask
% - (2) images with bayer sensor mask

clc; clear; close all;
ProjectPaths;

%% Parameters
%
% crop the image to remove the borders
[cx,cy] = deal(1:160,11:248);
% dimension of input image
[h,w] = deal(176,288);
[h,w] = deal(numel(cx),numel(cy));
% scale the intensity of image for better visualization 
scaling = 1.5;
% directory containing the raw noisy images
rawimagedir =  "data/noise_wrt_sensormask";
% directory containing groundtruth images
stackeddir = sprintf("%s/organized",rawimagedir);
% save images to 
savedir = "results/noise_wrt_sensormask"; mkdir(savedir);
% black level 
blacklevelpath = "data/alphabet_blacklvl/blacklevel.mat";
if ~isfile(blacklevelpath)
    ComputeBlackLevel("data/alphabet_blacklvl",h,w,blacklevelpath);
end
blacklvl = load(blacklevelpath); blacklvl = blacklvl.blacklvl;
% toggle to false for long runs
light_mode = false;
% sigmas 
input_sigma = 1;
% sensor mask type 
mask_type = "bayer";
% scene 
scene = "allblack";

[S,F] = deal(4,3);

M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("bayerdemosaic",h,w,F,S, 'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetDemosaicDemultiplexParams(light_mode);
params_admm_ratio = GetDemosaicDemultiplexParams(light_mode);

%% 

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s%d",stackeddir,"allblack"),h,w,S,'CropX',cx,'CropY',cy);

[allblackm.input_im,allblackm.input_ratio_im,allblackm.orig_noisy_im] = ReadInputIm(sprintf("%s/%s%d",rawimagedir,"allblack"),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'ForwardFunc',ForwardFunc);
[bayerm.input_im,bayerm.input_ratio_im,bayerm.orig_noisy_im] = ReadInputIm(sprintf("%s/%s%d",rawimagedir,"bayer"),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'ForwardFunc',ForwardFunc);

imshow(FlattenChannels(orig_im,orig_ratio_im,orig_noisy_im, ...
    cat(3,allblackm.input_im,allblackm.input_ratio_im),...
    cat(3,bayerm.input_im,bayerm.input_ratio_im))/255);

%% Problem: the bayer sensor mask 
%       trigger should be set after all projector patterns are projected to the scene
%       not right after each pattern is projected.
%       might be able to sum/average the images now, need to write more code