% Investigate ways to make admm faster
%
clc; clear; close all;
ProjectPaths;

%% Parameters
%
% crop the image to remove the borders
[cx,cy] = deal(1:160,10:247);
% dimension of input image
[h,w] = deal(176,288);
[h,w] = deal(numel(cx),numel(cy));
% scale the intensity of image for better visualization 
scaling = 1.5;
% dataset
dataset_alphabet = SceneNames("alphabet");
% directory containing the raw noisy images
rawimagedir =  "data/alphabet";
% directory containing groundtruth images
stackeddir = sprintf("%s/organized",rawimagedir);
% save images to 
savedir = "results/spatialspectral"; mkdir(savedir);
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
mask_type = "toeplitz";
% scene 
scene = "alphabet";


%% Prev
%

[S,F] = deal(4,3);

M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, 'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetDemosaicDemultiplexParams(light_mode);
params_admm_ratio = GetDemosaicDemultiplexParams(light_mode);
params_admm.denoiser_type = "tnrd";
params_admm.outer_iters = 50;
params_admm_ratio.denoiser_tyep = "tnrd";
params_admm_ratio.outer_iters = 50;

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s%d",stackeddir,scene,S),h,w,S,'CropX',cx,'CropY',cy);
[input_im,input_ratio_im,orig_noisy_im] = ReadInputIm(sprintf("%s/%s%d",rawimagedir,scene,S),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'ForwardFunc',ForwardFunc);

% 
[admm_intensity_im2,psnr_intensity2,ssim_out,~] = ADMM(input_im,H,InitEstFunc,params_admm,orig_im);

[admm_intensity_im,psnr_intensity,~] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);
[admm_ratio_im,psnr_ratio,~] = RunADMM_demosaic(input_ratio_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm_ratio,orig_ratio_im);

ratio_mult_inputsum_im = admm_ratio_im/255;
ratio_mult_inputsum_im = RatioToIntensity(ratio_mult_inputsum_im,sum(input_im,3));
psnr_ratio_mult_inputsum = ComputePSNR(orig_im,ratio_mult_inputsum_im);

denoised_input_im = Denoiser(sum(input_im,3),params_admm.effective_sigma,"tnrd");
ratio_mult_inputsum_denoised_im = admm_ratio_im/255;
ratio_mult_inputsum_denoised_im = RatioToIntensity(ratio_mult_inputsum_denoised_im,denoised_input_im);
psnr_ratio_mult_inputsum_denoised = ComputePSNR(orig_im,ratio_mult_inputsum_denoised_im);

fprintf("psnr_intensity                     %.4f\n",psnr_intensity);
fprintf("psnr_intensity(admm)               %.4f\n",admm_intensity_im2);
fprintf("psnr_ratio_mult_inputsum           %.4f\n",psnr_ratio_mult_inputsum);
fprintf("psnr_ratio_mult_inputsum_denoised  %.4f\n",psnr_ratio_mult_inputsum_denoised);


ims = scaling*FlattenChannels(orig_im,orig_ratio_im,admm_intensity_im,admm_ratio_im,ratio_mult_inputsum_im,ratio_mult_inputsum_denoised_im);
imshow(ims/255);

% [admm_intensity_im2,psnr_intensity2,ssim_out,~] = ADMM(input_im,H,InitEstFunc,params,orig_im);



