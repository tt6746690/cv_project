% Explore other other regularizers
%     - TV
%     - denoiser 
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
scaling = 2;
% directory containing the raw noisy images
rawimagedir =  "data/exp60";
% directory containing groundtruth images
stackeddir = sprintf("%s/organized",rawimagedir);
% save images to 
savedir = "results/explore_other_regularizers"; mkdir(savedir);
% black level 
blacklevelpath = "data/blacklevel_all1/blacklevel.mat";
if ~isfile(blacklevelpath)
    ComputeBlackLevel("data/blacklevel_all1",h,w,blacklevelpath);
end
blacklvl = load(blacklevelpath); blacklvl = blacklvl.blacklvl;
% toggle to false for long runs
light_mode = false;
% sigmas 
input_sigma = 1;
% sensor mask type 
mask_type = "toeplitz";
% scene 
scene = "head";
% projector height
hproj = 684;
% for visualizing disparity 
disparityFunc = @(corres,pos) (corres - 2.7*pos);
dispRange = [50, 160];
% bounds 
Bounds = load(sprintf('mian/CalibrationCode/%s.mat', 'Bounds'));
Bounds.yErrorLB = Bounds.yErrorLB(cx,cy); %  5;
Bounds.yErrorUB = Bounds.yErrorUB(cx,cy); % - 5;
tempshift = 1.2984; tempshift = 0.25; % tempshift = -0.34;
Bounds.LB = double(Bounds.yErrorLB)*2*pi/hproj + tempshift;
Bounds.UB = double(Bounds.yErrorUB)*2*pi/hproj + tempshift;

[X,Y] = meshgrid(1:w,1:h);

% params to admm
params_admm = GetDemosaicDemultiplexParams(light_mode);
params_admm_ratio = GetDemosaicDemultiplexParams(light_mode);

[scenes,shifts] = SceneNames("7patterns");
shift = find(scenes==scene);

take_indices = containers.Map( ...
    {4,5,6,7}, ...
    {
        [1 3 5 7],
        [1 3 4 5 7],
        [1 2 3 4 5 7],
        [1 2 3 4 5 6 7]
    });

%% 
S = 7;
F = S-1;
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, 'BucketMultiplexingMatrix',W,'SubsamplingMask',M);

%% 

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',shift);
[input_im,input_ratio_im,orig_noisy_im] = ReadInputIm(sprintf("%s/%s",rawimagedir,scene),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'CircShiftInputImageBy',shift,'ForwardFunc',ForwardFunc);


imholder = zeros(h,w,S);
imholder(:,:,1:2) = input_im;
imholder(:,:,3:4) = input_ratio_im;
imshow(FlattenChannels(orig_im,orig_ratio_im,imholder)/255);



%% 
take_idx = take_indices(S);

% 1. admm+tnrd+smooth in ratio space 
[radmmsmooth_ratio_im,~,~,~] = ADMMSmooth(input_ratio_im,H,InitEstFunc,params_admm_ratio,255*IntensityToRatio(orig_noisy_im(:,:,take_idx)));

radmmsmooth_im = radmmsmooth_ratio_im/255;
radmmsmooth_im = RatioToIntensity(radmmsmooth_im,sum(input_im,3));
[psnr_radmmsmooth,ssim_radmmsmooth] = ComputePSNRSSIM(orig_im(:,:,take_idx),radmmsmooth_im);

%% 

% 2. admm+tnrd in ratio space
[radmm_ratio_im,~,~,~] = ADMM(input_ratio_im,H,InitEstFunc,params_admm_ratio,255*IntensityToRatio(orig_noisy_im(:,:,take_idx)));
radmm_im = radmm_ratio_im/255;
radmm_im = RatioToIntensity(radmm_im,sum(input_im,3));
[psnr_radmm,ssim_radmm] = ComputePSNRSSIM(orig_im(:,:,take_idx),radmm_im);


fprintf("admm             without 1st order TV           %.4f/%.4f\n",psnr_radmm,ssim_radmm);
fprintf("admm smooth      with    1st order TV      %.4f/%.4f\n",psnr_radmmsmooth,ssim_radmmsmooth);


%% photometric stereo
projector_phase_shift = transpose((take_idx-1)*2*pi/7);

[orig_im_albedo,~,orig_im_phase] = DecodePhaseShiftWithDepthBound(orig_im(:,:,take_idx),W,Bounds,4,'Shifts',projector_phase_shift);
orig_im_disparity = disparityFunc((orig_im_phase*hproj/(2*pi)),Y);

[radmmsmooth_im_albedo,~,radmmsmooth_im_phase] = DecodePhaseShiftWithDepthBound(radmmsmooth_im,W,Bounds,4,'Shifts',projector_phase_shift);
radmmsmooth_im_disparity = disparityFunc((radmmsmooth_im_phase*hproj/(2*pi)),Y);

[radmm_im_albedo,~,radmm_im_phase] = DecodePhaseShiftWithDepthBound(radmm_im,W,Bounds,4,'Shifts',projector_phase_shift);
radmm_im_disparity = disparityFunc((radmm_im_phase*hproj/(2*pi)),Y);


% %% save images 

ims1 = scaling*FlattenChannels(orig_im,orig_ratio_im,radmm_ratio_im,radmmsmooth_ratio_im,radmmsmooth_im);
ims2 = zeros(3*h,w*S);
ims2(:,1:3*w) = [
    orig_im_albedo,255*orig_im_phase/(2*pi),orig_im_disparity;...
    radmm_im_albedo,255*radmm_im_phase/(2*pi),radmm_im_disparity;...
    radmmsmooth_im_albedo,255*radmmsmooth_im_phase/(2*pi),radmmsmooth_im_disparity];
ims = [ims1;ims2];
imshow(ims/255);


imwrite(uint8(ims),sprintf("%s/%s_%d.png",savedir,scene,S));
