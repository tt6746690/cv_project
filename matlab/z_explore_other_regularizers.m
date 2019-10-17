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
rawimagedir =  "data/7patterns";
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
light_mode = true;
% sigmas 
input_sigma = 1;
% sensor mask type 
mask_type = "toeplitz";
% scene 
scene = "giraffe";
% projector height
hproj = 684;
% for visualizing disparity 
disparityFunc = @(corres,pos) (corres - 2.7*pos);
dispRange = [50, 160];
% bounds 
Bounds = load(sprintf('mian/CalibrationCode/%s.mat', 'Bounds'));
Bounds.yErrorLB = Bounds.yErrorLB(cx,cy); %  5;
Bounds.yErrorUB = Bounds.yErrorUB(cx,cy); % - 5;
tempshift = 1.2984; tempshift = 0.25; tempshift = -0.34;
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
[admmsmooth_ratio_im,~,~,~] = ADMMSmooth(input_ratio_im,H,InitEstFunc,params_admm_ratio,255*IntensityToRatio(orig_noisy_im(:,:,take_idx)));
ratio_mult_inputsum_im = admmsmooth_ratio_im/255;
ratio_mult_inputsum_im = RatioToIntensity(ratio_mult_inputsum_im,sum(input_im,3));
[psnr_ratio_mult_inputsum,ssim_ratio_mult_inputsum] = ComputePSNRSSIM(orig_im(:,:,take_idx),ratio_mult_inputsum_im);
fprintf("admm smooth      psnr_ratio_mult_inputsum      %.4f/%.4f\n",psnr_ratio_mult_inputsum,ssim_ratio_mult_inputsum);

%% 

% % 2. admm+tnrd in ratio space
% [admm_ratio_im,~,~,~] = ADMM(input_ratio_im,H,InitEstFunc,params_admm_ratio,IntensityToRatio(orig_noisy_im(:,:,take_idx)));
% ratio_mult_inputsum_im = admm_ratio_im/255;
% ratio_mult_inputsum_im = RatioToIntensity(ratio_mult_inputsum_im,sum(input_im,3));
% [psnr_ratio_mult_inputsum,ssim_ratio_mult_inputsum] = ComputePSNRSSIM(orig_im(:,:,take_idx),ratio_mult_inputsum_im);
% fprintf("admm             psnr_ratio_mult_inputsum      %.4f/%.4f\n",psnr_ratio_mult_inputsum,ssim_ratio_mult_inputsum);


% %% photometric stereo
% projector_phase_shift = transpose((take_idx-1)*2*pi/7);

% [orig_im_albedo,~,orig_im_phase] = SLTriangulation(orig_im(:,:,take_idx),W,Bounds,4,'Shifts',projector_phase_shift);
% orig_im_disparity = disparityFunc((orig_im_phase*hproj/(2*pi)),Y);


% [ratio_im_albedo,~,ratio_im_phase] = SLTriangulation(ratio_mult_inputsum_im,W,Bounds,4,'Shifts',projector_phase_shift);
% ratio_im_disparity = disparityFunc((ratio_im_phase*hproj/(2*pi)),Y);


% %% save images 

% ims1 = scaling*FlattenChannels(orig_im,orig_ratio_im*scaling,admm_ratio_im*scaling,ratio_mult_inputsum_im);
% ims2 = zeros(2*h,w*S);
% ims2(:,1:3*w) = [
%     orig_im_albedo,255*orig_im_phase/(2*pi),orig_im_disparity;...
%     ratio_im_albedo,255*ratio_im_phase/(2*pi),ratio_im_disparity];
% ims = [ims1;ims2];
% imshow(ims/255);
% imwrite(uint8(ims),sprintf("%s/%s_%d.png",savedir,scene,S));


% % intensity_im_albedo,255*intensity_im_phase/(2*pi),intensity_im_disparity; ...
