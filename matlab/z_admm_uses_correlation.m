% investigate if admm really uses the correlation between channels 
%   - 7 random images with text and see how they fair with the algorithm 
clc; clear; close all;
ProjectPaths;

%% Parameters

[S,F] = deal(7,6);
scaling = 2;
[cx,cy] = deal(1:160,11:248);
[h,w] = deal(176,288);
[h,w] = deal(numel(cx),numel(cy));
datadir = "../data";
dataset_7patterns = SceneNames("7patterns");
rawimagedir =  sprintf("%s/7patterns",datadir);
stackeddir = sprintf("%s/7patterns/organized",datadir);
savedir = "results/does_admm_uses_correlation"; mkdir(savedir);
blacklevelpath = sprintf("%s/blacklevel_all1/blacklevel.mat",datadir);
if ~isfile(blacklevelpath)
    blackimsdir = sprintf("%s/blacklevel_all1",datadir);
    ComputeBlackLevel(blackimsdir,h,w,blacklevelpath);
end
blacklvl = load(blacklevelpath);
blacklvl = blacklvl.blacklvl;

% RED 
light_mode = false;
input_sigma = 1;
M = SubsamplingMask("toeplitz",h,w,F);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, ...
        'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetDemosaicDemultiplexParams(light_mode);
params_admm_ratio = GetDemosaicDemultiplexParams(light_mode);


%% 7 images ...

orig_mixed_im = zeros(h,w,S);
orig_noisy_mixed_im = zeros(h,w,S);

for i = 1:7
    scene = dataset_7patterns(i);
    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy);
    [input_im,input_ratio_im,orig_noisy_im] = ReadInputIm(sprintf("%s/%s",rawimagedir,scene),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'ForwardFunc',ForwardFunc);
    orig_mixed_im(:,:,i) = orig_im(:,:,1);
    orig_noisy_mixed_im(:,:,i) = orig_noisy_im(:,:,1);
end

input_mixed_im = ForwardFunc(orig_noisy_mixed_im);
initialest = InitEstFunc(input_mixed_im);

params_admm.outer_iters = 50;
[admm_im,psnr,ssim,~,iter_ims] = ADMM(input_mixed_im,H,InitEstFunc,params_admm,orig_mixed_im);

ims = scaling*FlattenChannels(orig_mixed_im,orig_noisy_mixed_im,cat(3,input_mixed_im/scaling,zeros(h,w,5)),initialest,admm_im);
imshow(ims/255);
imwrite(uint8(ims),sprintf("%s/random_S_images.png",savedir));


%% With texts

dataset_alphabet = SceneNames("alphabet");
rawimagedir =  "data/alphabet";
stackeddir = sprintf("%s/organized",rawimagedir);

orig_mixed_im = zeros(h,w,S);
orig_noisy_mixed_im = zeros(h,w,S);

for i = 1:S
    scene = "alphabet";
    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s%d",stackeddir,scene,S),h,w,S,'CropX',cx,'CropY',cy);
    [input_im,input_ratio_im,orig_noisy_im] = ReadInputIm(sprintf("%s/%s%d",rawimagedir,scene,S),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'ForwardFunc',ForwardFunc);
    orig_mixed_im(:,:,i) = orig_im(:,:,i);
    orig_noisy_mixed_im(:,:,i) = orig_noisy_im(:,:,i);
end

% do transformations

for i = 1:S
    rotated = imrotate(orig_mixed_im(:,:,i),(i-1)*25);
    [rh,rw] = size(rotated);
    orig_mixed_im(1:min(h,rh),1:min(w,rw),i) = rotated(1:min(h,rh),1:min(w,rw));

    rotated = imrotate(orig_noisy_mixed_im(:,:,i),(i-1)*25);
    [rh,rw] = size(rotated);
    orig_noisy_mixed_im(1:min(h,rh),1:min(w,rw),i) = rotated(1:min(h,rh),1:min(w,rw));
end

input_mixed_im = ForwardFunc(orig_noisy_mixed_im);
imshow(FlattenChannels(orig_mixed_im,orig_noisy_mixed_im,cat(3,input_mixed_im/scaling,zeros(h,w,5)))/255);

%%

initialest = InitEstFunc(input_mixed_im);

params_admm.outer_iters = 50;
[admm_im,psnr,ssim,~,iter_ims] = ADMM(input_mixed_im,H,InitEstFunc,params_admm,orig_mixed_im);

ims = scaling*FlattenChannels(orig_mixed_im,orig_noisy_mixed_im,cat(3,input_mixed_im/scaling,zeros(h,w,5)),initialest,admm_im);
imshow(ims/255);
imwrite(uint8(ims),sprintf("%s/alphabet_rotated.png",savedir));




im = orig_im(:,:,1)
noisy = im + 5*randn(size(im));
[~,denoised] = BM3D(1,noisy,5);
imshow([im noisy denoised]/255);