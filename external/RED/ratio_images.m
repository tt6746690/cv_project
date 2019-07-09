% Compare performance of RED when input_im is in {ratio,intensity} space
%       the idea is that in ratio space, texture removed, so might be easier to do reconstruction
clc; clear; close all;
addpath(genpath('./tnrd_denoising/'));
addpath(genpath('./minimizers/'));
addpath(genpath('./parameters/'));
addpath(genpath('./helper_functions/'));
addpath(genpath('./test_images/'));
addpath(genpath("./mian/helperFunctions/Camera"));
addpath(genpath("./mian/helperFunctions/ASNCC"));
addpath(genpath("./mian/helperFunctions/Algorithms"));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set to 1 if debug, 0 otherwise
short = 0;
% crop the image to remove the borders
[cx,cy] = deal(1:160,10:249);
% #patterns/frames
[S,F] = deal(4,3);
% dimension of input image
[h,w] = deal(176,288);
[h,w] = deal(numel(cx),numel(cy));
% scale the intensity of image for better visualization 
scaling = 2;
% scene
scene = "shoe";
% dataset
dataset_exp60 = SceneNames("exp60");
% directory containing the raw noisy images
rawimagedir =  "data/exp60";
% directory containing groundtruth images
stackeddir = "data/exp60/organized";
% save images to 
savedir = "results/ratio"; mkdir(savedir);
% black level 
blacklevelpath = "data/blacklevel_all1/blacklevel.mat";
if ~isfile(blacklevelpath)
    blackimsdir = "data/blacklevel_all1";
    ComputeBlackLevel(blackimsdir,h,w,blacklevelpath);
end
blacklvl = load(blacklevelpath);
blacklvl = blacklvl.blacklvl;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RED-specific parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RED less #iterations
light_mode = true;
% sigmas 
input_sigma = 1;
% mask 
M = BayerMask(h,w);
% two-bucket multiplexing matrix
W = BucketMultiplexingMatrix(S)*2;
% linear map from S patterened image -> two bucket image
[H,B,C] = SubsampleMultiplexOperator(S,M);
% args to RunADMM
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, ...
        'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetSuperResADMMParams(light_mode);
params_admm.beta = 1.5;
params_admm.lambda = 0.3;

if short == 1
    params_admm.outer_iters = 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compare RED on both ratio/intensity space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset_exp60 = ["shoe"];


for scene = dataset_exp60

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% read in image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % intensity space
    orig_im_noisy = zeros(h,w,S);
    orig_im = zeros(h,w,S);
    input_im = zeros(h,w,2);

    % ratio space (scale by 255)
    ratio_orig_im = zeros(h,w,S);
    ratio_input_im = zeros(h,w,2);

    files = dir(sprintf("%s/%s/*.png",rawimagedir,scene));
    [fnames,ffolders] = deal({files.name},{files.folder});
    folder = ffolders{1};
    for i = 1:S
        fname = fnames{i};
        splits = split(fname,' ');
        [bktno,id] = deal(splits{1},splits{2}); assert(bktno == "bucket1");
        impath = sprintf("%s/%s",folder,fname);
        im = double(BlackLevelRead(impath,blacklvl,1));
        orig_im_noisy(:,:,i) = im(cx,cy);
    end

    input_im = ForwardFunc(orig_im_noisy);
    ratio_input_im = intensity_to_ratio(input_im)*255;

    for s = 1:S
        im = double(imread(sprintf("%s/%s_%d.png",stackeddir,scene,s-1)));
        orig_im(:,:,s) = im(cx,cy);
    end

    ratio_orig_im = intensity_to_ratio(orig_im)*255;

    imshow([
        orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4)
        orig_im_noisy(:,:,1) orig_im_noisy(:,:,2) orig_im_noisy(:,:,3) orig_im_noisy(:,:,4)
        input_im(:,:,1) input_im(:,:,2) ratio_input_im(:,:,1) ratio_input_im(:,:,2)
        ratio_orig_im(:,:,1) ratio_orig_im(:,:,2) ratio_orig_im(:,:,3) ratio_orig_im(:,:,4)
    ]/255);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% run RED on ratio/intensity images
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [admm_intensity_im,psnr_intensity,~] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);
    [admm_ratio_im,psnr_ratio,~] = RunADMM_demosaic(ratio_input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,ratio_orig_im);

    im = ratio_to_intensity(ratio_input_im/255);
    im = im.*input_im;
    im = im*2;

    
    imshow([
        orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4)
        im(:,:,1) im(:,:,2) im(:,:,1) im(:,:,2)
    ]/255);
end


function ratio_im = intensity_to_ratio(im)
% Converts image in intensity space to ratio space
%
%   `im`        h x w x 2
%   `ratio_im`  h x w x S
%
    ratio_im = im ./ repmat(sum(im,3),1,1,size(im,3));
    ratio_im(isnan(ratio_im)) = 0.5;
end

function im = ratio_to_intensity(ratio_im)
% Converts image in intensity space to ratio space
%
%   `ratio_im`  h x w x S
%   `im`        h x w x 2
%
    im = ratio_im.*repmat(sum(ratio_im,3),1,1,size(ratio_im,3));
end