%% evaluate performance of RED/demosic on images with realistic noise
%
%       - red has better performance than demosaic, if use tndr denoiser!
%
clc; clear; close all;
addpath(genpath('./tnrd_denoising/'));
addpath(genpath('./minimizers/'));
addpath(genpath('./parameters/'));
addpath(genpath('./helper_functions/'));
addpath(genpath('./test_images/'));
addpath(genpath("./mian/helperFunctions/Camera"));
addpath(genpath("./mian/helperFunctions/ASNCC"));
addpath(genpath("./mian/helperFunctions/Algorithms"));

%% Parameters

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
scene = "flower";
% dataset
dataset_exp60 = SceneNames("exp60");
% directory containing the raw noisy images
rawimagedir =  "data/exp60";
% directory containing groundtruth images
stackeddir = "data/exp60/organized";
% save images to 
savedir = "results/realisticnoise"; mkdir(savedir);
% black level 
blacklevelpath = "data/blacklevel_all1/blacklevel.mat";
if ~isfile(blacklevelpath)
    blackimsdir = "data/blacklevel_all1";
    ComputeBlackLevel(blackimsdir,h,w,blacklevelpath);
end
blacklvl = load(blacklevelpath);
blacklvl = blacklvl.blacklvl;


% RED less #iterations
light_mode = true;
% sigmas 
input_sigma = 1;
% mask 
M = BayerMask(h,w);
% two-bucket multiplexing matrix
W = BucketMultiplexingMatrix(S);
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

%% loop over scenes, several noisy inputs 

m = containers.Map;

for scene = dataset_exp60

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% read in image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    orig_im_noisy = zeros(h,w,S);
    orig_im = zeros(h,w,S);
    input_im = zeros(h,w,2);

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

    for s = 1:S
        im = double(imread(sprintf("%s/%s_%d.png",stackeddir,scene,s-1)));
        orig_im(:,:,s) = im(cx,cy);
    end

    % imshow([
    %     orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4)
    %     orig_im_noisy(:,:,1) orig_im_noisy(:,:,2) orig_im_noisy(:,:,3) orig_im_noisy(:,:,4)
    %     input_im(:,:,1) input_im(:,:,2) zeros(h,w) zeros(h,w)
    % ]/255);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% run red and demosaic on cropped image 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [admm_im,psnr_admm,~] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);

    BayerDemosaicDemultiplex = InitialEstimateFunc("bayerdemosaic",h,w,F,S, ...
            'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
    prev_im = BayerDemosaicDemultiplex(input_im);
    psnr_prev = ComputePSNR(orig_im, prev_im);

    fprintf("psnr (bayerdemosaic)   %.3f\n",psnr_prev);
    fprintf("psnr (admm)            %.3f\n",psnr_admm);

    for s = 1:S
        imwrite(uint8(scaling*prev_im(:,:,s)),sprintf("%s/%s_%d_prev.png",savedir,scene,s));
        imwrite(uint8(scaling*admm_im(:,:,s)),sprintf("%s/%s_%d_admm.png",savedir,scene,s));
        imwrite(uint8(scaling*orig_im(:,:,s)),sprintf("%s/%s_%d_orig.png",savedir,scene,s));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save psnrs 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    data.psnr_prev = psnr_prev;
    data.psnr_admm = psnr_admm;
    m(scene) = data;
end


save(sprintf('%s/psnrs.mat',savedir),'m');

%% read in map just to check its 

m = load(sprintf('%s/psnrs.mat',savedir));
m = m.m;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experiment on hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ims = [
%     orig_im(:,:,1) prev_im(:,:,1) admm_im(:,:,1)
% ];
% figure; imshow(2*ims/255);


% hyperparameter guess ...
% shoe
% psnr (bayerdemosaic)   44.107

% fix lambda=0.008

% beta: 0.500
% psnr (admm)            43.616
% beta: 1.000
% psnr (admm)            43.067
% beta: 1.500
% psnr (admm)            42.817

% beta: 1.5 vary lambda

% lambda: 0.001
% psnr (admm)            42.368
% lambda: 0.010
% psnr (admm)            42.931
% lambda: 0.030
% psnr (admm)            43.733
% lambda: 0.100
% psnr (admm)            44.472
% lambda: 0.300
% psnr (admm)            44.532
% lambda: 1.000
% psnr (admm)            43.946
% lambda: 10.000
% psnr (admm)            41.004


% test: (beta=1.5, lambda=0.3)
% on flower, same level of performance
% psnr (bayerdemosaic)   44.796
% psnr (admm)            44.763
% psnr (admm)            45.155


% on shoe, 0.4 better
% psnr (bayerdemosaic)   44.107
% psnr (admm+medfilter)  44.532
% psnr (admm+tnrd)       45.135