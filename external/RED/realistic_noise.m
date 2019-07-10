%% evaluate performance of RED/demosic on images with realistic noise
%
%       - red has better performance than demosaic, if use tndr denoiser!
%       - no need to tune hyperparameters, tuning them might
%           - different initial guess makes method not convergent
%           - worse psnr values
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set to 1 if debug, 0 otherwise
short = 0;
% crop the image to remove the borders
[cx,cy] = deal(1:160,10:249);
% [cx,cy]=deal(11:30,11:30);
% #patterns/frames
[S,F] = deal(4,3);
% dimension of input image
[h,w] = deal(176,288);
[h,w] = deal(numel(cx),numel(cy));
% scale the intensity of image for better visualization 
scaling = 2;
% dataset
dataset_exp60 = SceneNames("exp60");
% directory containing the raw noisy images
rawimagedir =  "data/exp60";
% directory containing groundtruth images
stackeddir = "data/exp60/organized";
% save images to 
savedir = "results/realistic_noise"; mkdir(savedir);
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
W = BucketMultiplexingMatrix(S);
% linear map from S patterened image -> two bucket image
[H,B,C] = SubsampleMultiplexOperator(S,M);
% args to RunADMM
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, ...
        'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetSuperResADMMParams(light_mode);

if short == 1
    params_admm.outer_iters = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% loop over scenes, several noisy inputs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    params_admm.denoiser_type = "tnrd";
    [tnrd_admm_im,psnr_tnrd_admm,~] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);
    params_admm.denoiser_type = "medfilter";
    [medf_admm_im,psnr_medf_admm,~] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);

    BayerDemosaicDemultiplex = InitialEstimateFunc("bayerdemosaic",h,w,F,S,'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
    prev_im = BayerDemosaicDemultiplex(input_im);
    psnr_prev = ComputePSNR(orig_im, prev_im);

    fprintf("psnr (bayerdemosaic)   %.3f\n",psnr_prev);
    fprintf("psnr (admm+tnrd)       %.3f\n",psnr_tnrd_admm);
    fprintf("psnr (admm+medfilter)  %.3f\n",psnr_medf_admm);

    ims = [
        orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4)
        prev_im(:,:,1) prev_im(:,:,2) prev_im(:,:,3) prev_im(:,:,4)
        tnrd_admm_im(:,:,1) tnrd_admm_im(:,:,2) tnrd_admm_im(:,:,3) tnrd_admm_im(:,:,4)
        medf_admm_im(:,:,1) medf_admm_im(:,:,2) medf_admm_im(:,:,3) medf_admm_im(:,:,4)
    ]*scaling;
    imwrite(uint8(ims),sprintf("%s/%s.png",savedir,scene));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save psnrs 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    data.psnr_prev = psnr_prev;
    data.tnrd_admm_im = tnrd_admm_im;
    data.medf_admm_im = medf_admm_im;
    m(scene) = data;
end


save(sprintf('%s/psnrs.mat',savedir),'m');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Some plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = load(sprintf('results/realistic_noise_prev/psnrs.mat',savedir));
m = m.m;
ks = keys(m);

psnrs = zeros(2,numel(keys(m)));
for i = 1:size(ks,2)
    k = ks{i};
    psnrs(1,i) = m(k).psnr_prev;
    psnrs(2,i) = m(k).psnr_admm;
end

title("Performance (w.r.t. PSNR) for different objects using previous/admm method");
plot(1:size(ks,2),psnrs(1,:),'DisplayName',"prev"); hold on;
plot(1:size(ks,2),psnrs(2,:),'DisplayName','admm'); hold on;
set(gca,'xtick',1:size(ks,2),'xticklabel',ks);
legend();
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compare medfilter,tnrd,bayerdemosaic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = load(sprintf('results/realistic_noise/psnrs.mat',savedir));
m = m.m;
ks = keys(m);

psnrs = zeros(3,numel(keys(m)));
for i = 1:size(ks,2)
    k = ks{i};
    psnrs(1,i) = m(k).psnr_prev;
    psnrs(2,i) = m(k).psnr_medf_admm;
    psnrs(3,i) = m(k).psnr_tnrd_admm;
end

plot(1:size(ks,2),psnrs(1,:),'DisplayName',"prev"); hold on;
plot(1:size(ks,2),psnrs(2,:),'DisplayName','admm+medianfilter'); hold on;
plot(1:size(ks,2),psnrs(3,:),'DisplayName','admm+tnrd'); hold on;
set(gca,'xtick',1:size(ks,2),'xticklabel',ks);
legend();
xlabel("Scenes")
ylabel("PSNR")
title("Performance (w.r.t. PSNR) for different objects using previous/admm+medianfilter/admm+tnrd method");
saveas(gcf,sprintf("%s/compare_to_prev.png",savedir));
hold off;



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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

savedir = "results/realistic_noise_initialguess_mask"; mkdir(savedir);

mask_types = [
    "bayer"
    "toeplitz"
    "horz"
    "vert"
    "random"
]';

initialguesses = [
    "maxfilter"
    "zeroatunknown"
    "zero"
    "random"
]';

params_admm.denoiser = "tnrd";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experiment on initialguess and mask
%%      note: use default hyperparameter, since otherwise initialguess impact convergence very much, e.g. zero stays at zero
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for scene = dataset_exp60
    m = {}; iter = 1;
    for mask_type = mask_types
        for initialguess = initialguesses


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% RED-specific parameters
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            M = SubsamplingMask(mask_type,h,w,F);
            [H,B,C] = SubsampleMultiplexOperator(S,M);
            ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
            BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
            InitEstFunc = InitialEstimateFunc(initialguess,h,w,F,S, ...
                    'BucketMultiplexingMatrix',W,'SubsamplingMask',M);

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
            
            imshow([
                orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4)
                input_im(:,:,1) input_im(:,:,2) zeros(h,w) zeros(h,w)
            ]*2/255);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% run RED
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            [admm_im,psnr_admm,admm_statistics] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);

            fprintf("scene=%s mask_type=%s initialguess=%s\n",scene,mask_type,initialguess);
            fprintf("psnr (admm)            %.3f\n",psnr_admm);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% save 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            mkdir(sprintf("%s/%s",savedir,scene));

            ims = [
                orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4)
                admm_im(:,:,1) admm_im(:,:,2) admm_im(:,:,3) admm_im(:,:,4)
            ]*scaling;
            imwrite(uint8(ims),sprintf("%s/%s/%s_%s.png",savedir,scene,mask_type,initialguess));

            data.mask_type = mask_type;
            data.initialguess = initialguesses;
            data.M = M;
            data.psnr_admm = psnr_admm;
            data.admm_statistics = admm_statistics;

            m{iter} = data;
            iter = iter + 1;
        end
    end
    save(sprintf('%s/%s.mat',savedir,scene),'m');
end


