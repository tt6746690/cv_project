% demosaic+demultiplex a video sequence shot from c2b camera
clc; clear; close all;
addpath(genpath('./tnrd_denoising/'));
addpath(genpath('./minimizers/'));
addpath(genpath('./parameters/'));
addpath(genpath('./helper_functions/'));
addpath(genpath('./test_images/'));
addpath(genpath("./mian/helperFunctions/Camera"));
addpath(genpath("./mian/helperFunctions/ASNCC"));
addpath(genpath("./mian/helperFunctions/Algorithms"));

%% parameters 

S=4;
F=3;
[h,w] = deal(176,288);

videodir = '../../data/datafolder/Hand3';
outputdir = sprintf('%s_demosaiced',videodir);
mkdir(outputdir);

W = BucketMultiplexingMatrix(S);

blackLevel = load("./mian/CalibrationCode/BlackIms.mat");
blackLvl = blackLevel.blackLvl;

% arguments requried for previous method
OptimalMaskMatrix = [hadamard(4)];
OptimalMaskMatrix = OptimalMaskMatrix(2:end,:) > 0;
OptimalMaskMatrix = OptimalMaskMatrix(:,[4 1 2 3]);
DemosaicFunction = @(X) demosaic(X, 'bggr');
ASNCC.PatternCoeff = zeros(0,S);
Bounds.LB=5; Bounds.UP=-5;
opts.method='SL';
opts.verbose=false;
opts.interval = 30000;


%% read bucket1+bucket2 pair

listing = dir(sprintf('%s/*.png',videodir));
listingnames = {listing.name};
listingfolder = {listing.folder}; folder = listingfolder{1};

m = containers.Map;
for i = 1:size(listingnames,2)
    name = listingnames{i};
    splits = split(name,'_');
    [bkt,id] = deal(splits{1},splits{2});

    if ~isKey(m,id)
        m(id) = zeros(h,w,2);
    end

    im = m(id);
    if bkt == "bucket1"
        im(:,:,1) = double(BlackLevelRead(sprintf('%s/%s',folder,name),blackLvl,1));
    elseif bkt == 'bucket2'
        im(:,:,2) = double(BlackLevelRead(sprintf('%s/%s',folder,name),blackLvl,2));
    else
        warning('not one of bucket 1/2');
    end
    m(id) = im;
end

% do some cleaning ...

ks = keys(m);
for i = 1:(size(listingnames,2)/2)
    k = ks{i};
    im = m(k);
    assert(sum(im(:,:,1),'all') ~= 0);
    assert(sum(im(:,:,2),'all') ~= 0);

    imavg = im(:,:,1)+im(:,:,2);
    [ratio1,ratio2] = deal(im(:,:,1)./imavg,im(:,:,2)./imavg);
    [im(:,:,1),ratio1] = clean_dataASNCC(im(:,:,1),ratio1);
    [im(:,:,2),ratio2] = clean_dataASNCC(im(:,:,2),ratio2);

    m(k) = im;
end


%% run demosaicing 

save_im = @(im,p) ...
    imwrite(uint8([im(:,:,1) im(:,:,2);im(:,:,3) im(:,:,4)]),p);

for i = 1:(size(listingnames,2)/2)
    k = ks{i};
    input_im = m(k);
    [h,w,~] = size(input_im);

    tic;
    admm_im = red_demosaic_demultiplex(input_im);
    time_red = toc;

    tic;
    demosaic_im = ...
        reshape(...
            reshape( ...
                cat(3, ...
                    rgb2bgr(double(demosaic(uint8(input_im(:,:,1)), 'bggr'))), ...
                    rgb2bgr(double(demosaic(uint8(input_im(:,:,2)), 'bggr')))), ...
                [], 6) ...
            / W', ...
        h,w,S);
    time_demosaic = toc;
% 
%     ims = [input_im(:,:,1) input_im(:,:,2) zeros(h,w) zeros(h,w)
%            admm_im(:,:,1) admm_im(:,:,2) admm_im(:,:,3) admm_im(:,:,4)
%            demosaic_im(:,:,1) demosaic_im(:,:,2) demosaic_im(:,:,3) demosaic_im(:,:,4)];
%     imshow(ims/255);

    save_im(admm_im,sprintf('%s/red_%s',outputdir,k));
    save_im(demosaic_im,sprintf('%s/demosaic_%s',outputdir,k));

    v.input_im = input_im;
    v.admm_im = admm_im;
    v.demosaic_im = demosaic_im;
    v.time_red = time_red;
    v.time_demosaic = time_demosaic;
    v
    m(k) = v;
end

save(sprintf('%s/video.mat',outputdir),'m');
S = load(sprintf('%s/video.mat',outputdir));
m = S.m

%% write to video

vid = VideoWriter(sprintf('%s/video.avi',outputdir),'Uncompressed AVI');
open(vid);
sortedks = sort(keys(m));
for i = 1:size(sortedks,2)
    k = sortedks{i};
    admm_im = m(k).admm_im;
    demosaic_im = m(k).demosaic_im;
    writeVideo(vid,uint8([
        admm_im(:,:,1) admm_im(:,:,2) admm_im(:,:,3) admm_im(:,:,4)
        demosaic_im(:,:,1) demosaic_im(:,:,2) demosaic_im(:,:,3) demosaic_im(:,:,4)
    ]));
end
close(vid);

%%


function out_admm_im = red_demosaic_demultiplex(input_im)

    S=4;
    F=3;
    light_mode = true;
    
    [h,w] = deal(size(input_im,1),size(input_im,2));
    input_sigma = 1;
    M = BayerMask(h,w);
    W = BucketMultiplexingMatrix(S);
    [H,B,C] = SubsampleMultiplexOperator(S,M);
    
    ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
    BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
    InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S,W);

    orig_im = zeros(h,w,S); % not available ...
    params_admm = GetSuperResADMMParams(light_mode);
    [out_admm_im, psnr_admm, admm_statistics] = RunADMM_demosaic(input_im,...
                                                ForwardFunc,...
                                                BackwardFunc,...
                                                InitEstFunc,...
                                                input_sigma,...
                                                params_admm,...
                                                orig_im);
end