% Investigate how to train a denoiser over realistic noise from the camera
%   noise characteristic varies with scene
%                               and intensity of the pixel (black pixel-> no noise)   
%
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
% dataset
dataset_alphabet = SceneNames("alphabet");
% directory containing the raw noisy images
rawimagedir =  "data/alphabet";
% directory containing groundtruth images
stackeddir = sprintf("%s/organized",rawimagedir);
% save images to 
savedir = "results/noise_characteristics"; mkdir(savedir);
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
scene = "alphabet";

[S,F] = deal(4,3);

M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("bayerdemosaic",h,w,F,S, 'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetDemosaicDemultiplexParams(light_mode);
params_admm_ratio = GetDemosaicDemultiplexParams(light_mode);


%% realistic noise

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s%d",stackeddir,scene,S),h,w,S,'CropX',cx,'CropY',cy);
[input_im,input_ratio_im,orig_noisy_im] = ReadInputIm(sprintf("%s/%s%d",rawimagedir,scene,S),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'ForwardFunc',ForwardFunc);

inputs_holder = zeros(size(orig_im));
inputs_holder(:,:,1:2) = input_im; 
inputs_holder(:,:,3:4) = input_ratio_im;
imshow(FlattenChannels(orig_im,orig_ratio_im,orig_noisy_im,inputs_holder)/255);
imshow(FlattenChannels(orig_im,input_im,abs(orig_im-input_im))/255);

%% realistic noise

% videos 
videodir = '../data/datafolder/Hand3';
videoblackLevel = load("./mian/CalibrationCode/BlackIms.mat");
videoblackLvl = videoblackLevel.blackLvl;

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
        im_fullsize = double(BlackLevelRead(sprintf('%s/%s',folder,name),videoblackLvl,1));
        im(:,:,1) = im_fullsize(cx,cy);
    elseif bkt == 'bucket2'
        im_fullsize = double(BlackLevelRead(sprintf('%s/%s',folder,name),videoblackLvl,2));
        im(:,:,2) = im_fullsize(cx,cy);
    else
        warning('not one of bucket 1/2');
    end
    m(id) = im;
end

uncleaned = m;

% do some cleaning ...

cleaned = containers.Map;
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

    cleaned(k) = im;
end

%% 

ks = keys(uncleaned);

for i = 100:numel(ks)
    k = ks{i};
    denoised = Denoiser(cleaned(k),3,'tnrd');
    imshow(FlattenChannels(uncleaned(k),cleaned(k),denoised,abs(uncleaned(k)-denoised),10*abs(cleaned(k)-denoised))/255);
    pause;
    imshow(2*FlattenChannels(InitEstFunc(uncleaned(k)), ...
                             InitEstFunc(cleaned(k)),...
                             cat(3,cleaned(k),zeros(size(cleaned(k)))))/255);

    pause;
    % I = find(uncleaned(k)-cleaned(k)~=0);
    % foo = uncleaned(k)-cleaned(k);
    % foo = foo(I);
    % hist(reshape(foo,[],1),50);
    % pause;
end


%% 

clean_im = zeros(h,w,2)+127;
noise = zeros(size(clean_im));
noise(50,50,1) = 100;
noise(50,50,2) = -100;
noise(50:51,80:81,1) = 100;
noise(50:51,80:81,2) = -100;
noise(50:52,100:102,1) = 100;
noise(50:52,100:102,2) = -100;
input_im = clean_im + noise;


bayerdemosaic_im = InitEstFunc(input_im);
params_admm.outer_iters = 10;
params_admm.denoiser_type = "tnrd";
[admm_intensity_im,psnr_intensity,ssim_out,~] = ADMM(input_im,H,InitEstFunc,params_admm,InitEstFunc(clean_im));

imshow(FlattenChannels(cat(3,input_im,abs(noise)),...
abs(InitEstFunc(noise)),bayerdemosaic_im,admm_intensity_im,...
Denoiser(admm_intensity_im,3,"medfilter"),Denoiser(admm_intensity_im,3,"tnrd"),...
cat(3,ForwardFunc(abs(bayerdemosaic_im-InitEstFunc(clean_im))),ForwardFunc(abs(admm_intensity_im-InitEstFunc(clean_im)))))/255);

