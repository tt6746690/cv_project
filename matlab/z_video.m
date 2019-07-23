% demosaic+demultiplex a video sequence shot from c2b camera
clc; clear; close all;
ProjectPaths;

%% parameters 

[S,F] = deal(4,3);
scaling=3;

% crop the image to remove the borders
[cx,cy] = deal(1:160,11:248);
% dimension of input image
[h,w] = deal(176,288);
[h,w] = deal(numel(cx),numel(cy));
light_mode = false;

videodir = "../data/datafolder/Hand3";
outputdir = sprintf('%s_video',videodir); mkdir(outputdir);
blackLevel = load("./mian/CalibrationCode/BlackIms.mat");
blackLvl = blackLevel.blackLvl;

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
        im_fullsize = double(BlackLevelRead(sprintf('%s/%s',folder,name),blackLvl,1));
        im(:,:,1) = im_fullsize(cx,cy);
    elseif bkt == 'bucket2'
        im_fullsize = double(BlackLevelRead(sprintf('%s/%s',folder,name),blackLvl,2));
        im(:,:,2) = im_fullsize(cx,cy);
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

mkdir(sprintf("%s/iter_ims/",outputdir));

M = SubsamplingMask("bayer",h,w,F);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("bayerdemosaic",h,w,F,S, 'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetDemosaicDemultiplexParams(light_mode);

for i = 100:(size(listingnames,2)/2)
    k = ks{i};
    input_im = m(k);
    [h,w,~] = size(input_im);

    tic;
    params_admm.outer_iters = 10;
    params_admm.v_update_method = "fixed_point";
    [admm_im,psnr,ssim,~,iter_ims] = ADMM(input_im,H,InitEstFunc,params_admm,zeros(h,w,S));
    time_red = toc;
    for i = 1:size(iter_ims,3)
        imwrite(uint8(iter_ims(:,:,i)),sprintf("%s/iter_ims/%d.png",i));
    end

    tic;
    demosaic_im = InitEstFunc(input_im);
    time_demosaic = toc;

    ims = FlattenChannels(demosaic_im,admm_im);
    imwrite(uint8(3*ims),sprintf("%s/%s",outputdir,k));

    v.input_im = input_im;
    v.admm_im = admm_im;
    v.demosaic_im = demosaic_im;
    v.time_red = time_red;
    v.time_demosaic = time_demosaic;
    v
    m(k) = v;
    break;
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