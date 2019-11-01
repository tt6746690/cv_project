% Multispectral reconstruction 
%
clc; clear; close all;
ProjectPaths;

%% params 

outdir = 'results/multispectral';
mkdir(outdir);
scaling=2;
light_mode = true;
[S,F] = deal(5,4);
[h,w] =  deal(320,324);
M = SubsamplingMask("tiles",h,w,F,'Tile',[1 2; 3 4]);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S,'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetDemosaicDemultiplexParams(light_mode);

%% 

% scene = 'Bunny_FPS=26.54';
scene = 'Rubiks_FPS=25.83';
rawimagedir = sprintf('data/MultiSpectral/%s/',scene);
shift = 1;

i=292;
i=490;

ims = [];
im = [];

P = perms(1:S);

% for shift = 0:S-1
for j = 1:size(P,1)

perm = P(j,:)

input_im = imread(sprintf('%s/%04d.png',rawimagedir,i-1));
input_im = double(cat(3,input_im(:,1:w),input_im(:,(w+1):(2*w))));
% imshow(FlattenChannels(input_im)/255);

% %% 1: admm+tnrd in intensity space
% [admm_intensity_im,~,~,~] = ADMM(input_im,H,InitEstFunc,params_admm,zeros(h,w,S));


% imshow(FlattenChannels(admm_intensity_im)/255);
% imshow(admm_intensity_im(:,:,[3 2 1])/255);

% %% 2. admm+tnrd in ratio space
% input_ratio_im = IntensityToRatio(input_im);
% [admm_ratio_im,~,~,~] = ADMM(input_ratio_im,H,InitEstFunc,params_admm,zeros(h,w,S));
% ratio_mult_inputsum_im = admm_ratio_im/255;
% ratio_mult_inputsum_im = RatioToIntensity(ratio_mult_inputsum_im,sum(input_im,3));


% imshow(FlattenChannels(ratio_mult_inputsum_im));
% imshow(ratio_mult_inputsum_im(:,:,[3 2 1]));


demul_im = InitEstFunc(input_im);
% demul_im = circshift(demul_im,shift,3);
% demul_im = demul_im(:,:,[2 5 3 4 1]);
demul_im = demul_im(:,:,perm);

% imshow(scaling*FlattenChannels(demul_im)/255);
color_im = demul_im(:,:,[3 2 1]);
% imshow(scaling*color_im/255);

% im = [im;scaling*FlattenChannels(demul_im)];
% ims = [ims;scaling*color_im];

imwrite(uint8(scaling*color_im),sprintf('%s/%d-%d-%d-%d-%d.png',outdir,perm(1),perm(2),perm(3),perm(4),perm(5)));
end

% imshow(im/255);
% imshow(ims/255);