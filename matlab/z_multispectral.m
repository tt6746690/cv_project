% Multispectral reconstruction 
%
clc; clear; close all;
ProjectPaths;

%% params 


light_mode = false;
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

scene = 'Bunny_FPS=26.54';
rawimagedir = sprintf('data/MultiSpectral/%s/',scene);
shift = 1;

i=1;

input_im = imread(sprintf('%s/%04d.png',rawimagedir,i-1));
input_im = double(cat(3,input_im(:,1:w),input_im(:,(w+1):(2*w))));
input_im = circshift(input_im,shift,3);
imshow(FlattenChannels(input_im)/255);


[admm_intensity_im,~,~,~] = ADMM(input_im,H,InitEstFunc,params_admm,zeros(h,w,S));

imshow(FlattenChannels(admm_intensity_im)/255);
imshow(admm_intensity_im(:,:,[3 2 1])/255);

% demul_im = InitEstFunc(input_im);
% imshow(FlattenChannels(demul_im)/255);
% color_im = demul_im(:,:,[3 2 1]);
% imshow(color_im/255);