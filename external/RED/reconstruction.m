%% Structured light reconstruction for disparity map
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

% scene
scene = "shoe";
% crop the image to remove the borders
[cx,cy] = deal(1:160,10:247);
% #patterns/frames
[S,F] = deal(4,3);
% dimension of input image
[h,w] = deal(numel(cx),numel(cy));
% scale the intensity of image for better visualization 
scaling = 2;
% directory containing demosaiced image
rawimagedir =  "results/ratio";
% directory containing groundtruth images
stackeddir = "data/exp60/organized";
% save images to 
savedir = "results/structured_light"; mkdir(savedir);

% sensor mask
M = BayerMask(h,w);
% two-bucket multiplexing matrix
W = BucketMultiplexingMatrix(S);
% linear map from S patterened image -> two bucket image
[H,B,C] = SubsampleMultiplexOperator(S,M);
% args to RunADMM
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);


% illumination matrix
angle = (0:S-1)*2*pi/S;
Lighting = [ones(1,S);cos(angle);-sin(angle)];
% projector image size
hproj = 684;
% for visualizing disparity 
disparityFunc = @(corres,pos) (corres - 2.7*pos);
dispRange = [50, 160];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read in image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tiled = double(imread(sprintf("%s/%s.png",rawimagedir,scene)));

% groundtruth image
gt_im = reshape(tiled(1:h,:),h,w,[]);
imshow([gt_im(:,:,1) gt_im(:,:,2) gt_im(:,:,3) gt_im(:,:,4)]/255);

bkt1_lighting = Lighting * W(1:3,:)';
bkt2_lighting = Lighting * W(4:6,:)';
L = [bkt1_lighting, bkt2_lighting];

im = reshape(gt_im,[],S)*W'
g =  im/ L;

[phase, albedo] = extractPhase(g');
phase = reshape(phase', [h w]);
albedo = reshape(albedo', [h w]);
disparity = phase*hproj / (2*pi);


% visualization 
imshow(imresize(phase / (2*pi), 10, 'nearest'));

[X,Y] = meshgrid(1:w,1:h);
imagesc(disparityFunc((disparity), Y));