%% phase reconstruction using optimized patterns and ZNCC for decoding
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

% crop the image to remove the borders
[cx,cy] = deal(1:160,10:247);
% #patterns/frames
[S,F] = deal(4,3);
% dimension of input image
[h,w] = deal(numel(cx),numel(cy));
% scale the intensity of image for better visualization 
scaling = 1;
% directory containing demosaiced image
rawimagedir =  "results/alphabet_const_totalexp";
% save images to 
savedir = "results/reconstruction_zncc"; mkdir(savedir);
% black level 
blacklevelpath = "data/blacklevel_all1/blacklevel.mat";
if ~isfile(blacklevelpath)
    ComputeBlackLevel("data/blacklevel_all1",h,w,blacklevelpath);
end
blacklvl = load(blacklevelpath); blacklvl = blacklvl.blacklvl;


% sensor mask
M = BayerMask(h,w);
% two-bucket multiplexing matrix
W = BucketMultiplexingMatrix(S);
% linear map from S patterened image -> two bucket image
[H,B,C] = SubsampleMultiplexOperator(S,M);
% args to RunADMM
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
% spatial frequency of the spatial sinusoids
usedFreq = 4;


% illumination matrix
angle = (0:S-1)*2*pi/S;
Lighting = [ones(1,S);cos(angle);-sin(angle)];
% projector image size
hproj = 684;
% for visualizing disparity 
disparityFunc = @(corres,pos) (corres - 2.7*pos);
dispRange = [50, 160];
[X,Y] = meshgrid(1:w,1:h);


% bounds 
ProjectorInfoFolder = 'mian/CalibrationCode';
Bounds = load(sprintf('%s/%s.mat', ProjectorInfoFolder, 'Bounds'));
Bounds.yErrorLB = Bounds.yErrorLB(cx,cy); %  5;
Bounds.yErrorUB = Bounds.yErrorUB(cx,cy); % - 5;
Bounds.LB = double(Bounds.yErrorLB)*2*pi/hproj;
Bounds.UB = double(Bounds.yErrorUB)*2*pi/hproj;

%% synthetic pattern

PatternCoeff = 0.5 + 0.5*cos(usedFreq*(0:hproj-1)'*2*pi/hproj + linspace(0,3*pi/2, 4));
PatternCoeff = floor(PatternCoeff * 24) / 24;
imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

stackeddir = "data/exp60/organized";
scene = "pillow";

ims = [];

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy);

[phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB);
disparity = disparityFunc((phase*hproj/(2*pi)),Y);

ims = [ims; [orig_im(:,:,1) 255*phase/(2*pi) disparity]];

imshow(ims/255);

%% optimized pattern over `alphabet`


S=4;
PatternCoeff = GeneratePatternMatrix(hproj,S);
imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

stackeddir = "data/alphabet_const_totalexp/organized";
scene = sprintf("optimizedpattern_S=%d",S);

ims = [];

for shift = 0:S-1

    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',shift);

    % not rotational problem ...
    for s = 1:S
        orig_im(:,:,s) = imrotate(orig_im(:,:,s),0.5,'bilinear','crop');
    end

    [phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB);
    disparity = disparityFunc((phase*hproj/(2*pi)),Y);
    
    ims = [ims; [orig_im(:,:,shift+1) 255*phase/(2*pi) disparity]];
end

imshow(ims/255);


%% mannequin spatial sinusoids 

% directory containing demosaiced image
rawimagedir =  "data/mannequin";
% save images to 
savedir = "results/reconstruction_zncc"; mkdir(savedir);

% bounds
Bounds = load('data/mannequin/Bounds.mat');
Bounds.yErrorLB = Bounds.yErrorLB(cx,cy);
Bounds.yErrorUB = Bounds.yErrorUB(cx,cy);
Bounds.LB = double(Bounds.yErrorLB)*2*pi/hproj;
Bounds.UB = double(Bounds.yErrorUB)*2*pi/hproj;



%% find correct shifts for spatial sinusoids: shift=1

for usedFreq = [4 7]
ims = [];
for shift = 0:S-1

PatternCoeff = 0.5 + 0.5*cos(usedFreq*(0:hproj-1)'*2*pi/hproj + linspace(0,3*pi/2, 4));
PatternCoeff = floor(PatternCoeff * 24) / 24;
imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

stackeddir = sprintf('%s/organized',rawimagedir);
scene = sprintf('Freq%d',usedFreq);


orig_im = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',shift);


[phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB);
disparity = disparityFunc((phase*hproj/(2*pi)),Y);

ims = [ims; [orig_im(:,:,1) 255*phase/(2*pi) disparity]];


end
imwrite(uint8(ims),sprintf('%s/find_shift_Freq%d.png',savedir,usedFreq));
end

%% zncc on optimized pattern 

S=7;
patternMatrix = load('data/mannequin/PatternMat.mat');
patternMatrix = patternMatrix.patternMatrix;
PatternCoeff = zeros(hproj,S);
PatternCoeff(1:size(patternMatrix,1),1:end) = patternMatrix;
imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

stackeddir = sprintf('%s/organized',rawimagedir);
scene = 'optimized_pattern';
ims = [];

for shift = 0:S-1

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',shift);

% not rotational problem ...
for s = 1:S
    orig_im(:,:,s) = imrotate(orig_im(:,:,s),0.5,'bilinear','crop');
end

[phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB);
disparity = disparityFunc((phase*hproj/(2*pi)),Y);

ims = [ims; [orig_im(:,:,shift+1) 255*phase/(2*pi) disparity]];

end

imshow(ims/255);
imwrite(uint8(ims),sprintf('%s/find_shift_%s.png',savedir,scene));
