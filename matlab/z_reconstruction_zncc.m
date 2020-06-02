%% phase reconstruction using optimized patterns and ZNCC for decoding
%       
clc; clear; close all;
ProjectPaths;

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
Bounds.LB = Bounds.yErrorLB(cx,cy); %  5;
Bounds.UB = Bounds.yErrorUB(cx,cy);

expandby = 0; shiftby = 100;
Bounds.UB = min(shiftby + Bounds.yErrorLB(cx,cy) + expandby,hproj);
Bounds.LB = max(shiftby + Bounds.yErrorUB(cx,cy) - expandby,0);

%% sinusoidal patterns 

PatternCoeff = 0.5 + 0.5*cos(usedFreq*(0:hproj-1)'*2*pi/hproj + linspace(0,3*pi/2, 4));
PatternCoeff = floor(PatternCoeff * 24) / 24;
imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

% 
% S = 4;
% pattern = zeros(hproj,1,S);
% for s = 1:S
% for i = 1:24
%     im = imread(sprintf('./mian/Patterns/ProjectorPatterns/Sinusoids-freq=04_bins=24_subframes=04/Freq=004_pattern_%03d.bmp',i-1));
%     pattern(:,s) = im(:,1);
% end
% end
% imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));
%

stackeddir = "data/exp60/organized";
scene = "pillow";
% stackeddir = "data/alphabet_const_totalexp/organized";
% scene = sprintf("sinusoidalpattern_S=%d",S);

ims = [];

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy);

[phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB);
disparity = disparityFunc(phase,Y);

ims = [ims; [orig_im(:,:,1) mat2gray(phase)*255 mat2gray(disparity)*255]];

imshow(ims/255);

%% optimized pattern over `alphabet`


S=4;
PatternCoeff = GeneratePatternMatrix(hproj,S);
imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

stackeddir = "data/alphabet_const_totalexp/organized";
scene = sprintf("optimizedpattern_S=%d",S);

ims = [];

Ps = perms(1:S);

for i = 1:size(perms(1:S),1)
    P = Ps(i,:);

    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy);
    orig_im = orig_im(:,:,P);
    orig_ratio_im = orig_ratio_im(:,:,P);
    
%     % not rotational problem ...
     for s = 1:S
         orig_im(:,:,s) = imrotate(orig_im(:,:,s),0.2,'bilinear','crop');
     end

    [phase,zncc,I] = DecodeZNCC(orig_ratio_im,PatternCoeff,Bounds.LB,Bounds.UB);
    disparity = disparityFunc(phase,Y);
    
    ims = [ims; [orig_im(:,:,1) orig_ratio_im(:,:,1) mat2gray(phase)*255 mat2gray(disparity)*255]];
    
    
end


repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])

imshow([])

imwrite(uint8(ims),sprintf("%s/chart_all_permutation.png",savedir));
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

expandby = 0; shiftby = 0;
Bounds.UB = min(shiftby + Bounds.yErrorUB + expandby,hproj);
Bounds.LB = max(shiftby + Bounds.yErrorLB - expandby,0);

plot(Bounds.LB(:,1)); hold on;
plot(Bounds.UB(:,1)); hold off;

% find correct shifts for spatial sinusoids: shift=1

S = 4;
W = BucketMultiplexingMatrix(S);


for usedFreq = [4 7]
ims = [];
for shift = 0:S-1
PatternCoeff = 0.5 + 0.5*cos(usedFreq*(0:hproj-1)'*2*pi/hproj + linspace(0,3*pi/2, 4));
PatternCoeff = floor(PatternCoeff * 24) / 24;
% imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

stackeddir = sprintf('%s/organized',rawimagedir);
scene = sprintf('Freq%d',usedFreq);

orig_im = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',shift);


% works!
[phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB);

% works !
% [~,~,phase] = DecodePhaseShiftWithDepthBound(orig_im,W,Bounds.LB,Bounds.UB,hproj,usedFreq);
disparity = disparityFunc(phase,Y);

ims = [ims; [orig_im(:,:,1) 255*mat2gray(phase) 255*mat2gray(disparity)]];


end
% imwrite(uint8(ims),sprintf('%s/find_shift_Freq%d_decode_mps.png',savedir,usedFreq));
imwrite(uint8(ims),sprintf('%s/find_shift_Freq%d_decode_zncc.png',savedir,usedFreq));

end

%%

% true_disparity = disparity;

%% zncc on optimized pattern 

S=7;
patternMatrix = load('data/mannequin/PatternMat.mat');
patternMatrix = patternMatrix.patternMatrix;
PatternCoeff = zeros(hproj,S);
PatternCoeff(1:size(patternMatrix,1),1:end) = patternMatrix;
PatternCoeff = floor(PatternCoeff * 24) / 24;
% imshow(FlattenChannels(repmat(reshape(PatternCoeff,hproj,1,S),[1 w 1])));

stackeddir = sprintf('%s/organized',rawimagedir);
scene = 'optimized_pattern';
ims = [];

xy = []


for rotateby = -0.1:0.01:0.1
 
[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',0);

for s = 1:S
    orig_ratio_im(:,:,s) = imrotate(orig_ratio_im(:,:,s),rotateby,'bilinear','crop');
end  

[phase,zncc,I] = DecodeZNCC(orig_ratio_im,PatternCoeff,Bounds.LB,Bounds.UB,'NPixelNeighbors',3);
disparity = disparityFunc(phase,Y);
    
ims = [ims; [orig_ratio_im(:,:,shift+1) 255*mat2gray(phase) 255*mat2gray(disparity)]];

PSNR = ComputePSNR(true_disparity,disparity);

xy = [xy; [rotateby,PSNR]]

end

plot(xy(:,1),xy(:,2));

imshow(ims/255);
imwrite(uint8(ims),sprintf('%s/decode_zncc_rotate_S=%d_%s.png',savedir,S,scene));

%% pick n pixel neighborhood

rotateby = 0.01;
xy = []
ims = []

for NPixelNeighbors = [1,3,5]
 
[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',0);

for s = 1:S
    orig_ratio_im(:,:,s) = imrotate(orig_ratio_im(:,:,s),rotateby,'bilinear','crop');
end  

[phase,zncc,I] = DecodeZNCC(orig_ratio_im,PatternCoeff,Bounds.LB,Bounds.UB,'NPixelNeighbors',NPixelNeighbors);
disparity = disparityFunc(phase,Y);
    
ims = [ims; [orig_ratio_im(:,:,shift+1) 255*mat2gray(phase) 255*mat2gray(disparity)]];

PSNR = ComputePSNR(true_disparity,disparity);

xy = [xy; [NPixelNeighbors,PSNR]]
end

plot(xy(:,1),xy(:,2));
imwrite(uint8(ims),sprintf('%s/decode_zncc_npixelneighbor_S=%d_%s.png',savedir,S,scene));

%% use orig_im or ratio_im (about the same)


rotateby = 0.01;
NPixelNeighbors = 3;

ims = [];

[orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',0);

for s = 1:S
    orig_im(:,:,s) = imrotate(orig_im(:,:,s),rotateby,'bilinear','crop');
    orig_ratio_im(:,:,s) = imrotate(orig_ratio_im(:,:,s),rotateby,'bilinear','crop');
end  


[phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB,'NPixelNeighbors',NPixelNeighbors);
disparity = disparityFunc(phase,Y);
PSNR = ComputePSNR(true_disparity,disparity)
ims = [ims; [orig_im(:,:,1) 255*mat2gray(phase) 255*mat2gray(disparity)]];


[phase,zncc,I] = DecodeZNCC(orig_ratio_im,PatternCoeff,Bounds.LB,Bounds.UB,'NPixelNeighbors',NPixelNeighbors);
disparity = disparityFunc(phase,Y);
PSNR = ComputePSNR(true_disparity,disparity)
ims = [ims; [orig_ratio_im(:,:,1) 255*mat2gray(phase) 255*mat2gray(disparity)]];

imshow(ims/255);
imwrite(uint8(ims),sprintf('%s/decode_zncc_useratio_S=%d_%s.png',savedir,S,scene));

% 
znccr = reshape(zncc,h*w,[]);

for i = 1:50:h
    plot(znccr(i+int8(w/2),:)); hold on;
end
xlim([0 hproj]);
hold off;


%% projector pattern quantization  (does not matter)...
NPixelNeighbors=3;

patternMatrix = load('data/mannequin/PatternMat.mat');
patternMatrix = patternMatrix.patternMatrix;
PatternCoeff = zeros(hproj,S);
PatternCoeff(1:size(patternMatrix,1),1:end) = patternMatrix;
PatternCoeff = floor(PatternCoeff * 24) / 24;

ims = [];
xy = [];

for i = 10:30
    PatternCoeff = zeros(hproj,S);
    PatternCoeff(1:size(patternMatrix,1),1:end) = patternMatrix;
    PatternCoeff = floor(PatternCoeff * i) / i;
    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy);
    [phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB,'NPixelNeighbors',NPixelNeighbors);
    disparity = disparityFunc(phase,Y);
    PSNR = ComputePSNR(true_disparity,disparity);
    ims = [ims; [orig_im(:,:,1) 255*mat2gray(phase) 255*mat2gray(disparity)]];
    xy = [xy; [i PSNR]];
end

plot(xy(:,1),xy(:,2));
imshow(ims/255);
imwrite(uint8(ims),sprintf('%s/decode_zncc_quantization_S=%d_%s.png',savedir,S,scene));

%% have grountruth (asnccaccGC same as DecodeZNCC)


expandby = 100; shiftby = -200;
Bounds.LB = max(shiftby + Bounds.yErrorLB - expandby,0);
Bounds.UB = min(shiftby + Bounds.yErrorUB + expandby,hproj);
mesh(Bounds.UB); hold on; mesh(Bounds.LB);


PatternCoeff = zeros(hproj,S);
PatternCoeff(1:size(patternMatrix,1),1:end) = patternMatrix;
% [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy);
orig_im = repmat(reshape(PatternCoeff(100:100+h-1,:),h,[],S),[1 w 1]);

[phase,zncc,I] = DecodeZNCC(orig_im,PatternCoeff,Bounds.LB,Bounds.UB,'NPixelNeighbors',NPixelNeighbors);
disparity = disparityFunc(phase,Y);

imshow([FlattenChannels(orig_im) mat2gray(phase)])

% PSNR = ComputePSNR(true_disparity,disparity);

[idxnp,scores] = asnccaccGC(double(PatternCoeff), orig_im, Bounds.LB, Bounds.UB);
disparity2 = disparityFunc(idxnp,Y);

imshow([mat2gray(phase) mat2gray(idxnp) mat2gray(disparity) mat2gray(disparity2)])

