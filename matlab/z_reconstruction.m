%% Structured light reconstruction for disparity map
clc; clear; close all;
ProjectPaths;

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
Bounds.LB = Bounds.yErrorLB;
Bounds.UB = Bounds.yErrorUB;

tempshift = 1.2984;





%% [min(min(Bounds.UB-Bounds.LB)), max(max(Bounds.UB-Bounds.LB))]


% for tempshift = -pi:0.1:pi
for tempshift = (.8584-0.5):0.02:(.8584+0.5)
    tempshift
    % .8584 
    Bounds.UB = Bounds.yErrorUB + tempshift*684/(2*pi);
    Bounds.LB = Bounds.yErrorLB + tempshift*684/(2*pi);
    Bounds.UB = double(Bounds.UB)*2*pi/hproj;
    Bounds.LB = double(Bounds.LB)*2*pi/hproj;
    
    [max(max(Bounds.LB)) max(max(Bounds.UB))]


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Read in image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    tiled = double(imread(sprintf("%s/%s.png",rawimagedir,scene)));

    % groundtruth image
    gt_im = reshape(tiled(1:h,:),h,w,[]);
    % imshow([gt_im(:,:,1) gt_im(:,:,2) gt_im(:,:,3) gt_im(:,:,4)]/255);

    [albedo,wrapped_phase,phase] = SLTriangulation(gt_im,W,Bounds,4);


    % imshow([wrapped_phase phase]/(2*pi));
    % pause;

    disparity = phase*hproj / (2*pi);
    TwoBuckDisparity = disparityFunc((disparity), Y);
    imagesc(TwoBuckDisparity);
    pause;
end


