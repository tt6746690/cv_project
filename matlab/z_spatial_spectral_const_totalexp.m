% Investivagate trade-off between spectral vs. spatial resolution 
%       a dataset of resolution chart / alphabet
%       shot under {4,5,6,7} 
clc; clear; close all;
ProjectPaths;

%% Parameters
%
% crop the image to remove the borders
[cx,cy] = deal(1:160,10:247);
% dimension of input image
[h,w] = deal(176,288);
[h,w] = deal(numel(cx),numel(cy));
% scale the intensity of image for better visualization 
scaling = 1.5;
% directory containing the raw noisy images
rawimagedir =  "data/alphabet_const_totalexp";
% directory containing groundtruth images
stackeddir = sprintf("%s/organized",rawimagedir);
% save images to 
savedir = "results/spatialspectral_const_totalexp"; mkdir(savedir);
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
mask_type = "toeplitz";
% scene 
scene = "sinusoidalpattern_S=";
% projector height
hproj = 684;
% for visualizing disparity 
disparityFunc = @(corres,pos) (corres - 2.7*pos);
dispRange = [50, 160];
% bounds 
Bounds = load(sprintf('mian/CalibrationCode/%s.mat', 'Bounds'));
Bounds.yErrorLB = Bounds.yErrorLB(cx,cy); %  5;
Bounds.yErrorUB = Bounds.yErrorUB(cx,cy); % - 5;
tempshift = 1.2984; tempshift = 0.25;
Bounds.LB = double(Bounds.yErrorLB)*2*pi/hproj + tempshift;
Bounds.UB = double(Bounds.yErrorUB)*2*pi/hproj + tempshift;

[X,Y] = meshgrid(1:w,1:h);

%% 
%

Ss = [4 5 6 7];

m = {}; iter = 1;

%%

for s = 1:numel(Ss)
    [S,F] = deal(Ss(s),Ss(s)-1);
    

    M = SubsamplingMask(mask_type,h,w,F);
    W = BucketMultiplexingMatrix(S);
    [H,B,C] = SubsampleMultiplexOperator(S,M);
    ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
    BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
    InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, 'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
    params_admm = GetDemosaicDemultiplexParams(light_mode);
    params_admm_ratio = GetDemosaicDemultiplexParams(light_mode);

    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s%d",stackeddir,scene,S),h,w,S,'CropX',cx,'CropY',cy);
    [input_im,input_ratio_im,orig_noisy_im] = ReadInputIm(sprintf("%s/%s%d",rawimagedir,scene,S),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'ForwardFunc',ForwardFunc);

    
    imholder = zeros(h,w,S);
    imholder(:,:,1:2) = input_im;
    imholder(:,:,3:4) = input_ratio_im;
    imshow(FlattenChannels(orig_im,orig_ratio_im,orig_noisy_im,imholder)/255);


    %% Run RED

    % 1: admm+tnrd in intensity space
    [admm_intensity_im,psnr_intensity,~] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);

    % 2. admm+tnrd in ratio space
    [admm_ratio_im,psnr_ratio,~] = RunADMM_demosaic(input_ratio_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm_ratio,orig_ratio_im);
    ratio_mult_inputsum_im = admm_ratio_im/255;
    ratio_mult_inputsum_im = RatioToIntensity(ratio_mult_inputsum_im,sum(input_im,3));
    psnr_ratio_mult_inputsum = ComputePSNR(orig_im,ratio_mult_inputsum_im);
    
    fprintf("psnr_intensity                     %.4f\n",psnr_intensity);
    fprintf("psnr_ratio_mult_inputsum           %.4f\n",psnr_ratio_mult_inputsum);
    
    %% Photemetric stereo

    [orig_im_albedo,~,orig_im_phase] = SLTriangulation(orig_im,W,Bounds,4);
    [intensity_im_albedo,~,intensity_im_phase] = SLTriangulation(admm_intensity_im,W,Bounds,4);
    [ratio_im_albedo,~,ratio_im_phase] = SLTriangulation(ratio_mult_inputsum_im,W,Bounds,4);
    
    orig_im_disparity = disparityFunc((orig_im_phase*hproj/(2*pi)),Y);
    intensity_im_disparity = disparityFunc((intensity_im_phase*hproj/(2*pi)),Y);
    ratio_im_disparity = disparityFunc((ratio_im_phase*hproj/(2*pi)),Y);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save images
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    ims1 = scaling*FlattenChannels(orig_im,orig_ratio_im,admm_intensity_im,admm_ratio_im,ratio_mult_inputsum_im);
    ims2 = zeros(3*h,w*S);
    ims2(:,1:3*w) = [
        orig_im_albedo,255*orig_im_phase/(2*pi),orig_im_disparity;...
        intensity_im_albedo,255*intensity_im_phase/(2*pi),intensity_im_disparity; ...
        ratio_im_albedo,255*ratio_im_phase/(2*pi),ratio_im_disparity];
    ims = [ims1;ims2];
    imshow(ims/255);
    imwrite(uint8(ims),sprintf("%s/%s%d.png",savedir,scene,S));

    data.S = S;
    data.psnr_intensity = psnr_intensity;
    data.psnr_ratio_mult_inputsum   = psnr_ratio_mult_inputsum;

    m{iter} = data;
    iter = iter + 1;
end

save(sprintf('%s/spatialspectral.mat',savedir),'m');


%% Plots 
% 


m = load(sprintf('%s/spatialspectral.mat',savedir));
m = m.m;
nx = numel(Ss);

psnrs = zeros(2,nx);
for i = 1:nx
    data = m{i};
    psnrs(1,i) = data.psnr_intensity;
    psnrs(2,i) = data.psnr_ratio_mult_inputsum;
end

plot(1:nx,psnrs(1,:),'DisplayName',"intensity"); hold on;
plot(1:nx,psnrs(2,:),'DisplayName','ratio multiplied with inputsum'); hold on;
set(gca,'xtick',1:nx,'xticklabel',Ss);
legend();
xlabel("#Subframes")
ylabel("PSNR")
title("Spatial-spectral trade-off (constant exposure time)");
saveas(gcf,sprintf("%s/spatialspectraltradeoff.png",savedir));
hold off;