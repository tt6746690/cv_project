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
% dataset
dataset_alphabet = SceneNames("alphabet");
% directory containing the raw noisy images
rawimagedir =  "data/alphabet";
% directory containing groundtruth images
stackeddir = sprintf("%s/organized",rawimagedir);
% save images to 
savedir = "results/spatialspectral"; mkdir(savedir);
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
scene = "alphabet";

%% 
%

Ss = [4 5 6 7];
m = {}; iter = 1;

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

    % inputs_holder = zeros(size(orig_im));
    % inputs_holder(:,:,1:2) = input_im; 
    % inputs_holder(:,:,3:4) = input_ratio_im;
    % imshow(FlattenChannels(orig_im,orig_ratio_im,orig_noisy_im,inputs_holder)/255);

    %% Run RED

    % 1: admm+tnrd in intensity space
    [admm_intensity_im,psnr_intensity,~] = RunADMM_demosaic(input_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm,orig_im);

    % 2. admm+tnrd in ratio space
    [admm_ratio_im,psnr_ratio,~] = RunADMM_demosaic(input_ratio_im,ForwardFunc,BackwardFunc,InitEstFunc,input_sigma,params_admm_ratio,orig_ratio_im);
    
    % 3: admm+tnrd ratio images multiplied by total `input_im` intensity
    ratio_mult_inputsum_im = admm_ratio_im/255;
    ratio_mult_inputsum_im = RatioToIntensity(ratio_mult_inputsum_im,sum(input_im,3));
    psnr_ratio_mult_inputsum = ComputePSNR(orig_im,ratio_mult_inputsum_im);

    % 4: admm+tnrd ratio images multiplied by denoiseed (by tnrd) total `input_im` intensity
    denoised_input_im = Denoiser(sum(input_im,3),params_admm.effective_sigma,"tnrd");
    ratio_mult_inputsum_denoised_im = admm_ratio_im/255;
    ratio_mult_inputsum_denoised_im = RatioToIntensity(ratio_mult_inputsum_denoised_im,denoised_input_im);
    psnr_ratio_mult_inputsum_denoised = ComputePSNR(orig_im,ratio_mult_inputsum_denoised_im);

    fprintf("psnr_intensity                     %.4f\n",psnr_intensity);
    fprintf("psnr_ratio_mult_inputsum           %.4f\n",psnr_ratio_mult_inputsum);
    fprintf("psnr_ratio_mult_inputsum_denoised  %.4f\n",psnr_ratio_mult_inputsum_denoised);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save images
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    ims = scaling*FlattenChannels(orig_im,orig_ratio_im,admm_intensity_im,admm_ratio_im,ratio_mult_inputsum_im,ratio_mult_inputsum_denoised_im);
    imshow(ims/255);
    imwrite(uint8(ims),sprintf("%s/%s%d.png",savedir,scene,S));

    data.S = S;
    data.psnr_intensity = psnr_intensity;
    data.psnr_ratio_mult_inputsum   = psnr_ratio_mult_inputsum;
    data.psnr_ratio_mult_inputsum_denoised  = psnr_ratio_mult_inputsum_denoised;

    m{iter} = data;
    iter = iter + 1;
end

save(sprintf('%s/spatialspectral.mat',savedir),'m');


%% Plots 
% 


m = load(sprintf('%s/spatialspectral.mat',savedir));
m = m.m;
nx = numel(Ss);

psnrs = zeros(3,nx);
for i = 1:nx
    data = m{i};
    psnrs(1,i) = data.psnr_intensity;
    psnrs(2,i) = data.psnr_ratio_mult_inputsum;
    psnrs(3,i) = data.psnr_ratio_mult_inputsum_denoised;
end

plot(1:nx,psnrs(1,:),'DisplayName',"intensity"); hold on;
plot(1:nx,psnrs(2,:),'DisplayName','ratio multiplied with inputsum'); hold on;
plot(1:nx,psnrs(3,:),'DisplayName','ratio multiplied with denoised inputsum'); hold on;
set(gca,'xtick',1:nx,'xticklabel',Ss);
legend();
xlabel("#Subframes")
ylabel("PSNR")
title("Spatial-spectral trade-off");
saveas(gcf,sprintf("%s/spatialspectraltradeoff.png",savedir));
hold off;
