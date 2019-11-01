% Compare performance of ADMM on intensity/ratio images and see 
% how the performance carry over to disparity estimation
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
rawimagedir =  "data/7patterns";
% directory containing groundtruth images
stackeddir = sprintf("%s/organized",rawimagedir);
% save images to 
savedir = "results/spatialspectral_disparity_estimation"; mkdir(savedir);
% black level 
blacklevelpath = "data/blacklevel_all1/blacklevel.mat";
if ~isfile(blacklevelpath)
    ComputeBlackLevel("data/blacklevel_all1",h,w,blacklevelpath);
end
blacklvl = load(blacklevelpath); blacklvl = blacklvl.blacklvl;
% toggle to false for long runs
light_mode = true;
% sigmas 
input_sigma = 1;
% sensor mask type 
mask_type = "toeplitz";
% scene 
scene = "sponge";
% projector height
hproj = 684;
% for visualizing disparity 
disparityFunc = @(corres,pos) (corres - 2.7*pos);
dispRange = [50, 160];
% bounds 
Bounds = load(sprintf('mian/CalibrationCode/%s.mat', 'Bounds'));
Bounds.yErrorLB = Bounds.yErrorLB(cx,cy); %  5;
Bounds.yErrorUB = Bounds.yErrorUB(cx,cy); % - 5;
tempshift = 1.2984; tempshift = 0.25; tempshift = -0.34;
Bounds.LB = double(Bounds.yErrorLB)*2*pi/hproj + tempshift;
Bounds.UB = double(Bounds.yErrorUB)*2*pi/hproj + tempshift;

[X,Y] = meshgrid(1:w,1:h);


%% Finding the correct shift manually, ...

S = 7;
F = S-1;
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);

[scenes,shifts] = SceneNames("7patterns");
ims = zeros(numel(scenes)*h,w*(S+1));

for k = 1:numel(scenes)

    scene = scenes(k);
    shift = shifts(k);

    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',shift);

    [orig_im_albedo,~,orig_im_phase] = SLTriangulation(orig_im,W,Bounds,4);
    orig_im_disparity = disparityFunc((orig_im_phase*hproj/(2*pi)),Y);

    ims(((k-1)*h+1):k*h,:) = [orig_im_disparity FlattenChannels(orig_im)];
end

imwrite(uint8(ims),sprintf("%s/find_correct_ordering.png",savedir));

%% See if subsmaple the patterns worked ...

S = 7;
F = S-1;
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
params_admm = GetDemosaicDemultiplexParams(light_mode);
params_admm_ratio = GetDemosaicDemultiplexParams(light_mode);

take_indices = containers.Map( ...
    {4,5,6,7}, ...
    {
        [1 3 5 7],
        [1 3 4 5 7],
        [1 2 3 4 5 7],
        [1 2 3 4 5 6 7]
    });

mm = containers.Map;


for k = 1:numel(scenes)

    scene = scenes(k);
    shift = shifts(k); 

    [orig_im,orig_ratio_im] = ReadOrigIm(sprintf("%s/%s",stackeddir,scene),h,w,S,'CropX',cx,'CropY',cy,'CircShiftInputImageBy',shift);
    [~,~,orig_noisy_im] = ReadInputIm(sprintf("%s/%s",rawimagedir,scene),h,w,S,'CropX',cx,'CropY',cy,'BlackLevel',blacklvl,'CircShiftInputImageBy',shift);

    ims = [];

    m = {}; iter = 1;

    for S = 7:-1:4
        F=S-1;
        M = SubsamplingMask(mask_type,h,w,F);
        W = BucketMultiplexingMatrix(S);
        [H,B,C] = SubsampleMultiplexOperator(S,M);
        ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
        BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
        InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, 'BucketMultiplexingMatrix',W,'SubsamplingMask',M);

        take_idx = take_indices(S);
        input_im = ForwardFunc(orig_noisy_im(:,:,take_idx));
        input_ratio_im = ForwardFunc(IntensityToRatio(orig_noisy_im(:,:,take_idx)))*255;

        % 1: admm+tnrd in intensity space
        [admm_intensity_im,psnr_intensity,ssim_intensity,~] = ADMM(input_im,H,InitEstFunc,params_admm,orig_im(:,:,take_idx));

        % 2. admm+tnrd in ratio space
        [admm_ratio_im,~,~,~] = ADMM(input_ratio_im,H,InitEstFunc,params_admm_ratio,IntensityToRatio(orig_noisy_im(:,:,take_idx)));
        ratio_mult_inputsum_im = admm_ratio_im/255;
        ratio_mult_inputsum_im = RatioToIntensity(ratio_mult_inputsum_im,sum(input_im,3));
        [psnr_ratio_mult_inputsum,ssim_ratio_mult_inputsum] = ComputePSNRSSIM(orig_im(:,:,take_idx),ratio_mult_inputsum_im);
        
        %% photometric stereo

        projector_phase_shift = transpose((take_idx-1)*2*pi/7);

        [intensity_im_albedo,~,intensity_im_phase] = SLTriangulation(admm_intensity_im,W,Bounds,4,'Shifts',projector_phase_shift);
        intensity_im_disparity = disparityFunc((intensity_im_phase*hproj/(2*pi)),Y);
        
        [ratio_im_albedo,~,ratio_im_phase] = SLTriangulation(ratio_mult_inputsum_im,W,Bounds,4,'Shifts',projector_phase_shift);
        ratio_im_disparity = disparityFunc((ratio_im_phase*hproj/(2*pi)),Y);

        [orig_im_albedo,~,orig_im_phase] = SLTriangulation(orig_im(:,:,take_idx),W,Bounds,4,'Shifts',projector_phase_shift);
        orig_im_disparity = disparityFunc((orig_im_phase*hproj/(2*pi)),Y);


        [disparity_psnr_intensity,           disparity_ssim_intensity] = ComputePSNRSSIM(orig_im_disparity,intensity_im_disparity);
        [disparity_psnr_ratio_mult_inputsum, disparity_ssim_ratio_mult_inputsum] = ComputePSNRSSIM(orig_im_disparity,ratio_im_disparity);

        fprintf("admm      intensity                     %.4f/%.4f\n",psnr_intensity,ssim_intensity);
        fprintf("admm      psnr_ratio_mult_inputsum      %.4f/%.4f\n",psnr_ratio_mult_inputsum,ssim_ratio_mult_inputsum);
        fprintf("disparity intensity                     %.4f/%.4f\n",disparity_psnr_intensity,disparity_ssim_intensity);
        fprintf("disparity psnr_ratio_mult_inputsum      %.4f/%.4f\n",disparity_psnr_ratio_mult_inputsum,disparity_ssim_ratio_mult_inputsum);

        %% save

        ims1 = scaling*FlattenChannels(orig_im(:,:,take_idx),orig_ratio_im(:,:,take_idx),admm_intensity_im,admm_ratio_im,ratio_mult_inputsum_im);
        ims2 = zeros(3*h,w*S);
        ims2(:,1:3*w) = [
            orig_im_albedo,255*orig_im_phase/(2*pi),orig_im_disparity;...
            intensity_im_albedo,255*intensity_im_phase/(2*pi),intensity_im_disparity; ...
            ratio_im_albedo,255*ratio_im_phase/(2*pi),ratio_im_disparity];
        ims = [ims1;ims2];
        imshow(ims/255);
        imwrite(uint8(ims),sprintf("%s/%s_%d.png",savedir,scene,S));


        data.S = S;
        data.scene = scene;

        data.psnr_intensity = psnr_intensity;
        data.ssim_intensity = ssim_intensity;
        data.psnr_ratio_mult_inputsum = psnr_ratio_mult_inputsum;
        data.ssim_ratio_mult_inputsum = ssim_ratio_mult_inputsum;

        data.disparity_psnr_intensity = disparity_psnr_intensity;
        data.disparity_ssim_intensity = disparity_ssim_intensity;
        data.disparity_psnr_ratio_mult_inputsum = disparity_psnr_ratio_mult_inputsum;
        data.disparity_ssim_ratio_mult_inputsum = disparity_ssim_ratio_mult_inputsum;

        m{iter} = data;
        iter = iter + 1;
        break;
    end

    mm(scene) = m;
    break;
end

save(sprintf('%s/spatialspectral.mat',savedir),'mm');


%% plot 


mm = load(sprintf('%s/spatialspectral.mat',savedir));
mm = mm.mm;

% [scenes] x [psnr/ssim] x [intensity/ratio] x [4 5 6 7]
admm = zeros(numel(scenes),2,2,4);
disparity = zeros(numel(scenes),2,2,4);


for i = 1:numel(scenes)
    % scene = scenes(i);
    scene = "chameleon";
    m = mm(scene);
    for j = 1:4
        data = m{j};
        admm(i,1,1,j) = data.psnr_intensity;
        admm(i,2,1,j) = data.ssim_intensity;
        admm(i,1,2,j) = data.psnr_ratio_mult_inputsum;
        admm(i,2,2,j) = data.ssim_ratio_mult_inputsum;

        disparity(i,1,1,j) = data.disparity_psnr_intensity;
        disparity(i,2,1,j) = data.disparity_ssim_intensity;
        disparity(i,1,2,j) = data.disparity_psnr_ratio_mult_inputsum;
        disparity(i,2,2,j) = data.disparity_ssim_ratio_mult_inputsum;
    end
    break;
end


for i = 1:numel(scenes)
    % scene = scenes(i);
    scene = "chameleon";

    % admm 
    % psnr
    plot(1:4,reshape(admm(i,1,1,:),4,1),'DisplayName',"intensity"); hold on;
    plot(1:4,reshape(admm(i,1,2,:),4,1),'DisplayName','ratio multiplied with inputsum'); hold on;
    set(gca,'xtick',1:4,'xticklabel',[4 5 6 7]);
    legend();
    xlabel("#Subframes")
    ylabel("PSNR")
    title(sprintf("[%s] ADMM reconstruction vs. #subframes (constant exposure time)",scene));
    saveas(gcf,sprintf("%s/%s_admm_psnr.png",savedir,scene));
    hold off;
    % ssim 
    plot(1:4,reshape(admm(i,2,1,:),4,1),'DisplayName',"intensity"); hold on;
    plot(1:4,reshape(admm(i,2,2,:),4,1),'DisplayName','ratio multiplied with inputsum'); hold on;
    set(gca,'xtick',1:4,'xticklabel',[4 5 6 7]);
    legend();
    xlabel("#Subframes")
    ylabel("SSIM")
    title(sprintf("[%s] ADMM reconstruction vs. #subframes (constant exposure time)",scene));
    saveas(gcf,sprintf("%s/%s_admm_ssim.png",savedir,scene));
    hold off;

    % disparity
    % psnr
    plot(1:4,reshape(disparity(i,1,1,:),4,1),'DisplayName',"intensity"); hold on;
    plot(1:4,reshape(disparity(i,1,2,:),4,1),'DisplayName','ratio multiplied with inputsum'); hold on;
    set(gca,'xtick',1:4,'xticklabel',[4 5 6 7]);
    legend();
    xlabel("#Subframes")
    ylabel("PSNR")
    title(sprintf("[%s] disparity estimation vs. #subframes (constant exposure time)",scene));
    saveas(gcf,sprintf("%s/%s_disparity_psnr.png",savedir,scene));
    hold off;
    % ssim 
    plot(1:4,reshape(disparity(i,2,1,:),4,1),'DisplayName',"intensity"); hold on;
    plot(1:4,reshape(disparity(i,2,2,:),4,1),'DisplayName','ratio multiplied with inputsum'); hold on;
    set(gca,'xtick',1:4,'xticklabel',[4 5 6 7]);
    legend();
    xlabel("#Subframes")
    ylabel("SSIM")
    title(sprintf("[%s] disparity estimation vs. #subframes (constant exposure time)",scene));
    saveas(gcf,sprintf("%s/%s_disparity_ssim.png",savedir,scene));
    hold off;
    break;
end




% % 
% % -0.3416
% %% 
% 
% for tempshift = -pi:0.1:pi 
% % for tempshift = (-0.3416-0.5):0.1:(-0.3416+0.5)
%     
%     tempshift
%     % .8584 
%     Bounds.UB = Bounds.yErrorUB + tempshift*684/(2*pi);
%     Bounds.LB = Bounds.yErrorLB + tempshift*684/(2*pi);
%     Bounds.UB = double(Bounds.UB)*2*pi/hproj;
%     Bounds.LB = double(Bounds.LB)*2*pi/hproj;
%     
%     [max(max(Bounds.LB)) max(max(Bounds.UB))]
% 
% 
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %% Read in image
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     % tiled = double(imread(sprintf("%s/%s.png",rawimagedir,scene)));
% 
%     % % groundtruth image
%     % gt_im = reshape(tiled(1:h,:),h,w,[]);
%     % % imshow([gt_im(:,:,1) gt_im(:,:,2) gt_im(:,:,3) gt_im(:,:,4)]/255);
% 
% 
%     [orig_im_albedo,~,orig_im_phase] = SLTriangulation(orig_im,BucketMultiplexingMatrix(7),Bounds,4);
%     orig_im_disparity = disparityFunc((orig_im_phase*hproj/(2*pi)),Y);
%     
%     [subsampled_im_albedo,~,subsampled_im_phase] = SLTriangulation(subsampled_im,BucketMultiplexingMatrix(4),Bounds,4,'Shifts',transpose((0:4-1)*2*pi/S));
%     subsampled_im_disparity = disparityFunc((subsampled_im_phase*hproj/(2*pi)),Y);
% 
%     imagesc([orig_im_disparity subsampled_im_disparity]);
%     pause;
% end
% 
