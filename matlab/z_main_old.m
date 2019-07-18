clc; clear; close all;
addpath(genpath('./tnrd_denoising/'));
addpath(genpath('./minimizers/'));
addpath(genpath('./parameters/'));
addpath(genpath('./helper_functions/'));
addpath(genpath('./test_images/'));

%% parameters 

mask_types = [
    "toeplitz"
    "horz"
    "vert"
    "random"
];

% initialguesses = [
%     "groundtruth"
%     "maxfilter"
%     "zeroatunknown"
%     "zero"
%     "random"
% ];

initialguesses = [
    "maxfilter"
];

% snrs = [
%     25
%     30
%     35
%     40
% ];
snrs = [
    25
    30
    35
    40
];

% scenes = [
%     "bowl"
%     "buddha"
%     "building"
%     "candle"
%     "cups"
%     "flower"
%     "jfk"
%     "pens"
%     "pillow"
%     "shoe"
% ];
% 

scenes = [
    "sponge"
    "lamp"
    "cup"
    "chameleon"
    "giraffe"
    "head"
    "minion"
    "train"
    "totem"
    "cover"
];

[h,w] = deal(176,288);
% source image name
file_name = 'shoe';
% Light mode runs faster
light_mode = true;
% Signal to Noise ratio in [25,30,35,40]
snr = 35;
% Number of subframes
S = 7;
% Number of frames (F>=S-1)
F = 6;
% Mask 
mask_type = 'bayer';
% initial guess
initialguess = 'maxfilter';
% verbosity 
verbose = false;
% crop of orig_im
% crop = 50:(50+23);
crop = [];
% save results 
savedir = 'results/red/7patterns';
mkdir(savedir);
% groundtruth folder
groundtruth_folder = '../../data/7patterns/organized';


%% Run Experiment over [mask_types, initialguesses, objects, snrs]

% debug
% [scenes,snrs] = deal(["giraffe"],[35]);

for k = 1:size(scenes,1)
    scene = scenes(k);
    [keyset,valueset] = deal({},{});
    for l = 1:size(snrs,1)
        snr = snrs(l);
        for i = 1:size(mask_types,1)
            mask_type = mask_types(i);
            for j = 1:size(initialguesses,1)
                initialguess = initialguesses(j);
                
                tic;
                out = runRED(groundtruth_folder,scene,light_mode,snr,S,F,mask_type,initialguess,verbose,crop);
                out.time_elapsed = toc;

                keyset{end+1} = sprintf('%s-%s-%d',mask_type,initialguess,snr);
                valueset{end+1} = out;
            end
        end
    end
    rec = containers.Map(keyset,valueset);
    save(sprintf('%s/%s.mat',savedir,scene),'rec');
end

%% 

% Load from matfile
[keyset,valueset] = deal({},{});
for k = 1:size(scenes,1)
    scene = scenes(k);
    loaded = load(sprintf('%s/%s.mat',savedir,scene));
    keyset{k} = char(scene);
    valueset{k} = loaded.rec;
end
his = containers.Map(keyset,valueset);

%% reconstructed images

savedir_cur = sprintf("%s/ims",savedir);
mkdir(savedir_cur);
initialguess = "maxfilter";

for k = 1:size(scenes,1)
    scene = scenes(k);
    rec = his(scene);
    for l = 1:size(snrs,1)
        snr = snrs(l);
        ims = zeros((size(mask_types,1)+1)*h,(w+1)*S);
        for i = 1:size(mask_types,1)
            out = rec(sprintf('%s-%s-%d',mask_types(i),initialguess,snr));
            for s = 1:S
                ims(1:h,((s-1)*w+1):(w*s)) = out.orig_im(:,:,s);
                ims((i*h+1):(i+1)*h,((s-1)*w+1):(w*s)) = out.out_admm_im(:,:,s);
            end
            ims((i*h+1):(i+1)*h,(w*S+1):((S+1)*w)) = (out.M == 1)*255;
        end
        imwrite(uint8(ims),sprintf("%s/%s-%s-%d.png",savedir_cur,scene,initialguess,snr));
    end
end

%% inspect the mask on whole image

snr = 25;
scene = "giraffe";
initialguess = "maxfilter";
s = 1;
mask_type = "toeplitz";

out = rec(sprintf('%s-%s-%d',mask_type,initialguess,snr));
ims = [255*(out.M==s) out.orig_im(:,:,s) out.out_admm_im(:,:,s)];
imshow(ims/255);

%%

[cx,cy] = deal(80:110,90:110);
ims = [255*(out.M(cx,cy)==s) out.orig_im(cx,cy,s) out.out_admm_im(cx,cy,s)];
imshow(ims/255);


%% PSNR
fprintf('average psnr at (start->end) for all scenes:\n')
psnr_inputs = zeros(numel(snrs),numel(mask_types),numel(initialguesses),size(scenes,1));
psnr_admm = zeros(numel(snrs),numel(mask_types),numel(initialguesses),size(scenes,1));

for k = 1:size(scenes,1)
    rec = his(scenes(k));
    for l = 1:size(snrs,1)
        for i = 1:size(mask_types,1)
            for j = 1:size(initialguesses,1)
                out = rec(sprintf('%s-%s-%d',mask_types(i),initialguesses(j),snrs(l)));
                psnr_inputs(l,i,j,k) = out.psnr_input;
                psnr_admm(l,i,j,k) = out.psnr_admm;
            end
        end
    end
end


for l = 1:size(snrs,1)
    fprintf('\n\nSNR=%d\n',snrs(l));
    fprintf('%s\t',pad('',10));
    for i = 1:size(initialguesses,1)
        fprintf('%s\t',pad(initialguesses(i),10));
    end
    for i = 1:size(mask_types,1)
        fprintf('\n%s\t',pad(mask_types(i),10));
        for j = 1:size(initialguesses,1)
%             fprintf('(%.2f->%.2f)\t',mean(psnr_inputs(l,i,j,1:2)),mean(psnr_admm(l,i,j,1:2)));
            fprintf('(%.2f->%.2f)\t',mean(psnr_inputs(l,i,j,1:1)),mean(psnr_admm(l,i,j,1:1)));
        end
    end
end


%% Convergence Plots (PSNR/CostFunc vs. #iteration) with fixed SNR,scene,initialguess
fprintf('convergence\n');

snr = 25;
scene = 'cover';
rec = his(scene);
figure;
set(gcf, 'Position',  [0,0,1400,900])
statistics = 'psnrs';

for j = 1:size(initialguesses,1)
    subplot(1,1,j);
    for i = 1:size(mask_types,1)
        out = rec(sprintf('%s-%s-%d',mask_types(i),initialguesses(j),snr));
        if statistics == "psnrs"
            plot(full(0:10)*10,[out.psnr_input out.admm_statistics.(statistics)],'DisplayName',sprintf('%s',mask_types(i))); hold on;
        else
            plot((1:10)*10,out.admm_statistics.(statistics),'DisplayName',sprintf('%s',mask_types(i))); hold on;
        end
    end
    title(sprintf('%s vs. #Iteration (initialguess=%s)',statistics,initialguesses(j)));
    xlabel('#Iterations (ADMM)');
    if statistics == "psnrs"
        ylabel('PSNR');
%         ylim([35 40]);
    else
        ylabel('CostFunc');
    end
    legend('Location','East');
    hold off;
end
sgtitle(sprintf('%s vs. #iter (SNR: %d; scene: %s)',statistics,snr,scene));


%% Convergence Plots (PSNR/CostFunc vs. #iteration) with fixed SNR,scene,mask_type
fprintf('convergence\n');

savedir_cur = sprintf("%s/convergence_plot",savedir);
mkdir(savedir_cur);

initialguess = 'maxfilter';
set(0,'DefaultFigureVisible','off')
statistics = "costfunc"; % costfunc/psnrs

for l = 1:size(snrs,1)
    snr = snrs(l);

    %
    figure;
    set(gcf, 'DefaultLegendAutoUpdate','Off');
    set(gcf, 'Position',  [0,400,500,500])
    n_mts = size(mask_types,1);
    for i = 1:size(mask_types,1)

        xs = full(0:10)*5;
        ys = zeros(size(scenes,1),size(xs,2));

        for s = 1:size(scenes,1)
            scene = scenes(s);
            rec = his(scene);
            out = rec(sprintf('%s-%s-%d',mask_types(i),initialguess,snr));
            ys(s,:) = [out.psnr_input out.admm_statistics.(statistics)];
    %         plot(xs,ys(s,:)); hold on;
        end

        plot(xs,mean(ys,1),'DisplayName',mask_types(i),'LineWidth',3); hold on;
    end
    legend('Location','East');
    title(sprintf('%s vs. #iter (SNR: %d; initialguess: %s)',statistics,snr,initialguess));
    hold off;
    saveas(gcf, sprintf("%s/%s_vs_iterations-%s-%d.png",savedir_cur,statistics,initialguess,snr))

    %
end


% figure;
% set(gcf, 'Position',  [0,0,1800,350])
% statistics = 'psnrs';

% for i = 1:size(mask_types,1)
%     subplot(1,5,i);
%     for j = 1:size(initialguesses,1)
%         out = rec(sprintf('%s-%s-%d',mask_types(i),initialguesses(j),snr));
%         if statistics == "psnrs"
%             plot(full(0:10)*10,[out.psnr_input out.admm_statistics.(statistics)],'DisplayName',sprintf('%s',initialguesses(j))); hold on;
%         else
%             plot((1:10)*10,out.admm_statistics.(statistics),'DisplayName',sprintf('%s',initialguesses(j))); hold on;
%         end
%     end
%     title(sprintf('%s vs. #Iteration (mask=%s)',statistics,mask_types(i)));
%     xlabel('#Iterations (ADMM)');
%     if statistics == "psnrs"
%         ylabel('PSNR');
%         ylim([35 40]);
%     else
%         ylabel('CostFunc');
%     end
%     legend('Location','East');
%     hold off;
% end
% sgtitle(sprintf('%s vs. #iter (SNR: %d; scene: %s)',statistics,snr,scene));

%%

function out = runRED(groundtruth_folder,scene,light_mode,snr,S,F,mask_type,initialguess,verbose,crop)
    %% read the original image
    if verbose
        fprintf('Reading %s image...', scene);
    end
    for i = 1:S
        orig_im(:,:,i) = imread(sprintf('%s/%s_%d.png',groundtruth_folder,scene,i-1));
    end
    orig_im = double(orig_im);
    if numel(crop) ~= 0
        orig_im = orig_im(crop,crop,:);
    end
    assert(mod(size(orig_im,1),2)==0 && mod(size(orig_im,1),2)==0, 'image size multiple of 2');

    %% define the degradation model

    [h,w] = deal(size(orig_im,1),size(orig_im,2));
    input_sigma = mean(orig_im,'all')/snr;

    if verbose
        fprintf('Input sigma = %.5f\n',input_sigma);
    end

    M = SubsamplingMask(mask_type,h,w,F);
    W = BucketMultiplexingMatrix(S);
    [H,B,C] = SubsampleMultiplexOperator(S,M);

    % S patterned img -> 2 bucket measurements
    ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);

    % 2 bucket measurements -> S patterned img
    BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);

    switch initialguess
    case 'groundtruth'
        InitEstFunc = @(y) orig_im;
    otherwise
        InitEstFunc = InitialEstimateFunc(initialguess,h,w,F,S,W);
    end

    %% degrade the original image

    if verbose
        fprintf('Adding noise...\n');
    end

    % add noise before appling spatial subsampling operator `B`
    %       `fullres_im` is same for all InitEstFunc
    randn('state', 0);
    fullres_im = C*orig_im(:) + input_sigma*randn(h*w*2*F,1);
    input_im = reshape(B*fullres_im,h,w,2);

    % fprintf(' Adding noise ...\n');
    % input_im = ForwardFunc(orig_im);
    % input_im = input_im + input_sigma*randn(size(input_im));

    %% Initial Guess's PSNR

    x_est = InitEstFunc(input_im);
    psnr_input = ComputePSNR(orig_im, x_est);

    %% minimize the Laplacian regularization functional via ADMM
    if verbose
        fprintf('Restoring using RED: ADMM method\n');
    end

    params_admm = GetSuperResADMMParams(light_mode);
    [out_admm_im, psnr_admm, admm_statistics] = RunADMM_demosaic(input_im,...
                                                ForwardFunc,...
                                                BackwardFunc,...
                                                InitEstFunc,...
                                                input_sigma,...
                                                params_admm,...
                                                orig_im);

    out.scene = scene;
    out.S = S;
    out.F = F;
    out.snr = snr;
    out.mask_type = mask_type;
    out.initialguess = initialguess;
    out.orig_im = orig_im;
    out.input_sigma = input_sigma;
    out.M = M;
    out.W = W;
    out.H = H;
    out.fullres_im = fullres_im;
    out.input_im = input_im;
    out.psnr_input = psnr_input;
    out.x_est = x_est;
    out.out_admm_im = out_admm_im;
    out.psnr_admm = psnr_admm;
    out.admm_statistics = admm_statistics;
    out.params_admm = params_admm;



    % %% display final results

    % fprintf('Image name %s \n', scene);
    % fprintf('Input PSNR = %f \n', psnr_input);
    % fprintf('RED: ADMM PSNR = %f \n', psnr_admm);

    % s = 1;

    % cx = 20:120;
    % cy = 20:120;
    % imgs1 = [orig_im(:,:,s) x_est(:,:,s) out_admm_im(:,:,s)]/255;
    % imgs2 = [orig_im(cx,cy,s) x_est(cx,cy,s) out_admm_im(cx,cy,s)]/255;
    % figure; imshow(imgs1);
    % figure; imshow(imgs2);


    % cx = (176-100):176;
    % cy = 20:120;
    % imgs1 = [orig_im(:,:,s) x_est(:,:,s) out_admm_im(:,:,s)]/255;
    % imgs2 = [orig_im(cx,cy,s) x_est(cx,cy,s) out_admm_im(cx,cy,s)]/255;
    % figure; imshow(imgs1);
    % figure; imshow(imgs2);


    % % see the noise level

    % foo = ForwardFunc(orig_im);
    % foo = foo + input_sigma*randn(size(foo));
    % imgs3 = [orig_im(cx,cy,s) foo(cx,cy,s)]/255;
    % figure; imshow(imgs3);

    % % mesh(x_est(:,:,s)-out_admm_im(:,:,s));


    % % reconstructed - groundtruth

    % imgs3 = [orig_im(cx,cy,s) foo(cx,cy,s)]/255;

    % % figure; mesh(abs(orig_im(:,:,s)-x_est(:,:,s))); colormap('jet');
    % % figure; mesh(abs(orig_im(:,:,s)-out_admm_im(:,:,s))); colormap('jet');

    % figure; imagesc(abs(orig_im(:,:,s)-x_est(:,:,s)));
    % figure; imagesc(abs(orig_im(:,:,s)-out_admm_im(:,:,s)));


    % imshow([orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4); ...
    %         x_est(:,:,1) x_est(:,:,2) x_est(:,:,3) x_est(:,:,4); ...
    %         out_admm_im(:,:,1) out_admm_im(:,:,2) out_admm_im(:,:,3) out_admm_im(:,:,4)]/255);

    % %% write images

    % if ~exist('./results/','dir')
    %     mkdir('./results/');
    % end

    % fprintf('Writing the images to ./results...');

    % imwrite(uint8(orig_im),['./results/orig_' scene]);
    % imwrite(uint8(InitEstFunc(input_im)),['./results/input_' scene]);
    % % imwrite(uint8(out_fp_im),['./results/est_fp_' scene]);
    % imwrite(uint8(out_admm_im),['./results/est_admm_' scene]);
    % % imwrite(uint8(out_sd_im),['./results/est_sd_' scene]);

    % fprintf(' Done.\n');









    % %% minimize the Laplacian regularization functional via Fixed Point

    % fprintf('Restoring using RED: Fixed-Point method\n');

    % switch degradation_model
    %     case 'UniformBlur'
    %         params_fp = GetUniformDeblurFPParams(light_mode, psf, use_fft);
    %     case 'GaussianBlur'
    %         params_fp = GetGaussianDeblurFPParams(light_mode, psf, use_fft);
    %     case 'Downscale'
    %         assert(exist('use_fft','var') == 0);
    %         params_fp = GetSuperResFPParams(light_mode);
    %     otherwise
    %         error('Degradation model is not defined');
    % end

    % [est_fp_im, psnr_fp] = RunFP(input_luma_im,...
    %                              ForwardFunc,...
    %                              BackwardFunc,...
    %                              InitEstFunc,...
    %                              input_sigma,...
    %                              params_fp,...
    %                              orig_luma_im);
    % out_fp_im = MergeChannels(input_im,est_fp_im);

    % fprintf('Done.\n');

    % %% minimize the laplacian regularization functional via Steepest Descent

    % fprintf('Restoring using RED: Steepest-Descent method\n');

    % switch degradation_model
    %     case 'UniformBlur'
    %         params_sd = GetUniformDeblurSDParams(light_mode);
    %     case 'GaussianBlur'
    %         params_sd = GetGaussianDeblurSDParams(light_mode);
    %     case 'Downscale'
    %         params_sd = GetSuperResSDParams(light_mode);
    %     otherwise
    %         error('Degradation model is not defined');
    % end

    % [est_sd_im, psnr_sd] = RunSD(input_luma_im,...
    %                              ForwardFunc,...
    %                              BackwardFunc,...
    %                              InitEstFunc,...
    %                              input_sigma,...
    %                              params_sd,...
    %                              orig_luma_im);
    % % convert back to rgb if needed
    % out_sd_im = MergeChannels(input_im,est_sd_im);

    % fprintf('Done.\n');


    % %% display final results

    % fprintf('Image name %s \n', scene);
    % fprintf('Input PSNR = %f \n', psnr_input);
    % fprintf('RED: Fixed-Point PSNR = %f \n', psnr_fp);
    % fprintf('RED: ADMM PSNR = %f \n', psnr_admm);
    % fprintf('RED: Steepest-Decent PSNR = %f \n', psnr_sd);


    % %% write images

    % if ~exist('./results/','dir')
    %     mkdir('./results/');
    % end

    % fprintf('Writing the images to ./results...');

    % imwrite(uint8(input_im),['./results/input_' scene]);
    % imwrite(uint8(out_fp_im),['./results/est_fp_' scene]);
    % imwrite(uint8(out_admm_im),['./results/est_admm_' scene]);
    % imwrite(uint8(out_sd_im),['./results/est_sd_' scene]);

    % fprintf(' Done.\n');

end