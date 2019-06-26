clc; clear; close all;
addpath(genpath('./tnrd_denoising/'));
addpath(genpath('./minimizers/'));
addpath(genpath('./parameters/'));
addpath(genpath('./helper_functions/'));
addpath(genpath('./test_images/'));

%% parameters 

mask_types = [
    "bayer"
    "toeplitz"
    "horz3"
    "vert3"
    "random"
];

initialguesses = [
    "groundtruth"
    "maxfilter"
    "bayerdemosaic"
    "zeroatunknown"
    "zero"
    "random"
];

snrs = [
    25
    30
    35
    40
];

scenes = [
    "bowl"
    "buddha"
    "building"
    "candle"
    "cups"
    "flower"
    "jfk"
    "pens"
    "pillow"
    "shoe"
];

% source image name
file_name = 'shoe';
% Light mode runs faster
light_mode = true;
% Signal to Noise ratio in [25,30,35,40]
snr = 35;
% Number of subframes
S = 4;
% Number of frames (F>=S-1)
F = 3;
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
savedir = 'results/red';


%% Run Experiment over [mask_types, initialguesses, objects, snrs]

mkdir(savedir);

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
                out = runRED(scene,light_mode,snr,S,F,mask_type,initialguess,verbose,crop);
                out.time_elapsed = toc;

                keyset{end+1} = sprintf('%s-%s-%d',mask_type,initialguess,snr);
                valueset{end+1} = out;
            end
        end
    end
    rec = containers.Map(keyset,valueset);
    save(sprintf('%s/%s.mat',savedir,scene),'rec');
end


% Load from matfile
[keyset,valueset] = deal({},{});
for k = 1:size(scenes,1)
    scene = scenes(k);
    S = load(sprintf('%s/%s.mat',savedir,scene));
    keyset{k} = char(scene);
    valueset{k} = S.rec;
end
his = containers.Map(keyset,valueset);

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
            fprintf('(%.2f->%.2f)\t',mean(psnr_inputs(l,i,j,:)),mean(psnr_admm(l,i,j,:)));
        end
    end
end


%% Convergence Plots
fprintf('convergence\n');

figure;
for i = 1:size(mask_types,1)
    % for j = 1:size(initialguesses,1)
    for j = 1:3
        name = sprintf('%s-%s',mask_types(i),initialguesses(j));
        out = rec(name);
        plot(1:10,out.admm_statistics.psnrs,'DisplayName',name); hold on;
    end
    break
end
legend();
hold off;

%%

function out = runRED(scene,light_mode,snr,S,F,mask_type,initialguess,verbose,crop)
    %% read the original image
    if verbose
        fprintf('Reading %s image...', scene);
    end
    for i = 1:S
        orig_im(:,:,i) = imread(sprintf('../../data/exp60/organized/%s_%d.png',scene,i-1));
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

    switch mask_type
    case 'bayer'
        % 1 2
        % 2 3
        M = BayerMask(h,w);
    case 'random'
        M = floor(rand(h,w)/(1/F))+1;
    case 'horz3'
        % 1 2 3
        % 1 2 3
        M = zeros(h,w);
        M(:,1:3:end) = 1;
        M(:,2:3:end) = 2;
        M(:,3:3:end) = 3;
    case 'toeplitz'
        % 1 2 3 1 2
        % 2 1 2 3 1
        % 3 2 1 2 3
        p = repmat([1 2 3],1,ceil(max(w,h)/3));
        c = p(1,1:h);
        r = p(1,1:w);
        M = toeplitz(c,r);
    case 'vert3'
        % 1 1
        % 2 2
        % 3 3
        M = zeros(h,w);
        M(1:3:end,:) = 1;
        M(2:3:end,:) = 2;
        M(3:3:end,:) = 3;
    otherwise
        warning('mask not set properly');
    end

    W = BucketMultiplexingMatrix(S);
    [H,B,C] = SubsampleMultiplexOperator(S,M);

    % S patterned img -> 2 bucket measurements
    ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);

    % 2 bucket measurements -> S patterned img
    BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);

    switch initialguess
    case 'groundtruth'
        InitEstFunc = @(y) orig_im;
    case 'bayerdemosaic'
        assert(F == 3, 'F == 3');
        InitEstFunc = @(y) ...
        reshape(...
            reshape( ...
                cat(3, ...
                    rgb2bgr(double(demosaic(uint8(y(:,:,1)), 'bggr'))), ...
                    rgb2bgr(double(demosaic(uint8(y(:,:,2)), 'bggr')))), ...
                [], 6) ...
            / W', ...
        h,w,S);
    case 'maxfilter'
        mask = zeros(h,w,F);
        for k = 1:F
            mask(:,:,k) = double(M==k);
        end
        % max filter (3x3) over each channel of `im`
        max_filtering = @(im) reshape(cell2mat(arrayfun(@(i) ...
            ordfilt2(im(:,:,i),9,ones(3,3)),1:3,'UniformOutput',false)),h,w,[]);
        InitEstFunc = @(y) ...
            reshape(...
                reshape(cat(F, ...
                    max_filtering(mask.*y(:,:,1)), ...
                    max_filtering(mask.*y(:,:,2))),[],2*F) / W', ...
            h,w,S);
    case 'zeroatunknown'
        mask = zeros(h,w,F);
        for k = 1:F
            mask(:,:,k) = double(M==k);
        end
        InitEstFunc = @(y) ...
            reshape(...
                reshape(cat(F, mask.*y(:,:,1), mask.*y(:,:,2)),[],2*F) / W', ...
            h,w,S);
    case 'zero'
        InitEstFunc = @(y) zeros(h,w,S);
    case 'random'
        InitEstFunc = @(y) 255*rand(h,w,S);
    otherwise
        warning('initial guess not set properly');
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