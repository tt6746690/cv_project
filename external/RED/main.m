clc; clear; close all;
addpath(genpath('./tnrd_denoising/'));
addpath(genpath('./minimizers/'));
addpath(genpath('./parameters/'));
addpath(genpath('./helper_functions/'));
addpath(genpath('./test_images/'));

%% parameters 

% source image name
file_name = 'shoe';
% Light mode runs faster
light_mode = false;
% Signal to Noise ratio in [25,30,35,40]
snr = 35;
% Number of subframes
S = 4;
% Number of frames (F>=S-1)
F = 3;
% Mask 
mask = 'bayer';
% initial guess
initialguess = 'zeros';


%% read the original image

fprintf('Reading %s image...', file_name);
for i = 1:S
    orig_im(:,:,i) = imread(sprintf('../../data/exp60/organized/%s_%d.png',file_name,i-1));
end
orig_im = double(orig_im);
fprintf(' Done.\n');
assert(mod(size(orig_im,1),2)==0 && mod(size(orig_im,1),2)==0, 'image size multiple of 2');

%% define the degradation model

[h,w] = deal(size(orig_im,1),size(orig_im,2));
input_sigma = mean(orig_im,'all')/snr;
fprintf('Input sigma = %.5f\n',input_sigma);

switch mask
case 'bayer'
    M = BayerMask(h,w);
case 'random'
    M = floor(rand(h,w)/(1/F))+1;
otherwise
    warning('mask not set properly');
end

W = BucketMultiplexingMatrix(S);
H = SubsamplingOperator(S,M);

% S patterned img -> 2 bucket measurements
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);

% 2 bucket measurements -> S patterned img
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);

switch initialguess
case 'groundtruth'
    InitEstFunc = @(y) orig_im;
case 'demosaic'
    InitEstFunc = @(y) ...
        reshape(...
            reshape( ...
                cat(3, ...
                    rgb2bgr(double(demosaic(uint8(y(:,:,1)), 'bggr'))), ...
                    rgb2bgr(double(demosaic(uint8(y(:,:,2)), 'bggr')))), ...
                [], 6) ...
            / W', ...
        h,w,S);

case 'zeros'
    InitEstFunc = @(y) zeros(h,w,S);
otherwise
    warning('initial guess not set properly');
end


%% degrade the original image

fprintf('Demosaicing+Demultiplexing...');
input_im = ForwardFunc(orig_im);
randn('state', 0);

% add noise
fprintf(' Adding noise...\n');
input_im = input_im + input_sigma*randn(size(input_im));

x_est = InitEstFunc(input_im);
% imgs = [orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4); ...
%         x_est(:,:,1) x_est(:,:,2) x_est(:,:,3) x_est(:,:,4)]/255;
% imshow(imgs/255);
psnr_input = ComputePSNR(orig_im, x_est);


%% minimize the Laplacian regularization functional via ADMM

fprintf('Restoring using RED: ADMM method\n');

switch degradation_model
    case 'UniformBlur'
        params_admm = GetUniformDeblurADMMParams(light_mode, psf, use_fft);
    case 'GaussianBlur'
        params_admm = GetGaussianDeblurADMMParams(light_mode, psf, use_fft);
    case 'Downscale'
        assert(exist('use_fft','var') == 0);
        params_admm = GetSuperResADMMParams(light_mode);
    case 'demosaic'
        params_admm = GetSuperResADMMParams(light_mode);
    otherwise
        error('Degradation model is not defined');
end

[out_admm_im, psnr_admm] = RunADMM_demosaic(input_im,...
                                            ForwardFunc,...
                                            BackwardFunc,...
                                            InitEstFunc,...
                                            input_sigma,...
                                            params_admm,...
                                            orig_im);
% out_admm_im = MergeChannels(input_im,est_admm_im);

fprintf('Done.\n');

%% display final results

fprintf('Image name %s \n', file_name);
fprintf('Input PSNR = %f \n', psnr_input);
fprintf('RED: ADMM PSNR = %f \n', psnr_admm);

s = 1;

cx = 20:120;
cy = 20:120;
imgs1 = [orig_im(:,:,s) x_est(:,:,s) out_admm_im(:,:,s)]/255;
imgs2 = [orig_im(cx,cy,s) x_est(cx,cy,s) out_admm_im(cx,cy,s)]/255;
figure; imshow(imgs1);
figure; imshow(imgs2);


cx = (176-100):176;
cy = 20:120;
imgs1 = [orig_im(:,:,s) x_est(:,:,s) out_admm_im(:,:,s)]/255;
imgs2 = [orig_im(cx,cy,s) x_est(cx,cy,s) out_admm_im(cx,cy,s)]/255;
figure; imshow(imgs1);
figure; imshow(imgs2);


% see the noise level

foo = ForwardFunc(orig_im);
foo = foo + input_sigma*randn(size(foo));
imgs3 = [orig_im(cx,cy,s) foo(cx,cy,s)]/255;
figure; imshow(imgs3);

% mesh(x_est(:,:,s)-out_admm_im(:,:,s));


% reconstructed - groundtruth

imgs3 = [orig_im(cx,cy,s) foo(cx,cy,s)]/255;

% figure; mesh(abs(orig_im(:,:,s)-x_est(:,:,s))); colormap('jet');
% figure; mesh(abs(orig_im(:,:,s)-out_admm_im(:,:,s))); colormap('jet');

figure; imagesc(abs(orig_im(:,:,s)-x_est(:,:,s)));
figure; imagesc(abs(orig_im(:,:,s)-out_admm_im(:,:,s)));


% imshow([orig_im(:,:,1) orig_im(:,:,2) orig_im(:,:,3) orig_im(:,:,4); ...
%         x_est(:,:,1) x_est(:,:,2) x_est(:,:,3) x_est(:,:,4); ...
%         out_admm_im(:,:,1) out_admm_im(:,:,2) out_admm_im(:,:,3) out_admm_im(:,:,4)]/255);

% %% write images

% if ~exist('./results/','dir')
%     mkdir('./results/');
% end

% fprintf('Writing the images to ./results...');

% imwrite(uint8(orig_im),['./results/orig_' file_name]);
% imwrite(uint8(InitEstFunc(input_im)),['./results/input_' file_name]);
% % imwrite(uint8(out_fp_im),['./results/est_fp_' file_name]);
% imwrite(uint8(out_admm_im),['./results/est_admm_' file_name]);
% % imwrite(uint8(out_sd_im),['./results/est_sd_' file_name]);

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

% fprintf('Image name %s \n', file_name);
% fprintf('Input PSNR = %f \n', psnr_input);
% fprintf('RED: Fixed-Point PSNR = %f \n', psnr_fp);
% fprintf('RED: ADMM PSNR = %f \n', psnr_admm);
% fprintf('RED: Steepest-Decent PSNR = %f \n', psnr_sd);


% %% write images

% if ~exist('./results/','dir')
%     mkdir('./results/');
% end

% fprintf('Writing the images to ./results...');

% imwrite(uint8(input_im),['./results/input_' file_name]);
% imwrite(uint8(out_fp_im),['./results/est_fp_' file_name]);
% imwrite(uint8(out_admm_im),['./results/est_admm_' file_name]);
% imwrite(uint8(out_sd_im),['./results/est_sd_' file_name]);

% fprintf(' Done.\n');

