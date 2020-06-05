%% reconstruction on ParsaPattern
%       
clc; clear; close all;
ProjectPaths;

%% Parameters

[cx,cy] = deal(1:160,10:247);
[h,w] = deal(numel(cx),numel(cy));
savedir = "results/reconstruction_parsapattern"; mkdir(savedir);
blacklevelpath = "data/blacklevel_all1/blacklevel.mat";
blacklvl = load(blacklevelpath); blacklvl = blacklvl.blacklvl;
hproj = 608;
dispRange = [50, 160];
[X,Y] = meshgrid(1:w,1:h);
% bounds 
ProjectorInfoFolder = 'mian/CalibrationCode';
Bounds = load(sprintf('%s/%s.mat', ProjectorInfoFolder, 'Bounds'));
Bounds.LB = Bounds.yErrorLB(cx,cy);
Bounds.UB = Bounds.yErrorUB(cx,cy);
expandby = 1000; shiftby = 0;
Bounds.UB = min(shiftby + Bounds.yErrorLB(cx,cy) + expandby,hproj);
Bounds.LB = max(shiftby + Bounds.yErrorUB(cx,cy) - expandby,0);
assetsdir = '../writeup/assets';
disparityFunc = @(corres) double(corres)-2.35*Y;
CorrectPhase = @(phase) 0.89*phase - 0.5;
light_mode = true;



%% groundtruth 

Load = load(sprintf('%s/GroundTruthPhaseDisparity.mat', savedir));
gt = Load.GroundTruth;

%% phase PSNR with ZNCC/MPS decoder on sinusoids with spatial_freq=1

savedir_cur = sprintf('%s/InverseRecontruction',savedir); mkdir(savedir_cur);

% sinusoids 
spatial_freq = 1;
[I,P] = ParsaPatternSinusoidsGetStackedIm(hproj,spatial_freq);


psnrs.zncc = [];
psnrs.mps = [];
Ss = 2:30; Ss = [2:7 12 16 20 24];
for i = 1:size(Ss,2)
S = Ss(i);
is = ceil(linspace(1,30*(S-1)/S,S));
W = BucketMultiplexingMatrix(size(is,2));
P_ = P(:,is);
I_ = I(:,:,is);
[phase_zncc,~,~] = DecodeZNCC(I_,P_,Bounds.LB,Bounds.UB);
[~,~,phase_mps] = DecodePhaseShiftWithDepthBound(I_,W,Bounds.LB,Bounds.UB,hproj,spatial_freq);
psnrs.zncc = [psnrs.zncc ComputePSNR(gt.phase,phase_zncc)];
psnrs.mps = [psnrs.mps ComputePSNR(gt.phase,phase_mps)];
end

plot(Ss,psnrs.zncc,'-o','LineWidth',3,'MarkerSize',5,'DisplayName','ZNCC'); hold on;
plot(Ss,psnrs.mps,'-o','LineWidth',3,'MarkerSize',5,'DisplayName','MPS');
title(sprintf('Phase PSNR vs. Shifts (spatial freqency = %d)',spatial_freq));
legend;
hold off;
saveas(gcf,sprintf('%s/phase_psnr_stackedim_vs_shifts.png',savedir));

%% performance of x-update step 

% for S  = [7 24 48]
for S = [7]
% S = 7;
S
F = S-1;
mask_type = "toeplitz";
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[A,~,~] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(A*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(A'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, ...
    'BucketMultiplexingMatrix',W,'SubsamplingMask',M);

v = rand(S*h*w,1)*100;
y = rand(2*h*w,1)*100;
rho = 100;

[R,flag] = chol(A'*A + rho*speye(size(A,2),size(A,2)));

tic;
x = A'*y + rho*v;
y_backsolve1 = R\(R'\(x));
e = toc;
fprintf('back-solve update                   %f\n',e);

tic;
[R,flag] = chol(A'*A + rho*speye(size(A,2),size(A,2)));
x = A'*y + rho*v;
y_backsolve2 = R\(R'\(x));
e = toc;
fprintf('back-solve update inc factorization %f\n',e);

if isdiag(A*A') ~= 1
    warning("A*A' with optimal multiplexing matrix W is diagonal");
end

AAT = A*A';
PP = speye(S*h*w)/rho - A'*inv( speye(2*h*w)*rho + AAT )*A/rho;

tic;
x = A'*y + rho*v;
y_inversion_lemma1 = PP*x;
e = toc;
fprintf('matrix inversion                    %f\n',e);

tic;
PP = speye(S*h*w)/rho - A'*inv( speye(2*h*w)*rho + AAT )*A/rho;
x = A'*y + rho*v;
y_inversion_lemma2 = PP*x;
e = toc;
fprintf('matrix inversion inc factorization  %f\n',e);

zeta = full(diag(A*A'));

tic;
y_desci = v + A'*((y-A*v)./(rho + zeta));
e = toc;
fprintf('DeSCI (adaptive rho)                %f\n',e);
end
     
% S = 7
% back-solve update                   0.007146
% back-solve update inc factorization 0.074019
% matrix inversion                    0.002579
% matrix inversion inc factorization  0.033881
% DeSCI (adaptive rho)                0.002027
% 
% S = 24
% back-solve update                   0.044138
% back-solve update inc factorization 1.062932
% matrix inversion                    0.025854
% matrix inversion inc factorization  0.269067
% DeSCI (adaptive rho)                0.005039
% 
% S = 48
% back-solve update                   0.154915
% back-solve update inc factorization 5.637564
% matrix inversion                    0.133417
% matrix inversion inc factorization  1.628710
% DeSCI (adaptive rho)                0.010573


%% phase PSNR on X reconstructed using ADMM

S = 7;
F = S-1;
mask_type = "toeplitz";
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[A,~,~] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(A*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(A'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S, ...
    'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params = GetDemosaicDemultiplexParams(false)


[X,P] = ParsaPatternSinusoidsGetStackedIm(hproj,spatial_freq);
is = ceil(linspace(1,30*(S-1)/S,S));
X = X(:,:,is); P = P(:,is);

Y = ForwardFunc(X);
[im_out,psnr_out,ssim_out,history,iter_ims] = ADMM(Y,A,InitEstFunc,params,X);
[phase,~,~] = DecodeZNCC(im_out,P,Bounds.LB,Bounds.UB);
ComputePSNR(gt.phase,phase)

plot(1:10:100, history.psnrs)
imshow(FlattenChannels(im_out)/255)


%% 

function [I,P] = ParsaPatternSinusoidsGetStackedIm(hproj,spatial_freq)
    imagedir=sprintf('results/reconstruction_parsapattern/Sinusoids/Freq%02d',spatial_freq);
    files = dir(sprintf("%s/*.png",imagedir));
    [fnames,ffolders] = deal({files.name},{files.folder});
    K = size(fnames,2);
    I = [];
    for k = 1:K
        I = cat(3,I,double(imread(sprintf('%s/%s',ffolders{k},fnames{k}))));
    end
    shifts = 1:30;
    P = 0.5 + 0.5*cos(spatial_freq*(0:hproj-1)'*2*pi/hproj + (shifts-1)*2*pi/30 );
end
