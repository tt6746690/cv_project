%% reconstruction on ParsaPattern
%       
clc; clear; close all;
ProjectPaths;


%% Parameters

[cx,cy] = deal(1:160,10:247);
[h,w] = deal(numel(cx),numel(cy));
savedir = "results/reconstruction_parsapattern";
if ~exist(savedir,'dir'); mkdir(savedir); end
blacklevelpath = "data/blacklevel_all1/blacklevel.mat";
blacklvl = load(blacklevelpath); blacklvl = blacklvl.blacklvl;
hproj = 608;
dispRange = [50, 160];
[X,Y] = meshgrid(1:w,1:h);
% bounds 
ProjectorInfoFolder = '../external/mian/CalibrationCode';
Bounds = load(sprintf('%s/%s.mat', ProjectorInfoFolder, 'Bounds'));
Bounds.LB = Bounds.yErrorLB(cx,cy);
Bounds.UB = Bounds.yErrorUB(cx,cy);
expandby = 1000; shiftby = 0;
Bounds.UB = min(shiftby + Bounds.yErrorLB(cx,cy) + expandby,hproj);
Bounds.LB = max(shiftby + Bounds.yErrorUB(cx,cy) - expandby,0);
assetsdir = '../writeup/assets';
disparityFunc = @(corres) double(corres)-2.35*Y;
CorrectPhase = @(phase) 0.89*phase - 0.5;
Load = load(sprintf('%s/GroundTruthPhaseDisparity.mat', savedir));
gt = Load.GroundTruth;

%% phase PSNR with ZNCC/MPS decoder on sinusoids (given full res image) with spatial_freq=1

savedir_cur = sprintf('%s/InverseRecontruction',savedir); mkdir(savedir_cur);

% sinusoids 
spatial_freq = 1;
[I,P] = ParsaPatternSinusoidsGetStackedIm(hproj,spatial_freq);

zncc.psnrs =[]; zncc.ssims = [];
mps.psnrs = []; mps.ssims = [];
Ss = 2:30; Ss = [2:7 12 16 20 24];
for i = 1:size(Ss,2)
S = Ss(i);
is = ceil(linspace(1,30*(S-1)/S,S));
W = BucketMultiplexingMatrix(size(is,2));
P_ = P(:,is);
I_ = I(:,:,is);
[phase_zncc,~,~] = DecodeZNCC(I_,P_,Bounds.LB,Bounds.UB);
[~,~,phase_mps] = DecodePhaseShiftWithDepthBound(I_,W,Bounds.LB,Bounds.UB,hproj,spatial_freq);
zncc.psnrs = [zncc.psnrs ComputePSNR(gt.phase,phase_zncc)];
mps.psnrs = [mps.psnrs ComputePSNR(gt.phase,phase_mps)];
zncc.ssims = [zncc.ssims ComputeSSIM(gt.phase,phase_zncc)];
mps.ssims = [mps.ssims ComputeSSIM(gt.phase,phase_mps)];
end


figure('Renderer', 'painters', 'Position', [10 10 300 300])
plot(Ss,zncc.psnrs,'-o','LineWidth',3,'MarkerSize',5,'DisplayName','ZNCC'); hold on;
plot(Ss,mps.psnrs,'--o','LineWidth',3,'MarkerSize',5,'DisplayName','MPS');
title(sprintf('Phase PSNR vs. Shifts (spatial freqency = %d)',spatial_freq));
xlabel("#Shifts (S)"); ylabel("Phase PNSR");
legend('Location','southeast'); grid on;
hold off;
saveas(gcf,sprintf('%s/SinusoidsSpatialFreq1PhaseUpperBoundPSNR.png',savedir));
saveas(gcf,sprintf('%s/SinusoidsSpatialFreq1PhaseUpperBoundPSNR.png',assetsdir));


figure('Renderer', 'painters', 'Position', [10 10 300 300])
plot(Ss,zncc.ssims,'-o','LineWidth',3,'MarkerSize',5,'DisplayName','ZNCC'); hold on;
plot(Ss,mps.ssims,'--o','LineWidth',3,'MarkerSize',5,'DisplayName','MPS');
title(sprintf('Phase SSIM vs. Shifts (spatial freqency = %d)',spatial_freq));
xlabel("#Shifts (S)"); ylabel("Phase SSIM");
legend('Location','southeast'); grid on;
hold off;
saveas(gcf,sprintf('%s/SinusoidsSpatialFreq1PhaseUpperBoundSSIM.png',savedir));
saveas(gcf,sprintf('%s/SinusoidsSpatialFreq1PhaseUpperBoundSSIM.png',assetsdir));

%% faster denoising step using parfor

Ss = [7 30];
denoiser_types = {'mf','tnrd'};

for i = 1:size(denoiser_types,2)
for si = 1:size(Ss,2)
S = Ss(si);
x_est = rand(h,w,S);
denoiser_type = denoiser_types{i};

tic;
Denoiser(x_est,3,denoiser_type,'MaxNumWorkers',1);
elapsed1 = toc;
fprintf('(S=%2d) serial-%s %3.3f s\n',S,denoiser_type,elapsed1);

tic;
Denoiser(x_est,3,denoiser_type,'MaxNumWorkers',100);
elapsed2 = toc;
fprintf('(S=%2d) parall-%s %3.3f s\n',S,denoiser_type,elapsed2);

end
end


% Using 8 cores: about 3~4x speedup by using parfor
%
% (S= 7) serial-mf 0.106 s
% (S= 7) parall-mf 0.025 s
% (S=30) serial-mf 0.049 s
% (S=30) parall-mf 0.019 s
% (S= 7) serial-tnrd 4.056 s
% (S= 7) parall-tnrd 1.216 s
% (S=30) serial-tnrd 4.165 s
% (S=30) parall-tnrd 1.172 s

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


%% phase PSNR on X reconstructed using ADMM (varying denoiser/#shifts)

spatial_freq = 1;
[XX,PP] = ParsaPatternSinusoidsGetStackedIm(hproj,spatial_freq);

clear histories xhats;
denoiser_types = {'mf','tnrd'};
Ss = [2:7 12 16 20 24];
for si = 1:size(Ss,2)
for di = 1:size(denoiser_types,2)
S = Ss(si); F = S-1;
denoiser_type = denoiser_types{di};
key = sprintf("ADMM_%s_S_%d",upper(denoiser_type),S);
% parameters for optimization 
mask_type = "toeplitz";
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[A,~,~] = SubsampleMultiplexOperator(S,M);
Aop  = @(X) reshape(A*X(:),h,w,2);
ATop = @(Y) reshape(A'*Y(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S,'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
SaveIterateDirectory = sprintf('%s/Reconstruction/SinusoidsNoiseless/Recon_Sinusoids_%s',savedir,key);
params = GetDemosaicDemultiplexParams('SaveIterateDirectory',SaveIterateDirectory);
params.denoiser_type = denoiser_type;
% simulate missing data
is = ceil(linspace(1,30*(S-1)/S,S));
X = XX(:,:,is); P = PP(:,is);
Y = Aop(X);
% admm
[im_out,history] = ADMM(Y,A,InitEstFunc,params,X);
[phase,~,~] = DecodeZNCC(im_out,P,Bounds.LB,Bounds.UB);
fprintf('Phase SSIM %s:%2.3f\n',upper(denoiser_type),ComputeSSIM(gt.phase,phase));
histories.(key) = history;
xhats.(key) = im_out;
end
end
%% Some plotting 


% subplot(1,2,1);
% plot(1:params.outer_iters, histories.mf.psnrs,'LineWidth',3,'DisplayName','ADMM-MF'); hold on;
% plot(1:params.outer_iters, histories.tnrd.psnrs,'LineWidth',3,'DisplayName','ADMM-TNRD');
% legend; hold off;

phases = [];
Ss = [2:7 12 16 20 24];

for si = 1:size(Ss,2)
for di = [1,2]    
S = Ss(si); F = S-1;
if ~any(S==[3 5 7 14])
    continue
end
denoiser_type = denoiser_types{di};
key = sprintf("ADMM_%s_S_%d",upper(denoiser_type),S);
is = ceil(linspace(1,30*(S-1)/S,S)); P = PP(:,is);
[phase,~,~] = DecodeZNCC(xhats.(key),P,Bounds.LB,Bounds.UB);
[psnr,ssim] = ComputePSNRSSIM(gt.phase,phase);
fprintf('%s Phase PSNR/SSIM: %2.3f/%.3f\n',key,psnr,ssim);
phases = [phases phase];
% loss curve
plot(histories.(key).ssims,'DisplayName',strrep(key,'_','-'),'LineWidth',3); hold on;
end
end
legend('Location','southeast');
% imshow([gt.phase phases]/hproj);


%% phase PSNR on X reconstructed using ADMM (varying denoiser/#shifts) on noisy inputs


S = 7;

freq_and_shifts = [repmat([1],30,1) (1:30)'];
[XX,PP] = ParsaPatternSinusoidsGetNoisyIm(freq_and_shifts,1,blacklvl,hproj,cx,cy);

clear histories xhats;
denoiser_types = {'mf','tnrd'};
Ss = [2:7 12 16 20 24];
for si = 1:size(Ss,2)
for di = 1:size(denoiser_types,2)
S = Ss(si); F = S-1;
denoiser_type = denoiser_types{di};
key = sprintf("ADMM_%s_S_%d",upper(denoiser_type),S);
% parameters for optimization 
mask_type = "toeplitz";
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[A,~,~] = SubsampleMultiplexOperator(S,M);
Aop  = @(X) reshape(A*X(:),h,w,2);
ATop = @(Y) reshape(A'*Y(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S,'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
SaveIterateDirectory = sprintf('%s/Reconstruction/SinusoidsFreq1NoisyInputIm/%s',savedir,key);
params = GetDemosaicDemultiplexParams('SaveIterateDirectory',SaveIterateDirectory);
params.denoiser_type = denoiser_type;
% simulate missing data
is = ceil(linspace(1,30*(S-1)/S,S));
X = XX(:,:,is); P = PP(:,is);
Y = Aop(X);
% admm
[im_out,history] = ADMM(Y,A,InitEstFunc,params,X);
[phase,~,~] = DecodeZNCC(im_out,P,Bounds.LB,Bounds.UB);
[psnr,ssim]=ComputePSNRSSIM(gt.phase,phase);
fprintf('(%s) Phase PSNR/SSIM %2.3f/%2.3f\n',upper(denoiser_type),psnr,ssim);
histories.(key) = history;
xhats.(key) = im_out;
end
end

%% Some plotting 

for di = [1,2]    
denoiser_type = denoiser_types{di};
psnrs.(denoiser_type) = [];
ssims.(denoiser_type) = [];
end

phases = [];
Ss = [2:7 12 16 20 24];
for si = 1:size(Ss,2)
for di = [1 2]    
S = Ss(si); F = S-1;
% if ~any(S==[12 16 20 24])
%     continue
% end
denoiser_type = denoiser_types{di};
key = sprintf("ADMM_%s_S_%d",upper(denoiser_type),S);
% Pattern 
is = ceil(linspace(1,30*(S-1)/S,S));
P = PP(:,is);
% Last Iterate Image
SaveIterateDirectory = sprintf('%s/Reconstruction/SinusoidsFreq1NoisyInputIm/%s',savedir,key);
files = dir(sprintf("%s/*.png",SaveIterateDirectory)); last_file_i = size(files,1);
[fnames,ffolders] = deal({files.name},{files.folder});
impath = sprintf('%s/%s',ffolders{last_file_i},fnames{last_file_i});
X = double(imread(impath));
X = reshape(I,h,w,[]);
% Decoding
[phase,~,~] = DecodeZNCC(X,P,Bounds.LB,Bounds.UB);
[psnr,ssim] = ComputePSNRSSIM(gt.phase,phase);
psnrs.(denoiser_type) = [psnrs.(denoiser_type) psnr];
ssims.(denoiser_type) = [ssims.(denoiser_type) ssim];
fprintf('%s Phase PSNR/SSIM: %2.3f/%.3f\n',key,psnr,ssim);
phases = [phases phase];
% loss curve
% plot(histories.(key).psnrs,'DisplayName',strrep(key,'_','-'),'LineWidth',3); hold on;
end
end
% legend('Location','southeast');
% imshow([gt.phase phases]/hproj);
%% phase PSNR/SSIM table

% varNames = {}; varTypes = {};
% for si = 1:size(Ss,2)
%     S = Ss(si);
%     varTypes{si} = 'char';
%     varNames{si} = sprintf('S=%d',S);
% end
% colNames = {'ADMM-MF','ADMM-TNRD'};
% T = table('Size',[2 size(varNames,2)],'VariableTypes',varTypes,...
%         'VariableNames',varNames,'colNames',colNames);
%     
% T(1,:) = arrayfun(@(a,b) sprintf('%2.2f/%.3f',a,b),psnrs.('mf'),ssims.('mf'),'UniformOutput',false);
% T(2,:) = arrayfun(@(a,b) sprintf('%2.2f/%.3f',a,b),psnrs.('tnrd'),ssims.('tnrd'),'UniformOutput',false);

T = table()

input.data = T;
input.tableBorders = 0;
latex = LatexTable(input)
for i = 1:size(latex,1)
    fprintf("%s\n",latex{i});
end

%% See if the image formation model under different illuminations are the same

% denoiser_type = 'tnrd'; S = 7;
% key = sprintf("ADMM_%s_S_%d",upper(denoiser_type),S);
% is = ceil(linspace(1,30*(S-1)/S,S));
% P = PP(:,is);
% % Last Iterate Image
% SaveIterateDirectory = sprintf('%s/Reconstruction/SinusoidsFreq1NoisyInputIm/%s',savedir,key);
% files = dir(sprintf("%s/*.png",SaveIterateDirectory)); last_file_i = size(files,1);
% [fnames,ffolders] = deal({files.name},{files.folder});
% impath = sprintf('%s/%s',ffolders{last_file_i},fnames{last_file_i});
% X = double(imread(impath));
% X = reshape(X,h,w,[]);
% % Decoding

S = 7;
[XX,PP] = ParsaPatternSinusoidsGetStackedIm(hproj,1);
is = ceil(linspace(1,30*(S-1)/S,S));
X = XX(:,:,is); P = PP(:,is);

[albedo,~,phase] = DecodePhaseShiftWithDepthBound(X,Bounds.LB,Bounds.UB,hproj,1,'Shifts',(is-1)/30);
[psnr,ssim] = ComputePSNRSSIM(gt.phase, phase)
albedo = albedo*1.65;

phase = phase + eps;
Iq = zeros(h,w,S);
for i = 1:h
for j = 1:w
    Iq(i,j,:) = P(ceil(phase(i,j)),:);
end
end
Xhat = albedo.*Iq;

% Iq \in [0,1] since pattern is in [0,1]
imshow([albedo/255 disparityFunc(phase)/255]);
imshow(FlattenChannels(Iq));
% imshow([FlattenChannels(Xhat)/255; FlattenChannels(X-Xhat)/255]);
mesh()

fprintf('S=%d Mean X=%2.3f albedo=%2.3f phase=%2.3f xhat=%2.3f\n', ...
    S,mean(X,'all'),mean(albedo,'all'),mean(phase,'all'),mean(Xhat,'all'));


%%






