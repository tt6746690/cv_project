%% See if can use 1st order continuous optimization to recover phase, subject to some regularizer
%
clc; clear; close all;
ProjectPaths;

%% Setup

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
noisy_input_im_index = 1;

%% Given groundtruth albedo / ambient illumination, find phase 

% sinusoids 
spatial_freq = 2;
S = 24;
is = ceil(linspace(1,30*(S-1)/S,S));
freq_and_shifts = [repmat([1],S,1) is'];
[XX,PP] = ParsaPatternSinusoidsGetNoisyIm(freq_and_shifts,1,blacklvl,hproj,cx,cy);
X = XX; P = PP;

% hamiltonian 
S = 7;
coding_scheme = 'Hamiltonian';
[X,P] = ParsaPatternGetNoisyIm(coding_scheme,noisy_input_im_index,blacklvl,hproj,cx,cy);
LinInterp = @(x,y,c) (1-c).*x + c.*y;
[psnr,ssim] = ComputePSNRSSIM(gt.phase, DecodeZNCC(X,P,Bounds.LB,Bounds.UB))  

%%
% parameters for optimization 
F = S-1;
denoiser_type = 'tnrd';
mask_type = "toeplitz";
M = SubsamplingMask(mask_type,h,w,F);
W = BucketMultiplexingMatrix(S);
[A,~,~] = SubsampleMultiplexOperator(S,M);
Aop  = @(X) reshape(A*X(:),h,w,2);
ATop = @(Y) reshape(A'*Y(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S,'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
%SaveIterateDirectory = sprintf('%s/Reconstruction/SinusoidsRegularized/Recon_Sinusoids_%s',savedir,key);
params = GetDemosaicDemultiplexParams('SaveIterateDirectory','');
params.denoiser_type = denoiser_type;
% simulate missing data
Y = Aop(X);
% admm
[im_out,history] = ADMM(Y,A,InitEstFunc,params,X);
[phase,~,~] = DecodeZNCC(im_out,P,Bounds.LB,Bounds.UB);
[psnr,ssim] = ComputePSNRSSIM(gt.phase, phase)  
[psnr,ssim] = ComputePSNRSSIM(gt.phase, Denoiser(phase,5,'mf'))

imshow([disparityFunc( phase  ) disparityFunc( Denoiser(phase,5,'mf')  )]/255)

%%
% freq=1,S=7,noisy im           26.8787, 0.864
% freq=1,S=7,reconstructed im   19.4505, 0.8396
% freq=1,S=7,denoised           20.9802, 0.8827 *
% freq=1,S=7,finetune phase     20.1797, 0.8581
% freq=1,S=7,denoise phase      21.8536, 0.9218
% after readjust for albedo     ~       ,0.8979
% freq=1,S=7,denoise again      22.0902, 0.9236
% after readjust for albedo     20.7488, 0.8578
% freq=1,S=7,denoise again      22.4455, 0.9239 *
%                               +1.46    +0.041

% freq=1,S=24
% noisy im (gt)                 29.0545  0.9096
% reconstructed                 23.4951  0.8476
% denoised                      23.7977  0.8584 *
% finetune phase                25.7843  0.9062
% denoised                      26.7942  0.9405
% adjust for albedo
% finetune phase                25.9786  0.9079
% denoised                      26.8550  0.9430
%                               +3.057   +0.0846 *

% S=7, Hamiltonian              
% noisy im (gt)                 15.4316  0.8655
% reconstructed                 14.4423  0.7912
% denoised                      15.1446  0.8373
%

% should be even better if evaluate l2 loss in c2b 2 image domain, not in
%     reconstructed domain ... 
% what pattern to use should not be of too much concern, even if the method
%     works with sinusoids with f=1, would still be useful potentially

% 21.8861, 0.6588
%
% 
b = zeros(h,w);
albedo = Denoiser(mean(X,3),5,'mf');
phase = phase + eps;
Iz = zeros(h,w,S);
for i = 1:h
for j = 1:w
    Iz(i,j,:) = P(ceil(phase(i,j)),:);
end
end
Xhat = albedo.*Iz;
mesh(FlattenChannels(Xhat-X));
imshow([10*albedo/255 disparityFunc(phase)/255 b/255]);

%% 
lambda = 0;
h_= h; w_ = w;
npixels = h_*w_;

toim = @(x) reshape(x,h_,w_);

phase = 0.1+0.8*rand(h*w,1)*hproj;
zt = phase(1:npixels)'/hproj;
X_ = reshape(X,[],S);
[Gx,Gy] = ImGrad([h_ w_]);
G = [Gx;Gy];

lb = zeros(npixels,1) + 1/hproj;
ub = ones(npixels,1) - 1/hproj;
z0 = (lb+ub)/2+0.1*(rand(size(lb)));
% z0 = (lb+ub)/2;
z0 = 0.5*rand(size(lb));
%z0 = 0.5+0.5*rand(size(lb));
z0 = reshape(phase,h*w,1)/hproj;
%z0 = z;
%z0 = zt + 0.1*rand(size(lb));

% [0,1] -> \R^S
% sinusoids
I = @(z) 0.5 + 0.5*cos(2*pi*freq_and_shifts(:,1)'.*z + (freq_and_shifts(:,2)'-1)*2*pi/30 );
Igrad = @(z) -spatial_freq*pi*sin(2*pi*freq_and_shifts(:,1)'.*z + (freq_and_shifts(:,2)'-1)*2*pi/30 );
% Hamiltonian
I = @(z) LinInterp(P(floor(z*hproj),:), P(ceil(z*hproj),:), mod(z*hproj,1));
Igrad = @(z) (I(z)>0 & I(z)<1)*(2^S-2)/hproj;
% l2 norm at pixel p \in 1,2,...,h*w
f1p = @(z,p) norm(X_(p,:) - albedo(p)*I(z(p)) - b(p)*ones(1,S));
% regularized objective 
f1 = @(z) sum(arrayfun(@(p) f1p(z,p),1:npixels));
f2 = @(z) lambda*norm(G*z,1);
f = @(z) f1(z) + f2(z);
% gradient 
g1p = @(z,p) -2*albedo(p)*sum( (X_(p,:) - albedo(p)*I(z(p)) - b(p)).*Igrad(z(p)),2 );
g1 = @(z) arrayfun(@(p) g1p(z,p),1:npixels);
g2 = @(z) lambda*G'*sign(G*z);
g = @(z) g1(z) + g2(z);

% convexity of per-pixel function of f1
z = 0.1+0.8*rand(h*w,1);
p=500; ep = zeros(size(z)); ep(p) = 1; z(p)=0; ys = arrayfun(@(mult) f1(z+mult*ep),0.1:0.02:0.9); plot(ys)

%%

options = optimoptions('fmincon','Diagnostics','on','Display',...
    'iter-detailed','MaxIterations',100,'Algorithm','interior-point',...
    'HessianApproximation','lbfgs','SpecifyObjectiveGradient',true,...
    'PlotFcn','optimplotfval','OutputFcn',@outfun,'MaxFunctionEvaluations',100000);
[z,fval,exitflag,output,M,ggrad,~] = fmincon(@(z) problem(z,f1,g1),z0,[],[],[],[],lb,ub,'',options);

imshow([reshape(zt,h_,w_) reshape(z0,h_,w_) reshape(z,h_,w_) ...
    mat2gray([reshape(g1(z),h_,w_) reshape(g2(z),h_,w_)]) ]);
norm(g1(z)),norm(g2(z))

[psnr,ssim]= ComputePSNRSSIM(gt.phase, reshape(z,h,w)*hproj)  % 28.1196 0.8864
[psnr,ssim]= ComputePSNRSSIM(gt.phase, (Denoiser(reshape(z*hproj,h,w),5,'mf')))  % 28.1196 0.8864


%% optimize for a,b again 

z = reshape(z,[],1);
Iz = I(z);

M = [
    spdiags([Iz(:,1)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,2)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,3)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,4)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,5)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,6)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,7)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,8)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,9)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,10)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,11)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,12)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,13)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,14)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,15)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,16)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,17)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,18)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,19)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,20)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,21)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,22)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,23)],0,h*w,h*w) speye(h*w);
    spdiags([Iz(:,24)],0,h*w,h*w) speye(h*w);
    ];

ab = M\X(:);

albedon = ab(1:h*w);
bn = ab(h*w+1:2*h*w);
Xn = reshape(albedon .* Iz + bn,h,w,[])
albedon = reshape(albedon,h,w);
bn = reshape(bn,h,w);

imshow([albedo albedon (albedon-albedo) bn*20]/255);
albedo = albedon; b = bn;


function [fz,gz] = problem(z,f,g)
    fz = f(z);
    if nargout > 1 % gradient required
        gz = g(z);
    end
end
function stop = outfun(x,optimValues,state)
stop = false;
 
   switch state
       case 'init'
           hold on
       case 'iter'
           imshow([reshape(x,160,238)]);
% 
%            % Concatenate current point and objective function
%            % value with history. x must be a row vector.
%            history.fval = [history.fval; optimValues.fval];
%            history.x = [history.x; x];
%            % Concatenate current search direction with 
%            % searchdir.
%            searchdir = [searchdir;...
%                         optimValues.searchdirection'];
%            plot(x(1),x(2),'o');
%            % Label points with iteration number.
%            % Add .15 to x(1) to separate label from plotted 'o'
%            text(x(1)+.15,x(2),num2str(optimValues.iteration));
       case 'done'
           hold off
       otherwise
   end
end

%% 
