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

%% Given groundtruth albedo / ambient illumination, find phase 


S = 7;
spatial_freq = 1;
[XX,PP] = ParsaPatternSinusoidsGetStackedIm(hproj,spatial_freq);
is = ceil(linspace(1,30*(S-1)/S,S));
X = XX(:,:,is); P = PP(:,is);

[albedo,~,phase,b] = DecodePhaseShiftWithDepthBound(X,Bounds.LB,Bounds.UB,hproj,1,'Shifts',(is-1)/30);
albedo = 1.74*albedo;

phase = phase + eps;
Iq = zeros(h,w,S);
for i = 1:h
for j = 1:w
    Iq(i,j,:) = P(ceil(phase(i,j)),:);
end
end
Xhat = albedo.*Iq;

imshow([albedo/255 disparityFunc(phase)/255 b/255]);
mesh(FlattenChannels(Xhat-X));

% multiply albedo by 1.74
% b = zeros(h,w);
% alphas = 0:0.02:5;
% err = arrayfun(@(alpha) norm(FlattenChannels((alpha*albedo).*Iq + b - X)), alphas);
% [M,I] = min(err)
% alphas(I)

%% 
lambda = 1000;
h_= h; w_ = w;
npixels = h_*w_;

zt = phase(1:npixels)'/hproj;
b = zeros(h,w);
X_ = reshape(X,[],S);
[Gx,Gy] = ImGrad([h_ w_]);
G = [Gx;Gy];

lb = zeros(npixels,1);
ub = lb + 1;
% z0 = (lb+ub)/2+0.5*(rand(size(lb)));
z0 = zt + 0.1*rand(size(lb));

% [0,1] -> \R^S
I = @(z) 0.5 + 0.5*cos(spatial_freq*2*pi*z + (is-1)*2*pi/30 );
Igrad = @(z) -0.5*sin(spatial_freq*2*pi*z + (is-1)*2*pi/30 );
% l2 norm at pixel p \in 1,2,...,h*w
fp = @(z,p) norm(X_(p,:) - albedo(p)*I(z(p)) - b(p)*ones(1,S));
% regularized objective 
f = @(z) sum(arrayfun(@(p) fp(z,p),1:npixels)) + lambda*norm(G*z,1);
% gradient 
g = @(z) sum((X_ - albedo(:).*I(z) - b(:).*ones(1,S)) .* Igrad(z), 2);

options = optimoptions('fmincon','Diagnostics','on','Display',...
    'iter-detailed','MaxIterations',100,'Algorithm','interior-point',...
    'HessianApproximation','lbfgs','SpecifyObjectiveGradient',true,'PlotFcn','optimplotstepsize');
% funcon on P pixels too high dimensional to do effective optimization
[x,fval,exitflag,output,lambda,~,~] = fmincon(@(z) problem(z,f,g),z0,[],[],[],[],lb,ub,'',options);

% plot(z0);hold on; plot(z); hold on; plot(zt);
% legend('z0','z','true'); hold off;

imshow([reshape(z0,h_,w_) mat2gray(reshape(g(z),h_,w_)) reshape(z,h_,w_) reshape(zt,h_,w_)]);

function [fz,gz] = problem(z,f,g)
    fz = f(z);
    if nargout > 1 % gradient required
        gz = g(z);
    end
end
