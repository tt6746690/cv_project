% Multispectral reconstruction 
%
clc; clear; close all;
ProjectPaths;
warning('off', 'MATLAB:MKDIR:DirectoryExists');

%% params 

outdir = 'results/multispectral';
scaling=2;
light_mode = false;
[S,F] = deal(5,4);
[h,w] =  deal(320,324);
M = SubsamplingMask("tiles",h,w,F,'Tile',[1 2; 3 4]);
W = BucketMultiplexingMatrix(S);
[H,B,C] = SubsampleMultiplexOperator(S,M);
ForwardFunc = @(in_im) reshape(H*in_im(:),h,w,2);
BackwardFunc = @(in_im) reshape(H'*in_im(:),h,w,S);
InitEstFunc = InitialEstimateFunc("maxfilter",h,w,F,S,'BucketMultiplexingMatrix',W,'SubsamplingMask',M);
params_admm = GetDemosaicDemultiplexParams(light_mode);
perm = [5 2 4 3 1];
scenes = [
    "Rubiks_FPS=25.83"
    "Bunny_FPS=26.54"
    "Giraffe_FPS=25.61"
    "Cloud_FPS=22.02"
];

scene = scenes(1);
rawimagedir = sprintf('data/MultiSpectral/%s/',scene);

%% 


%% initfunc 
% 

for si = 1:size(scenes,1)
scene = scenes(si);
subdir = 'initfunc';
mkdir(sprintf('%s/%s/%s',outdir,scene,subdir));
for i = 1:999
    scene,i
    ims = reconstruct_im(i,h,w,perm,InitEstFunc,scaling,rawimagedir,scene,InitEstFunc);
    imwrite(uint8(scaling*ims),sprintf('%s/%s/%s/%.4d.png',outdir,scene,subdir,i-1));
end
break;
end

%% admmparams

subdir = 'admmparams';
mkdir(sprintf('%s/%s/%s',outdir,scene,subdir));
denoiser_types = [
    "medfilter",
    "tnrd" 
];

for outer_iters = [5 10 20 40 80]
params_admm.outer_iters = outer_iters;

for di = [1 2]
    denoiser_type = denoiser_types(di);
    params_admm.denoiser_type = denoiser_type;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    outer_iters
    denoiser_type

    input_im = imread(sprintf('%s/%04d.png',rawimagedir,i-1));
    input_im = double(cat(3,input_im(:,1:w),input_im(:,(w+1):(2*w))));

    reconfunc = @(input_im) run_admm(input_im,H,InitEstFunc,params_admm,zeros(h,w,S),'RatioIntensity','ratio','ADMMFunc',@ADMMSmooth);
    ims = reconstruct_im(i,h,w,perm,InitEstFunc,scaling,rawimagedir,scene,reconfunc);
    imwrite(uint8(scaling*ims),sprintf('%s/%s/%s/smooth_%s_%d_%.4d.png',outdir,scene,subdir,denoiser_type,outer_iters,i-1));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
end

%% create video with medfilter,outer_iters=10,ADMMSmooth

params_admm.outer_iters = 10;
params_admm.denoiser_type = 'medfilter';

for si = 1:size(scenes,1)
scene = scenes(si);
subdir = 'smooth_medfilter_10';
mkdir(sprintf('%s/%s/%s',outdir,scene,subdir));
for i = 1:999
    scene,i
    reconfunc = @(y) run_admm(y,H,InitEstFunc,params_admm,zeros(h,w,S),'RatioIntensity','intensity','ADMMFunc',@ADMMSmooth);
    ims = reconstruct_im(i,h,w,perm,InitEstFunc,scaling,rawimagedir,scene,reconfunc);
    imwrite(uint8(scaling*ims),sprintf('%s/%s/%s/%.4d.png',outdir,scene,subdir,i-1));
end
break;
end


%%

function y = to_color_im(x)
    y = x;
    y(:,:,3) = 0.4*x(:,:,3)+0.4*x(:,:,4); % white balance
    y = y(:,:,[3 2 1]); % bgr -> rgb
end


function ims = gather_multispectral_ims(im)
    [h,w,S]=size(im);
    ims = [im(:,:,1) im(:,:,2) im(:,:,3)
        im(:,:,4) im(:,:,5) zeros(size(im(:,:,1)))];
    ims = repmat(ims,[1 1 3]);
    ims(h+1:end,2*w+1:end,:) = to_color_im(im);
end


function ims = reconstruct_im(i,h,w,perm,InitEstFunc,scaling,rawimagedir,scene,reconfunc)
    input_im = imread(sprintf('%s/%04d.png',rawimagedir,i-1));
    input_im = double(cat(3,input_im(:,1:w),input_im(:,(w+1):(2*w))));

    demul_im = reconfunc(input_im);
    demul_im = demul_im(:,:,perm);

    ims = gather_multispectral_ims(demul_im);
end