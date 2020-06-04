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
disparityFunc = @(corres,pos) (corres - 2.7*pos);
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


%% stack images for Sinusoids
%  light all goes to bkt-1
%  from sinusoids with Freq [1 2 5 17 31]
%  each shifted by (i-1)*2*pi/30, i.e. 30 shifts

savedir_cur = sprintf("%s/Sinusoids",savedir); mkdir(savedir_cur);

spatial_freqs = [1 2 5 17 31];
shifts = 1:30;

for spatial_freq = spatial_freqs
for shift_i = shifts
    
% problematic case ... missing data
if all([spatial_freq shift_i] == [5 16])
    continue
end

if spatial_freq == 1
imagedir = sprintf('data/ParsaPatterns/Sinusoids/Freq%02d/Shift%02d', ...
        spatial_freq, shift_i-1);
else
imagedir = sprintf('data/ParsaPatterns/Sinusoids/Freq%02d/P%d', ...
        spatial_freq, shift_i);
end

files = dir(sprintf("%s/bucket1*.png",imagedir));
[fnames,ffolders] = deal({files.name},{files.folder});

n_im = size(fnames,2);
ims = zeros(h,w,n_im);
for i = 1:n_im
    impath = sprintf('%s/%s',ffolders{i},fnames{i});
    im = double(BlackLevelRead(impath,blacklvl,1));
    im = im(cx,cy);
    ims(:,:,i) = im;
end

stackedim = mean(ims,3);
im = [mat2gray(ims(:,:,1)) mat2gray(stackedim)];
imshow(im);

mkdir(sprintf("%s/Freq%02d",savedir_cur,spatial_freq));
imwrite(uint8(stackedim),sprintf("%s/Freq%02d/Shift%02d.png", ...
        savedir_cur,spatial_freq,shift_i));

end
end


%% create groundtruth disparity for sinusoids patterns

savedir_cur = sprintf("%s/Sinusoids",savedir); mkdir(savedir_cur);

% closed form 
spatial_freq = 5;

imagedir=sprintf('results/reconstruction_parsapattern/Sinusoids/Freq%02d',spatial_freq);
files = dir(sprintf("%s/*.png",imagedir));
[fnames,ffolders] = deal({files.name},{files.folder});

K = size(fnames,2);
ims = zeros(h,w,K);
for k = 1:K
    ims(:,:,k) = imread(sprintf('%s/%s',ffolders{k},fnames{k}));
end

if spatial_freq == 5
    shifts = [0:15 17:K];
else
    shifts = 1:K
end

phase_shifts = (shifts-1)*2*pi/K;

im = []
for nimages = [4 10 30]
    is = int8(linspace(1,K,nimages));
    im = [im PhaseShiftingSolveRelativePhase(ims(:,:,is), phase_shifts(is))];
end

imshow(im)
imwrite(uint8(im*255),sprintf("%s/relphase_vs_K.png",savedir_cur));
    
    
 
%% solve for disparity using zncc

relphases = zeros(h,w,size(spatial_freqs,2));
stacked_ims = [];
freqs = []; % spatial freqs
shifts = []; % shifts

for i = 1:size(spatial_freqs,2)
spatial_freq = spatial_freqs(i);

imagedir=sprintf('results/reconstruction_parsapattern/Sinusoids/Freq%02d',spatial_freq);
files = dir(sprintf("%s/*.png",imagedir));
[fnames,ffolders] = deal({files.name},{files.folder});

K = size(fnames,2);
stacked_ims_ = zeros(h,w,K);
for k = 1:K
    stacked_ims_(:,:,k) = imread(sprintf('%s/%s',ffolders{k},fnames{k}));
end

if spatial_freq == 5
    varphi = [0:15 17:K];
else
    varphi = 1:K;
end
phase_shifts = (varphi-1)*2*pi/K;

relphase = PhaseShiftingSolveRelativePhase(stacked_ims_,phase_shifts);
relphases(:,:,i) = relphase;

freqs = [freqs repmat([spatial_freq],1,size(varphi,2))];
shifts = [shifts varphi];
stacked_ims = cat(3,stacked_ims,stacked_ims_);
end

imshow(FlattenChannels(relphases))
imwrite(uint8(FlattenChannels(mat2gray(relphases))*255),sprintf("%s/relphases.png",savedir));
imwrite(uint8(FlattenChannels(mat2gray(relphases))*255),sprintf("%s/relphases.png",assetsdir));

    
P = zeros(hproj, 149);
for i = 1:size(freqs,2)
    freq = freqs(i);
    shift = shifts(i);
    if i > 30 && i <= 59
        P(:,i) = 0.5 + 0.5*cos(freq*(0:hproj-1)'*2*pi/hproj + (shift-1)*2*pi/29 );
    else
        P(:,i) = 0.5 + 0.5*cos(freq*(0:hproj-1)'*2*pi/hproj + (shift-1)*2*pi/30 );
    end 
end


tbs = [];
for nIs = [5 7 9 11 13 50 149]
    Is = datasample(RandStream('mlfg6331_64'),1:149,nIs,'Replace',false);
    [phase,zncc,I] = DecodeZNCC(stacked_ims(:,:,Is),P(:,Is),Bounds.LB,Bounds.UB);
    tbs = [tbs phase];
end

imshow(mat2gray(tbs))
imwrite(uint8(255*tbs/hproj),sprintf("%s/sinusoids_disparity_vs_random_shifts.png",savedir));
imwrite(uint8(255*tbs/hproj),sprintf("%s/sinusoids_disparity_vs_random_shifts.png",assetsdir));


phase_gt = phase;
disparity_gt = disparityFunc(phase,Y);
imwrite(uint8(disparity_gt),sprintf('%s/disparity_gt.png', savedir));
imwrite(uint8(255*phase_gt/hproj),sprintf('%s/phase_gt.png', savedir));


%% stack images for Hamiltonian/MPS/Optimized-MDE/Optimized-Top0/Optimized-Top1/Optimized-Top2
%  7 patterns

S = 7;
coding_schemes = {'Hamiltonian','MPS','Optimized-MDE','Optimized-Top0','Optimized-Top1','Optimized-Top2'};

for ii = 1:size(coding_schemes,2)
coding_scheme = coding_schemes{ii}
savedir_cur = sprintf("%s/%s",savedir,coding_scheme); mkdir(savedir_cur);
for s = 1:S

imagedir = sprintf('./data/ParsaPatterns/%s/P%d/',coding_scheme,s);
files = dir(sprintf("%s/bucket1*.png",imagedir));
[fnames,ffolders] = deal({files.name},{files.folder});

% 250 noisy images 
n_im = size(fnames,2);
ims = zeros(h,w,n_im);
for i = 1:n_im
    impath = sprintf('%s/%s',ffolders{i},fnames{i});
    im = double(BlackLevelRead(impath,blacklvl,1));
    im = im(cx,cy);
    ims(:,:,i) = im;
end

stackedim = mean(ims,3);
im = [mat2gray(ims(:,:,1)) mat2gray(stackedim)];
imshow(im);

imwrite(uint8(stackedim),sprintf("%s/P%d.png",savedir_cur,s));
end
end

%% zncc decoding for Hamiltonian/MPS/Optimized-MDE/Optimized-Top0/Optimized-Top1/Optimized-Top2

savedir_phase = sprintf("%s/ZNCCDecodingPhase",savedir); mkdir(savedir_phase);
savedir_disparity = sprintf("%s/ZNCCDecodingDisparity",savedir); mkdir(savedir_disparity);

coding_schemes = {'Hamiltonian','MPS','Optimized-MDE','Optimized-Top0','Optimized-Top1','Optimized-Top2'};
disparitys = [];
phases = [];
psnrs_disparity = [];
psnrs_phase = [];

use_mat2gray = false;
% if use_mat2gray
%     disparity_gt_ = mat2gray(disparity_gt);
%     phase_gt_ = mat2gray(phase_gt);
% else
%     disparity_gt_ = disparity_gt;
%     phase_gt_ = phase_gt;
% end
%     

for ii = 1:size(coding_schemes,2)
coding_scheme = coding_schemes{ii}
if strcmp(coding_scheme,'Hamiltonian')
    PatternMatrix = load('./data/ParsaPatterns/Hamiltonian/Pi_Ham_608_7_1.mat');
    P = PatternMatrix.Pi';
else
    PatternMatrix = load('./data/ParsaPatterns/MPS/PatternMat.mat');
    P = PatternMatrix.patternMatrix;
end
imagedir=sprintf('results/reconstruction_parsapattern/%s',coding_scheme);
files = dir(sprintf("%s/*.png",imagedir));
[fnames,ffolders] = deal({files.name},{files.folder});
K = size(fnames,2);
I = zeros(h,w,K);
for k = 1:K
    I(:,:,k) = imread(sprintf('%s/%s',ffolders{k},fnames{k}));
end
[phase,~,~] = DecodeZNCC(I,P,Bounds.LB,Bounds.UB);
disparity = disparityFunc(phase,Y);
imwrite(uint8(255*phase/hproj), sprintf('%s/ZNCCDecodingPhase/%s.png',savedir,coding_scheme));
imwrite(uint8(disparity), sprintf('%s/ZNCCDecodingDisparity/%s.png',savedir,coding_scheme));

psnrs_disparity = [psnrs_disparity ComputePSNR(disparity_gt,disparity)];
psnrs_phase = [psnrs_phase ComputePSNR(phase_gt,phase)];
phases = [phases phase];
disparitys = [disparitys disparity];
end

figure('Renderer', 'painters', 'Position', [10 10 300*2 300])
subplot(1,2,1);
bar(categorical(coding_schemes),psnrs_disparity);
subplot(1,2,2);
bar(categorical(coding_schemes),psnrs_phase);

ims = [disparity_gt_ disparitys];
imwrite(uint8(ims), sprintf('%s/zncc_decoding_disparity.png',savedir));
imwrite(uint8(ims), sprintf('%s/zncc_decoding_disparity.png',assetsdir));

ims = 255*[phase_gt_ phases]/hproj;
imwrite(uint8(ims), sprintf('%s/zncc_decoding_phase.png',savedir));
imwrite(uint8(ims), sprintf('%s/zncc_decoding_phase.png',assetsdir));
