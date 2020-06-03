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
hproj = 684;
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
imwrite(uint8(im*255),sprintf("%s/relphase_vs_K.png", ...
        savedir_cur));
    
 
%% get absolute phase / disparity using chinese reminder

relphases = zeros(h,w,size(spatial_freqs,2));

for i = 1:size(spatial_freqs,2)
spatial_freq = spatial_freqs(i);

imagedir=sprintf('results/reconstruction_parsapattern/Sinusoids/Freq%02d',spatial_freq);
files = dir(sprintf("%s/*.png",imagedir));
[fnames,ffolders] = deal({files.name},{files.folder});

K = size(fnames,2);
stacked_ims = zeros(h,w,K);
for k = 1:K
    stacked_ims(:,:,k) = imread(sprintf('%s/%s',ffolders{k},fnames{k}));
end

if spatial_freq == 5
    shifts = [0:15 17:K];
else
    shifts = 1:K;
end
phase_shifts = (shifts-1)*2*pi/K;

relphase = PhaseShiftingSolveRelativePhase(stacked_ims,phase_shifts);
relphases(:,:,i) = relphase;
end

imshow(FlattenChannels(relphases))
imwrite(uint8(FlattenChannels(mat2gray(relphases))*255),sprintf("%s/relphases.png", ...
        savedir_cur));
    
    
% denoise
% I = find(~isfinite(relphases));
% relphases(I) = eps;
% for i = 1:size(relphases,3)
%     relphases(:,:,i) = wdenoise2(relphases(:,:,i));
% end

imshow(FlattenChannels(relphases))
imwrite(uint8(FlattenChannels(mat2gray(relphases))*255),sprintf("%s/relphases_denoised.png", ...
        savedir_cur));

%%
 
P = 0.5 + 0.5*cos(spatial_freqs.*(0:hproj-1)'*2*pi/hproj);
P = floor(P * 24) / 24;

imagesc(P);

absphases = zeros(size(P,1),1);
for i = 1:size(P,1)
    absphase = Chinese(P(i,:),spatial_freqs);
    absphases(i) = absphase;
end

% nearest neighbor search 

D = pdist2(P,reshape(relphases,[],5),'euclidean');
[~,I] = min(D,[],1);
Phi = absphases(I);

Phi = reshape(Phi,h,w);
I = reshape(I,h,w);
imshow([mat2gray(I) mat2gray(Phi)]);

% CF = [4 5]; % choose spatial frequency
% relphasesp = relphases(:,:,CF);
% relphasesp = reshape(relphasesp,[],size(spatial_freqs(CF),2));
% 
% X = [];
% for i = 1:size(relphasesp,1)
%     per_pixel_relphase = relphasesp(i,:);
%     x = Chinese(per_pixel_relphase, spatial_freqs(CF));
%     X = [X;x];
% end
% 
% im = reshape(X/prod(spatial_freqs(CF)),h,w);
% imagesc(im)


%% get disparity using zncc

PatternCoeff = 0.5 + 0.5*cos(spatial_freqs(CF).*(0:hproj-1)'*2*pi/hproj);
PatternCoeff = floor(PatternCoeff * 24) / 24;

[phase,zncc,I] = DecodeZNCC(relphases,PatternCoeff,Bounds.LB,Bounds.UB);

imshow(mat2gray(phase))
