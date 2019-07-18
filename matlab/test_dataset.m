% generate stacked noiseless images 
clc; clear; close all;
addpath(genpath('./tnrd_denoising/'));
addpath(genpath('./minimizers/'));
addpath(genpath('./parameters/'));
addpath(genpath('./helper_functions/'));
addpath(genpath('./test_images/'));
addpath(genpath("./mian/helperFunctions/Camera"));
addpath(genpath("./mian/helperFunctions/ASNCC"));
addpath(genpath("./mian/helperFunctions/Algorithms"));

%% Parameters

% location of raw images captured
rawimagedir =  "~/github/cv_project/data/exp60";
% image size 
[h,w] = deal(176,288);
% number of subframes
S = 4;
% number of frames per pattern 
n_frames_total = 1000;
% pick oneself
n_frames_per_pattern = 250;
% number of scenes 
n_scenes = 10;
% black level 
blackLevel = load("./mian/CalibrationCode/BlackIms.mat");
blackLvl = blackLevel.blackLvl;


%% 

files = dir(sprintf("%s/bowl/*.png",rawimagedir));
fnames = {files.name};
folders = {files.folder}; folder = folders{1};


for i = 1:size(fnames,2)
    fname = fnames{i};
    splits = split(fname,' ');
    [bkt,id] = deal(splits{1},splits{2});
    impath = sprintf("%s/%s",folder,fname);
    im1 = double(imread(impath));
    im2 = double(BlackLevelRead(impath,blackLvl,2));
    
    imshow([im1 im2]/255);

    break
end

