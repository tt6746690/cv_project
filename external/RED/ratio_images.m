%% evaluate performance of ratio/non-ratio images
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

% #patterns/frames
[S,F] = deal(4,3);
% dimension of input image
[h,w] = deal(176,288);
% initial guess

% data directory 





%% 


savedir = 'results/red/7patterns';