[h,w] = deal(176,512);

%% For bayer

[F,S] = deal(3,4);
mask_type = "bayer";
filepath = sprintf("./mian/Patterns/MaskPatterns/%s_S=%d",mask_type,S);
SensorMask = GenerateSensorMask(mask_type,F,S,h,w,filepath);

foo = SensorMask(


%% For other masks

mask_types = [
    "horz"
    "vert"
    "toeplitz"
    "random"
]';

for S = 3:7
    F = S-1;
    for mask_type = mask_types
        filepath = sprintf("./mian/Patterns/MaskPatterns/%s_S=%d",mask_type,S);
        SensorMask = GenerateSensorMask(mask_type,F,S,h,w,filepath);
    end
end


%% for a fixed exposure time, trig_value, S, compute exposure/subframe to a fixed exposure (in ms)


total_exposure = 420; % ms
S = 4;
trigger_value = 24;
exposure_per_subframe = ((total_exposure/0.01)/S)-trigger_value





% (((total_exposure/0.01)/S)+205-trig_value)/trig_value = int((205+exposure/trig_value))


% (int((205+exposure)/trig_value)*trig_value + trig_value - 205)*n_subframes * 0.01ms


60*0.01



% % 84 
% subframes = 7;
% numImages = 12;

% % 96
% subframes = 6;
% numImages = 16;

% % 80
% subframes = 5;
% numImages = 16;

% % 96
% subframes = 4;
% numImages = 24;


