
function [X,P] = ParsaPatternSinusoidsGetNoisyIm(freq_and_shifts,noisy_input_im_index,blacklvl,hproj,cx,cy)
    n_ims = size(freq_and_shifts,1);
    X = [];

    for i = 1:n_ims
    spatial_freq = freq_and_shifts(i,1);
    shift = freq_and_shifts(i,2);
    if spatial_freq == 5 && shift == 16
        error('Missing Data');
    end

    if spatial_freq == 1
    imagedir = sprintf('data/ParsaPatterns/Sinusoids/Freq%02d/Shift%02d', ...
            spatial_freq, shift-1);
    else
    imagedir = sprintf('data/ParsaPatterns/Sinusoids/Freq%02d/P%d', ...
            spatial_freq, shift);
    end

    files = dir(sprintf("%s/bucket1*.png",imagedir));
    [fnames,ffolders] = deal({files.name},{files.folder});
    impath = sprintf('%s/%s',ffolders{noisy_input_im_index},fnames{noisy_input_im_index});
    im = double(BlackLevelRead(impath,blacklvl,1));  % note all light go to bkt-1
    im = im(cx,cy);
    X = cat(3,X,im);
    end

    P = zeros(hproj,n_ims);
    for i = 1:n_ims
    spatial_freq = freq_and_shifts(i,1);
    shift = freq_and_shifts(i,2);
    P(:,i) = 0.5 + 0.5*cos(spatial_freq*(0:hproj-1)'*2*pi/hproj + (shift-1)*2*pi/30 );
    end
end

