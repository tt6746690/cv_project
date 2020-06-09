
function [I,P] = ParsaPatternSinusoidsGetStackedIm(hproj,spatial_freq)
    % get stackedim and pattern for sinusoidal coding scheme
    imagedir=sprintf('results/reconstruction_parsapattern/Sinusoids/Freq%02d',spatial_freq);
    files = dir(sprintf("%s/*.png",imagedir));
    [fnames,ffolders] = deal({files.name},{files.folder});
    K = size(fnames,2);
    I = [];
    for k = 1:K
        I = cat(3,I,double(imread(sprintf('%s/%s',ffolders{k},fnames{k}))));
    end
    shifts = 1:30;
    P = 0.5 + 0.5*cos(spatial_freq*(0:hproj-1)'*2*pi/hproj + (shifts-1)*2*pi/30 );
end
