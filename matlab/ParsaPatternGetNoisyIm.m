function [X,P] = ParsaPatternGetNoisyIm(coding_scheme,noisy_input_im_index,blacklvl,hproj,cx,cy)

    S = 7;
    X = [];
    
    for s = 1:S
    imagedir = sprintf('./data/ParsaPatterns/%s/P%d/',coding_scheme,s);
    files = dir(sprintf("%s/bucket1*.png",imagedir));
    [fnames,ffolders] = deal({files.name},{files.folder});
    impath = sprintf('%s/%s',ffolders{noisy_input_im_index},fnames{noisy_input_im_index});
    im = double(BlackLevelRead(impath,blacklvl,1));  % note all light go to bkt-1
    im = im(cx,cy);
    X = cat(3,X,im);
    end
    
    if strcmp(coding_scheme,'Hamiltonian')
        PatternMatrix = load('./data/ParsaPatterns/Hamiltonian/Pi_Ham_608_7_1.mat');
        P = PatternMatrix.Pi';
    else
        PatternMatrix = load('./data/ParsaPatterns/MPS/PatternMat.mat');
        P = PatternMatrix.patternMatrix;
    end
    
end