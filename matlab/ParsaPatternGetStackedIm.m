function [X,P] = ParsaPatternGetStackedIm(coding_scheme)
    % get stackedim and pattern for Hamiltonian/MPS/Optimized-etc.
    imagedir=sprintf('results/reconstruction_parsapattern/%s',coding_scheme);
    files = dir(sprintf("%s/*.png",imagedir));
    [fnames,ffolders] = deal({files.name},{files.folder});
    K = size(fnames,2); X = [];
    for k = 1:K
        X = cat(3,X,double(imread(sprintf('%s/%s',ffolders{k},fnames{k}))));
    end

    if strcmp(coding_scheme,'Hamiltonian')
        PatternMatrix = load('./data/ParsaPatterns/Hamiltonian/Pi_Ham_608_7_1.mat');
        P = PatternMatrix.Pi';
    else
        PatternMatrix = load('./data/ParsaPatterns/MPS/PatternMat.mat');
        P = PatternMatrix.patternMatrix;
    end
end


