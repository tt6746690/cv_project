% pattern matrix for optimized code!
%       C = GeneratePatternMatrix(684,S);
%
function patternMatrix = GeneratePatternMatrix(numCodes,numImages)

    curFolder = sprintf('mian/Patterns/ProjectorPatterns/optimized_patterns_pt01-608-%d-64', numImages);
    allImages = dir(sprintf('%s/*.bmp', curFolder));
    
    num_all = numel(allImages);
    numSteps = num_all / numImages;

    patternMatrix = zeros(numCodes, numImages);

    counter = 1;
    
    for i = 1 : num_all
        curImage = double(imread(sprintf('%s/Pat_%03d.bmp', curFolder, mod(i, num_all))));
        patternMatrix(:, counter) = patternMatrix(:, counter) + curImage(:,1);
    end

    patternMatrix = patternMatrix / (max(patternMatrix(:)));
end