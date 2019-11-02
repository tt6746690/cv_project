function [orig_im,orig_ratio_im] = ReadOrigIm(impath,h,w,S,varargin)
    % Reads groundtruth (stacked) images and its ratios
    % 
    %   Assumes there is `S` images with name as follwos
    %       - `{impath}_0.png` ... `{impath}_{S-1}.png`
    %
    %   varargin.CropX      crop along x-dim
    %   varargin.CropY      crop along y-dim
    %
    %   Returns
    %       - orig_im           \in [0,255]
    %       - orig_ratio_im     \in [0,255]
    %
    circshiftby = 0;
    
    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'CropX','CropY','CircShiftInputImageBy'}, ...
        {'cx','cy','circshiftby'});
    v = 1;
    while v <= numel(varargin)
        param_name = varargin{v};
        if isKey(params_to_variables,param_name)
            assert(v+1<=numel(varargin));
            v = v+1;
            % Trick: use feval on anonymous function to use assignin to this workspace
            feval(@()assignin('caller',params_to_variables(param_name),varargin{v}));
        else
            error('Unsupported parameter: %s',varargin{v});
        end
        v=v+1;
    end

    orig_im = zeros(h,w,S);
    orig_ratio_im = zeros(h,w,S);
    
    for s = 1:S
        im = double(imread(sprintf("%s_%d.png",impath,s-1)));
        if ~all([exist('cx','var') exist('cy','var')])
            orig_im(:,:,s) = im;
        else
            orig_im(:,:,s) = im(cx,cy);
        end
    end

    orig_im = circshift(orig_im,circshiftby,3);
    orig_ratio_im = IntensityToRatio(orig_im)*255;
end