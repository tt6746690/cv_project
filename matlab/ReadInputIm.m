function [input_im,input_ratio_im,orig_noisy_im] = ReadInputIm(imagedir,h,w,S,varargin)
    % Reads groundtruth (stacked) images and its ratios
    % 
    %   Assumes the first S images shot in `rawimagedir` are distint subframes
    %
    %   varargin.CropX          crop along x-dim
    %   varargin.CropY          crop along y-dim
    %   varargin.BlackLevel     blacklevel to be subtracted from `input_im`
    %   varargin.ForwardFunc    function from hxwxS -> hxwx2
    %
    %   Returns
    %       - input_im           \in [0,255]
    %       - input_ratio_im     \in [0,255]
    %       - orig_noisy_im      \in [0,255]
    %

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'CropX','CropY','BlackLevel','ForwardFunc','CircShiftInputImageBy'}, ...
        {'cx','cy','blacklevel','ForwardFunc','circshiftby'});
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

    if ~exist('circshiftby','var')
        circshiftby = 0;
    end

    input_im        = zeros(h,w,2);
    input_ratio_im  = zeros(h,w,2);
    orig_noisy_im   = zeros(h,w,S);

    files = dir(sprintf("%s/*.png",imagedir));
    [fnames,ffolders] = deal({files.name},{files.folder});
    folder = ffolders{1};

    for i = 1:S
        fname = fnames{i};
        splits = split(fname,' ');
        [bktno,id] = deal(splits{1},splits{2}); assert(bktno == "bucket1");
        impath = sprintf("%s/%s",folder,fname);
        im = double(BlackLevelRead(impath,blacklevel,1));
        orig_noisy_im(:,:,i) = im(cx,cy);
    end

    orig_noisy_im = circshift(orig_noisy_im,circshiftby,3);

    if exist('ForwardFunc','var')
        input_im = ForwardFunc(orig_noisy_im);
        input_ratio_im = ForwardFunc(IntensityToRatio(orig_noisy_im))*255;
    else
        input_im = [];
        input_ratio_im = [];
    end

end