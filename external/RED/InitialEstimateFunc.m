function InitEstFunc = InitialEstimateFunc(est_type,h,w,F,S,varargin)
%   Define `InitEstFunc` which does demosaic+demultiplex
%       which act as initial guess image to optimization methods
%
%   InitEstFunc: hxwx2 -> hxwxS
%       takes two bucket image, and outputs S reconstructed images
%   
%    est_type
%        zero: function zero initializes the image
%        random: function randomly initializes the image
%        zeroatunknown: function fills zeros for unknown pixel values
%        maxfilter: function does a 3x3 max filter over the image output by `zeroatunknown`
%        bayerdemosaic: do bayer demosaicing then demultiplexing
%
%   varargin
%       'BucketMultiplexingMatrix',W       needed for est_type \in {bayerdemosaic, maxfilter,zeroatunknown}
%       'SubsamplingMask',M                needed for est_type \in {maxfilter,zeroatunknown}
%
%   [h,w] = deal(4,4);
%   f = InitialEstimateFunc("zero",h,w,3,4,[]);
%   f(ones(h,w,2);
%
    assert(S >= (F+1));

    
    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'BucketMultiplexingMatrix','SubsamplingMask'}, ...
        {'W','M'});
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


    switch est_type
    case 'zero'
        InitEstFunc = @(y) zeros(h,w,S);
    case 'random'
        InitEstFunc = @(y) 255*rand(h,w,S);
    case 'zeroatunknown'
        mask = zeros(h,w,F);
        for k = 1:F
            mask(:,:,k) = double(M==k);
        end
        InitEstFunc = @(y) ...
            reshape(...
                reshape(cat(F, mask.*y(:,:,1), mask.*y(:,:,2)),[],2*F) / W', ...
            h,w,S);
    case 'bayerdemosaic'
        assert(F == 3, 'F == 3');
        InitEstFunc = @(y) ...
        reshape(...
            reshape( ...
                cat(3, ...
                    rgb2bgr(double(demosaic(uint8(y(:,:,1)), 'bggr'))), ...
                    rgb2bgr(double(demosaic(uint8(y(:,:,2)), 'bggr')))), ...
                [], 6) ...
            / W', ...
        h,w,S);
    case 'maxfilter'
        mask = zeros(h,w,F);
        for k = 1:F
            mask(:,:,k) = double(M==k);
        end
        max_filtering = @(im) reshape(cell2mat(arrayfun(@(i) ...
            ordfilt2(im(:,:,i),9,ones(3,3)),1:F,'UniformOutput',false)),h,w,[]);
        InitEstFunc = @(y) ...
            reshape(...
                reshape(cat(F, ...
                    max_filtering(mask.*y(:,:,1)), ...
                    max_filtering(mask.*y(:,:,2))),[],2*F) / W', ...
            h,w,S);
    otherwise
        warning('initial estimate function not set properly');
    end
end