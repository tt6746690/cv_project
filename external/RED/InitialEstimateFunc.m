function InitEstFunc = InitialEstimateFunc(est_type,h,w,F,S,W)
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
%   W is needed only for est_type \in {bayerdemosaic, maxfilter}
%
%   [h,w] = deal(4,4);
%   f = InitialEstimateFunc("zero",h,w,3,4,[]);
%   f(ones(h,w,2);
%

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