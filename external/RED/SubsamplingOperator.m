function H = SubsamplingOperator(mask)
% Create subsampling operator, given any mask
%
%       mask        hxw     integer-valued mask, where
%                           mask[i,j]=k => pixel i,j has value of frame_k
%
%       H           hw x hwK    subsampling operator applied to vec(im)
%
    assert(all(arrayfun(@(x) isInteger(x),mask),'all'), 'Mask should be discrete');
    assert(min(mask,[],'all') == 1, 'Mask should start from 1');

    P = size(mask,1)*size(mask,2);
    K = max(mask,[],'all');

    S = {};
    for k = 1:K
        S{k} = spdiags(double(reshape(mask==k,[],1)),0,P,P);
    end

    H = horzcat(S{:});
end