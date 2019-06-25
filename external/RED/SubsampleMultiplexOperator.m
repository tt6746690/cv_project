function H = SubsampleMultiplexOperator(S,mask)
% vectorized linear operator that maps S full-res images to 2 bucket measurements
%       for a given subsampling mask
%
%   S = 4;
%   x = reshape(1:16,2,2,S);
%   mask = BayerMask(2,2);
%   H = SubsampleMultiplexOperator(S,mask);
%   y = reshape(H*(x(:)),2,2,2);
%
    P = size(mask,1)*size(mask,2);

    W = BucketMultiplexingMatrix(S);
    B = SubsamplingOperator(mask);
    B = blkdiag(B,B);
    
    H = B * kron(W,speye(P));
end