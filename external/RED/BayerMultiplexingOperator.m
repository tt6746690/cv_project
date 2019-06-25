function H = BayerMultiplexingOperator(S,x_size)
% vectorized linear operator that maps S full-res images to 2 bucket measurements
%
%   S = 4;
%   x = reshape(1:16,2,2,S);
%   H = BayerMultiplexingOperator(S,[2 2]);
%   y = reshape(H*(x(:)),2,2,2);
%
    P = x_size(1)*x_size(2);
    
    C = (hadamard(S)+1)/2;
    C = C(2:end,:);
    W = [C; 1-C];
    W = W/2;

    B = BayerSubsamplingOperator(x_size);
    B = blkdiag(B,B);
    
    H = B * kron(W,speye(P));
end