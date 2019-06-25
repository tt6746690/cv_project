function W = BucketMultiplexingMatrix(S)
% Per-pixel bucket multiplexing matrix of size 2(S-1) x S
%
    C = (hadamard(S)+1)/2;
    C = C(2:end,:);
    W = [C; 1-C];
    % take account that the optimal C matrix has 2 1s' on each row
    W = W/2;
end