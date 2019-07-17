function W = BucketMultiplexingMatrix(S)
% Per-pixel bucket multiplexing matrix of size 2(S-1) x S
%
    if S == 7
        C = [
            1 1 1 1 1 0 0
            1 1 1 0 0 0 1
            1 1 0 0 1 1 0
            1 0 1 0 1 1 0
            1 0 0 1 0 1 0
            1 0 0 0 1 0 1
        ];
    else
        C = hadamard(S);
        C = (C+1)/2;
        C = C(2:end,:);
    end

    W = [C; 1-C];
end