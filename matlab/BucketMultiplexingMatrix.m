function W = BucketMultiplexingMatrix(S)
% Per-pixel bucket multiplexing matrix of size 2(S-1) x S
%

    switch S
    case 3
        C = [
            1 0 0
            0 1 0
        ];
    % case 4
    %     C = [
    %         1 0 1 0
    %         1 1 0 0
    %         1 0 0 1
    %     ];
    case 5
        C = [
            1 1 0 0 0
            1 0 1 0 0
            1 0 0 1 0
            1 0 0 0 1
        ];
    case 6
        C = [
            1 1 1 0 0 0
            1 1 0 0 1 0
            1 0 1 1 1 0
            1 0 1 0 1 1
            1 0 0 1 0 1
        ];
    case 7
        C = [
            1 1 1 1 1 0 0
            1 1 1 0 0 0 1
            1 1 0 0 1 1 0
            1 0 1 0 1 1 0
            1 0 0 1 0 1 0
            1 0 0 0 1 0 1
        ];
    otherwise
        C = hadamard(S);
        C = (C+1)/2;
        C = C(2:end,:);
    end

    W = [C; 1-C];
end