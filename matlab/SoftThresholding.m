function y = SoftThresholding(x,k)
    %% Soft threshold operator
    %      S_k(x) = sign(x)(|x_i|-\lambda)_+
    %
    y = sign(x).*max(abs(x)-k,0);
end