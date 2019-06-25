% Creates `H`, opeartor for spatial subsampling according to bayer `bggr` 
%       y = Hx + e
%
function H = BayerSubsamplingOperator(x_size)
    
    [h,w] = deal(x_size(1),x_size(2));
    Hnnz = h*w;

    I = zeros(Hnnz,1);
    J = zeros(Hnnz,1);
    V = zeros(Hnnz,1);

    for i = 1:h
        for j = 1:w
            y_idx = (j-1)*h+i;
            if ( mod(i,2)==1 && mod(j,2)==1 )
                J(y_idx) = (j-1)*h+i;
            elseif ( mod(i,2)==0 && mod(j,2)==0 )
                J(y_idx) = (j-1)*h+i+2*h*w;
            else
                J(y_idx) = (j-1)*h+i+h*w;
            end
            I(y_idx) = y_idx;
            V(y_idx) = 1;
        end
    end

    H = sparse(I,J,V,h*w,3*h*w);
end