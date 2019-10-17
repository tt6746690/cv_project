function [Gx,Gy] = ImGrad(imsize)
    %   Returns a sparse matrix G = [Gx;Gy] \in\R^{2*h*w x h*w} that computes 
    %       discrete forward difference for a 2d image of (h,w) = size

    assert(numel(imsize) == 2);

    h = imsize(1);
    w = imsize(2);

    B = sparse(1:h-1,1:h-1,-1*ones(h-1,1),h,h) + sparse(1:h-1,2:h,1*ones(h-1,1),h,h);
    Gy = kron(speye(w),B);
    
    Gx = kron(spdiags([ones(w-1,1);0],0,w,w),-1*speye(h)) + kron(spdiags(ones(w,1),1,w,w),speye(h));
end