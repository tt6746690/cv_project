function [im_out,psnr_out,ssim_out,statistics,iter_ims] = ADMMSmooth(y,A,InitEstFunc,params,orig_im)
    %   Run ADMM to solve minimize 
    %       E(x) = ||Ax-y||_2^2 + lambda_2*0.5*x'*(x-denoise(x)) + lambda_3*||Grad(x)||_1
    %   
    % Inputs:
    %   y                        input image
    %   A                        multiplexing and spatial subsampling operator
    %   InitEstFunc              initial guess of `y` based on `x`
    %   params.
    %       lambda2              weight to laplacian-based regularizer
    %       lambda3              weight to total variation regularizer
    %       rho                  augmented lagrangian parameter
    %       outer_iters          number of ADMM iterations
    %       inner_denoiser_iters number of fixed iterations in v-minimization step
    %       denoiser_type        denoiser used in v-minimization step
    %       effective_sigma      input noise level to denoiser
    % 
    % Outputs:
    %   im_out                   imputed image
    %   psnr_out                 psnr of `im_out` to `orig_im`
    %   ssim_out                 ssim of `im_out` to `orig_im`
    %   statistics               psnrs at each iterations
 
    QUIET = 0;
    PRINT_MOD = floor(params.outer_iters/10);
    if ~QUIET
        fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR/SSIM', 'objective');
    end

    lambda2 = params.lambda2;
    lambda3 = params.lambda3;
    rho = params.rho;
    outer_iters = params.outer_iters;
    inner_denoiser_iters = params.inner_denoiser_iters;
    denoiser_type = params.denoiser_type;
    effective_sigma = params.effective_sigma;
    v_update_method = params.v_update_method;

    x_init = InitEstFunc(y);
    [h,w,S] = size(x_init);
    hwS = h*w*S;

    l = [hwS hwS hwS*2];
    l = [0 cumsum(l)];

    [Gx,Gy] = ImGrad([h w]);
    G = [Gx;Gy];

    x{1} = x_init(:);
    x{2} = x_init(:);
    x{3} = G*x_init(:);

    z{1} = x{1};
    z{2} = x{2};
    z{3} = x{3};

    u{1} = zeros(size(x{1}));
    u{2} = zeros(size(x{2}));
    u{3} = zeros(size(x{3}));

    X = [x{1};x{2};x{3}];
    Z = [z{1};z{2};z{3}];
    U = [u{1};u{2};u{3}];

    history = zeros(3,0);
    save_iter = 1;

    ToIm = @(x) reshape(x,h,w,[]);
    
    [R,flag] = chol(speye(size(A,2)) + (2/rho)*A'*A);
    if flag ~= 0
        warning("A'A+(rho/2)*I should be symmetric positive definite");
    end

    I = speye(hwS);
    Z = sparse(hwS,hwS);
    H = [
        I -I  Z  Z
        Gx Z -I  Z
        Gy Z  Z -I
    ];

    iter_ims = zeros(5*h,S*w,outer_iters);
    
    for k = 1:outer_iters
        % iter_ims(:,:,k) = 3*FlattenChannels(x_est,v_est,u_est,...
        % Clip(ToIm(R\(R'\(reshape(ToIm(A'*y(:))+rho*(v_est-u_est),[],1)))),0,255)+u_est, ...
        % Denoiser(Clip(ToIm(R\(R'\(reshape(ToIm(A'*y(:))+rho*(v_est-u_est),[],1)))),0,255)+u_est,effective_sigma,denoiser_type));
        % imshow(iter_ims(:,:,k)/255);
        % pause;

        for i = 1:3
            x_old{i} = X( l(i)+1 : l(i+1) );
            z_old{i} = Z( l(i)+1 : l(i+1) );
            u_old{i} = U( l(i)+1 : l(i+1) );
        end

        x1 = (2/rho)*ToIm(A'*y(:))+(z_old{1}-u_old{1});
        x1 = Clip(R\(R'\(x1(:))),0,255);
        x{1} = x1;

        switch v_update_method
        case "fixed_point"
            for j = 1:1:inner_denoiser_iters
                Dx = Denoiser(ToIm(x_old{2}),effective_sigma,denoiser_type);
                x2 = (lambda2*Dx + rho*(ToIm(z_old{2}-u_old{2})))/(lambda2 + rho);
            end
        case "denoiser"
            x2 = Denoiser(ToIm(z_old{2}-u_old{2}),lambda2/rho,denoiser_type);
        otherwise
            warning("v-update method not correct");
        end

        x{2} = x2(:);
        x{3} = SoftShrinkage(z_old{3}-u_old{3},lambda3/rho);

        U_old = [u_old{1};u_old{2};u_old{3}];
        X = [x{1};x{2};x{3}];
        V = X + U_old;
        Z = V - lsqminnorm(H,H*V);

        U = U_old + X - Z;
    
        if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
            im = ToIm(x{1});

            f_est = Denoiser(im,effective_sigma,denoiser_type);
            costfunc = norm(reshape(ToIm(A*im(:))-y,[],1)) + lambda*im(:)'*(im(:)-f_est(:));

            im_out = im(1:size(orig_im,1), 1:size(orig_im,2),:);
            [psnr,ssim] = ComputePSNRSSIM(orig_im, im_out);

            fprintf('%7i %.5f/%.5f %12.5f \n',k,psnr,ssim,costfunc);
            history(:,save_iter) = [psnr;ssim;costfunc];
            save_iter = save_iter + 1;

            imshow(2*FlattenChannels(orig_im,ToIm(x{1}),ToIm(x{2}),ToIm(x{3}))/255);
        end
    end
    
    im = ToIm(x{1});
    im_out = im(1:size(orig_im,1), 1:size(orig_im,2),:);
    im_out = Clip(im,0,255);
    [psnr_out,ssim_out] = ComputePSNRSSIM(orig_im, im_out);

    statistics.psnr     = history(1,:);
    statistics.ssim     = history(2,:);
    statistics.costfunc = history(3,:);
end